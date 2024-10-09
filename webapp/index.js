import express from 'express';
import { createServer } from 'node:http';
import { resolve } from 'node:path';
import { Server } from 'socket.io';
import sqlite3 from 'sqlite3';
import { open } from 'sqlite';

import { spawn } from 'child_process';

const db = await open({
  filename: 'chatbot.db',
  driver: sqlite3.Database
});

await db.exec(`
  CREATE TABLE IF NOT EXISTS q_and_a (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_sequence_number TEXT UNIQUE,
    query TEXT,
    answer TEXT
  );
`);

const app = express();
const server = createServer(app);
const io = new Server(server, {
  connectionStateRecovery: {},
});

app.get('/', (req, res) => {
  res.sendFile(resolve('index.html'));
});

app.delete('/clear', async (req,res) => {
    try {
        await db.run('DELETE FROM q_and_a');
        await db.run('DELETE from sqlite_sequence where name=\'q_and_a\'');
        console.log('Clearing messages...');
        res.sendStatus(204);
    } catch (e) {
        res.sendStatus(404);
        console.log('Message clearing failed.');
    }
})

io.on('connection', async (socket) => {
  socket.on('query', async (query, clientSequenceNumber, callback) => {
    try {
      let res;
      res = await db.run('INSERT INTO q_and_a (query, client_sequence_number) VALUES (?, ?)', query, clientSequenceNumber);
      console.log(`LAST ID: ${res.lastID}`);
      io.emit('query', query, res.lastID);

      let chatbot;
      chatbot = spawn('/opt/homebrew/Caskroom/miniconda/base/envs/webapp/bin/python', ['aiscu.py', query]);
      console.log(`QUERY RECEIVED. PROCESSING...`);
      chatbot.stdout.on('data', async (answer) => {
        io.emit('answer', `${answer}`);
        await db.run('UPDATE q_and_a SET answer = ? WHERE id = ?', `${answer}`, res.lastID);
      })
      callback();
    } catch (e) {
      if (e.errno === 19) { //SQLITE_CONSTRAINT violated, e.g. UNIQUE constraint on client_sequence_number
        callback();
      } else {
        // make client try again, do nothing
      }
      return;
    }
  });

  if (!socket.recovered) {
    try {
      await db.each('SELECT id, query, answer FROM q_and_a WHERE id > ?',
        [socket.handshake.auth.serverSequenceNumber || 0],
        (_err, row) => {
          socket.emit('query', row.query, row.id);
          socket.emit('answer', row.answer);
        }
      )
    } catch (e) {
      // error
    }
  }
});

server.listen(3000, () => {
  console.log(`server running at http://localhost:3000`);
});