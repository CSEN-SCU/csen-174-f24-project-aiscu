import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

#Avg Chat Length
conn = sqlite3.connect('db.sqlite3')
query = "SELECT * FROM chatbot_avgchatlength"
df = pd.read_sql_query(query, conn)
conn.close()

df['created_at'] = pd.to_datetime(df['created_at'])
df['created_at'] = df['created_at'].dt.strftime('%b %d\n%I:%M:%S %p')

ax=df.plot(kind='bar', x='created_at', y='length',legend=False)
ax.set_xlabel("Date")
ax.set_ylabel("Average Chat Length")
ax.set_title("Average Chat Length Over Time")
plt.xticks(rotation=45, ha='center')  

plt.tight_layout()  
plt.show()

#Avg Response Time#
conn = sqlite3.connect('db.sqlite3')
query = "SELECT * FROM chatbot_avgresponsetime"
df = pd.read_sql_query(query, conn)
conn.close()

df['created_at'] = pd.to_datetime(df['created_at'])
df['created_at'] = df['created_at'].dt.strftime('%b %d\n%I:%M:%S %p')

ax=df.plot(kind='bar', x='created_at', y='time',legend=False)
ax.set_xlabel("Date")
ax.set_ylabel("Average Response Time")
ax.set_title("Average Response Time Over Time")
plt.xticks(rotation=45, ha='center')  

plt.tight_layout()  
plt.show()


conn = sqlite3.connect('db.sqlite3')
query = "SELECT * FROM chatbot_counters"
df = pd.read_sql_query(query, conn)
conn.close()

df['created_at'] = pd.to_datetime(df['created_at'])
df['created_at'] = df['created_at'].dt.strftime('%b %d\n%I:%M:%S %p')

ax=df.plot(kind='bar', x='created_at', y='counter',legend=False)
ax.set_xlabel("Date")
ax.set_ylabel("Chat Usage Frequency")
ax.set_title("Chatbot Usage Frequency Over Time")
plt.xticks(rotation=45, ha='center')  

plt.tight_layout()  
plt.show()