from .models import DevOpsMetrics

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
from langchain.llms import OpenAI
from langchain_openai import OpenAIEmbeddings

#for chat history
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
#
#for DevOp measurements
import time

load_dotenv(override=True)

def ask_openai(chat_history, message, request):
    openai_api_key = os.getenv('OPENAI_API_KEY') # Your OpenAI API key here 
    os.environ["OPENAI_API_KEY"] = openai_api_key  # OPENAI API KEY

    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pc = Pinecone(api_key=pinecone_api_key)  # PINECONE API KEY
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    llm=OpenAI()
    
    def split_https(entries):
        result = []
        for entry in entries:
            parts = entry.split('https:')
            for part in parts:
                if part:
                    result.append('https:' + part)
        return result
    
    # Run To Select/Swap Database
    index_name = request.session.get('index','general-index') #(wanted, default fallback)
    index = pc.Index(index_name)

    def create_chain(vectorStore):
        system_prompt = (
            "You are an AI assistant designed to help students at Santa Clara University (SCU) navigate university resources, based on their personal needs."
            "Be friendly, and approachable."
            "Do NOT attempt to guess or complete unfinished questions."
            "If what is being asked of you appears to be incomplete, do not complete it, and instead respond saying it looks incomplete."
            "If you cannot find the answer in the context, say you cannot find it, rather than answer it."
            "Answer based on this context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])
        chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt
        )
        retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

        retrival_prompt = (
            "Given a chat history and the latest user question which might reference the chat history above,"
            "formulate a standalone question which can be understood without the chat history."
            "Do NOT answer the question or attempt to complete it if it looks incomplete."
            "Just reformulate it, if it looks complete, and otherwise return it as is."
        )

        retriever_prompt = ChatPromptTemplate.from_messages([
            ("system", retrival_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
         ])

        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=retriever_prompt
        )
        retrieval_chain = create_retrieval_chain(
            history_aware_retriever,
            chain
        )
        return retrieval_chain

    docsearch = PineconeVectorStore(index=index, embedding=embeddings)
    chain = create_chain(docsearch)

    # Run For User To Interact With Chatbot
    deserial_chat_history = [HumanMessage(content=j) if i%2==0 else AIMessage(content=j) for i,j in enumerate(chat_history)]


    # Passes User_Request to OpenAI
    # DevOp measures the time before this line & time after
    begin = time.time()
    result = chain.invoke({
            "chat_history": deserial_chat_history,
            "input": message,
    })
    end = time.time()
    time_diff = end - begin
    print(f"Time for response time: {time_diff}")

    old_obj = DevOpsMetrics.objects.filter(chatbot_index=index_name,metric_type='avgresponsetime').last()
    total =  DevOpsMetrics.objects.filter(chatbot_index=index_name,metric_type='avgresponsetime').count()
    old_value = float(old_obj.metric_value) if old_obj else 0.0
    obj = DevOpsMetrics.objects.create(chatbot_index=index_name, metric_type='avgresponsetime', metric_value= ((old_value * total) + (time_diff))/(total+1))
    
    sources = [doc.metadata["source"] for doc in result["context"]]
    sources = set(split_https(sources))

    print(result["answer"])
    print(sources)
    
    return result["answer"].split(":", 1)[1] if ':' in result['answer']