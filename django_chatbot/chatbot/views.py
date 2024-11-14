from django.shortcuts import render, redirect
from django.http import JsonResponse
import openai
from .models import Counters
from .models import AvgResponseTime
from .models import AvgChatLength
from django.utils import timezone

#for serializing arrays
import json
#
import os
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY') # Your OpenAI API key here 
pinecone_api_key = os.getenv('PINECONE_API_KEY')
openai.api_key = openai_api_key

def ask_openai(chat_history, message, request):
    os.environ["OPENAI_API_KEY"] = openai_api_key  # OPENAI API KEY

    from pinecone import Pinecone
    pc = Pinecone(api_key=pinecone_api_key)  # PINECONE API KEY

    from langchain_pinecone import PineconeVectorStore
    from pinecone import ServerlessSpec
    from langchain.llms import OpenAI
    from langchain_openai import OpenAIEmbeddings

    #for chat history
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain_core.prompts import MessagesPlaceholder
    from langchain.chains.history_aware_retriever import create_history_aware_retriever
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    #
    #for DevOp measurements
    import time
    #
    
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
    index_name = request.session.get('index','langchain-test-index') #(wanted, default fallback)
    index = pc.Index(index_name)

    def create_chain(vectorStore):
        system_prompt = (
            "You are an AI assistant designed to help students at Santa Clara University (SCU) navigate university resources, based on their personal needs." 
            "Your goal is to provide quick, clear, and accurate guidance by suggesting relevant SCU resources."
            "Be friendly, and approachable."
            "Provide specific contacts whenever possible."
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
        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
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
    old_obj = AvgResponseTime.objects.last()
    if old_obj is None:
        old_obj = obj = AvgResponseTime.objects.create()
    else:
        obj = AvgResponseTime.objects.create()
    obj.time = ((old_obj.time * old_obj.total) + (time_diff))/(old_obj.total+1)
    obj.total = old_obj.total + 1
    obj.save()

    sources = [doc.metadata["source"] for doc in result["context"]]
    sources = set(split_https(sources))

    print(result["answer"])
    print(sources)
    print(f"Time for response time: {time_diff}")

    return result["answer"].split(":")[1] if ':' in result['answer'] else result['answer'], list(sources)

# Chatbot view to handle user interaction
def chatbot(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        chat_history = json.loads(request.POST.get('chatHistory'))
        print(message)
        print(chat_history)
        response, sources = ask_openai(chat_history, message, request)
        return JsonResponse({'message': message, 'response': response, 'sources': sources})
    return render(request, 'chatbot.html')

# Index function to set the selected index
def index(request):
    if request.method == 'POST':
        request.session['index'] = request.POST.get('message', 'langchain-test-index')

        # DevOp tracks number of times an index/specialized chatbot is used
        obj, _ = Counters.objects.get_or_create(
            index_name=request.session['index']
        )
        obj.counter+=1
        obj.save()
    return redirect('chatbot')

def clear(request):
    if request.method == 'POST':
        # Outputs number of user messages + number of chatbot message
        # DevOp that we try to minimize
        length = int(request.POST.get('length', 5))
        old_obj = AvgChatLength.objects.last()
        if old_obj is None:
            old_obj = obj = AvgChatLength.objects.create()
        else:
            obj = AvgChatLength.objects.create()
        obj.length = (old_obj.length * old_obj.total + length)/(old_obj.total+1)
        obj.total = old_obj.total + 1
        obj.save()
        request.session.flush()
    return redirect('chatbot')

def home(request):
    return render(request, 'home.html')