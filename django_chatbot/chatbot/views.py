from django.shortcuts import render, redirect
from django.http import JsonResponse
import openai
from django.contrib.auth.decorators import login_required
from django.contrib import auth
from django.contrib.auth.models import User
from .models import Chat
from .models import Counters
from .models import AvgResponseTime
from .models import AvgChatLength
from django.utils import timezone

#for serializing arrays
import json
#

openai_api_key = ""  # Your OpenAI API key here 
openai.api_key = openai_api_key

def ask_openai(message, request):
    import os
    os.environ["OPENAI_API_KEY"] = ""  # OPENAI API KEY

    from pinecone import Pinecone
    pc = Pinecone(api_key="")  # PINECONE API KEY

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
    
    # DevOp tracks number of times an index/specialized chatbot is used
    prev_index_name = request.session['prev_index'] 
    if prev_index_name != index_name:
        obj, created = Counters.objects.get_or_create(
            index_name=prev_index_name
        )
        obj.counter+=1
        obj.save()
        print(f"{prev_index_name}: {obj.counter}")

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

    # Run For User To Interact With Chatbot
    chat_history = json.loads(request.session.get('chat_history', '[]'))
    chain = create_chain(docsearch)

    deserial_chat_history = [HumanMessage(content=j) if i%2==0 else AIMessage(content=j) for i,j in enumerate(chat_history)]

    # Passes User_Request to OpenAI
    # DevOp measures the time before this line & time after
    begin = time.time()
    result = chain.invoke({
            "chat_history": deserial_chat_history[-6:],
            "input": message,
    })
    end = time.time()
    time_diff = end - begin
    print(f"Time for response time: {time_diff}")
    obj = AvgResponseTime.objects.first()
    if obj is None:
        obj = AvgResponseTime.objects.create(time=0,total=0)
    obj.time = ((obj.time * obj.total) + (time_diff))/(obj.total+1)
    obj.total += 1
    obj.save()

    print(result["answer"])
    sources = [doc.metadata["source"] for doc in result["context"]]
    sources = set(split_https(sources))
    print(sources)

    chat_history.append(message)
    chat_history.append(result["answer"].split(":")[1] if ':' in result['answer'] else result['answer'])
    request.session['chat_history'] = json.dumps(chat_history)

    # Outputs number of user messages + number of chatbot message
    # DevOp that we try to minimize
    length = len(chat_history)
    if prev_index_name != index_name:
        obj = AvgChatLength.objects.first()
        if obj is None:
            obj = AvgChatLength.objects.create(length=0,total=0)
        obj.length = ((obj.length * obj.total) + (length))/(obj.total+1)
        obj.total += 1
        obj.save()
        print(f"Current chat length: {length}")

    return result["answer"].split(":")[1] if ':' in result['answer'] else result['answer'], json.dumps(list(sources))

# Chatbot view to handle user interaction
def chatbot(request):
    chats = Chat.objects.filter(user=request.user)

    if request.method == 'POST':
        message = request.POST.get('message')
        response, sources = ask_openai(message, request)

        sources = json.loads(sources)
        #print(sources)

        chat = Chat(user=request.user, message=message, response=response, created_at=timezone.now())
        chat.save()

        return JsonResponse({'message': message, 'response': response, 'sources': sources})

    return render(request, 'chatbot.html', {'chats': chats})



# Index function to set the selected index
def index(request):
    if request.method == 'POST':
        request.session['prev_index'] = request.session.get('index','langchain-test-index')
        request.session['index'] = request.POST.get('message')
        request.session['chat_history'] = '[]'
    return redirect('chatbot')

def clear(request):
    if request.method == 'GET':
        Chat.objects.all().delete()
    return redirect('chatbot')

# Other unchanged views...
def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('chatbot')
        else:
            error_message = 'Invalid username or password'
            return render(request, 'login.html', {'error_message': error_message})
    return render(request, 'login.html')


def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 == password2:
            try:
                user = User.objects.create_user(username, email, password1)
                user.save()
                auth.login(request, user)
                return redirect('chatbot')
            except:
                error_message = 'Error creating account'
                return render(request, 'register.html', {'error_message': error_message})
        else:
            error_message = 'Passwords do not match'
            return render(request, 'register.html', {'error_message': error_message})
    return render(request, 'register.html')


def logout(request):
    auth.logout(request)
    return redirect('login')


def home(request):
    return render(request, 'home.html')


#end of new code


    # Run To Select/Swap Database
    # select = 0
    # while select != 1 and select != 2:
    #     print("[1] Tutoring")
    #     print("[2] Safety")
    #     select = int(input("Which do you need help with?"))
    #     if select == 2:
    #         index_name = "safety-test-index"
    #     elif select == 1:
    #         index_name = "tutor-test-index"
   

    
    # Run For User To Interact With Chatbot
'''
    while True:
    query = input("What do you need help with?")
    if query.lower() == "exit":
        break
    result = qa_with_sources({"query": query})
    print(result["result"])
    sources = [doc.metadata["source"] for doc in result["source_documents"]]
    def split_https(entries):
        result = []
        for entry in entries:
            parts = entry.split('https:')
            for part in parts:
                if part:
                    result.append('https:' + part)
        return result
    sources = set(split_https(sources))
    print(sources)
'''

'''
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ]
    )
    answer = response.choices[0].message['content'].strip()  # Access message content correctly
    return answer
'''

