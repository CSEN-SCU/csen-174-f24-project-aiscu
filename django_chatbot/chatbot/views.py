from django.shortcuts import render, redirect
from django.http import JsonResponse
import openai
from django.contrib.auth.decorators import login_required
from django.contrib import auth
from django.contrib.auth.models import User
from .models import Chat

from django.utils import timezone



openai_api_key = "" #put your Openai API key here 
openai.api_key = openai_api_key


def ask_openai(message):
    # Needs to be run only once as a setup
    import os
    os.environ["OPENAI_API_KEY"] = "" #OPENAI API KEY#

    from pinecone import Pinecone
    pc = Pinecone(api_key="") #PINECONE API KEY#

    from langchain_pinecone import PineconeVectorStore
    from pinecone import ServerlessSpec
    from langchain.chains import RetrievalQA
    from langchain.llms import OpenAI
    from langchain_openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    llm=OpenAI()

    
    # Run To Select/Swap Database
    select = 0
    while select != 1 and select != 2:
    print("[1] Tutoring")
    print("[2] Safety")
    select = int(input("Which do you need help with?"))
    if select == 2:
        index_name = "safety-test-index"
    elif select == 1:
        index_name = "tutor-test-index"
    
    #index_name = "langchain-test-index"  # change if desired
    index = pc.Index(index_name)
    docsearch = PineconeVectorStore(index=index, embedding=embeddings)

    qa_with_sources = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)

    '''
    # Run For User To Interact With Chatbot
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

    import sys
    query = sys.argv[1] #"I have a Math 11 midterm coming up. Is there any place I can get help?"#

    result = qa_with_sources({"query": message})
    return result["result"]
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


# Create your views here.
@login_required(login_url='login')  #ensure only logged-in users can access the chatbot
def chatbot(request):
    chats = Chat.objects.filter(user=request.user)

    if request.method == 'POST':
        message = request.POST.get('message')
        response = ask_openai(message)

        chat = Chat(user=request.user, message=message, response=response, created_at=timezone.now())
        chat.save()
        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chatbot.html', {'chats': chats})


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
    else:
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
            error_message = 'Password dont match'
            return render(request, 'register.html', {'error_message': error_message})
    return render(request, 'register.html')

def logout(request):
    auth.logout(request)
    return redirect('login')

def home(request):
    return render(request, 'home.html')
