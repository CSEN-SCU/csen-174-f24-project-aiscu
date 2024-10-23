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

'''
# Needs to be run only once as a setup
import os
# os.environ["OPENAI_API_KEY"] = "INSERT_OPENAI_API_KEY"
from pinecone import Pinecone
# pc = Pinecone(api_key="INSERT_PINECONE_API_KEY")
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
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
# Run To Select/Swap Database
# Potential DevOp is tracking number of times an index/specialized chatbot is used
select = 0
while select != 1 and select != 2:
  print("[1] Tutoring")
  print("[2] Safety")
  select = int(input("Which do you need help with?"))
if select == 2:
  index_name = "safety-test-index"
elif select == 1:
  index_name = "tutor-test-index"
index = pc.Index(index_name)
docsearch = PineconeVectorStore(index=index, embedding=embeddings)
# Run For User To Interact With Chatbot
chat_history = []
chain = create_chain(docsearch)
while True:
  query = input("What do you need help with?")
  if query.lower() == "exit":
    break
    # Passes User_Request to OpenAI
    # DevOp that can be measured is time before this line & time after
  result = chain.invoke({
        "chat_history": chat_history,
        "input": query,
  })
  print(result["answer"])
  sources = [doc.metadata["source"] for doc in result["context"]]
  sources = set(split_https(sources))
  print(sources)
  chat_history.append(HumanMessage(content=query))
  chat_history.append(AIMessage(content=result["answer"]))
# Outputs number of user messages + number of chatbot message
# DevOp that we try to minimize
print(len(chat_history))
'''

def ask_openai(message, request):
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
    # select = 0
    # while select != 1 and select != 2:
    #     print("[1] Tutoring")
    #     print("[2] Safety")
    #     select = int(input("Which do you need help with?"))
    #     if select == 2:
    #         index_name = "safety-test-index"
    #     elif select == 1:
    #         index_name = "tutor-test-index"
   
    #index_name = "langchain-test-index"  # change if desired
    index_name = request.session.get('index') #(wanted, default fallback)
    index = pc.Index(index_name)
    docsearch = PineconeVectorStore(index=index, embedding=embeddings)

    qa_with_sources = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)

    
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
    

    import sys
    query = sys.argv[1] #"I have a Math 11 midterm coming up. Is there any place I can get help?"#

    result = qa_with_sources({"query": message})


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
        response = ask_openai(message, request)

        chat = Chat(user=request.user, message=message, response=response, created_at=timezone.now())
        chat.save()
        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chatbot.html', {'chats': chats})

def index(request):
    if request.method == 'POST':
        request.session['index'] = request.POST.get('message')
    return redirect('chatbot')

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