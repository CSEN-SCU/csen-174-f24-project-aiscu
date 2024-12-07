from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import DevOpsMetrics
from django.utils import timezone
import json
from .utils import ask_openai

# Chatbot view to handle user interaction
def chatbot(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        chat_history = json.loads(request.POST.get('chatHistory'))
        print(message)
        print(chat_history)
        response, sources = ask_openai(chat_history, message, request)
        return JsonResponse({'message': message, 'response': response, 'sources': sources})
    
    if request.session.get('index') == 'technology-index':
        return render(request, 'chatbot_technology.html')
    elif request.session.get('index') == 'academic-index':
        return render(request, 'chatbot_academic.html')
    elif request.session.get('index') == 'health-and-safety-index':
        return render(request, 'chatbot_health_safety.html')
    elif request.session.get('index') == 'services-index':
        return render(request, 'chatbot_services.html')
    elif request.session.get('index') == 'general-index':
        return render(request, 'chatbot_general.html')
    else:
        return render(request, 'home.html')

# Index function to set the selected index
def index(request):
    if request.method == 'POST':
        request.session['index'] = request.POST.get('message', 'general-index')
        index_name=request.session['index']

        # DevOp tracks number of times an index/specialized chatbot is used
        old_obj = DevOpsMetrics.objects.filter(chatbot_index=index_name,metric_type='chatbotcounter').last()
        old_value = int(old_obj.metric_value) if old_obj else 0 
        obj = DevOpsMetrics.objects.create(chatbot_index=index_name, metric_type='chatbotcounter', metric_value= old_value+1)
        
    return redirect('chatbot')

def clear(request):
    if request.method == 'POST':
        # Outputs number of user messages + number of chatbot message
        # DevOp that we try to minimize
        index_name = request.session.get('index','general-index')
        length = int(request.POST.get('length', 5))

        old_obj = DevOpsMetrics.objects.filter(chatbot_index=index_name,metric_type='avgchatlength').last()
        total =  DevOpsMetrics.objects.filter(chatbot_index=index_name,metric_type='avgchatlength').count()
        old_value = int(old_obj.metric_value) if old_obj else 0
        obj = DevOpsMetrics.objects.create(chatbot_index=index_name, metric_type='avgchatlength', metric_value=(int(old_value) * total + length)/(total+1))

        request.session.flush()
    return redirect('chatbot')

def home(request):
    return render(request, 'home.html')