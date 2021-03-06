from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import os
from medic_hero4.settings import BASE_DIR
from django.http import JsonResponse
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random

model = load_model("assets\\chat_files\\chatbot_model.h5")
intents = json.loads(open("assets\\chat_files\\intents.json").read())
words = pickle.load(open("assets\\chat_files\\words.pkl",'rb'))
classes = pickle.load(open("assets\\chat_files\\classes.pkl",'rb'))

model2 = load_model("assets\\chat_files\\chatbot_medic_model.h5")
intents2 = json.loads(open("assets\\chat_files\\symp.json").read())
words2 = pickle.load(open("assets\\chat_files\\words2.pkl",'rb'))
classes2 = pickle.load(open("assets\\chat_files\\classes2.pkl",'rb'))

context=1
dis=""
user=[]
yes = ["yes","Yes","YES","y","Y"]
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))
    
def bow2(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words2)  
    for s in sentence_words:
        for i,w in enumerate(words2):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
    
def predict_class2(sentence, model):
    # filter out predictions below a threshold
    p = bow2(sentence, words2,show_details=False)
    res = model2.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes2[r[0]], "probability": str(r[1])})
    return return_list

def get_remedy(ints,intents_json, symp, msg):
    global dis
    global context
    tag=dis
    list_of_intents = symp['intents']
    context=1
    for i in list_of_intents:
        if(i['tag']== tag):
            if(msg in yes):
                result = " "
                result = random.choice(i['remedy'])
            else:
                result = "Have a nice day."
            break
    return result
    
def get_med_Response_disease(symp, msg):
    global dis
    global context
    tag=dis
    list_of_intents = symp['intents']
    context=1
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            context=5
            break
    return result
    
def get_med_Response_temp(msg):    
    global dis
    global context
    context=4
    user.append(msg)
    fever=0
    if(dis=="Common cold"):
        for mesg in user:
            for wrd in mesg.split():
                if(wrd.isdigit()):
                    fever=int(wrd)
            
        if(fever!=0):
            reply="you got "+str(fever)+" fever ?"
        else:
            reply = "Is your body temperature above 98 ?"
    else:
        reply=get_med_Response_disease(intents2, msg)
    return reply
    
   
    
def get_med_Response(ints,intents_json, symp, msg):
    global context
    global dis
    tag = ints[0]['intent']
    list_of_intents = symp['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            context=3
            dis=tag                
            result = random.choice(i['responses'])
            break
    return result

def getResponse(ints, intents_json,symp, msg):
    global context
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            if(tag=="diagonised"):
                context=2
                ints2 = predict_class2(msg, model2)
                result = get_med_Response(ints2, intents_json,symp, msg)     
                result = "any more symptomes?"
                break
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(ctx,msg):
    global user
    global context
    global dis
    context = ctx
    user.append(msg)    
    if(ctx==1):
        ints = predict_class(msg, model)
        res = getResponse(ints, intents,intents2,msg)
    elif (ctx==2):
        ints = predict_class2(msg,model2)
        res = get_med_Response(ints,intents,intents2, msg) 
    elif(ctx==3):        
        res = get_med_Response_temp(msg)
    elif(ctx==4):
        ints = predict_class2(msg,model2)
        res = get_med_Response_disease(intents2, msg)
    elif(ctx==5):
        ints = predict_class2(msg,model2)
        res = get_remedy(ints,intents,intents2, msg)        
    return res


# Create your views here.

@csrf_exempt
def reply(request):
    global context    
    x = request.POST["msg"]
    ctx = request.POST["ctx"]
    ctx = int(ctx)
    resp = chatbot_response(ctx,x)
    return HttpResponse(json.dumps({"m":resp,"c":context}),content_type='application/json')


def home(request):
    return render(request,"index.html")


def chat(request):    
    return render(request,"chat.html")