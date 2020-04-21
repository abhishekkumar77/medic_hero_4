import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import json
import random
from keras.models import load_model

lemmatizer = WordNetLemmatizer()

model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

model2 = load_model('chatbot_medic_model.h5')
intents2 = json.loads(open('symp.json').read())
words2 = pickle.load(open('words2.pkl','rb'))
classes2 = pickle.load(open('classes2.pkl','rb'))

context=1
dis=""
user=[]

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

def get_med_Response_disease(ints,intents_json, symp, msg):
    global dis
    global context
    #tag = ints[0]['intent']
    tag=dis
    list_of_intents = symp['intents']
    context=1
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result
    
def get_med_Response_temp(msg):    
    global dis
    global context
    context=4
    user.append(msg)
    fever=0
    for mesg in user:
        for wrd in mesg.split():
            if(wrd.isdigit()):
                fever=int(wrd)
            
    if(fever!=0):
        reply="you got "+str(fever)+" fever ?"
    else:
        reply = "Is your body temperature above 98 ?"
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
                result = "any more known symptomes?"
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
        #ints = predict_class2(msg,model2)
        #res = get_med_Response_temp(ints,intents,intents2, msg)
        res = get_med_Response_temp(msg)
    elif(ctx==4):
        ints = predict_class2(msg,model2)
        res = get_med_Response_disease(ints,intents,intents2, msg)
        
    return res




while(1):
    con = int(input("Context:"))
    msg = str(input("Msg:"))
    response = chatbot_response(con,msg)
    c = str(context)
    print("Context:"+c+"  Bot:"+response)



#---]medichero[---#

