import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, jsonify, request
from keras.models import load_model
from flask_cors import CORS
from fuzzywuzzy import fuzz

# Inicialización
app = Flask(__name__)
CORS(app)
lemmatizer = WordNetLemmatizer()

# Cargar archivos
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Normalización
def normalize_text(text):
    return ''.join(char.lower() for char in text if char.isalnum() or char.isspace()).strip()

# Procesamiento de oración
def clean_up_sentence(sentence):
    sentence = normalize_text(sentence)
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words if word.isalnum()]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, threshold=0.3):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.argmax(res)
    confidence = res[max_index]
    if confidence > threshold:
        print(f"Predicción aceptada con confianza {confidence}: {classes[max_index]}")
        return classes[max_index]
    else:
        print(f"Confianza insuficiente: {confidence}")
        return None

# Coincidencia parcial
def match_intent(user_message):
    best_match = {"tag": None, "score": 0}
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            score = fuzz.ratio(normalize_text(user_message), normalize_text(pattern))
            if score > best_match["score"]:
                best_match = {"tag": intent["tag"], "score": score}
    
    print(f"Coincidencia parcial para '{user_message}': {best_match}")
    if best_match["score"] > 70:  # Ajusta según sea necesario
        return next(intent for intent in intents["intents"] if intent["tag"] == best_match["tag"])
    return None

def respuesta(message):
    print(f"Mensaje recibido: {message}")
    tag = predict_class(message)
    if not tag:
        matched_intent = match_intent(message)
        if matched_intent:
            tag = matched_intent["tag"]
    
    if not tag:
        print("No se encontró ninguna coincidencia.")
        return "Lo siento, no entiendo tu mensaje. ¿Puedes reformularlo?"
    
    return get_response(tag, intents)

def get_response(tag, intents_json):
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Lo siento, no puedo procesar tu solicitud en este momento."

@app.route("/api/chat", methods=["POST"])
def chat():
    message = request.json.get('message')
    response = respuesta(message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
