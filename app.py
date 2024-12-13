# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from chatbot import predict_class, get_response, respuesta  # Asegúrate de que existan estas funciones en chatbot.py

app = Flask(__name__)
CORS(app)  # Habilitar CORS para que React pueda acceder al backend

@app.route("/api/chat", methods=["POST"])
def chat():
    message = request.json.get('message')  # Obtén el mensaje del frontend
    if not message:
        return jsonify({"response": "Por favor envía un mensaje válido."}), 400
    print(f"Mensaje recibido: {message}")  # Verifica si el mensaje llega bien
    response = respuesta(message)  # Llama a tu función respuesta
    print(f"Respuesta generada: {response}")  # Verifica si la respuesta es correcta
    return jsonify({"response": response}), 200

if __name__ == "__main__":
    app.run(debug=True)
