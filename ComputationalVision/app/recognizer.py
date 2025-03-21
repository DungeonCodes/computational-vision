from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import os
from app.recognizer import FaceRecognizer

app = Flask(__name__)
CORS(app)  # Permite requisições de outros domínios (útil para frontend)

# Inicializa o reconhecedor facial
recognizer = FaceRecognizer()

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'API de Reconhecimento Facial está online!'})

@app.route('/train', methods=['POST'])
def train_faces():
    """Endpoint para treinar/recalcular embeddings faciais"""
    recognizer.encode_faces()
    return jsonify({'message': 'Base de rostos treinada com sucesso!'})

@app.route('/recognize', methods=['POST'])
def recognize_face():
    """Recebe uma imagem e tenta reconhecer um rosto"""
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400

    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # Converte imagem para formato de reconhecimento
    temp_image_path = "temp.jpg"
    cv2.imwrite(temp_image_path, frame)

    result = recognizer.recognize(temp_image_path)

    # Remove a imagem temporária após o reconhecimento
    os.remove(temp_image_path)

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
