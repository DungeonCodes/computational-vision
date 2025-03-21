from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import face_recognition

app = Flask(__name__)
CORS(app)  # Permite acesso externo (útil para integração com outras apps)

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'API de Reconhecimento Facial está online!'})

@app.route('/recognize', methods=['POST'])
def recognize_face():
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400

    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # Detecta rostos (versão básica, sem banco de dados ainda)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    if not face_locations:
        return jsonify({'message': 'Nenhum rosto detectado.'})

    # (Placeholder) Aqui retornamos apenas a quantidade de rostos
    return jsonify({
        'message': 'Rostos detectados!',
        'quantidade': len(face_locations)
    })

if __name__ == '__main__':
    app.run(debug=True)
