from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

# Cargar el modelo
model = load_model('face_validation_model.h5')  # Asegúrate de que este archivo esté en el repositorio

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar si se envió un archivo
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Procesar la imagen
    img = load_img(file, target_size=(128, 128))  # Ajusta el tamaño según tu modelo
    img_array = img_to_array(img) / 255.0  # Normalizar la imagen
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de batch

    # Hacer la predicción
    prediction = model.predict(img_array)
    result = 'Válida' if prediction[0][0] > 0.5 else 'Inválida'  # Ajusta según tu lógica

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))  # Usa el puerto proporcionado por Render