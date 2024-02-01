# Exemple avec Flask
from flask import Flask, request
import base64
import os

app = Flask(__name__)

@app.route('/save-image', methods=['POST'])
def save_image():
    data = request.json
    image_data = data['imageData']
    class_name = data['className']

    # Convertir les données d'image de base64 en image
    image_data = base64.b64decode(image_data.split(',')[1])
    image_path = os.path.join('training_data', class_name, 'image.png')  # Chemin à ajuster

    # Sauvegarder l'image
    with open(image_path, 'wb') as f:
        f.write(image_data)

    return "Image sauvegardée", 200

if __name__ == '__main__':
    app.run(debug=True)