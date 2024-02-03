# Exemple avec Flask

from flask import Flask, request
from flask_cors import CORS  # Importez CORS depuis Flask-CORS
import base64
import os
import logging
import time

# Configurez les paramètres de logging
logging.basicConfig(
    filename='capture.log',  # Nom du fichier de log
    level=logging.DEBUG,  # Niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Format du message de log
)

logging.debug('passage par ici')

app = Flask(__name__)
CORS(app)  # Activez CORS pour votre application Flask

@app.route('/save-image', methods=['POST'])
def save_image():
    try:
        data = request.json

        logging.debug('passage route')
        
        image_data = data['imageData']
        class_name = data['className']

        # Créez le répertoire s'il n'existe pas
        image_dir = os.path.join('training_data', class_name)
        os.makedirs(image_dir, exist_ok=True)

        logging.debug("image_dir")
        logging.debug(image_dir)

        # Convertir les données d'image de base64 en image
        image_data = base64.b64decode(image_data.split(',')[1])
        # Utilisez un timestamp pour rendre le nom de fichier unique
        timestamp = str(int(time.time() * 1000))
        image_path = os.path.join(image_dir, f'image_{timestamp}.png')

        logging.debug("image_path")
        logging.debug(image_path)

        # Sauvegarder l'image
        with open(image_path, 'wb') as f:
            f.write(image_data)

        # Retournez une réponse JSON indiquant que l'image a été sauvegardée
        response_data = {'message': 'Image sauvegardée'}
        return jsonify(response_data), 200
    
    except Exception as e:
        logging.error(str(e))
        return "Une erreur s'est produite", 500

if __name__ == '__main__':
    app.run(debug=True)