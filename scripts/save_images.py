# Exemple avec Flask
from flask import Flask, request, jsonify
from flask_cors import CORS  # Importez CORS depuis Flask-CORS
import base64
import os
import logging
import time

#tensorflow
import tensorflow as tf
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

#convert tensorflow h5 to .json
import subprocess

#generate classes
import json

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

@app.route('/generate-model', methods=['POST'])
def generate_model():

    largest_dimension = 224  # Exemple : réduire la plus grande dimension à 224
    aspect_ratio = 1949 / 2835  # Calcul du ratio d'aspect (hauteur / largeur)

    # Calcul des nouvelles dimensions tout en conservant le ratio d'aspect
    new_width = largest_dimension
    new_height = int(largest_dimension * aspect_ratio)

    print("new dimension")
    print(new_width, new_height)  # Affiche les nouvelles dimensions


    # Remplacez ceci par le chemin du répertoire de données
    TRAINING_DIR = "/home/ludovic/Documents/sites/api-recognition/training_data"

    #nom du model pour l'export
    name_model = "/home/ludovic/Documents/sites/api-recognition/training_data/book.h5"

    # Chemin pour enregistrer le fichier JSON des classes
    classes_json_path = "/home/ludovic/Documents/sites/api-recognition/book/modele/classes.json"


    logging.debug("Début de la section ImageDataGenerator")
    # Utilisez ImageDataGenerator pour prétraiter les images
    training_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    logging.debug("Début de la section flow_from_directory")

    # Chargement des données d'entraînement
    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(new_width, new_height),  # Utilisez les dimensions calculées
        class_mode='categorical',
        batch_size=8)

    # Nombre de catégories (à ajuster en fonction du nombre de sous répertoire)
    nombre_de_categories = len(train_generator.class_indices)

    print("Nombre de catégories :", nombre_de_categories)

    steps_per_epoch = max(1, nombre_de_categories // train_generator.batch_size)

    logging.debug("Début de la section Construction du modèle")

    # Construction du modèle
    model = tf.keras.models.Sequential([
        # Première couche de convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(new_width, new_height, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Ajoutez ici des couches de convolution et de pooling supplémentaires si nécessaire
        
        # Aplatir la sortie pour la connecter à des couches denses
        tf.keras.layers.Flatten(),
        
        # Couche dense, vous pouvez ajuster le nombre de neurones
        tf.keras.layers.Dense(64, activation='relu'),

        # Dernière couche dense, le nombre de neurones doit correspondre au nombre de classes
        tf.keras.layers.Dense(nombre_de_categories, activation='softmax')
    ])

    model.summary()


    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    logging.debug("Entraînement du modèle")

    # Entraînement du modèle
    history = model.fit(train_generator, epochs=25, steps_per_epoch=steps_per_epoch, verbose=1)

    logging.debug("history")

    logging.debug(history)

    logging.debug("Sauvegarde du modèle")

    # Sauvegarde du modèle
    model.save(name_model)

    logging.debug("destination du fichier:")

    logging.debug(name_model)


    #generation des classes


    # Obtenir la liste des noms des sous-répertoires (classes)
    model_classes = [d for d in os.listdir(TRAINING_DIR) if os.path.isdir(os.path.join(TRAINING_DIR, d))]
    model_classes.sort()  # Trié si nécessaire

    # Enregistrer la liste des classes dans un fichier JSON
    with open(classes_json_path, 'w') as json_file:
        json.dump(model_classes, json_file)

    print(f"Fichier de classes généré à {classes_json_path}")

    # TODO A DEBUG

    # command = "sudo tensorflowjs_converter --input_format=keras /home/ludovic/Documents/sites/api-recognition/training_data/book.h5 /home/ludovic/Documents/sites/api-recognition/book/modele"

    # result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # logging.debug(result)

    response_data = {'message': '"Modèle généré avec succès'}
    return jsonify(response_data), 200


if __name__ == '__main__':
    app.run(debug=True)