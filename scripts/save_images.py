# Exemple avec Flask
from flask import Flask, request, jsonify
from flask_cors import CORS  # Importez CORS depuis Flask-CORS
import base64
import os
import logging
import time
import datetime

#tensorflow
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

#Mobilnet
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

#convert tensorflow h5 to .json
import subprocess

#generate classes
import json

#visualisation du modèle
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend sans GUI

import matplotlib.pyplot as plt
import numpy as np

# Configurez les paramètres de logging
logging.basicConfig(
    filename='capture.log',  # Nom du fichier de log
    level=logging.DEBUG,  # Niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Format du message de log
)

logging.debug('passage par ici')

app = Flask(__name__)
CORS(app)  # Activez CORS pour votre application Flask

# Définir le chemin du répertoire de logs TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

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

# Fonction pour charger et adapter le modèle MobileNetV2
def load_pretrained_model(num_classes):
    # Chargement du modèle MobileNetV2 pré-entraîné sans les couches fully-connected
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Congelez les couches du modèle pré-entraîné pour empêcher l'entraînement
    for layer in base_model.layers:
        layer.trainable = False
    
    # Ajoutez des couches fully-connected personnalisées adaptées à votre nombre de classes
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Ajoutez une couche de pooling
    x = Dense(512, activation='relu')(x)  # Ajoutez une couche fully-connected
    predictions = Dense(num_classes, activation='softmax')(x)  # Couche de sortie avec softmax pour la classification
    
    # Créez le modèle final en spécifiant les entrées et les sorties
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    
    return model

# Vos opérations de génération de graphiques
def generate_plots(history):
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

@app.route('/generate-model', methods=['POST'])
def generate_model():
    try:
        largest_dimension = 224  # Exemple : réduire la plus grande dimension à 224
        aspect_ratio = 1949 / 2835  # Calcul du ratio d'aspect (hauteur / largeur)

        # Calcul des nouvelles dimensions tout en conservant le ratio d'aspect
        new_width = largest_dimension
        new_height = int(largest_dimension * aspect_ratio)

        # Remplacez ceci par le chemin du répertoire de données
        TRAINING_DIR = "/home/ludovic/Documents/sites/api-recognition/training_data"

        name_model = "/home/ludovic/Documents/sites/api-recognition/training_data/book.h5"

        # Chemin pour enregistrer le fichier JSON des classes
        classes_json_path = "/home/ludovic/Documents/sites/api-recognition/book/modele/classes.json"

        # Obtenez une liste de tous les éléments dans le répertoire
        all_items = os.listdir(TRAINING_DIR)

        # Filtrez les dossiers uniquement
        folders = [item for item in all_items if os.path.isdir(os.path.join(TRAINING_DIR, item))]
        num_classes = len(folders)

        logging.debug("nombres de dossier:")
        logging.debug(num_classes)

        # logging.debug("nombres de dossier:")
        # logging.debug(folders)

        # Chargement du modèle pré-entraîné
        model = load_pretrained_model(num_classes)

        # Configurer TensorBoard pour surveiller les métriques d'entraînement
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

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

        # Chargement des données d'entraînement
        train_generator = training_datagen.flow_from_directory(
            TRAINING_DIR,
            target_size=(new_width, new_height),
            class_mode='categorical',
            batch_size=32)

        # Entraînement du modèle
        history = model.fit(train_generator, epochs=25, verbose=1, callbacks=[tensorboard_callback])

        # Générer et afficher les graphiques d'entraînement
        generate_plots(history)

        # Sauvegarde du modèle
        model.save(name_model)
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
    
    except Exception as e:
        logging.error(str(e))
        return "Une erreur s'est produite lors de la génération du modèle", 500


if __name__ == '__main__':
    app.run(debug=True)