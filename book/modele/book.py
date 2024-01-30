import os
import logging
import tensorflow as tf
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

#loger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Remplacez ceci par le chemin du répertoire de données
TRAINING_DIR = "/home/ludovic/Documents/FAR-planches-brochure"



#nom du model pour l'export
name_model = "/home/ludovic/Documents/FAR-planches-brochure/book.h5"

logger.info("Début de la section ImageDataGenerator")
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

logger.info("Début de la section flow_from_directory")

# Chargement des données d'entraînement
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(2835, 1949), #taille des images en entrée
    class_mode='categorical',
    batch_size=126)

# Nombre de catégories (à ajuster en fonction du nombre de sous répertoire)
nombre_de_categories = len(train_generator.class_indices)

logger.info("Début de la section Construction du modèle")

# Construction du modèle
model = tf.keras.models.Sequential([
    # Couches de convolution et pooling
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # ...
    # Couche Dense finale avec nombre_de_categories neurones
    tf.keras.layers.Dense(nombre_de_categories, activation='softmax')
])

model.summary()


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

logger.info("Entraînement du modèle")

# Entraînement du modèle
history = model.fit(train_generator, epochs=25, steps_per_epoch=20, verbose=1)

logger.info("Sauvegarde du modèle")

# Sauvegarde du modèle
model.save(name_model)

logger.info("destination du fichier:")

logger.info(name_model)