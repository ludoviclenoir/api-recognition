import os
import logging
import tensorflow as tf
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

#loger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

largest_dimension = 224  # Exemple : réduire la plus grande dimension à 224
aspect_ratio = 1949 / 2835  # Calcul du ratio d'aspect (hauteur / largeur)

# Calcul des nouvelles dimensions tout en conservant le ratio d'aspect
new_width = largest_dimension
new_height = int(largest_dimension * aspect_ratio)

print("new dimension")
print(new_width, new_height)  # Affiche les nouvelles dimensions


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
    target_size=(new_width, new_height),  # Utilisez les dimensions calculées
    class_mode='categorical',
    batch_size=8)

# Nombre de catégories (à ajuster en fonction du nombre de sous répertoire)
nombre_de_categories = len(train_generator.class_indices)

print("Nombre de catégories :", nombre_de_categories)

steps_per_epoch = max(1, nombre_de_categories // train_generator.batch_size)

logger.info("Début de la section Construction du modèle")

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

logger.info("Entraînement du modèle")

# Entraînement du modèle
history = model.fit(train_generator, epochs=25, steps_per_epoch=steps_per_epoch, verbose=1)

logger.info("history")

logger.info(history)

logger.info("Sauvegarde du modèle")

# Sauvegarde du modèle
model.save(name_model)

logger.info("destination du fichier:")

logger.info(name_model)