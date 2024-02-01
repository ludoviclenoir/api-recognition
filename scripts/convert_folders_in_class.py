import os
import json

# Chemin du répertoire contenant vos données d'entraînement
TRAINING_DIR = "/Users/macpro3/Downloads/FAR-planches-brochure"

# Obtenir la liste des noms des sous-répertoires (classes)
model_classes = [d for d in os.listdir(TRAINING_DIR) if os.path.isdir(os.path.join(TRAINING_DIR, d))]
model_classes.sort()  # Trié si nécessaire

# Chemin pour enregistrer le fichier JSON des classes
classes_json_path = "/Users/macpro3/sites/api-recognition/book/modele/classes.json"

# Enregistrer la liste des classes dans un fichier JSON
with open(classes_json_path, 'w') as json_file:
    json.dump(model_classes, json_file)

print(f"Fichier de classes généré à {classes_json_path}")
