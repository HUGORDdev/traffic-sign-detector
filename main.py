from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# chargement du fichier.json
with open('Labels_map.json','r',encoding='utf-8')as f:
    labels_map = json.load(f)
# chargement du templates
templates = Jinja2Templates(directory="templates")
# Charger ton modèle
model = tf.keras.models.load_model("modele_traffic.hdf5")

# Constantes
IMG_HEIGHT, IMG_WIDTH = 30, 30
NUM_CATEGORIES = 43

# Créer ton app FastAPI
app = FastAPI(title="Traffic Sign Recognition API")

# Endpoint pour tester si l'API fonctionne 
@app.get("/",response_class=HTMLResponse)
def home(request:Request):
    return templates.TemplateResponse('index.html',{'request':request})

# Endpoint pour recevoir une image et faire une prédiction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lire le contenu du fichier
    image = Image.open(file.file).convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch

    # Prédiction
    prediction = model.predict(img_array)
    # predicted_class = np.argmax(prediction)
    predicted_class = np.argmax(prediction)
    label_info = labels_map[str(predicted_class)]
    # result = labels_map.get(predicted_class, {"type": "Inconnu", "description": "Panneau non reconnu."})
    return {
        "class_id": int(predicted_class),
        "type": label_info["type"],
        "description": label_info["description"]
    }

    # return {
    #     "predicted_class": int(predicted_class),
    #     "confidence": float(np.max(prediction))
    # }
