import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Chargement du modèle
@st.cache_data( )
def obtenir_modele():
    modele = tf.keras.models.load_model('model_ml.hdf5')
    return modele

# Traitement de l'image et prédiction
def traiter_image_et_predire(image, modele):
    # Prétraitement de l'image pour qu'elle corresponde à l'entrée attendue du modèle
    image_traitée = image.resize((356, 536))  # ou la taille requise par votre modèle
    image_traitée = np.array(image_traitée) / 255.0  # si nécessaire, assurez-vous que les valeurs des pixels sont normalisées
    image_traitée = np.expand_dims(image_traitée, axis=0)

    # Faire la prédiction
    predictions = modele.predict(image_traitée)
    return predictions

# Interface principale
def interface_principale():


    fichier = st.file_uploader("Télécharger une image de rétine", type=["jpg", "png"])

    if fichier is not None:
        image = Image.open(fichier)
        st.image(image, caption='Image téléchargée', use_column_width=True)
        st.write("")
        st.write("Analyse...")

        modele = obtenir_modele()
        predictions = traiter_image_et_predire(image, modele)

        # Supposez que 'predictions' est un tableau 1D de probabilités
        if predictions is not None:
            st.subheader("Résultats de la prédiction")
            
            # Si vous avez des étiquettes pour chaque prédiction, vous pouvez les afficher ainsi :
            etiquettes_maladies = [ 'Disease_Risk', 'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN', 'ERM', 'LS',
                    'MS', 'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE', 'ST', 'AION', 'PT',
                    'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP', 'CWS', 'CB', 'ODPM', 'PRH',
                    'MNF', 'HR', 'CRAO', 'TD', 'CME', 'PTCR', 'CF', 'VH', 'MCA', 'VS', 'BRAO',
                    'PLQ', 'HPED', 'CL']  # Remplacer par vos vraies étiquettes
            for label, prob in zip(etiquettes_maladies, predictions[0]):
                st.write(f"{label}: {prob * 100:.2f}%")

            # Si vous voulez juste lister les probabilités sans étiquettes spécifiques :
            # for prob in predictions[0]:
            #     st.write(f"Probabilité: {prob * 100:.2f}%")

def principal():
    interface_principale()

if __name__ == "__main__":
    principal()
