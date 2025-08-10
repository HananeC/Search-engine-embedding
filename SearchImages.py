import os
from pymongo import MongoClient
from transformers import AutoImageProcessor, ViTModel
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# takes as input an image and returns its embeddings
def generate_image_embeddings(filepath):
    image = Image.open(filepath).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.tolist()
    return embeddings[0][0]

# Connect to the MongoDB database
MONGODB_URI = "mongodb://localhost:27017/" #Replace with your MongoDB URI

# Connect to MongoDB cluster with MongoClient
client = MongoClient(MONGODB_URI)
db = client.imageDB
image_DB_collection = db.imageVec

# Loading the Model and feature extractor
feature_extractor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTModel.from_pretrained("google/vit-base-patch16-224")

# principal function for the Streamlit application
def main():
    st.title("Image Search")

    # select an image from the local file system
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.')
        ext = uploaded_file.name.split(".")[-1]
        image_path = Path("uploaded_image." + ext).resolve()  # crée un chemin absolu propre
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Search for similar images in the database
        search_image(image_path)

# function to search for similar images in the database
def search_image(filepath):

    filepath = Path(filepath).resolve()

    # generate embeddings for the image to search
    q_image_embeddings = generate_image_embeddings(filepath)

    # request for vector search
    # Note: Ensure that the index "p_similarity_search" exists in the MongoDB collection
    results = image_DB_collection.aggregate([
    {
        "$vectorSearch": {
            "queryVector": q_image_embeddings,
            "path": "embeddings",
            "numCandidates": 5,
            "limit": 5,
            "index": "p_similarity_search",
        }
    },
    {
        "$project": {
            "image_path": 1,
            "_id": 0
        }
    }
      ])


    # Verify if any results are returned
    if not results:
        st.warning("No similar images found in the database.")
        return

    images_to_plot = [filepath]
    
    for document in results:
        img_path = os.path.normpath(document['image_path'])  # clean up path
        
        if os.path.exists(img_path):
            images_to_plot.append(img_path)
        else:
            st.warning(f"Image not found: {img_path}")

    images_to_plot = [Image.open(Path(img_path).resolve()).resize((224, 224)) for img_path in images_to_plot]

    # create a grid to display the images
    fig, axs = plt.subplots(2, 3, figsize=(8, 8))
    axs = axs.ravel()
    for i, img in enumerate(images_to_plot):
        axs[i].imshow(np.asarray(img))
        if i == 0:
            axs[i].set_title("Original Image")
        else:
            axs[i].set_title(f"Retrieved Image {i}") 
        axs[i].axis('off')

    st.pyplot(fig)

if __name__ == "__main__":
    main()
