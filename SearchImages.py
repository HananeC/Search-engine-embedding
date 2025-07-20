import os
from pymongo import MongoClient
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Function to generate the embeddings of an image
def generate_image_embeddings(filepath):
    image = Image.open(filepath).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.tolist()
    return embeddings[0][0]

# Connect to the MongoDB database
MONGODB_URI = "mongodb://localhost:27017/" #Replace with your MongoDB URI

client = MongoClient(MONGODB_URI)
db = client.imageDB
image_DB_collection = db.imageVec

# Load the ViT model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTModel.from_pretrained("google/vit-base-patch16-224")

# Main function for the Streamlit application
def main():
    st.title("Image Search")

    # Select an image from the local file system
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        image_path = "uploaded_image." + uploaded_file.name.split(".")[-1]
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Search for similar images in the database
        search_image(image_path)

# Function to search for similar images in the database
def search_image(filepath):
    # Generate the embeddings of the image to search
    q_image_embeddings = generate_image_embeddings(filepath)

    # Vector search query
    results = image_DB_collection.aggregate([
        {   "$vectorSearch": {
                "queryVector": q_image_embeddings,
                "path": "embeddings",
                "numCandidates": 5,
                "limit": 5,
                "index": "p_similarity_serach",
            }
        }
    ])

    # Display the retrieved images
    images_to_plot = [filepath]
    for document in results:
        images_to_plot.append(document['image_path'])

    images_to_plot = [Image.open(img_path).resize((224, 224)) for img_path in images_to_plot]

    # Create a window to display the images
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
