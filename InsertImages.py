from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
from glob import glob
import os
from pymongo import MongoClient

img_path = "C:/Users/yourname/images"# share the path of all your images in local folder
labels_list = [label for label in os.listdir(img_path)]
print(f"Dataset Labels: {labels_list}")

images = list()
for i in glob(img_path+"/*/*"):
    images.append(i)
len(images) # find the size of all the images

# Loading the Model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTModel.from_pretrained("google/vit-base-patch16-224")

# takes as input an image and returns its embeddings
def generate_image_embeddings(myImage):
    image = Image.open(myImage).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.tolist()
    return embeddings[0][0]

alldata = list()
for index, y in enumerate(images): # store images and their corresponding embeddings
    alldata.append({
        "image_path1" : images[index],
        "embeddings1" : generate_image_embeddings(images[index])
    })

MONGODB_URI = "mongodb://localhost:27017/" #Replace with your MongoDB URI

# Connect to MongoDB cluster with MongoClient
client = MongoClient(MONGODB_URI)

db = client.imageDB # database name
image_DB_collection = db.imageVec # collection name

result = image_DB_collection.insert_many(alldata)

document_ids = result.inserted_ids
print("# of documents inserted: "+str(len(document_ids)))
print("Process Completed!")