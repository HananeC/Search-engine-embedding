from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
from glob import glob
import os
from pymongo import MongoClient

img_path = "/Users/your_username/Downloads/images" # Replace with your local folder path 

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
        "image_path" : images[index],
        "embeddings" : generate_image_embeddings(images[index])
    })

MONGODB_URI = "mongodb://localhost:27017/" # Replace with your MongoDB URI
# If you are using MongoDB Atlas, it will look like this:  # MONGODB_URI = "mongodb+srv://<username>:<password>@cluster.mongodb.net/test?retryWrites=true&w=majority"
# Connect to MongoDB cluster with MongoClient
client = MongoClient(MONGODB_URI)

db = client.imageDB # database name
image_DB_collection = db.imageVec # collection name

result = image_DB_collection.insert_many(alldata)

document_ids = result.inserted_ids
print("# of documents inserted: "+str(len(document_ids)))
print("Process Completed!")