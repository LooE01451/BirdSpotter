import pandas as pd
import numpy as np
import os
import urllib.request
from tqdm import tqdm

observationData = "observations-576490.csv/observations-576490.csv"

clean_df = pd.read_csv(observationData)

print(clean_df.head())
clean_df = clean_df.dropna(subset=["image_url", "common_name"])
clean_df = clean_df.drop_duplicates(subset = ["image_url"])

species_counts = clean_df["common_name"].value_counts()
filtered_df = clean_df[clean_df["common_name"].isin(species_counts[species_counts > 10].index)]

print(filtered_df.head())
rows,columns = filtered_df.shape
print(f"{rows}: {columns}")

print("Number of species:", filtered_df["common_name"].nunique())
print(filtered_df["common_name"].value_counts().head(10))
print(filtered_df[["common_name", "image_url"]].head())


num_images = filtered_df["image_url"].notnull().sum()
print(f"Total images to download: {num_images}")

#test_df = filtered_df.sample(10)  
# Set your output directory
output_dir = "C:/Users/yakis/Documents/GitHub/BirdSpotter/birdDataset"
os.makedirs(output_dir, exist_ok=True)
"""
for i, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
    label = row["common_name"].strip().replace("/", "_")
    image_url = row["image_url"]
    bird_id = row["id"]

    # Create folder for the species
    species_folder = os.path.join(output_dir, label)
    os.makedirs(species_folder, exist_ok=True)

    # Define image path
    image_path = os.path.join(species_folder, f"{bird_id}.jpg")

    # Download image
    try:
        urllib.request.urlretrieve(image_url, image_path)
    except Exception as e:
        print(f"Failed to download {image_url}: {e}")
        """