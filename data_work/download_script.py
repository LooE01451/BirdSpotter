import os
import pandas as pd
import urllib.request

# === Config ===
CSV_FILE = "filteredBirdList_with_counts.csv"  # or your CSV with 'observation_count'
OUTPUT_DIR = "test_bird_dataset"
MIN_OBSERVATIONS = 100
MAX_DOWNLOADS = 500  # Limit for test

# === Load and filter data ===
df = pd.read_csv(CSV_FILE)
df = df[df["observation_count"] >= MIN_OBSERVATIONS]

# Limit to a few rows for testing
df = df.head(MAX_DOWNLOADS)

# === Download images ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

for _, row in df.iterrows():
    species = row["common_name"].replace("/", "_").strip()
    bird_id = row["id"]
    image_url = row["image_url"]

    species_dir = os.path.join(OUTPUT_DIR, species)
    os.makedirs(species_dir, exist_ok=True)

    file_path = os.path.join(species_dir, f"{bird_id}.jpg")
    if os.path.exists(file_path):
        print(f"[✓] Already exists: {file_path}")
        continue

    try:
        urllib.request.urlretrieve(image_url, file_path)
        print(f"[→] Downloaded: {file_path}")
    except Exception as e:
        print(f"[✗] Failed to download {image_url}: {e}")
