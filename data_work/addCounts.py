import pandas as pd
import re

# Load your CSV
df = pd.read_csv("filteredBirdList_with_counts.csv")
print(df.columns)
"""df = df.drop(columns= ['observed_on_string','description','species_guess'])
print(df.columns)
# Load and parse allSpeciesList.txt
species_counts = {}
with open("allSpeciesList.txt", encoding="utf-8") as f:
    for line in f:
        match = re.match(r"(.+?)\s+(\d+)$", line.strip())
        if match:
            name, count = match.groups()
            species_counts[name.strip()] = int(count)

# Map observation counts to DataFrame
df["observation_count"] = df["common_name"].map(species_counts)

# Save new file
df.to_csv("filteredBirdList_with_counts.csv", index=False)

print("✅ Done. Saved to 'filteredBirdList_with_counts.csv'")"""
