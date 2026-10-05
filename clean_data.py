import pandas as pd
import numpy as np

print("⏳ Starting Data Cleaning Process...")

# 1. Load the raw dataset
try:
    df = pd.read_csv("placementdata.csv")
    print(f"✅ Loaded dataset. Original shape: {df.shape} (Rows, Columns)")
except FileNotFoundError:
    print("❌ Error: placementdata.csv not found in this folder!")
    exit()

# 2. Fix Text Columns (Strip accidental spaces)
if 'PlacementStatus' in df.columns:
    df['PlacementStatus'] = df['PlacementStatus'].astype(str).str.strip()

# 3. Handle Missing Values (Imputation)
# Fill missing numerical columns with their respective column median
num_cols = ['CGPA', 'AptitudeTestScore', 'SoftSkillsRating', 'Projects', 'Internships', 'Workshops/Certifications']
for col in num_cols:
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"🔧 Filled {missing_count} missing values in '{col}' with median: {median_value}")

# 4. Remove Outliers / Invalid Rows
# Keep only realistic CGPA data points (between 0 and 10)
if 'CGPA' in df.columns:
    initial_rows = len(df)
    df = df[(df['CGPA'] >= 0.0) & (df['CGPA'] <= 10.0)]
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"🗑️ Dropped {dropped_rows} rows due to invalid CGPA metrics.")

# 5. Encode Categorical Status into Binary Numbers (0 and 1)
if 'PlacementStatus' in df.columns:
    df['PlacementStatus_Encoded'] = df['PlacementStatus'].map({'Placed': 1, 'NotPlaced': 0})
    # Fill any parsing failures with 0
    df['PlacementStatus_Encoded'].fillna(0, inplace=True)
    print("🔢 Encoded 'PlacementStatus' text columns into numeric 'PlacementStatus_Encoded' (1/0).")

# 6. Export the Cleaned Dataset
cleaned_filename = "placementdata_cleaned.csv"
df.to_csv(cleaned_filename, index=False)
print(f"🎉 Success! Cleaned file saved separately as: '{cleaned_filename}'")
print(f"📊 Final clean shape: {df.shape}")