import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from PIL import Image
import joblib

# Load labels
df = pd.read_csv("bark_labels.csv")
print("📄 Labels loaded:")
print(df.head())

# Folder with spectrogram images
image_folder = "spectrograms"

# Prepare features and labels
X = []
y = []

# Loop through each row in the dataframe
for _, row in df.iterrows():
    filename = row["Filename"]
    emotion = row["Emotion"]
    image_path = os.path.join(image_folder, filename)

    try:
        img = Image.open(image_path).convert("L")  # Grayscale
        img = img.resize((128, 128))
        img_array = np.array(img).flatten()

        X.append(img_array)
        y.append(emotion)  # ✅ Correct label from CSV

    except Exception as e:
        print(f"❌ Could not process {filename}: {e}")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Print label counts
print("\n📊 Emotion counts:")
print(pd.Series(y).value_counts())

# Train classifier on the full dataset
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Predict on training set (just to see how it fits the data)
y_pred = model.predict(X)

print("\n🧠 Emotion Classification Report (on training set):")
print(classification_report(y, y_pred))

# Save the trained model
joblib.dump(model, "bark_classifier.pkl")
print("\n✅ Model saved as 'bark_classifier.pkl'")
