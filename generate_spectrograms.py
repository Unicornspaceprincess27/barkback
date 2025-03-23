import os
import librosa
import matplotlib.pyplot as plt
import numpy as np

print("Script is running!")

# Paths
input_folder = "audio_clips"
output_folder = "spectrograms"
os.makedirs(output_folder, exist_ok=True)

# Loop through audio files
for filename in os.listdir(input_folder):
    if filename.endswith(".wav"):
        name = os.path.splitext(filename)[0]
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"{name}.png")

        print(f"Generating spectrogram: {filename}")

        # Load audio
        y, sr = librosa.load(input_path, sr=44100)

        # Create mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Plot and save
        plt.figure(figsize=(4, 4))
        plt.axis('off')
        librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, cmap='magma')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

print("\n✅ All spectrograms saved to:", output_folder)
