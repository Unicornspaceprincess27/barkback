import os
import subprocess

# Set paths
input_folder = "clipped_sounds"
output_folder = "audio_clips"
os.makedirs(output_folder, exist_ok=True)

# Loop through all video files in clipped_sounds/
for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
        name = os.path.splitext(filename)[0]
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"{name}.wav")

        # Run ffmpeg command to extract audio
        command = [
            os.path.expanduser("~/bin/ffmpeg"),
            "-y",
            "-i", input_path,
            "-vn",  # no video
            "-acodec", "pcm_s16le",  # raw PCM 16-bit WAV
            "-ar", "44100",  # sample rate
            "-ac", "1",  # mono
            output_path
        ]

        print(f"Extracting audio: {filename} → {name}.wav")
        subprocess.run(command)

print("\n✅ All audio files saved in:", output_folder)
