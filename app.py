import streamlit as st
import os
import tempfile
import subprocess
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import joblib
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray

# Load trained model
model = joblib.load("bark_classifier.pkl")

# 🌈 Page config
st.set_page_config(page_title="BarkBack 🐶✨", layout="centered")
st.markdown("<h1 style='text-align: center;'>🐶 BarkBack</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>✨ What is your dog really saying? Let's find out! ✨</h3>", unsafe_allow_html=True)

# Optional: placeholder for logo
st.markdown("### 🧬 *Powered by science and lots of barks!*")

# Upload video
video_file = st.file_uploader(
    "🎥 Upload a short video of your dog barking (MP4 or MOV)", 
    type=["mp4", "mov"]
)

if video_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(video_file.read())
        video_path = tmp_video.name

    st.video(video_path)

    # Step 1: Extract audio
    st.write("🔊 Extracting audio...")
    audio_path = video_path.replace(".mp4", ".wav")
    command = [
        os.path.expanduser("~/bin/ffmpeg"), "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Step 2: Generate spectrogram
    st.write("📸 Generating spectrogram...")
    y, sr = librosa.load(audio_path, sr=44100)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(6, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set(title='Spectrogram of Bark')
    plt.tight_layout()

    spectrogram_path = video_path.replace(".mp4", "_spec.png")
    plt.savefig(spectrogram_path)
    plt.close(fig)

    st.image(Image.open(spectrogram_path), caption="🐾 Bark Spectrogram", use_container_width=True)

    # Step 3: Predict emotion
    st.markdown("---")
    st.subheader("🧠 BarkBack is analyzing the bark...")

    try:
        image = imread(spectrogram_path)
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        gray_image = rgb2gray(image)
        image_resized = resize(gray_image, (128, 128), anti_aliasing=True)
        image_flat = image_resized.flatten().reshape(1, -1)

        prediction = model.predict(image_flat)[0]
        emoji_map = {
            "cuddly": "🥰", "playful": "😜", "needy": "🫶", "anxious": "😟",
            "excited": "🤩", "scared": "😱", "happy": "😄", "over_excited": "🚀",
            "answer": "🐺", "dreaming": "💤", "other": "❓"
        }
        emotion_emoji = emoji_map.get(prediction.lower(), "💬")
        st.success(f"🎉 BarkBack thinks your dog is feeling: **{prediction.upper()}** {emotion_emoji}")

        st.markdown("#### Did we get it right?")
        agree = st.radio(
            "🎯 Was this emotion accurate?",
            ["Yes!", "Not quite..."],
            horizontal=True
        )

        if agree == "Not quite...":
            emotion_options = [
                "cuddly", "playful", "needy", "anxious", "excited",
                "scared", "happy", "over_excited", "answer", "dreaming", "other"
            ]
            selected_emotions = st.multiselect(
                "💡 What do *you* think your dog was feeling?",
                options=emotion_options,
                help="You can select up to 3",
                max_selections=3
            )
            if "other" in selected_emotions:
                st.text_input("💬 Describe the emotion in your own words:")

    except Exception as e:
        st.error(f"❌ Could not predict emotion: {e}")

    st.markdown("---")
    # Step 4: Contribution opt-in
    if st.checkbox("✅ I want to contribute this clip to help improve future models.", value=False):
        st.success("🐾 Thank you! You're helping us build a better BarkBack and bring science closer to every pup's heart.")
        st.text_input("📬 Want updates? Drop your email (optional):")

    st.markdown("---")
    st.markdown("<div style='text-align: center;'>🌍 Made with 💖 by Marie & ChatGPT</div>", unsafe_allow_html=True)

