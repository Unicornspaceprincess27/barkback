import os
import subprocess

# Input full video file
input_video = os.path.expanduser("dog_sounds_full.mp4")

# Create output folder
output_folder = "clipped_sounds"
os.makedirs(output_folder, exist_ok=True)

# List of clips (start/end in seconds)
clips = [
    {"start": 27, "end": 30, "filename": "youtube_playful_laugh_1.mp4"},
    {"start": 79, "end": 83, "filename": "youtube_answer_howl_1.mp4"},
    {"start": 125, "end": 129, "filename": "youtube_overexcited_sneeze_1.mp4"},
    {"start": 262, "end": 265, "filename": "youtube_dreaming_whimper_1.mp4"},
    {"start": 291, "end": 293, "filename": "youtube_needy_whine_1.mp4"},
    {"start": 412, "end": 416, "filename": "youtube_cuddly_grumble_1.mp4"},
    {"start": 432, "end": 435, "filename": "youtube_scared_growl_1.mp4"},
]

# Function to extract each clip
def extract_clip(start, end, filename):
    duration = end - start
    output_path = os.path.join(output_folder, filename)
    command = [
        os.path.expanduser("~/bin/ffmpeg"),
        "-y",
        "-i", input_video,
        "-ss", str(start),
        "-t", str(duration),
        "-c", "copy",
        output_path
    ]
    print(f"Clipping: {filename} ({start}s → {end}s)")
    subprocess.run(command)

# Run all clips
for clip in clips:
    extract_clip(clip["start"], clip["end"], clip["filename"])

print("\n✅ Done! Clips saved to:", output_folder)
