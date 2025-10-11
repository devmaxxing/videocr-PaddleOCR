from videocr import save_subtitles_to_file

#@title OCR All Files In Folder (Parallel Batch Processing)
import os
from multiprocessing import Pool, cpu_count
from functools import partial

# ==========================================================
# Configuration parameters
# ==========================================================
input_folder = "/Users/sunwarder/Documents/AI-Tools/video-to-srt-ocr/ocrsample/b9nno_sample_ocr"  #@param {type:"string"}
output_folder = "/Users/sunwarder/Documents/AI-Tools/video-to-srt-ocr/ocrsample/result_ocr/zh/b9nno_sample_ocr"   #@param {type:"string"}

language_code = "ch" #@param {type:"string"}
use_gpu = True #@param {type:"boolean"}
start_time = "00:00" #@param {type:"string"}
end_time = "" #@param {type:"string"}
confidence_threshold = 75 #@param {type:"number"}
similarity_threshold = 80 #@param {type:"number"}
frames_to_skip = 0 #@param {type:"integer"}
crop_x = 0 #@param {type:"integer"}
crop_y = 0 #@param {type:"integer"}
crop_width = 1080 #@param {type:"integer"}
crop_height = 300 #@param {type:"integer"}
use_fullframe = True #@param {type: "boolean"}

# ==========================================================
# Ensure output folder exists
# ==========================================================
os.makedirs(output_folder, exist_ok=True)

# ==========================================================
# Helper: process a single video file
# ==========================================================
# === Run multi-process inference ===
video_extensions = ('.mp4', '.mkv', '.mov', '.avi')
for filename in os.listdir(input_folder):
    if filename.lower().endswith(video_extensions):
        input_file_path = os.path.join(input_folder, filename)
        name_no_ext = os.path.splitext(filename)[0]
        output_file_path = os.path.join(output_folder, f"{name_no_ext}_zh.srt")
        # Skip if output already exists
        if os.path.exists(output_file_path):
            print(f"✅ Skipping {filename} (already exists)")
            continue
        print(f"Processing: {filename}")
        print(f"Saving to: {output_file_path}")
        save_subtitles_to_file(
            input_dir=input_folder,
            output_dir=output_folder,
            lang=language_code,
            use_fullframe=use_fullframe,
            time_start=start_time,
            time_end=end_time,
            conf_threshold=confidence_threshold,
            sim_threshold=similarity_threshold,
            frames_to_skip=frames_to_skip,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_width=crop_width,
            crop_height=crop_height,
        )

print("✅ All videos processed and subtitles saved.")

