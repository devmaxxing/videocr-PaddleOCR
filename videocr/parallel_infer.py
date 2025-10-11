# parallel_infer.py

import os
import multiprocessing
from functools import partial
import paddle
from .api import save_subtitles_to_file


def detect_gpus():
    """Detect available GPUs via PaddlePaddle."""
    try:
        gpu_count = paddle.device.cuda.device_count()
        if gpu_count > 0:
            return list(range(gpu_count))
        else:
            print("⚠️ No GPUs detected. Falling back to CPU mode.")
            return []
    except Exception as e:
        print(f"⚠️ GPU detection failed: {e}")
        return []


def process_video(video_path, output_dir, gpu_id=None, lang='ch', **kwargs):
    """Worker process to OCR one video using a specific GPU or CPU."""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[GPU {gpu_id}] Starting: {os.path.basename(video_path)}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print(f"[CPU] Starting: {os.path.basename(video_path)}")

    video_name = os.path.basename(video_path)
    srt_path = os.path.join(output_dir, os.path.splitext(video_name)[0] + ".srt")

    try:
        save_subtitles_to_file(
            video_path=video_path,
            file_path=srt_path,
            lang=lang,
            use_gpu=(gpu_id is not None),
            **kwargs
        )
        status = "✅"
    except Exception as e:
        status = "❌"
        print(f"[{status}] Error processing {video_name}: {e}")

    print(f"[{status}] Finished: {video_name}")


def multi_process_infer(
    input_dir,
    output_dir,
    gpu_ids=None,
    lang='ch',
    num_processes=None,
    **kwargs
):
    """
    Multi-GPU, multi-process subtitle generation for all videos in a directory.

    Args:
        input_dir (str): Folder containing input video files.
        output_dir (str): Folder to save generated SRT subtitles.
        gpu_ids (list[int] | None): GPU device IDs. If None, auto-detects.
        lang (str): OCR language.
        num_processes (int | None): Number of processes to use.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Detect GPUs if not provided
    if gpu_ids is None:
        gpu_ids = detect_gpus()

    use_gpu = len(gpu_ids) > 0

    # Gather video files
    video_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov'))
    ]

    if not video_files:
        print("❌ No video files found in:", input_dir)
        return

    # Decide how many processes to launch
    if num_processes is None:
        if use_gpu:
            num_processes = min(len(video_files), len(gpu_ids))
        else:
            num_processes = min(len(video_files), multiprocessing.cpu_count())

    pool = multiprocessing.Pool(processes=num_processes)

    # Assign GPUs in round-robin fashion (or None for CPU)
    tasks = []
    for idx, video_path in enumerate(video_files):
        gpu_id = gpu_ids[idx % len(gpu_ids)] if use_gpu else None
        tasks.append((video_path, gpu_id))

    process_func = partial(process_video, output_dir=output_dir, lang=lang, **kwargs)

    print(f"🚀 Starting OCR on {len(video_files)} videos using {num_processes} process(es)...")
    pool.starmap(process_func, tasks)
    pool.close()
    pool.join()
    print("✅ All videos processed successfully.")
