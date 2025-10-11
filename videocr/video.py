from __future__ import annotations
from typing import List
import cv2
import numpy as np
import os
import warnings

from . import utils
from .models import PredictedFrames, PredictedSubtitle
from .opencv_adapter import Capture
from paddleocr import PaddleOCR


class Video:
    path: str
    lang: str
    use_fullframe: bool
    det_model_dir: str
    rec_model_dir: str
    num_frames: int
    fps: float
    height: int
    width: int
    ocr: PaddleOCR
    pred_frames: List[PredictedFrames]
    pred_subs: List[PredictedSubtitle]

    def __init__(self, path: str, det_model_dir: str, rec_model_dir: str):
        self.path = path
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        with Capture(path) as v:
            self.num_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = v.get(cv2.CAP_PROP_FPS)
            self.height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.width = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))

    def run_ocr(self, use_gpu: bool, lang: str, time_start: str, time_end: str,
                conf_threshold: int, use_fullframe: bool, brightness_threshold: int, similar_image_threshold: int, similar_pixel_threshold: int, frames_to_skip: int,
                crop_x: int, crop_y: int, crop_width: int, crop_height: int) -> None:
        conf_threshold_percent = float(conf_threshold/100)
        self.lang = lang
        self.use_fullframe = use_fullframe
        self.pred_frames = []

        # Fix for PaddleOCR versions greater or equal than 3.0.3
        if utils.needs_conversion():
            ocr = PaddleOCR(
                lang=self.lang,
                text_recognition_model_dir=self.rec_model_dir,
                text_detection_model_dir=self.det_model_dir,
                text_detection_model_name=utils.get_model_name_from_dir(self.det_model_dir),
                text_recognition_model_name=utils.get_model_name_from_dir(self.rec_model_dir),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="gpu" if use_gpu else "cpu",
                enable_hpi=True
            )
        else:
            ocr = PaddleOCR(
                lang=self.lang,
                rec_model_dir=self.rec_model_dir,
                det_model_dir=self.det_model_dir,
                use_gpu=use_gpu,
                enable_hpi=True
            )

        ocr_start = utils.get_frame_index(time_start, self.fps) if time_start else 0
        ocr_end = utils.get_frame_index(time_end, self.fps) if time_end else self.num_frames

        if ocr_end < ocr_start:
            raise ValueError('time_start is later than time_end')
        num_ocr_frames = ocr_end - ocr_start

        crop_x_start = None
        crop_y_start = None
        crop_x_end = None
        crop_y_end = None

        if self.use_fullframe:
            if any(p is not None for p in [crop_x, crop_y, crop_width, crop_height]):
                warnings.warn("use_fullframe=True: Ignoring provided crop parameters.")
        else:
            if all(p is None for p in [crop_x, crop_y, crop_width, crop_height]):
                warnings.warn(r"use_fullframe=False and no crop provided: defaulting to lower 30% region of the frame.")
            else:
                # infer missing crop parameters
                inferred_x = 0 if crop_x is None else crop_x
                inferred_y = 0 if crop_y is None else crop_y
                inferred_width = (self.width - inferred_x) if crop_width is None else crop_width
                inferred_height = (self.height - inferred_y) if crop_height is None else crop_height

                # clamp to valid ranges
                inferred_x = max(0, min(int(inferred_x), self.width))
                inferred_y = max(0, min(int(inferred_y), self.height))
                inferred_width = max(0, min(int(inferred_width), self.width - inferred_x))
                inferred_height = max(0, min(int(inferred_height), self.height - inferred_y))
                if inferred_width == 0 or inferred_height == 0:
                    warnings.warn("resolved crop has zero width/height. Defaulting to lower 30% region of the frame.")
                else:
                    crop_x_start = inferred_x
                    crop_y_start = inferred_y
                    crop_x_end = inferred_x + inferred_width
                    crop_y_end = inferred_y + inferred_height

                    # warn if not all parameters were explicitly provided
                    if None in [crop_x, crop_y, crop_width, crop_height]:
                        warnings.warn(
                            f"incomplete crop provided. Using inferred crop: x={crop_x_start}, y={crop_y_start}, "
                            f"width={inferred_width}, height={inferred_height}.")

        # get frames from ocr_start to ocr_end
        with Capture(self.path) as v:
            v.set(cv2.CAP_PROP_POS_FRAMES, ocr_start)
            prev_grey = None
            predicted_frames = None
            modulo = frames_to_skip + 1
            pbar = None
            frames_to_process = (num_ocr_frames + modulo - 1) // modulo
            try:
                from tqdm import tqdm as _tqdm
                pbar = _tqdm(total=frames_to_process, desc="Processing frames", unit="frame", dynamic_ncols=True, leave=False, mininterval=120.0)
            except Exception:
                pass
            batch_size = 100
            frames_batch = []
            frames_index = []
            
            for i in range(num_ocr_frames):
                ret, frame = v.read()
                if not ret:
                    break

                if i % modulo != 0:
                    continue

                # Crop frame
                if not self.use_fullframe:
                    if crop_x_end is not None and crop_y_end is not None:
                        frame = frame[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
                    else:
                        # default: bottom third
                        frame = frame[2 * self.height // 3:, :]

                # Brightness filter
                if brightness_threshold:
                    frame = cv2.bitwise_and(frame, frame, mask=cv2.inRange(frame, 
                                    (brightness_threshold, brightness_threshold, brightness_threshold), 
                                    (255, 255, 255)))

                # Skip similar frames
                if similar_image_threshold:
                    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_grey is not None:
                        _, absdiff = cv2.threshold(cv2.absdiff(prev_grey, grey),
                                                similar_pixel_threshold, 255, cv2.THRESH_BINARY)
                        if np.count_nonzero(absdiff) < similar_image_threshold:
                            predicted_frames.end_index = i + ocr_start
                            prev_grey = grey
                            if pbar is not None:
                                pbar.update(1)
                            continue
                    prev_grey = grey

                # Thêm frame vào batch
                frames_batch.append(frame)
                frames_index.append(i + ocr_start)

                # Nếu đủ batch hoặc cuối video → predict
                if len(frames_batch) == batch_size or i == num_ocr_frames - 1:
                    # Tạo boxes list cho batch (toàn frame)
                    boxes_list = [[[0,0], [frame.shape[1]-1,0], 
                                [frame.shape[1]-1, frame.shape[0]-1], [0, frame.shape[0]-1]]
                                for frame in frames_batch]

                    preds_batch = ocr.predict(frames_batch, boxes_list)

                    # Lưu kết quả
                    for idx, pred in zip(frames_index, preds_batch):
                        predicted_frames = PredictedFrames(idx, pred, conf_threshold_percent)
                        self.pred_frames.append(predicted_frames)

                    # Reset batch
                    frames_batch = []
                    frames_index = []

                    if pbar is not None:
                        pbar.update(len(preds_batch))
            if pbar is not None:
                pbar.close()
        

    def get_subtitles(self, sim_threshold: int) -> str:
        self._generate_subtitles(sim_threshold)
        return ''.join(
            '{}\n{} --> {}\n{}\n\n'.format(
                i,
                utils.get_srt_timestamp(sub.index_start, self.fps),
                utils.get_srt_timestamp(sub.index_end + 1, self.fps),
                sub.text)
            for i, sub in enumerate(self.pred_subs, start=1))

    def _generate_subtitles(self, sim_threshold: int) -> None:
        self.pred_subs = []

        if self.pred_frames is None:
            raise AttributeError(
                'Please call self.run_ocr() first to perform ocr on frames')

        max_frame_merge_diff = int(0.09 * self.fps)
        for frame in self.pred_frames:
            self._append_sub(PredictedSubtitle([frame], sim_threshold), max_frame_merge_diff)
        self.pred_subs = [sub for sub in self.pred_subs if len(sub.frames[0].lines) > 0]

    def _append_sub(self, sub: PredictedSubtitle, max_frame_merge_diff: int) -> None:
        if len(sub.frames) == 0:
            return

        # merge new sub to the last subs if they are not empty, similar and within 0.09 seconds apart
        if self.pred_subs:
            last_sub = self.pred_subs[-1]
            if len(last_sub.frames[0].lines) > 0 and sub.index_start - last_sub.index_end <= max_frame_merge_diff and last_sub.is_similar_to(sub):
                del self.pred_subs[-1]
                sub = PredictedSubtitle(last_sub.frames + sub.frames, sub.sim_threshold)

        self.pred_subs.append(sub)
