from typing import Iterator

import cv2
import numpy as np
import imageio.v3 as iio


class VideoFrameLoader:

    def __init__(self, video_path: str):
        self._video_capture = cv2.VideoCapture(video_path)
        self._video_path = video_path

    def get_n_frames(self) -> int:
        n_frames = int(self._video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return n_frames

    def load_frame(self, frame_id: int) -> np.ndarray:
        # set frame position
        self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, image = self._video_capture.read()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def load_all_frames(self) -> Iterator[np.ndarray]:
        return iio.imiter(self._video_path, plugin='pyav')

        # self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # while True:
        #     ret, image = self._video_capture.read()
        #     if not ret:
        #         break
        #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #     yield image

        # return mediapy.read_video(self._video_path)
