import random
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union


class ClipInfo(NamedTuple):
    """
    Named-tuple for clip information with:
        clip_start_sec  (Union[float, Fraction]): clip start time.
        clip_end_sec (Union[float, Fraction]): clip end time.
        clip_index (int): clip index in the video.
        aug_index (int): augmentation index for the clip. Different augmentation methods
            might generate multiple views for the same clip.
        is_last_clip (bool): a bool specifying whether there are more clips to be
            sampled from the video.
    """

    clip_start_sec: Union[float, Fraction]
    clip_end_sec: Union[float, Fraction]
    clip_index: int
    aug_index: int
    is_last_clip: bool


class ClipInfoList(NamedTuple):
    """
    Named-tuple for clip information with:
        clip_start_sec  (float): clip start time.
        clip_end_sec (float): clip end time.
        clip_index (int): clip index in the video.
        aug_index (int): augmentation index for the clip. Different augmentation methods
            might generate multiple views for the same clip.
        is_last_clip (bool): a bool specifying whether there are more clips to be
            sampled from the video.
    """

    clip_start_sec: List[float]
    clip_end_sec: List[float]
    clip_index: List[float]
    aug_index: List[float]
    is_last_clip: List[float]


class ClipSampler(ABC):
    """
    Interface for clip samplers that take a video time, previous sampled clip time,
    and returns a named-tuple ``ClipInfo``.
    """

    def __init__(self, clip_duration: Union[float, Fraction]) -> None:
        self._clip_duration = Fraction(clip_duration)
        self._current_clip_index = 0
        self._current_aug_index = 0

    @abstractmethod
    def __call__(
        self,
        last_clip_end_time: Union[float, Fraction],
        video_duration: Union[float, Fraction],
        annotation: Dict[str, Any],
    ) -> ClipInfo:
        pass

    def reset(self) -> None:
        """Resets any video-specific attributes in preperation for next video"""
        pass


class UniformClipSampler(ClipSampler):
    """
    Evenly splits the video into clips of size clip_duration.
    """

    def __init__(
        self,
        clip_duration: Union[float, Fraction],
        stride: Optional[Union[float, Fraction]] = None,
        backpad_last: bool = False,
        eps: float = 1e-6,
    ):
        """
        Args:
            clip_duration (Union[float, Fraction]):
                The length of the clip to sample (in seconds).
            stride (Union[float, Fraction], optional):
                The amount of seconds to offset the next clip by
                default value of None is equivalent to no stride => stride == clip_duration.
            eps (float):
                Epsilon for floating point comparisons. Used to check the last clip.
            backpad_last (bool):
                Whether to include the last frame(s) by "back padding".

                For instance, if we have a video of 39 frames (30 fps = 1.3s)
                with a stride of 16 (0.533s) with a clip duration of 32 frames
                (1.0667s). The clips will be (in frame numbers):

                with backpad_last = False
                - [0, 31]

                with backpad_last = True
                - [0, 31]
                - [8, 39], this is "back-padded" from [16, 48] to fit the last window
        Note that you can use Fraction for clip_duration and stride if you want to
        avoid float precision issue and need accurate frames in each clip.
        """
        super().__init__(clip_duration)
        self._stride = stride if stride is not None else self._clip_duration
        self._eps = eps
        self._backpad_last = backpad_last

        assert self._stride > 0, "stride must be positive"

    def _clip_start_end(
        self,
        last_clip_end_time: Union[float, Fraction],
        video_duration: Union[float, Fraction],
        backpad_last: bool,
    ) -> Tuple[Fraction, Fraction]:
        """
        Helper to calculate the start/end clip with backpad logic
        """
        delta = self._stride - self._clip_duration
        last_end_time = -delta if last_clip_end_time is None else last_clip_end_time
        clip_start = Fraction(last_end_time + delta)
        clip_end = Fraction(clip_start + self._clip_duration)
        if backpad_last:
            buffer_amount = max(0, clip_end - video_duration)
            clip_start -= buffer_amount
            clip_start = Fraction(max(0, clip_start))  # handle rounding
            clip_end = Fraction(clip_start + self._clip_duration)

        return clip_start, clip_end

    def __call__(
        self,
        last_clip_end_time: Optional[float],
        video_duration: float,
        annotation: Dict[str, Any],
    ) -> ClipInfo:
        """
        Args:
            last_clip_end_time (float): the last clip end time sampled from this video. This
                should be 0.0 if the video hasn't had clips sampled yet.
            video_duration: (float): the duration of the video that's being sampled in seconds
            annotation (Dict): Not used by this sampler.
        Returns:
            clip_info: (ClipInfo): includes the clip information (clip_start_time,
            clip_end_time, clip_index, aug_index, is_last_clip), where the times are in
            seconds and is_last_clip is False when there is still more of time in the video
            to be sampled.
        """
        clip_start, clip_end = self._clip_start_end(
            last_clip_end_time, video_duration, backpad_last=self._backpad_last
        )

        # if they both end at the same time - it's the last clip
        _, next_clip_end = self._clip_start_end(
            clip_end, video_duration, backpad_last=self._backpad_last
        )
        if self._backpad_last:
            is_last_clip = abs(next_clip_end - clip_end) < self._eps
        else:
            is_last_clip = (next_clip_end - video_duration) > self._eps

        clip_index = self._current_clip_index
        self._current_clip_index += 1

        if is_last_clip:
            self.reset()

        return ClipInfo(clip_start, clip_end, clip_index, 0, is_last_clip)

    def reset(self):
        self._current_clip_index = 0


class MiddleClipSampler(ClipSampler):
    """
    Middle samples clip of size clip_duration from the videos.
    """

    def __call__(
        self,
        last_clip_end_time: float,
        video_duration: float,
    ) -> ClipInfo:
        """
        Args:
            last_clip_end_time (float): Not used for RandomClipSampler.
            video_duration: (float): the duration (in seconds) for the video that's
                being sampled
            annotation (Dict): Not used by this sampler.
        Returns:
            clip_info (ClipInfo): includes the clip information of (clip_start_time,
            clip_end_time, clip_index, aug_index, is_last_clip). The times are in seconds.
            clip_index, aux_index and is_last_clip are always 0, 0 and True, respectively.
        """
        if video_duration < self._clip_duration:   # todo: new add
            self._clip_duration = video_duration
        
        max_possible_clip_start = max(video_duration - self._clip_duration, 0)
        middle_clip_start_sec = Fraction(video_duration / 2)
        half_clip = Fraction(self._clip_duration / 2)
        middle_clip_start_sec = Fraction(middle_clip_start_sec - half_clip)
        clip_start_sec = max(0, middle_clip_start_sec)
        clip_start_sec = min(clip_start_sec, max_possible_clip_start)
        
        return ClipInfo(
            clip_start_sec, clip_start_sec + self._clip_duration, 0, 0, True
        )

