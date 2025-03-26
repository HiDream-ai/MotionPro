import io
import logging
from typing import Any, Dict
from iopath.common.file_io import g_pathmgr
from enum import Enum
from abc import ABC, abstractmethod
from typing import BinaryIO, Dict, Optional
import torch
from iopath.common.file_io import g_pathmgr


class Video(ABC):
    """
    Video provides an interface to access clips from a video container.
    """

    @abstractmethod
    def __init__(
        self,
        file: BinaryIO,
        video_name: Optional[str] = None,
        decode_audio: bool = True,
    ) -> None:
        """
        Args:
            file (BinaryIO): a file-like object (e.g. io.BytesIO or io.StringIO) that
                contains the encoded video.
        """
        pass

    @property
    @abstractmethod
    def duration(self) -> float:
        """
        Returns:
            duration of the video in seconds
        """
        pass

    @abstractmethod
    def get_clip(
        self, start_sec: float, end_sec: float
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Retrieves frames from the internal video at the specified start and end times
        in seconds (the video always starts at 0 seconds).

        Args:
            start_sec (float): the clip start time in seconds
            end_sec (float): the clip end time in seconds
        Returns:
            video_data_dictonary: A dictionary mapping strings to tensor of the clip's
                underlying data.

        """
        pass

    def close(self):
        pass


logger = logging.getLogger(__name__)

class DecoderType(Enum):
    PYAV = "pyav"
    TORCHVISION = "torchvision"
    DECORD = "decord"


def select_video_class(decoder: str) -> Video:
    """
    Select the class for accessing clips based on provided decoder string

    Args:
        decoder (str): Defines what type of decoder used to decode a video.
    """
    if DecoderType(decoder) == DecoderType.PYAV:
        from .encoded_video_pyav import EncodedVideoPyAV
        video_cls = EncodedVideoPyAV
    else:
        raise NotImplementedError(f"Unknown decoder type {decoder}")

    return video_cls


class EncodedVideo(Video):
    """
    EncodedVideo is an abstraction for accessing clips from an encoded video.
    It supports selective decoding when header information is available.
    """

    @classmethod
    def from_path(
        cls,
        file_path: str,
        decode_video: bool = True,
        decode_audio: bool = True,
        decoder: str = "pyav",
        **other_args: Dict[str, Any],
    ):
        """
        Fetches the given video path using PathManager (allowing remote uris to be
        fetched) and constructs the EncodedVideo object.

        Args:
            file_path (str): a PathManager file-path.
        """
        # We read the file with PathManager so that we can read from remote uris.
        with g_pathmgr.open(file_path, "rb") as fh:
            video_file = io.BytesIO(fh.read())

        video_cls = select_video_class(decoder)
        return video_cls(
            file=video_file,
            #video_name=pathlib.Path(file_path).name,
            video_name=file_path,
            decode_video=decode_video,
            decode_audio=decode_audio,
            **other_args,
        )
