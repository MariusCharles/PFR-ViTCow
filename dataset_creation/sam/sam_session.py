import logging
from typing import Dict

import numpy as np
from sam3.model_builder import build_sam3_video_predictor

logger = logging.getLogger(__name__)


class SAMSession:
    """Wrapper around SAM3 video predictor sessions."""

    def __init__(self) -> None:
        """Initialize the SAM3 video predictor."""
        logger.info("Initializing SAM3 predictor")
        self.predictor = build_sam3_video_predictor()

    def start(self, video_path: str) -> str:
        """Start a new SAM session for a video.

        Args:
            video_path: Path to the input video.

        Returns:
            Session identifier.
        """
        logger.info("Starting session for %s", video_path)
        response = self.predictor.handle_request(
            dict(type="start_session", resource_path=video_path)
        )
        return response["session_id"]

    def add_prompt(
        self,
        session_id: str,
        text: str,
        frame_index: int = 0,
    ) -> Dict[int, np.ndarray]:
        """Add a text prompt to a SAM session.

        Args:
            session_id: Identifier of the SAM session.
            text: Text prompt describing the target object.
            frame_index: Index of the frame where the prompt is applied.

        Returns:
            Mapping from object ID to segmentation mask for the frame.
        """
        logger.info("Adding prompt '%s' to session %s", text, session_id)
        response = self.predictor.handle_request(
            dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_index,
                text=text,
            )
        )
        return response["outputs"]

    def propagate(self, session_id: str) -> Dict[int, Dict[int, np.ndarray]]:
        """Propagate segmentation through the video.

        Args:
            session_id: Identifier of the SAM session.

        Returns:
            Mapping from frame index to object masks.
        """
        logger.info("Propagating session %s", session_id)

        outputs: Dict[int, Dict[int, np.ndarray]] = {}
        for r in self.predictor.handle_stream_request(
            dict(type="propagate_in_video", session_id=session_id)
        ):
            outputs[r["frame_index"]] = r["outputs"]

        logger.info("Propagation done for session %s", session_id)
        return outputs
import logging
from typing import Dict

import numpy as np
from sam3.model_builder import build_sam3_video_predictor

logger = logging.getLogger(__name__)


class SAMSession:
    """Wrapper around SAM3 video predictor sessions."""

    def __init__(self) -> None:
        """Initialize the SAM3 video predictor."""
        logger.info("Initializing SAM3 predictor")
        self.predictor = build_sam3_video_predictor()

    def start(self, video_path: str) -> str:
        """Start a new SAM session for a video.

        Args:
            video_path: Path to the input video.

        Returns:
            Session identifier.
        """
        logger.info("Starting session for %s", video_path)
        response = self.predictor.handle_request(
            dict(type="start_session", resource_path=video_path)
        )
        return response["session_id"]

    def add_prompt(
        self,
        session_id: str,
        text: str,
        frame_index: int = 0,
    ) -> Dict[int, np.ndarray]:
        """Add a text prompt to a SAM session.

        Args:
            session_id: Identifier of the SAM session.
            text: Text prompt describing the target object.
            frame_index: Index of the frame where the prompt is applied.

        Returns:
            Mapping from object ID to segmentation mask for the frame.
        """
        logger.info("Adding prompt '%s' to session %s", text, session_id)
        response = self.predictor.handle_request(
            dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_index,
                text=text,
            )
        )
        return response["outputs"]

    def propagate(self, session_id: str) -> Dict[int, Dict[int, np.ndarray]]:
        """Propagate segmentation through the video.

        Args:
            session_id: Identifier of the SAM session.

        Returns:
            Mapping from frame index to object masks.
        """
        logger.info("Propagating session %s", session_id)

        outputs: Dict[int, Dict[int, np.ndarray]] = {}
        for r in self.predictor.handle_stream_request(
            dict(type="propagate_in_video", session_id=session_id)
        ):
            outputs[r["frame_index"]] = r["outputs"]

        logger.info("Propagation done for session %s", session_id)
        return outputs
import logging
from typing import Dict

import numpy as np
from sam3.model_builder import build_sam3_video_predictor

logger = logging.getLogger(__name__)


class SAMSession:
    """Wrapper around SAM3 video predictor sessions."""

    def __init__(self) -> None:
        """Initialize the SAM3 video predictor."""
        logger.info("Initializing SAM3 predictor")
        self.predictor = build_sam3_video_predictor()

    def start(self, video_path: str) -> str:
        """Start a new SAM session for a video.

        Args:
            video_path: Path to the input video.

        Returns:
            Session identifier.
        """
        logger.info("Starting session for %s", video_path)
        response = self.predictor.handle_request(
            dict(type="start_session", resource_path=video_path)
        )
        return response["session_id"]

    def add_prompt(
        self,
        session_id: str,
        text: str,
        frame_index: int = 0,
    ) -> Dict[int, np.ndarray]:
        """Add a text prompt to a SAM session.

        Args:
            session_id: Identifier of the SAM session.
            text: Text prompt describing the target object.
            frame_index: Index of the frame where the prompt is applied.

        Returns:
            Mapping from object ID to segmentation mask for the frame.
        """
        logger.info("Adding prompt '%s' to session %s", text, session_id)
        response = self.predictor.handle_request(
            dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_index,
                text=text,
            )
        )
        return response["outputs"]

    def propagate(self, session_id: str) -> Dict[int, Dict[int, np.ndarray]]:
        """Propagate segmentation through the video.

        Args:
            session_id: Identifier of the SAM session.

        Returns:
            Mapping from frame index to object masks.
        """
        logger.info("Propagating session %s", session_id)

        outputs: Dict[int, Dict[int, np.ndarray]] = {}
        for r in self.predictor.handle_stream_request(
            dict(type="propagate_in_video", session_id=session_id)
        ):
            outputs[r["frame_index"]] = r["outputs"]

        logger.info("Propagation done for session %s", session_id)
        return outputs