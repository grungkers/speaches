import logging

import torch
from numpy import float32
from numpy.typing import NDArray
from pyannote.audio import Pipeline

logger = logging.getLogger(__name__)


def run_diarization(audio: NDArray[float32], pipeline: Pipeline, sample_rate: int = 16000) -> list[dict]:
    """Run speaker diarization on audio data.

    Args:
        audio: Audio array (mono, float32)
        pipeline: Pyannote diarization pipeline
        sample_rate: Sample rate of the audio

    Returns:
        Dictionary mapping (start_time, end_time) tuples to speaker labels
    """
    logger.debug("Running speaker diarization")
    # Run diarization
    output = pipeline({"waveform": torch.from_numpy(audio[None, :]), "sample_rate": sample_rate})

    # Convert pyannote output to our format
    speaker_segments = []
    for segment, speaker in output.speaker_diarization:
        speaker_segments.append({
            "start": segment.start,
            "end": segment.end,
            "speaker": speaker,
        })
    #
    logger.debug(f"Diarization found {len(speaker_segments)} speaker segments")
    return speaker_segments
