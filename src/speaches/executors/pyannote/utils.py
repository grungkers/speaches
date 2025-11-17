import io
import logging

import torchaudio
from numpy import float32
from numpy.typing import NDArray
from pyannote.audio import Pipeline
from pyannote.core import Annotation
import soundfile as sf
from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput

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

    # Convert numpy array to audio format that pyannote can process
    # Create an in-memory file-like object
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)

    # 2. Load buffer with torchaudio (returns tensor)
    waveform, sr = torchaudio.load(buffer)
    # Run diarization
    output = pipeline({"waveform": waveform, "sample_rate": sr})

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
