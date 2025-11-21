from collections.abc import Iterable
from typing import Literal

import faster_whisper.transcribe
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from speaches.text_utils import segments_to_text


# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L10909
class TranscriptionWord(BaseModel):
    start: float
    end: float
    word: str
    probability: float
    speaker: str | None = None

    @classmethod
    def from_segments(cls, segments: Iterable["TranscriptionSegment"]) -> list["TranscriptionWord"]:
        words: list[TranscriptionWord] = []
        for segment in segments:
            # NOTE: a temporary "fix" for https://github.com/speaches-ai/speaches/issues/58.
            # TODO: properly address the issue
            assert segment.words is not None, (
                "Segment must have words. If you are using an API ensure `timestamp_granularities[]=word` is set"
            )
            words.extend(segment.words)
        return words

    def offset(self, seconds: float) -> None:
        self.start += seconds
        self.end += seconds


# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L10938
class TranscriptionSegment(BaseModel):
    avg_logprob: float
    compression_ratio: float
    end: float
    id: int
    no_speech_prob: float
    seek: int
    start: float
    temperature: float
    text: str
    tokens: list[int] | None = None
    words: (
        list[TranscriptionWord] | None
    )  # TODO: why is here? It's not a field defined in the [OpenAI API spec](https://platform.openai.com/docs/api-reference/audio/verbose-json-object)
    # TODO: add `usage` field: https://platform.openai.com/docs/api-reference/audio/verbose-json-object#audio/verbose-json-object-usage
    speaker: str | None = None

    @classmethod
    def from_faster_whisper_segments(
        cls, segments: Iterable[faster_whisper.transcribe.Segment], diarization: list[dict]
    ) -> Iterable["TranscriptionSegment"]:
        diarize_df = pd.DataFrame(diarization)
        for segment in segments:
            speaker = assign_speaker(diarize_df, segment.start, segment.end)
            yield cls(
                id=segment.id,
                seek=segment.seek,
                start=segment.start,
                end=segment.end,
                text=segment.text,
                temperature=segment.temperature or 0,  # FIX: hardcoded
                avg_logprob=segment.avg_logprob,
                compression_ratio=segment.compression_ratio,
                no_speech_prob=segment.no_speech_prob,
                speaker=speaker,
                words=[
                    TranscriptionWord(
                        start=word.start,
                        end=word.end,
                        word=word.word,
                        probability=word.probability,
                        speaker=speaker,
                    )
                    for word in segment.words
                ]
                if segment.words is not None
                else None,
            )

    @classmethod
    def regenerate_segments(
            cls, segments: Iterable["TranscriptionSegment"],
            min_silence: float = 0.5
    ) -> Iterable["TranscriptionSegment"]:
        # 1. Flatten all words
        all_words = []
        for seg in segments:
            if seg.words:
                all_words.extend(seg.words)

        all_words.sort(key=lambda w: w.start)

        # 2. Group words into new segments
        new_segments = []
        current_words = []
        seg_id = 0

        for i, word in enumerate(all_words):
            if not current_words:
                current_words.append(word)
                continue

            last_word = current_words[-1]
            # Check for silence or speaker change
            if (word.start - last_word.end) > min_silence or word.speaker != last_word.speaker:
                # create a new segment
                seg_id += 1
                new_seg = TranscriptionSegment(
                    avg_logprob=sum(w.probability for w in current_words) / len(current_words),
                    compression_ratio=1.0,  # could compute if needed
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    id=seg_id,
                    no_speech_prob=0.0,
                    seek=0,
                    temperature=0.0,
                    text="".join(w.word for w in current_words),
                    words=current_words.copy(),
                    speaker=current_words[0].speaker
                )
                new_segments.append(new_seg)
                current_words = [word]
            else:
                current_words.append(word)

        # Add last segment
        if current_words:
            seg_id += 1
            new_segments.append(
                TranscriptionSegment(
                    avg_logprob=sum(w.probability for w in current_words) / len(current_words),
                    compression_ratio=1.0,
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    id=seg_id,
                    no_speech_prob=0.0,
                    seek=0,
                    temperature=0.0,
                    text="".join(w.word for w in current_words),
                    words=current_words.copy(),
                    speaker=current_words[0].speaker
                )
            )

        return new_segments


def assign_speaker(df, start, end):
    """Return speaker label based on overlap with diarization."""
    if df is None or len(df) == 0:
        return None

    df["intersection"] = np.minimum(df["end"], end) - np.maximum(df["start"], start)
    df["intersection"] = df["intersection"].clip(lower=0)

    if df["intersection"].sum() == 0:
        return None

    return df.groupby("speaker")["intersection"].sum().idxmax()

# https://platform.openai.com/docs/api-reference/audio/json-object
# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L10924
class CreateTranscriptionResponseJson(BaseModel):
    text: str
    # NOTE: there's also a `logprobs` field it's only supported by non-whisper models, so we don't include it here (we can't `faster-whisper` doesn't provide it)
    # TODO: add `usage` field: https://platform.openai.com/docs/api-reference/audio/json-object#audio/json-object-usage

    @classmethod
    def from_segments(cls, segments: list[TranscriptionSegment]) -> "CreateTranscriptionResponseJson":
        return cls(text=segments_to_text(segments))


# https://platform.openai.com/docs/api-reference/audio/verbose-json-object
# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L11007
class CreateTranscriptionResponseVerboseJson(BaseModel):
    # NOTE: there's also a `logprobs` field it's only supported by non-whisper models, so we don't include it here (we can't `faster-whisper` doesn't provide it)
    task: str = "transcribe"
    language: str
    duration: float
    text: str
    words: list[TranscriptionWord] | None
    segments: list[TranscriptionSegment]
    speaker_segments: list[dict] | None = None,

    @classmethod
    def from_segment(
        cls, segment: TranscriptionSegment, transcription_info: faster_whisper.transcribe.TranscriptionInfo,
        speaker_segments: list[dict] | None = None,
    ) -> "CreateTranscriptionResponseVerboseJson":
        return cls(
            language=transcription_info.language,
            duration=segment.end - segment.start,
            text=segment.text,
            words=segment.words if transcription_info.transcription_options.word_timestamps else None,
            segments=[segment],
            speaker_segments=speaker_segments,
        )

    @classmethod
    def from_segments(
        cls, segments: list[TranscriptionSegment], transcription_info: faster_whisper.transcribe.TranscriptionInfo,
        speaker_segments: list[dict] | None = None,
        is_stereo: bool = False,
    ) -> "CreateTranscriptionResponseVerboseJson":
        return cls(
            language=transcription_info.language,
            duration=transcription_info.duration,
            text=segments_to_text(segments),
            segments=segments,
            words=TranscriptionWord.from_segments(segments)
            if transcription_info.transcription_options.word_timestamps
            else None,
            speaker_segments=speaker_segments,
        )


ModelTask = Literal["automatic-speech-recognition", "text-to-speech"]  # TODO: add "voice-activity-detection"


# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L11146
class Model(BaseModel):
    """There may be additional fields in the response that are specific to the model type."""

    id: str
    """The model identifier, which can be referenced in the API endpoints."""
    created: int = 0
    """The Unix timestamp (in seconds) when the model was created."""
    object: Literal["model"] = "model"
    """The object type, which is always "model"."""
    owned_by: str
    """The organization that owns the model."""
    language: list[str] | None = None
    """List of ISO 639-3 supported by the model. It's possible that the list will be empty. This field is not a part of the OpenAI API spec and is added for convenience."""

    task: ModelTask  # TODO: make a list?

    model_config = ConfigDict(extra="allow")


# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L8730
class ListModelsResponse(BaseModel):
    data: list[Model]
    object: Literal["list"] = "list"


# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L10909
TimestampGranularities = list[Literal["segment", "word"]]


DEFAULT_TIMESTAMP_GRANULARITIES: TimestampGranularities = ["segment"]
TIMESTAMP_GRANULARITIES_COMBINATIONS: list[TimestampGranularities] = [
    [],  # should be treated as ["segment"]. https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-timestamp_granularities
    ["segment"],
    ["word"],
    ["word", "segment"],
    ["segment", "word"],  # same as ["word", "segment"] but order is different
]
