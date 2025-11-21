import asyncio
import itertools
import logging
from collections import defaultdict
from collections.abc import Generator, Iterable
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    Form,
    HTTPException,
    Request,
    Response,
)
from fastapi.responses import StreamingResponse
from faster_whisper.transcribe import BatchedInferencePipeline, TranscriptionInfo
from huggingface_hub.utils._cache_manager import _scan_cached_repo

from speaches.api_types import (
    DEFAULT_TIMESTAMP_GRANULARITIES,
    TIMESTAMP_GRANULARITIES_COMBINATIONS,
    CreateTranscriptionResponseJson,
    CreateTranscriptionResponseVerboseJson,
    TimestampGranularities,
    TranscriptionSegment,
)
from speaches.dependencies import (
    AudioFileDependency,
    ConfigDependency,
    ParakeetModelManagerDependency,
    PyannoteModelManagerDependency,
    WhisperModelManagerDependency,
)
from speaches.executors.parakeet import utils as nemo_conformer_tdt_utils
from speaches.executors.pyannote.utils import run_diarization
from speaches.executors.whisper import utils as whisper_utils
from speaches.hf_utils import (
    MODEL_CARD_DOESNT_EXISTS_ERROR_MESSAGE,
    get_model_card_data_from_cached_repo_info,
    get_model_repo_path,
)
from speaches.model_aliases import ModelId
from speaches.text_utils import segments_to_srt, segments_to_text, segments_to_vtt

logger = logging.getLogger(__name__)

router = APIRouter(tags=["automatic-speech-recognition"])

type ResponseFormat = Literal["text", "json", "verbose_json", "srt", "vtt"]

# https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-response_format
DEFAULT_RESPONSE_FORMAT: ResponseFormat = "json"
MIN_SILENCE_DURATION_MS = 250


def segments_to_response(
    segments: Iterable[TranscriptionSegment],
    transcription_info: TranscriptionInfo,
    response_format: ResponseFormat,
    speaker_segments: list[dict] | None = None,
) -> Response:
    segments = list(segments)
    match response_format:
        case "text":
            return Response(segments_to_text(segments), media_type="text/plain")
        case "json":
            return Response(
                CreateTranscriptionResponseJson.from_segments(segments).model_dump_json(),
                media_type="application/json",
            )
        case "verbose_json":
            return Response(
                CreateTranscriptionResponseVerboseJson.from_segments(segments, transcription_info, speaker_segments).model_dump_json(),
                media_type="application/json",
            )
        case "vtt":
            return Response(
                "".join(segments_to_vtt(segment, i) for i, segment in enumerate(segments)), media_type="text/vtt"
            )
        case "srt":
            return Response(
                "".join(segments_to_srt(segment, i) for i, segment in enumerate(segments)), media_type="text/plain"
            )


def format_as_sse(data: str) -> str:
    return f"data: {data}\n\n"


def segments_to_streaming_response(
    segments: Iterable[TranscriptionSegment],
    transcription_info: TranscriptionInfo,
    response_format: ResponseFormat,
    speaker_segments: list[dict] | None = None,
) -> StreamingResponse:
    def segment_responses() -> Generator[str, None, None]:
        for i, segment in enumerate(segments):
            if response_format == "text":
                data = segment.text
            elif response_format == "json":
                data = CreateTranscriptionResponseJson.from_segments([segment]).model_dump_json()
            elif response_format == "verbose_json":
                data = CreateTranscriptionResponseVerboseJson.from_segment(
                    segment, transcription_info, speaker_segments
                ).model_dump_json()
            elif response_format == "vtt":
                data = segments_to_vtt(segment, i)
            elif response_format == "srt":
                data = segments_to_srt(segment, i)
            yield format_as_sse(data)

    return StreamingResponse(segment_responses(), media_type="text/event-stream")


@router.post(
    "/v1/audio/translations",
    response_model=str | CreateTranscriptionResponseJson | CreateTranscriptionResponseVerboseJson,
)
def translate_file(
    config: ConfigDependency,
    pyannote_manager: PyannoteModelManagerDependency,
    whisper_model_manager: WhisperModelManagerDependency,
    request: Request,
    audio: AudioFileDependency,
    model: Annotated[ModelId, Form()],
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = DEFAULT_RESPONSE_FORMAT,
    temperature: Annotated[float, Form()] = 0.0,
    stream: Annotated[bool, Form()] = False,
    vad_filter: Annotated[bool | None, Form()] = None,
    diarization: Annotated[bool | None, Form()] = None,
) -> Response | StreamingResponse:
    # Use config default if vad_filter not explicitly provided
    effective_vad_filter = vad_filter if vad_filter is not None else config._unstable_vad_filter  # noqa: SLF001
    effective_diarization = diarization if diarization is not None else config._unstable_diarization  # noqa: SLF001
    timestamp_granularities = asyncio.run(get_timestamp_granularities(request))

    if isinstance(audio, tuple):
        left, right = audio
        channels = [left, right]
    else:
        channels = [audio]
    # Run diarization if enabled
    speaker_segments = None
    if effective_diarization:
        with pyannote_manager.load_model("pyannote/speaker-diarization-community-1") as pipeline:
            speaker_segments = run_diarization(audio, pipeline)

    with whisper_model_manager.load_model(model) as whisper:
        whisper_model = BatchedInferencePipeline(model=whisper) if config.whisper.use_batched_mode else whisper
        all_set_segments = []
        all_transcription_info = []
        for ch_audio in channels:
            segments, transcription_info = whisper_model.transcribe(
                ch_audio,
                task="translate",
                initial_prompt=prompt,
                temperature=temperature,
                word_timestamps="word" in timestamp_granularities,
                vad_filter=effective_vad_filter,
                vad_parameters=dict(min_silence_duration_ms=MIN_SILENCE_DURATION_MS),
            )
            all_transcription_info.append(transcription_info)
            all_set_segments.append(segments)

        grouped = defaultdict(list)
        for seg in speaker_segments:
            grouped[seg["speaker"]].append(seg)
        diarize_groups = list(grouped.values())

        all_segments = []
        if len(all_set_segments) == 2:
            all_set_segments = [list(g) for g in all_set_segments]
            all_set_segments.sort(key=lambda group: group[0].start)
            for idx, segment in enumerate(all_set_segments):
                all_segments.extend(TranscriptionSegment.from_faster_whisper_segments(segment, diarize_groups[idx]))
            all_segments = TranscriptionSegment.regenerate_segments(all_segments)
        else:
            all_segments = list(itertools.chain(*[list(g) for g in all_set_segments]))
            all_segments = TranscriptionSegment.from_faster_whisper_segments(all_segments, speaker_segments)

        if stream:
            return segments_to_streaming_response(all_segments, transcription_info, response_format, speaker_segments)
        else:
            return segments_to_response(all_segments, transcription_info, response_format, speaker_segments)


# HACK: Since Form() doesn't support `alias`, we need to use a workaround.
async def get_timestamp_granularities(request: Request) -> TimestampGranularities:
    form = await request.form()
    if form.get("timestamp_granularities[]") is None:
        return DEFAULT_TIMESTAMP_GRANULARITIES
    timestamp_granularities = form.getlist("timestamp_granularities[]")
    assert timestamp_granularities in TIMESTAMP_GRANULARITIES_COMBINATIONS, (
        f"{timestamp_granularities} is not a valid value for `timestamp_granularities[]`."
    )
    return timestamp_granularities  # type: ignore[return-value]


# https://platform.openai.com/docs/api-reference/audio/createTranscription
# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L8915
@router.post(
    "/v1/audio/transcriptions",
    response_model=str | CreateTranscriptionResponseJson | CreateTranscriptionResponseVerboseJson,
)
def transcribe_file(  # noqa: C901
    config: ConfigDependency,
    pyannote_manager: PyannoteModelManagerDependency,
    whisper_model_manager: WhisperModelManagerDependency,
    parakeet_model_manager: ParakeetModelManagerDependency,
    request: Request,
    audio: AudioFileDependency,
    model: Annotated[ModelId, Form()],
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = DEFAULT_RESPONSE_FORMAT,
    temperature: Annotated[float, Form()] = 0.0,
    timestamp_granularities: Annotated[
        TimestampGranularities,
        # WARN: `alias` doesn't actually work.
        Form(alias="timestamp_granularities[]"),
    ] = ["segment"],
    stream: Annotated[bool, Form()] = False,
    hotwords: Annotated[str | None, Form()] = None,
    vad_filter: Annotated[bool | None, Form()] = None,
    without_timestamps: Annotated[bool | None, Form()] = None,
    diarization: Annotated[bool | None, Form()] = None,
) -> Response | StreamingResponse:
    # Use config default if vad_filter not explicitly provided
    effective_vad_filter = vad_filter if vad_filter is not None else config._unstable_vad_filter  # noqa: SLF001
    effective_diarization = diarization if diarization is not None else config._unstable_diarization  # noqa: SLF001

    timestamp_granularities = asyncio.run(get_timestamp_granularities(request))
    if timestamp_granularities != DEFAULT_TIMESTAMP_GRANULARITIES and response_format != "verbose_json":
        logger.warning(
            "It only makes sense to provide `timestamp_granularities[]` when `response_format` is set to `verbose_json`. See https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-timestamp_granularities."
        )

    # Run diarization if enabled
    speaker_segments = None
    if effective_diarization:
        with pyannote_manager.load_model("pyannote/speaker-diarization-community-1") as pipeline:
            speaker_segments = run_diarization(audio, pipeline)

    model_repo_path = get_model_repo_path(model)
    if model_repo_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' is not installed locally. You can download the model using `POST /v1/models`",
        )
    cached_repo_info = _scan_cached_repo(model_repo_path)
    model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
    if model_card_data is None:
        raise HTTPException(
            status_code=500,
            detail=MODEL_CARD_DOESNT_EXISTS_ERROR_MESSAGE.format(model_id=model),
        )
    if whisper_utils.hf_model_filter.passes_filter(model, model_card_data):
        with whisper_model_manager.load_model(model) as whisper:
            whisper_model = BatchedInferencePipeline(model=whisper) if config.whisper.use_batched_mode else whisper
            # Check if audio is stereo (tuple returned by decode_audio)
            if isinstance(audio, tuple):
                left, right = audio
                channels = [left, right]
            else:
                channels = [audio]

            all_set_segments = []
            all_transcription_info = []
            for ch_audio in channels:
                segments, transcription_info = whisper_model.transcribe(
                    ch_audio,
                    task="transcribe",
                    language=language,
                    initial_prompt=prompt,
                    word_timestamps="word" in timestamp_granularities,
                    temperature=temperature,
                    vad_filter=effective_vad_filter,
                    vad_parameters=dict(min_silence_duration_ms=MIN_SILENCE_DURATION_MS),
                    hotwords=hotwords,
                    without_timestamps=without_timestamps,
                )
                all_transcription_info.append(transcription_info)
                all_set_segments.append(segments)

            grouped = defaultdict(list)
            for seg in speaker_segments:
                grouped[seg["speaker"]].append(seg)
            diarize_groups = list(grouped.values())

            all_segments = []
            if len(all_set_segments) == 2:
                all_set_segments = [list(g) for g in all_set_segments]
                all_set_segments.sort(key=lambda group: group[0].start)
                for idx, segment in enumerate(all_set_segments):
                    all_segments.extend(TranscriptionSegment.from_faster_whisper_segments(segment, diarize_groups[idx]))
                all_segments = TranscriptionSegment.regenerate_segments(all_segments)
            else:
                all_segments = list(itertools.chain(*[list(g) for g in all_set_segments]))
                all_segments = TranscriptionSegment.from_faster_whisper_segments(all_segments, speaker_segments)

            if stream:
                return segments_to_streaming_response(all_segments, transcription_info, response_format, speaker_segments)
            else:
                return segments_to_response(all_segments, transcription_info, response_format, speaker_segments)
    elif nemo_conformer_tdt_utils.hf_model_filter.passes_filter(model, model_card_data):
        if stream:
            raise HTTPException(status_code=500, detail=f"Model '{model}' does not support streaming yet.")
        if response_format not in ("text", "json"):
            raise HTTPException(
                status_code=500, detail=f"Model '{model}' only supports 'text' and 'json' response formats for now."
            )
        with parakeet_model_manager.load_model(model) as parakeet:
            # TODO: issue warnings when client specifies unsupported parameters like `prompt`, `temperature`, `hotwords`, etc.
            timestamped_result = parakeet.with_timestamps().recognize(audio)

            match response_format:
                case "text":
                    return Response(timestamped_result.text, media_type="text/plain")
                case "json":
                    return Response(
                        CreateTranscriptionResponseJson(
                            text=timestamped_result.text,
                        ).model_dump_json(),
                        media_type="application/json",
                    )
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' is not supported. If you think this is a mistake, please open an issue.",
        )
