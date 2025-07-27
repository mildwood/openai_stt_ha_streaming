from __future__ import annotations

from collections.abc import AsyncIterable
import io
import logging
import wave

import httpx
import voluptuous as vol

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    Provider,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
)
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.httpx_client import get_async_client

_LOGGER = logging.getLogger(__name__)


CONF_API_KEY = "api_key"
CONF_API_URL = "api_url"
CONF_MODEL = "model"
CONF_PROMPT = "prompt"
CONF_TEMP = "temperature"

DEFAULT_API_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_PROMPT = ""
DEFAULT_TEMP = 0

SUPPORTED_MODELS = [
    "whisper-1",
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe",
]

SUPPORTED_LANGUAGES = [
    "af",
    "ar",
    "hy",
    "az",
    "be",
    "bs",
    "bg",
    "ca",
    "zh",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "gl",
    "de",
    "el",
    "he",
    "hi",
    "hu",
    "is",
    "id",
    "it",
    "ja",
    "kn",
    "kk",
    "ko",
    "lv",
    "lt",
    "mk",
    "ms",
    "mr",
    "mi",
    "ne",
    "no",
    "fa",
    "pl",
    "pt",
    "ro",
    "ru",
    "sr",
    "sk",
    "sl",
    "es",
    "sw",
    "sv",
    "tl",
    "ta",
    "th",
    "tr",
    "uk",
    "ur",
    "vi",
    "cy",
]

MODEL_SCHEMA = vol.In(SUPPORTED_MODELS)

PLATFORM_SCHEMA = cv.PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_API_KEY): cv.string,
        vol.Optional(CONF_API_URL, default=DEFAULT_API_URL): cv.string,
        vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): MODEL_SCHEMA,
        vol.Optional(CONF_PROMPT, default=DEFAULT_PROMPT): cv.string,
        vol.Optional(CONF_TEMP, default=DEFAULT_TEMP): cv.positive_int,
    }
)


async def async_get_engine(hass, config, discovery_info=None):
    """Set up the OpenAI STT component."""
    api_key = config[CONF_API_KEY]
    api_url = config.get(CONF_API_URL, DEFAULT_API_URL)
    model = config.get(CONF_MODEL, DEFAULT_MODEL)
    prompt = config.get(CONF_PROMPT, DEFAULT_PROMPT)
    temperature = config.get(CONF_TEMP, DEFAULT_TEMP)
    return OpenAISTTProvider(hass, api_key, api_url, model, prompt, temperature)


class OpenAISTTProvider(Provider):
    """The OpenAI STT provider."""

    def __init__(self, hass, api_key, api_url, model, prompt, temperature) -> None:
        """Init OpenAI STT service."""
        self.hass = hass
        self.name = "OpenAI STT"

        self._api_key = api_key
        self._api_url = api_url
        self._model = model
        self._prompt = prompt
        self._temperature = temperature
        self._client = get_async_client(hass)

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return SUPPORTED_LANGUAGES

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return a list of supported formats."""
        return [AudioFormats.WAV, AudioFormats.OGG]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return a list of supported codecs."""
        return [AudioCodecs.PCM, AudioCodecs.OPUS]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return a list of supported bitrates."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return a list of supported samplerates."""
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return a list of supported channels."""
        return [AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Process audio stream in real-time chunks."""
        _LOGGER.debug(
            "Start streaming audio processing for language: %s", metadata.language
        )
    
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
    
        # Initialize streaming session with your backend
        session_data = {
            "model": self._model,
            "language": "hu-HU" if metadata.language == "hu" else metadata.language,  # FIX: Ensure proper language format
            "prompt": self._prompt,
            "temperature": self._temperature,
            "response_format": "json",
            "stream": True,  # Enable streaming mode
            "sample_rate": metadata.sample_rate,
            "channels": metadata.channel,
            "bit_rate": metadata.bit_rate
        }
    
        session_id = None  # Initialize session_id
        
        try:
            # Start streaming session
            session_url = f"{self._api_url}/audio/transcriptions/stream"
            
            _LOGGER.debug("Initializing streaming session with data: %s", session_data)
            
            # Initialize the streaming session
            init_response = await self._client.post(
                session_url,
                headers=headers,
                json=session_data,
                timeout=httpx.Timeout(30.0)
            )
            init_response.raise_for_status()
            session_info = init_response.json()
            session_id = session_info.get("session_id")
            
            if not session_id:
                _LOGGER.error("Failed to initialize streaming session - no session_id received")
                return SpeechResult("", SpeechResultState.ERROR)
    
            _LOGGER.info("Streaming session initialized: %s", session_id)
    
            # Stream audio chunks in real-time
            chunk_url = f"{self._api_url}/audio/transcriptions/stream/{session_id}/chunk"
            chunk_count = 0
            
            async for chunk in stream:
                if not chunk:
                    continue
                    
                chunk_count += 1
                _LOGGER.debug("Sending audio chunk #%d: %d bytes", chunk_count, len(chunk))
                
                # Convert raw audio chunk to WAV format
                wav_chunk = self._convert_chunk_to_wav(chunk, metadata)
                
                # Send chunk to streaming endpoint
                files = {
                    "audio_chunk": ("chunk.wav", wav_chunk, "audio/wav"),
                }
                
                # FIX: Remove session_id from files, it's in the URL
                chunk_response = await self._client.post(
                    chunk_url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    files=files,
                    timeout=httpx.Timeout(10.0)  # Increased timeout
                )
                
                if chunk_response.status_code == 200:
                    chunk_result = chunk_response.json()
                    partial_text = chunk_result.get("partial_text", "")
                    is_final = chunk_result.get("is_final", False)
                    
                    if partial_text:
                        _LOGGER.debug("Partial transcription: %s (final: %s)", partial_text, is_final)
                else:
                    _LOGGER.warning("Chunk processing failed with status %s: %s", 
                                  chunk_response.status_code, chunk_response.text)
    
            _LOGGER.info("Finished sending %d audio chunks, finalizing session %s", chunk_count, session_id)
    
            # FIX: Critical - Finalize the streaming session and handle response properly
            finalize_url = f"{self._api_url}/audio/transcriptions/stream/{session_id}/finalize"
            final_response = await self._client.post(
                finalize_url,
                headers={"Authorization": f"Bearer {self._api_key}"},  # FIX: Use proper headers
                timeout=httpx.Timeout(20.0)  # Increased timeout for processing
            )
            
            _LOGGER.debug("Finalize response status: %s", final_response.status_code)
            
            if final_response.status_code == 200:
                final_result = final_response.json()
                final_text = final_result.get("text", "")  # FIX: Server returns "text", not "final_text"
                
                _LOGGER.info("Final transcription received: '%s'", final_text)
                
                # FIX: Check if final_text is empty or None
                if not final_text or final_text.strip() == "":
                    _LOGGER.error("Empty final transcription received from server")
                    return SpeechResult("", SpeechResultState.ERROR)
                
                return SpeechResult(final_text.strip(), SpeechResultState.SUCCESS)
            
            elif final_response.status_code == 400:
                # Handle "No speech detected" error
                error_msg = "No speech detected or transcription failed"
                try:
                    error_detail = final_response.json().get("error", error_msg)
                    _LOGGER.warning("Transcription failed: %s", error_detail)
                except:
                    _LOGGER.warning("Transcription failed with 400 error")
                return SpeechResult("", SpeechResultState.ERROR)
            
            elif final_response.status_code == 408:
                # Handle timeout error
                _LOGGER.error("Speech recognition timed out")
                return SpeechResult("", SpeechResultState.ERROR)
            
            else:
                _LOGGER.error("Failed to finalize session with status %s: %s", 
                            final_response.status_code, final_response.text)
                return SpeechResult("", SpeechResultState.ERROR)
    
        except httpx.TimeoutException as err:
            _LOGGER.error("Request timeout during streaming: %s", err)
            return SpeechResult("", SpeechResultState.ERROR)
        except httpx.HTTPStatusError as err:
            _LOGGER.error("HTTP status error: %s - %s", err.response.status_code, err.response.text)
            return SpeechResult("", SpeechResultState.ERROR)
        except httpx.HTTPError as err:
            _LOGGER.error("HTTP error during streaming: %s", err)
            return SpeechResult("", SpeechResultState.ERROR)
        except Exception as err:
            _LOGGER.error("Unexpected error during streaming: %s", err, exc_info=True)
            return SpeechResult("", SpeechResultState.ERROR)
        
        finally:
            # FIX: Cleanup - if session was created but not properly finalized
            if session_id and hasattr(self, '_cleanup_needed'):
                try:
                    cleanup_url = f"{self._api_url}/audio/transcriptions/stream/{session_id}/cleanup"
                    await self._client.delete(cleanup_url, timeout=httpx.Timeout(5.0))
                except:
                    pass  # Ignore cleanup errors
    
    def _convert_chunk_to_wav(self, chunk_data: bytes, metadata: SpeechMetadata) -> bytes:
        """Convert raw audio chunk to WAV format."""
        try:
            wav_stream = io.BytesIO()
            
            with wave.open(wav_stream, "wb") as wf:
                wf.setnchannels(metadata.channel)
                wf.setsampwidth(metadata.bit_rate // 8)
                wf.setframerate(metadata.sample_rate)
                wf.writeframes(chunk_data)
            
            wav_data = wav_stream.getvalue()
            _LOGGER.debug("Converted %d bytes of raw audio to %d bytes WAV", len(chunk_data), len(wav_data))
            return wav_data
            
        except Exception as e:
            _LOGGER.error("Error converting chunk to WAV: %s", e)
            # Return empty WAV header as fallback
            wav_stream = io.BytesIO()
            with wave.open(wav_stream, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"")
            return wav_stream.getvalue()
