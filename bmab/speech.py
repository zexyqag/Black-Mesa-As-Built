"""Transcribe voice files with Whisper (optional dependency: openai-whisper, a CUDA GPU helps).

WAVs are decoded here (8 or 16-bit PCM) and resampled to 16 kHz, so no ffmpeg is needed.
"""
import io
import re
import wave

import numpy as np

RATE = 16000


def load_wav(data):
    with wave.open(io.BytesIO(data)) as w:
        n, width, rate, ch = w.getnframes(), w.getsampwidth(), w.getframerate(), w.getnchannels()
        raw = w.readframes(n)
    if width == 1:
        a = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    elif width == 2:
        a = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    else:
        raise ValueError(f"unsupported sample width {width}")
    a = a.reshape(-1, ch).mean(1)
    if rate != RATE and len(a) > 1:
        t = np.arange(0, len(a) / rate, 1 / RATE)
        a = np.interp(t, np.arange(len(a)) / rate, a).astype(np.float32)
    return a


class Transcriber:
    def __init__(self, model="large-v3"):
        import whisper  # optional dependency
        self.model = whisper.load_model(model)
        self.name = model

    def __call__(self, wav_bytes, prompt=None):
        audio = load_wav(wav_bytes)
        if len(audio) < RATE // 4:
            return {"text": "", "no_speech": 1.0}
        r = self.model.transcribe(audio, language="en", initial_prompt=prompt, condition_on_previous_text=False,
                                  temperature=0.0, fp16=True)
        segs = r.get("segments") or []
        text = r["text"].strip()
        if prompt and _echoes(text, prompt):  # on non-speech, Whisper can repeat the vocabulary hint
            return {"text": "", "no_speech": 1.0, "prompt_echo": True}
        return {"text": text,
                "no_speech": round(max((s["no_speech_prob"] for s in segs), default=1.0), 2),
                "logprob": round(min((s["avg_logprob"] for s in segs), default=-9.0), 2)}


def _echoes(text, prompt):
    words = set(re.findall(r"[a-z]+", text.lower()))
    hint = set(re.findall(r"[a-z]+", prompt.lower()))
    return bool(words) and len(words & hint) / len(words) > 0.7
