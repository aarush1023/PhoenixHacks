import os
import wave
import struct
from pydub import AudioSegment
from faster_whisper import WhisperModel
from pvrecorder import PvRecorder
import struct
import wave
from pydub import AudioSegment

import sounddevice as sd
import soundfile as sf
import sys

def record_to_mp3(filename, samplerate=44100, channels=2):
    print(f"Recording {filename}... Press Ctrl+C to stop.")
    try:
        recording = sd.rec(frames=0, samplerate=samplerate, channels=channels, dtype='int16', blocking=True)
        sf.write(filename, recording, samplerate, format='mp3')
        print(f"Recording saved as {filename}")
    except KeyboardInterrupt:
        print("Recording stopped.")

# Example usage:
record_to_mp3("output.mp3")  # Record until interrupted by Ctrl+C

model_size = "large-v3"

model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("output.mp3", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

segments, _ = model.transcribe(
    "output.mp3",
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
)
segments = list(segments)
