from faster_whisper import WhisperModel
import pyaudio
import wave
import pyaudio
import wave
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=2048)
print("Recording... Press Ctrl+C to stop")
frames = []
try:
    while True:
        data = stream.read(2048)
        frames.append(data)
except KeyboardInterrupt:
    print("Recording stopped")
stream.stop_stream()
stream.close()
p.terminate()
filename = "speech.wav"
wf = wave.open(filename, 'wb')
wf.setnchannels(1)
wf.setsampwidth(2)
wf.setframerate(44100)
wf.writeframes(b''.join(frames))
wf.close()
print(f"Recording saved to {filename}")
model_size = "large-v3"
model = WhisperModel(model_size, device="cpu", compute_type="int8")
segments, info = model.transcribe("speech.wav", beam_size=5)
print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
segments, _ = model.transcribe(
    "speech.wav",
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
)
segments = list(segments)
with open('output.txt', 'w') as f:
    f.write(info.language+'\n')
    for segment in segments:
        f.write(segment.text+'\n')