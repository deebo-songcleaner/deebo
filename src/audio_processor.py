import whisper_timestamped as whisper
import json

file = "/Users/aranyeosakwe/deebo/deebo/data/Melanie Martinez - Play Date (Official Audio).mp3"
audio = whisper.load_audio(file)

model = whisper.load_model("medium", device="cpu")

result = whisper.transcribe(model, audio, language="en")

print(result)
