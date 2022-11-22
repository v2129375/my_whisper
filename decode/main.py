import whisper
model = whisper.load_model("large")
result = model.transcribe("data/55_Channel-46_20200501001931_896_20200501001946_814_.mp3")
print(result["text"])