import assemblyai as aai
aai.settings.api_key = "f1ffc7a655454ac6908cf8411c91cf13"
transcriber = aai.Transcriber()
audio_file = "./audio.mp3"
config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.slam_1, disfluencies=True)
transcript = transcriber.transcribe(audio_file, config=config)
if transcript.error:
  print(transcript.error)
  exit(1)
print(transcript.text)
text=transcript.text

junk_words = ["um", "uh", "you know", "like", "I mean","I guess"]
junk_count = sum(text.lower().count(word) for word in junk_words)

print("Disfluency Count:", junk_count)
print("suggested change")
config=aai.TranscriptionConfig(speech_model=aai.SpeechModel.slam_1)
i=transcriber.transcribe(audio_file,config)
print(i.text)

persuasive_words = ["guarantee", "proven", "effective", "must", "should", "need to", "save", "discover", "limited time"]
score = sum(text.lower().count(word) for word in persuasive_words)
print("Persuasiveness Score:", score)
if score==0:

  print("add more persuasive words like")
  for w in persuasive_words:
    print(w)


from transformers import pipeline

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
results = emotion_classifier(text)

top_emotion = max(results[0], key=lambda x: x['score'])
print(f"Dominant Emotion: {top_emotion['label']} ({top_emotion['score']:.2f})")