from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import assemblyai as aai
import tempfile
import os
import textstat
import re
from transformers import pipeline
from pydub import AudioSegment
from pydub.utils import which

# 🔧 Set up local ffmpeg path (portable fallback)
ffmpeg_folder = os.path.join(os.getcwd(), "ffmpeg")
if os.path.exists(ffmpeg_folder):
    bin_dir = os.path.join(ffmpeg_folder, os.listdir(ffmpeg_folder)[0], "bin")
    ffmpeg_path = os.path.join(bin_dir, "ffmpeg.exe")
    AudioSegment.converter = ffmpeg_path

# 🗝️ AssemblyAI API key
aai.settings.api_key = "de3767299c094bc690cf6110e6ee46cf"
transcriber = aai.Transcriber()

# 🔈 Transcription config
config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.slam_1)

# 🤖 Load emotion classifier
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

# 🚀 FastAPI app
app = FastAPI()

# 🌐 CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🎧 Helper: Convert uploaded audio to WAV
def convert_to_wav(path):
    audio = AudioSegment.from_file(path)
    wav_path = path.rsplit(".", 1)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

# 📤 POST: Analyze Audio
@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        print("Received:", file.filename)
        suffix = os.path.splitext(file.filename)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            temp_file.write(await file.read())

        wav_path = convert_to_wav(temp_path)

        # 🎙️ Transcribe audio
        result = transcriber.transcribe(wav_path, config=config)
        if result.error:
            raise Exception(result.error)
        text = result.text
        print("Transcript:", text)

        # 📊 Readability score
        readability = textstat.flesch_reading_ease(text)

        # 🔁 Disfluency count
        junk_words = ["um", "uh", "you know", "like", "i mean", "ah", "hmm", "erm"]
        disfluency_count = sum(len(re.findall(rf"\b{w}\b", text.lower())) for w in junk_words)

        # 😊 Emotion detection
        emotion_scores = emotion_classifier(text)[0]
        top_emotion = max(emotion_scores, key=lambda x: x["score"])
        emotion_breakdown = [{"emotion": e["label"], "score": round(e["score"], 2)} for e in emotion_scores]

        # 🗣️ Persuasiveness
        persuasive_keywords = [
            "guarantee", "proven", "effective", "must", "should", "need to",
            "save", "discover", "limited time", "now"
        ]
        persuasiveness_score = sum(
            len(re.findall(rf"\b{word}\b", text.lower())) for word in persuasive_keywords
        )

        # 🧼 Suggested change (clean transcript)
        clean_result = transcriber.transcribe(wav_path, config=config)
        suggested_change = clean_result.text

        # 🧹 Cleanup
        os.remove(temp_path)
        os.remove(wav_path)

        # ✅ Return results
        return {
            "transcript": text,
            "readability": round(readability, 2),
            "disfluencyCount": disfluency_count,
            "dominantEmotion": top_emotion["label"],
            "emotionScores": emotion_breakdown,
            "persuasivenessScore": min(persuasiveness_score, 10),
            "suggestedChange": suggested_change
        }

    except Exception as e:
        return {"error": str(e)}