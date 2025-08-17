import os
import torchaudio
import json
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel

# ✅ Hugging Face authentication (no secrets in code)
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("❌ No Hugging Face token found. Run `huggingface-cli login` or set HF_TOKEN env variable.")

# ✅ Input audio
audio_file = r"D:\PROJECTS\AudioPs6\output\cleaned_audio1.wav"
json_output_path = r"D:\PROJECTS\AudioPs6\output\diarised_transcript.json"

print("🔥 Token detected. Starting processing...")

# ✅ Load diarisation model
print("🔊 Loading diarisation model...")
diarisation_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HF_TOKEN
)

# ✅ Load Whisper model
print("🧠 Loading Faster-Whisper...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# ✅ Diarise audio
print(f"📂 Analyzing: {audio_file}")
diarisation = diarisation_pipeline(audio_file)

# ✅ Load waveform
waveform, sample_rate = torchaudio.load(audio_file)

results = []

# ✅ Loop through diarisation turns
for turn, _, speaker in diarisation.itertracks(yield_label=True):
    start_sample = int(turn.start * sample_rate)
    end_sample = int(turn.end * sample_rate)
    segment_audio = waveform[:, start_sample:end_sample]

    temp_file = "temp_segment.wav"
    torchaudio.save(temp_file, segment_audio, sample_rate)

    segments, _ = whisper_model.transcribe(temp_file)
    transcript = " ".join([seg.text for seg in segments]).strip()

    results.append({
        "start_time": round(turn.start, 2),
        "end_time": round(turn.end, 2),
        "speaker": speaker,
        "transcript": transcript
    })

with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

os.remove("temp_segment.wav")
print(f"\n✅ Done! JSON saved to {json_output_path}")
