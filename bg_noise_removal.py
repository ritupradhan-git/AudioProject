import os
import argparse
import torch
import torchaudio
import soundfile as sf
from scipy.signal import wiener
from pathlib import Path

print("\n>>> SCRIPT STARTED")

parser = argparse.ArgumentParser(description="Audio Cleaner Script")
parser.add_argument("input_path", type=str, help="Path to the input audio file")
parser.add_argument("-o", "--output_path", type=str, help="Optional path to save cleaned audio")
parser.add_argument("-d", "--denoise", action="store_true", help="Apply denoising filter")
args = parser.parse_args()

print(f">>> Input: {args.input_path}")
print(f">>> Output: {args.output_path if args.output_path else '[auto-generated]'}")
print(f">>> Denoise: {args.denoise}")

print("\n--- Starting Audio Processing ---")

# 1. Load Silero VAD model (fixed: direct download, no Hugging Face secret)
print("1. Loading Silero VAD model...")
torch.set_num_threads(1)  # avoid threading issues

model_url = "https://models.silero.ai/models/vad_models/en/v5_silero_vad.jit"
model_path = torch.hub.download_url_to_file(model_url, "silero_vad.jit")
model = torch.jit.load(model_path)
model.eval()

# Import VAD utils (no huggingface/github dependency)
try:
    from silero_vad import utils
    (get_speech_timestamps, _, read_audio, _, _) = utils
except ImportError:
    raise ImportError("Please install silero-vad utils:\n pip install silero-vad")

# 2. Load and preprocess audio
print(f"2. Loading and preprocessing audio: {args.input_path}")
waveform, sample_rate = torchaudio.load(args.input_path)
waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
sample_rate = 16000
waveform = waveform.squeeze().numpy()
print(f"   - Loaded. Duration: {len(waveform)/sample_rate:.2f}s | Sample Rate: {sample_rate}")

# 3. Run VAD
print("3. Running Voice Activity Detection (VAD)...")
speech_segments = get_speech_timestamps(waveform, model, sampling_rate=sample_rate)
print(f"   - Detected {len(speech_segments)} speech segments.")

# 4. Clip and concatenate segments
print("4. Clipping and concatenating segments...")
cleaned = []
for segment in speech_segments:
    start, end = segment['start'], segment['end']
    cleaned.extend(waveform[start:end])
cleaned = torch.tensor(cleaned)
print(f"   - Cleaned duration: {len(cleaned)/sample_rate:.2f}s")

# 5. Apply denoising (optional)
if args.denoise:
    print("5. Applying Wiener filter for denoising...")
    cleaned = torch.tensor(wiener(cleaned.numpy()))
else:
    print("5. Skipping denoising step.")

# 6. Save output
print("6. Saving output audio...")

if args.output_path:
    output_path = Path(args.output_path)
    if output_path.is_dir():
        base_name = Path(args.input_path).stem
        output_path = output_path / f"cleaned_{base_name}.wav"
else:
    base_name = Path(args.input_path).stem
    output_path = Path(f"cleaned_{base_name}.wav")

try:
    sf.write(output_path, cleaned.numpy(), sample_rate)
    print(f"✅ Saved to: {output_path}")
except Exception as e:
    print(f"[ERROR] Could not save audio: {e}")

print("\n✅ Script Completed!")
