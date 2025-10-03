# import torch
# import torchaudio
# import sounddevice as sd
# import numpy as np
# import whisper
# from transformers import AutoModel

# # Load Whisper for language detection
# whisper_model = whisper.load_model("medium")  # small/medium better accuracy on CPU

# # Load Indic Conformer for ASR
# conformer_model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)

# # Recording settings
# DURATION = 7
# SAMPLE_RATE = 16000
# print(f"🎤 Speak now for {DURATION} seconds...")
# audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
# sd.wait()
# audio = np.squeeze(audio)

# # Whisper expects 30-sec audio → pad/truncate
# audio_30s = whisper.pad_or_trim(torch.from_numpy(audio))

# # Convert to Mel spectrogram
# mel = whisper.log_mel_spectrogram(audio_30s).to(whisper_model.device)

# # Detect language
# print("Detecting language...")
# _, probs = whisper_model.detect_language(mel)
# lang_detected = max(probs, key=probs.get)
# confidence = probs[lang_detected]
# print(f"Detected Language: {lang_detected} (confidence: {confidence:.2f})")

# # Map Whisper → IndicConformer language codes
# lang_map = {
#     "hi": "hi", "bn": "bn", "ta": "ta", "te": "te", "gu": "gu",
#     "ml": "ml", "pa": "pa", "kn": "kn", "mr": "mr", "ur": "ur",
#     "as": "as", "brx": "brx", "doi": "doi", "ks": "ks", "kok": "kok",
#     "mai": "mai", "mni": "mni", "ne": "ne", "or": "or", "sa": "sa",
#     "sd": "sd", "sat": "sat", "en": "en"
# }

# # Fallback if unsupported or low confidence
# if lang_detected not in lang_map or confidence < 0.5:
#     print("⚠️ Low confidence or unsupported language → defaulting to Hindi")
#     lang_code = "hi"
# else:
#     lang_code = lang_map[lang_detected]

# # Run Indic Conformer ASR
# wav = torch.tensor(audio).unsqueeze(0)
# print(f"Transcribing in language: {lang_code} ...")
# transcription = conformer_model(wav, lang_code, "ctc")
# print("Final Transcription:", transcription)


import torch
import torchaudio
import sounddevice as sd
import numpy as np
from transformers import AutoModel

# ---------- Settings ----------
MODEL_NAME = "ai4bharat/indic-conformer-600m-multilingual"
LANG = "pa"  # Change language code: hi, bn, te, ta, etc.
DURATION = 5  # Recording duration in seconds
SAMPLE_RATE = 16000  # Model expects 16kHz

# ---------- Load Model ----------
device = "cpu"  # CPU inference
print(f"Loading model for language: {LANG}")
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)

# ---------- Record Audio ----------
print(f"Speak now... Recording {DURATION} seconds of audio")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
audio = np.squeeze(audio)  # shape: (samples,)

# Convert numpy to torch tensor
wav = torch.tensor(audio).unsqueeze(0)  # (1, samples)

# Resample if needed (not needed since we record at 16kHz)
if SAMPLE_RATE != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=SAMPLE_RATE, new_freq=16000)
    wav = resampler(wav)

# ---------- Perform ASR ----------
print("Running CTC decoding...")
transcription_ctc = model(wav, LANG, "ctc")
print("CTC Transcription:", transcription_ctc)

print("Running RNNT decoding...")
transcription_rnnt = model(wav, LANG, "rnnt")
print("RNNT Transcription:", transcription_rnnt)
