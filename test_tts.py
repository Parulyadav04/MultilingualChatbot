# import torch
# import wave
# import numpy as np
# from transformers import AutoModel

# # Patch torch.compile for Windows
# if hasattr(torch, "compile"):
#     torch.compile = lambda *args, **kwargs: (lambda model: model)

# # Load the model
# model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)

# # Text input
# text = "नमस्ते, आप कैसे हैं?"

# # Run inference
# audio_tensor = model(
#     text,
#     ref_audio_path="dummy.wav",
#     ref_text=""
# )

# # Convert tensor to numpy
# if isinstance(audio_tensor, torch.Tensor):
#     audio_np = audio_tensor.detach().cpu().numpy()
# else:
#     audio_np = np.array(audio_tensor)

# # Ensure audio is in int16 format
# audio_int16 = np.int16(audio_np * 32767)  # scale float [-1,1] → int16

# # Save WAV
# sample_rate = 16000
# with wave.open("output.wav", "w") as f:
#     f.setnchannels(1)
#     f.setsampwidth(2)
#     f.setframerate(sample_rate)
#     f.writeframes(audio_int16.tobytes())

# print("TTS synthesis complete. Output saved as output.wav")



# import torch
# from transformers import AutoProcessor, AutoModelForTextToWaveform
# import soundfile as sf

# # 1. Load a pretrained TTS model (no reference audio needed)
# model_id = "facebook/mms-tts-hin"  # Hindi TTS model; change if you want English etc.

# processor = AutoProcessor.from_pretrained(model_id)
# model = AutoModelForTextToWaveform.from_pretrained(model_id)

# # 2. Text you want to convert to speech
# text = "नमस्ते, आप कैसे हैं?"

# # 3. Process text
# inputs = processor(text=text, return_tensors="pt")

# # 4. Generate speech
# with torch.no_grad():
#     speech = model(**inputs).waveform.squeeze().cpu().numpy()

# # 5. Save output as WAV
# sf.write("output.wav", speech, model.config.sampling_rate)
# print("TTS synthesis complete. Output saved as output.wav")


# from transformers import AutoModel
# import soundfile as sf
# import numpy as np

# # Load IndicF5
# model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)

# # New text you want to synthesize
# text = "नमस्ते, आप कैसे हैं?"

# # Reference audio details
# ref_audio_path = r"C:\Users\Admin\Chatbot\Indic-TTS\ref_audio.wav"
# ref_text = "नमस्ते, यह एक परीक्षण ऑडियो है।"  # what was spoken in reference audio

# # Generate speech
# audio = model(text=text, ref_audio_path=ref_audio_path, ref_text=ref_text)

# # Convert to float32 for saving
# if isinstance(audio, np.ndarray):
#     if audio.dtype == np.int16:
#         audio = audio.astype(np.float32) / 32768.0
#     elif audio.dtype != np.float32:
#         audio = audio.astype(np.float32)

# sf.write("output.wav", audio, samplerate=24000)
# print("TTS synthesis complete. Output saved as output.wav")
# import sounddevice as sd
# import soundfile as sf
# import numpy as np
# import wave
# from transformers import AutoModel

# # -----------------------------
# # Step 1: Record reference audio
# # -----------------------------
# def record_audio(filename, duration=5, samplerate=16000):
#     print(f"Recording reference audio for {duration} seconds... Speak now!")
#     audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
#     sd.wait()
#     sf.write(filename, audio, samplerate)
#     print(f"Reference audio saved as {filename}")

# # -----------------------------
# # Step 2: Load IndicF5 model
# # -----------------------------
# print("Loading IndicF5 model...")
# model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)

# # -----------------------------
# # Step 3: Record a real reference audio
# # -----------------------------
# ref_audio = "ref_audio.wav"
# record_audio(ref_audio, duration=6)

# # The text you actually spoke in the reference audio
# ref_text = "नमस्ते, मैं आज आपके लिए एक ऑडियो रिकॉर्ड कर रहा हूँ।"

# # -----------------------------
# # Step 4: Text to synthesize
# # -----------------------------
# text_to_speak = "नमस्ते, आप कैसे हैं? यह आवाज़ एआई द्वारा जनरेट की गई है।"

# # -----------------------------
# # Step 5: Run TTS
# # -----------------------------
# print("Synthesizing speech...")
# audio_bytes = model(
#     text_to_speak,
#     ref_audio_path=ref_audio,
#     ref_text=ref_text
# )

# # -----------------------------
# # Step 6: Save proper WAV output
# # -----------------------------
# audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

# out_file = "output.wav"
# with wave.open(out_file, "w") as wf:
#     wf.setnchannels(1)
#     wf.setsampwidth(4)
#     wf.setframerate(16000)
#     wf.writeframes(audio_array.tobytes())

# print(f"TTS synthesis complete. Output saved as {out_file}")

# # -----------------------------
# # Step 7: Play audio 
# # -----------------------------
# print(" Playing generated audio...")
# sd.play(audio_array, 16000)
# sd.wait()
# import torch
# from parler_tts import ParlerTTSForConditionalGeneration
# from transformers import AutoTokenizer
# import soundfile as sf

# device = "cuda:0" if torch.cuda.is_available() else "cpu"

# model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
# tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
# description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

# prompt = "अरे, तुम आज कैसे हो?"
# description = "Divya's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."

# description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
# prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)

# generation = model.generate(input_ids=description_input_ids.input_ids, attention_mask=description_input_ids.attention_mask, prompt_input_ids=prompt_input_ids.input_ids, prompt_attention_mask=prompt_input_ids.attention_mask)
# audio_arr = generation.cpu().numpy().squeeze()
# sf.write("indic_tts_out.wav", audio_arr, model.config.sampling_rate)


# import torch
# from parler_tts import ParlerTTSForConditionalGeneration
# from transformers import AutoTokenizer
# import soundfile as sf
# import os

# device = "cuda:0" if torch.cuda.is_available() else "cpu"

# # Load the model and tokenizer
# model_name = "ai4bharat/indic-parler-tts"
# model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

# # Output directory
# os.makedirs("tts_outputs", exist_ok=True)

# # Sample prompt per language (you can modify if needed)
# prompts = {
#     "Assamese": "আপুনি আজিকালি কেনে আছেন?",
#     "Bengali": "তুমি আজ কেমন আছ?",
#     "Bodo": "निःशब्द मोन हाबो?",
#     "Chhattisgarhi": "तें आज कइसने हस?",
#     "Dogri": "तुम आज कैसे हो?",
#     "English": "Hello, how are you today?",
#     "Gujarati": "તમે આજે કેમ છો?",
#     "Hindi": "अरे, तुम आज कैसे हो?",
#     "Kannada": "ನೀವು ಇಂದು ಹೇಗಿದ್ದೀರಾ?",
#     "Malayalam": "നിങ്ങൾ ഇന്ന് എങ്ങനെയാണ്?",
#     "Manipuri": "নিংগা আজা কসী অছু?",
#     "Marathi": "तुम्ही आज कसे आहात?",
#     "Nepali": "तिमी आज कस्तो छौ?",
#     "Odia": "ଆପଣ କେମିତି ଅଛନ୍ତି?",
#     "Punjabi": "ਤੁਸੀਂ ਅੱਜ ਕਿਵੇਂ ਹੋ?",
#     "Sanskrit": "भवान् अद्य कथम् अस्ति?",
#     "Tamil": "நீங்கள் இன்று எப்படி இருக்கிறீர்கள்?",
#     "Telugu": "మీరు ఈరోజు ఎలా ఉన్నారు?"
# }

# # Recommended speakers per language
# speakers = {
#     "Assamese": "Amit",
#     "Bengali": "Arjun",
#     "Bodo": "Bikram",
#     "Chhattisgarhi": "Bhanu",
#     "Dogri": "Karan",
#     "English": "Thoma",
#     "Gujarati": "Yash",
#     "Hindi": "Divya",
#     "Kannada": "Suresh",
#     "Malayalam": "Anjali",
#     "Manipuri": "Laishram",
#     "Marathi": "Sanjay",
#     "Nepali": "Amrita",
#     "Odia": "Manas",
#     "Punjabi": "Divjot",
#     "Sanskrit": "Aryan",
#     "Tamil": "Jaya",
#     "Telugu": "Prakash"
# }

# # Generate TTS for all languages
# for lang, prompt in prompts.items():
#     speaker = speakers[lang]
#     description = f"{speaker}'s voice, clear and natural."

#     # Tokenize
#     description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
#     prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)

#     # Generate audio
#     with torch.no_grad():
#         generation = model.generate(
#             input_ids=description_input_ids.input_ids,
#             attention_mask=description_input_ids.attention_mask,
#             prompt_input_ids=prompt_input_ids.input_ids,
#             prompt_attention_mask=prompt_input_ids.attention_mask
#         )

#     # Convert to numpy and save
#     audio_arr = generation.cpu().numpy().squeeze()
#     out_file = f"tts_outputs/{lang}_{speaker}.wav"
#     sf.write(out_file, audio_arr, model.config.sampling_rate)
#     print(f"Saved TTS for {lang} ({speaker}) -> {out_file}")

# import torch
# from parler_tts import ParlerTTSForConditionalGeneration
# from transformers import AutoTokenizer
# import soundfile as sf
# import os

# device = "cuda:0" if torch.cuda.is_available() else "cpu"

# # Load the model and tokenizer
# model_name = "ai4bharat/indic-parler-tts"
# model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

# # Output directory
# os.makedirs("tts_outputs", exist_ok=True)

# # Sample prompt per language (you can modify if needed)
# prompts = {
#     "Assamese": "আপুনি আজিকালি কেনে আছেন?",
#     "Bengali": "তুমি আজ কেমন আছ?",
#     "Bodo": "निःशब्द मोन हाबो?",
#     "Chhattisgarhi": "तें आज कइसने हस?",
#     "Dogri": "तुम आज कैसे हो?",
#     "English": "Hello, how are you today?",
#     "Gujarati": "તમે આજે કેમ છો?",
#     "Hindi": "अरे, तुम आज कैसे हो?",
#     "Kannada": "ನೀವು ಇಂದು ಹೇಗಿದ್ದೀರಾ?",
#     "Malayalam": "നിങ്ങൾ ഇന്ന് എങ്ങനെയാണ്?",
#     "Manipuri": "নিংগা আজা কসী অছু?",
#     "Marathi": "तुम्ही आज कसे आहात?",
#     "Nepali": "तिमी आज कस्तो छौ?",
#     "Odia": "ଆପଣ କେମିତି ଅଛନ୍ତି?",
#     "Punjabi": "ਤੁਸੀਂ ਅੱਜ ਕਿਵੇਂ ਹੋ?",
#     "Sanskrit": "भवान् अद्य कथम् अस्ति?",
#     "Tamil": "நீங்கள் இன்று எப்படி இருக்கிறீர்கள்?",
#     "Telugu": "మీరు ఈరోజు ఎలా ఉన్నారు?"
# }

# # Recommended speakers per language
# speakers = {
#     "Assamese": "Amit",
#     "Bengali": "Arjun",
#     "Bodo": "Bikram",
#     "Chhattisgarhi": "Bhanu",
#     "Dogri": "Karan",
#     "English": "Thoma",
#     "Gujarati": "Yash",
#     "Hindi": "Divya",
#     "Kannada": "Suresh",
#     "Malayalam": "Anjali",
#     "Manipuri": "Laishram",
#     "Marathi": "Sanjay",
#     "Nepali": "Amrita",
#     "Odia": "Manas",
#     "Punjabi": "Divjot",
#     "Sanskrit": "Aryan",
#     "Tamil": "Jaya",
#     "Telugu": "Prakash"
# }

# # Generate TTS for all languages
# for lang, prompt in prompts.items():
#     speaker = speakers[lang]
#     description = f"{speaker}'s voice, clear and natural."

#     # Tokenize
#     description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
#     prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)

#     # Generate audio
#     with torch.no_grad():
#         generation = model.generate(
#             input_ids=description_input_ids.input_ids,
#             attention_mask=description_input_ids.attention_mask,
#             prompt_input_ids=prompt_input_ids.input_ids,
#             prompt_attention_mask=prompt_input_ids.attention_mask
#         )

#     # Convert to numpy and save
#     audio_arr = generation.cpu().numpy().squeeze()
#     out_file = f"tts_outputs/{lang}_{speaker}.wav"
#     sf.write(out_file, audio_arr, model.config.sampling_rate)
#     print(f"Saved TTS for {lang} ({speaker}) -> {out_file}")


import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer
model_name = "ai4bharat/indic-parler-tts"
model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

# Output directory
os.makedirs("tts_outputs", exist_ok=True)

# Prompts for all 22 Indian languages + English
prompts = {
    "Assamese": "আপুনি আজিকালি কেনে আছেন?",
    "Bengali": "তুমি আজ কেমন আছ?",
    "Bodo": "निःशब्द मोन हाबो?",
    "Chhattisgarhi": "तें आज कइसने हस?",
    "Dogri": "तुम आज कैसे हो?",
    "English": "Hello, how are you today?",
    "Gujarati": "તમે આજે કેમ છો?",
    "Hindi": "अरे, तुम आज कैसे हो?",
    "Kannada": "ನೀವು ಇಂದು ಹೇಗಿದ್ದೀರಾ?",
    "Kashmiri": "تُہند کیا حال آہ؟",
    "Malayalam": "നിങ്ങൾ ഇന്ന് എങ്ങനെയാണ്?",
    "Manipuri": "নিংগা আজা কসী অছু?",
    "Marathi": "तुम्ही आज कसे आहात?",
    "Nepali": "तिमी आज कस्तो छौ?",
    "Odia": "ଆପଣ କେମିତି ଅଛନ୍ତି?",
    "Punjabi": "ਤੁਸੀਂ ਅੱਜ ਕਿਵੇਂ ਹੋ?",
    "Sanskrit": "भवान् अद्य कथम् अस्ति?",
    "Sindhi": "توهان اڄ ڪيئن آهيو؟",
    "Tamil": "நீங்கள் இன்று எப்படி இருக்கிறீர்கள்?",
    "Telugu": "మీరు ఈరోజు ఎలా ఉన్నారు?",
    "Urdu": "آپ آج کیسے ہیں؟",
    "Maithili": "अहाँ आज केहेन छी?"
}

# Speakers: recommended where available, otherwise use female speaker
speakers = {
    "Assamese": "Amit",
    "Bengali": "Arjun",
    "Bodo": "Bikram",
    "Chhattisgarhi": "Bhanu",
    "Dogri": "Karan",
    "English": "Thoma",
    "Gujarati": "Yash",
    "Hindi": "Divya",
    "Kannada": "Suresh",
    "Kashmiri": "FemaleSpeaker",
    "Malayalam": "Anjali",
    "Manipuri": "Laishram",
    "Marathi": "Sanjay",
    "Nepali": "Amrita",
    "Odia": "Manas",
    "Punjabi": "Divjot",
    "Sanskrit": "Aryan",
    "Sindhi": "FemaleSpeaker",
    "Tamil": "Jaya",
    "Telugu": "Prakash",
    "Urdu": "FemaleSpeaker",
    "Maithili": "FemaleSpeaker"
}

# Generate TTS for all languages
for lang, prompt in prompts.items():
    speaker = speakers[lang]
    description = f"{speaker}'s voice, clear and natural."

    # Tokenize
    description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate audio
    with torch.no_grad():
        generation = model.generate(
            input_ids=description_input_ids.input_ids,
            attention_mask=description_input_ids.attention_mask,
            prompt_input_ids=prompt_input_ids.input_ids,
            prompt_attention_mask=prompt_input_ids.attention_mask
        )

    # Convert to numpy and save
    audio_arr = generation.cpu().numpy().squeeze()
    out_file = f"tts_outputs/{lang}_{speaker}.wav"
    sf.write(out_file, audio_arr, model.config.sampling_rate)
    print(f"Saved TTS for {lang} ({speaker}) -> {out_file}")
