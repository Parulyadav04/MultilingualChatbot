# import json
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer, CrossEncoder
# from llama_cpp import Llama
# import time
# import os
# import torch
# import sounddevice as sd
# import soundfile as sf
# from transformers import AutoModel, AutoTokenizer
# from parler_tts import ParlerTTSForConditionalGeneration
# import sys

# # Import translation module with error handling
# try:
#     from translation_module import IndicTranslator
#     TRANSLATION_AVAILABLE = True
# except ImportError as e:
#     print(f"⚠️  Warning: Could not import translation module: {e}")
#     TRANSLATION_AVAILABLE = False

# # --- CONFIGURATION ---
# FAISS_INDEX_PATH = 'meity_faiss.index'
# CHUNKS_PATH = 'meity_chunks.json'
# LLM_MODEL_PATH = 'Meta-Llama-3-8B-Instruct.Q4_K_M.gguf'
# EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
# RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# # STT and TTS Models
# STT_MODEL_NAME = "ai4bharat/indic-conformer-600m-multilingual"
# TTS_MODEL_NAME = "ai4bharat/indic-parler-tts"

# class IntegratedMeityRAGChatbot:
#     def __init__(self, index_path, chunks_path, llm_path, embed_model_name, 
#                  reranker_model_name, enable_stt=False, enable_tts=False, 
#                  enable_translation=True, skip_translation=False):
#         print("=" * 60)
#         print("Initializing MeitY RAG Chatbot")
#         print("=" * 60)
        
#         self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         print(f"Device: {self.device}")
        
#         # 1. Load FAISS index and text chunks
#         print("\n[1/7] Loading FAISS index and text chunks...")
#         try:
#             self.index = faiss.read_index(index_path)
#             with open(chunks_path, 'r', encoding='utf-8') as f:
#                 self.chunks = json.load(f)
#             print(f"✓ Loaded {len(self.chunks)} chunks")
#         except Exception as e:
#             print(f"❌ Failed to load FAISS/chunks: {e}")
#             raise
        
#         # 2. Load Embedding Model
#         print("\n[2/7] Loading embedding model...")
#         try:
#             self.embedding_model = SentenceTransformer(embed_model_name)
#             print("✓ Embedding model loaded")
#         except Exception as e:
#             print(f"❌ Failed to load embedding model: {e}")
#             raise
        
#         # 3. Load Reranker Model
#         print("\n[3/7] Loading reranker model...")
#         try:
#             self.reranker_model = CrossEncoder(reranker_model_name)
#             print("✓ Reranker model loaded")
#         except Exception as e:
#             print(f"❌ Failed to load reranker: {e}")
#             raise
        
#         # 4. Load LLM
#         print(f"\n[4/7] Loading LLM from: {llm_path}...")
#         try:
#             n_threads = os.cpu_count() or 4
#             self.llm = Llama(
#                 model_path=llm_path,
#                 n_ctx=4096,
#                 n_threads=n_threads,
#                 verbose=False
#             )
#             print(f"✓ LLM loaded (using {n_threads} threads)")
#         except Exception as e:
#             print(f"❌ Failed to load LLM: {e}")
#             raise
        
#         # 5. Load Translation Module - FIXED VERSION
#         self.translator = None
#         if skip_translation:
#             print("\n[5/7] Translation skipped by user - English-only mode")
#         elif enable_translation and TRANSLATION_AVAILABLE:
#             print("\n[5/7] Loading IndicTrans2 translation module...")
#             print("=" * 60)
#             print("⚠️  IMPORTANT: Translation models are LARGE (~8.5GB total)")
#             print("⚠️  First-time download may take 10-30 minutes")
#             print("⚠️  Models: ai4bharat/indictrans2-indic-en-1B (4GB)")
#             print("⚠️           ai4bharat/indictrans2-en-indic-1B (4GB)")
#             print("=" * 60)
#             print("\nOptions:")
#             print("  1) Wait for download (press Enter)")
#             print("  2) Skip translation for now (type 'skip' + Enter)")
#             print("  3) Cancel and restart in text-only mode (Ctrl+C)")
            
#             choice = input("\nYour choice: ").strip().lower()
            
#             if choice == 'skip':
#                 print("\n✓ Skipping translation - English-only mode")
#                 self.translator = None
#             else:
#                 print("\n⏳ Loading translation models...")
#                 print("💡 This will show 'Loading English→Indic model...' for a long time")
#                 print("💡 That's normal - models are downloading in the background\n")
                
#                 try:
#                     self.translator = IndicTranslator(device=self.device)
#                     print("\n✓ Translation module loaded successfully!")
#                 except KeyboardInterrupt:
#                     print("\n⚠️  Translation loading interrupted by user")
#                     print("Continuing without translation support (English-only mode)")
#                     self.translator = None
#                 except Exception as e:
#                     print(f"\n❌ Translation failed: {e}")
#                     print("Continuing without translation support (English-only mode)")
#                     self.translator = None
#         elif enable_translation and not TRANSLATION_AVAILABLE:
#             print("\n[5/7] Translation module not available (import failed)")
#             print("Continuing in English-only mode")
#         else:
#             print("\n[5/7] Translation disabled - English-only mode")
        
#         # 6. Load STT Model (Optional)
#         self.stt_model = None
#         if enable_stt:
#             print("\n[6/7] Loading Speech-to-Text model...")
#             try:
#                 self.stt_model = AutoModel.from_pretrained(
#                     STT_MODEL_NAME, 
#                     trust_remote_code=True
#                 ).to(self.device)
#                 print("✓ STT model loaded")
#             except Exception as e:
#                 print(f"❌ Failed to load STT model: {e}")
#                 print("Continuing without STT support")
#         else:
#             print("\n[6/7] STT disabled - skipping")
        
#         # 7. Load TTS Model (Optional)
#         self.tts_model = None
#         self.tts_tokenizer = None
#         self.tts_desc_tokenizer = None
#         if enable_tts:
#             print("\n[7/7] Loading Text-to-Speech model...")
#             try:
#                 self.tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
#                     TTS_MODEL_NAME
#                 ).to(self.device)
#                 self.tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_NAME)
#                 self.tts_desc_tokenizer = AutoTokenizer.from_pretrained(
#                     self.tts_model.config.text_encoder._name_or_path
#                 )
#                 print("✓ TTS model loaded")
#             except Exception as e:
#                 print(f"❌ Failed to load TTS model: {e}")
#                 print("Continuing without TTS support")
#         else:
#             print("\n[7/7] TTS disabled - skipping")
        
#         # Speaker mapping for TTS
#         self.speakers = {
#             "as": "Amit", "bn": "Arjun", "brx": "Bikram", "doi": "Karan",
#             "en": "Thoma", "gu": "Yash", "hi": "Divya", "kn": "Suresh",
#             "ks": "FemaleSpeaker", "ml": "Anjali", "mni": "Laishram",
#             "mr": "Sanjay", "ne": "Amrita", "or": "Manas", "pa": "Divjot",
#             "sa": "Aryan", "sd": "FemaleSpeaker", "ta": "Jaya",
#             "te": "Prakash", "ur": "FemaleSpeaker", "mai": "FemaleSpeaker"
#         }
        
#         print("\n" + "=" * 60)
#         print("✓ Chatbot initialization complete!")
#         self._print_features()
#         print("=" * 60 + "\n")
    
#     def _print_features(self):
#         """Print available features"""
#         print("\n🎯 Available Features:")
#         print(f"  ✓ Text Query (RAG)")
#         print(f"  {'✓' if self.translator else '✗'} Translation (22 Indian languages)")
#         print(f"  {'✓' if self.stt_model else '✗'} Speech Input (STT)")
#         print(f"  {'✓' if self.tts_model else '✗'} Speech Output (TTS)")
    
#     def speech_to_text(self, audio_file_or_array, lang_code, sample_rate=16000):
#         """Convert speech to text using Indic Conformer"""
#         if self.stt_model is None:
#             raise RuntimeError("STT model not loaded. Initialize with enable_stt=True")
        
#         if isinstance(audio_file_or_array, str):
#             audio_arr, sr = sf.read(audio_file_or_array)
#             if sr != 16000:
#                 import torchaudio
#                 resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
#                 audio_arr = resampler(torch.tensor(audio_arr)).numpy()
#         else:
#             audio_arr = audio_file_or_array
        
#         wav = torch.tensor(audio_arr).unsqueeze(0)
#         print("🔄 Transcribing audio...")
#         transcription = self.stt_model(wav, lang_code, "rnnt")
        
#         return transcription
    
#     def record_audio(self, duration=5, sample_rate=16000):
#         """Record audio from microphone"""
#         print(f"\n🎤 Recording {duration} seconds of audio...")
#         print("🔴 Starting in 1 second... Get ready to speak!")
#         time.sleep(1)
        
#         audio = sd.rec(
#             int(duration * sample_rate), 
#             samplerate=sample_rate, 
#             channels=1, 
#             dtype='float32'
#         )
        
#         for i in range(duration, 0, -1):
#             print(f"   Recording... {i} seconds remaining", end='\r', flush=True)
#             time.sleep(1)
        
#         sd.wait()
#         print("\n✓ Recording complete!                    ")
#         return np.squeeze(audio)
    
#     def record_and_transcribe(self, lang_code, duration=5, save_audio=False):
#         """Record audio and transcribe it"""
#         if self.stt_model is None:
#             raise RuntimeError("STT model not loaded")
        
#         audio_array = self.record_audio(duration=duration)
        
#         if save_audio:
#             timestamp = time.strftime("%Y%m%d_%H%M%S")
#             audio_file = f"recorded_query_{lang_code}_{timestamp}.wav"
#             sf.write(audio_file, audio_array, 16000)
#             print(f"💾 Audio saved to: {audio_file}")
        
#         transcription = self.speech_to_text(audio_array, lang_code)
#         return transcription
    
#     def text_to_speech(self, text, lang_code, output_path=None):
#         """Convert text to speech"""
#         if self.tts_model is None:
#             raise RuntimeError("TTS model not loaded")
        
#         speaker = self.speakers.get(lang_code, "FemaleSpeaker")
#         description = f"{speaker}'s voice, clear and natural."
        
#         desc_inputs = self.tts_desc_tokenizer(description, return_tensors="pt").to(self.device)
#         prompt_inputs = self.tts_tokenizer(text, return_tensors="pt").to(self.device)
        
#         print("🔊 Generating speech...")
#         with torch.no_grad():
#             generation = self.tts_model.generate(
#                 input_ids=desc_inputs.input_ids,
#                 attention_mask=desc_inputs.attention_mask,
#                 prompt_input_ids=prompt_inputs.input_ids,
#                 prompt_attention_mask=prompt_inputs.attention_mask
#             )
        
#         audio_arr = generation.cpu().numpy().squeeze()
        
#         if output_path:
#             os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
#             sf.write(output_path, audio_arr, self.tts_model.config.sampling_rate)
#             print(f"💾 Audio saved to: {output_path}")
        
#         return audio_arr
    
#     def _retrieve_context(self, query, k_initial=10):
#         """Retrieve contexts"""
#         query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
#         query_embedding = np.array(query_embedding).astype('float32')
#         distances, indices = self.index.search(query_embedding, k_initial)
#         return [self.chunks[i] for i in indices[0] if i != -1]
    
#     def _rerank_context(self, query, chunks, k_rerank=3):
#         """Rerank chunks"""
#         if not chunks:
#             return ""
#         pairs = [[query, chunk] for chunk in chunks]
#         scores = self.reranker_model.predict(pairs)
#         scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
#         return "\n\n---\n\n".join([c for c, s in scored[:k_rerank]])
    
#     def _generate_response(self, query, context):
#         """Generate response"""
#         prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a helpful assistant for MeitY, India. Answer based ONLY on the context provided.

# <|eot_id|><|start_header_id|>user<|end_header_id|>
# CONTEXT:
# {context}

# QUESTION:
# {query}

# ANSWER:
# <|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """
#         response = self.llm(prompt, max_tokens=500, stop=["<|eot_id|>"], echo=False)
#         return response['choices'][0]['text'].strip()
    
#     def ask(self, query, source_lang="en", return_audio=False, audio_output_path=None):
#         """Main ask method"""
#         print(f"\n{'='*60}")
#         print(f"📝 Query ({source_lang}): {query}")
#         print(f"{'='*60}\n")
        
#         # Translate to English if needed
#         if source_lang != "en" and self.translator:
#             try:
#                 query_en = self.translator.translate_to_english(query, source_lang)
#                 print(f"🔄 English: {query_en}\n")
#             except Exception as e:
#                 print(f"⚠️  Translation failed: {e}. Using original query.")
#                 query_en = query
#         else:
#             query_en = query
#             if source_lang != "en" and not self.translator:
#                 print("⚠️  Translation not available. Processing as English.\n")
        
#         # RAG pipeline
#         print("🔍 Searching knowledge base...")
#         chunks = self._retrieve_context(query_en)
#         context = self._rerank_context(query_en, chunks)
        
#         if not context:
#             response_en = "I don't have enough information to answer this question."
#         else:
#             print("🤖 Generating answer...")
#             response_en = self._generate_response(query_en, context)
        
#         # Translate back if needed
#         if source_lang != "en" and self.translator:
#             try:
#                 response_local = self.translator.translate_from_english(response_en, source_lang)
#             except Exception as e:
#                 print(f"⚠️  Back-translation failed: {e}")
#                 response_local = response_en
#         else:
#             response_local = response_en
        
#         result = {"text": response_local}
        
#         # TTS if requested
#         if return_audio and self.tts_model:
#             try:
#                 audio_arr = self.text_to_speech(response_local, source_lang, audio_output_path)
#                 result["audio"] = audio_arr
#             except Exception as e:
#                 print(f"⚠️  TTS failed: {e}")
        
#         return result
    
#     def voice_query(self, lang_code, duration=5, return_audio=False, save_recordings=False):
#         """Complete voice interaction"""
#         query_text = self.record_and_transcribe(lang_code, duration, save_audio=save_recordings)
#         print(f"\n✅ You said: {query_text}\n")
        
#         audio_path = None
#         if return_audio and save_recordings:
#             timestamp = time.strftime("%Y%m%d_%H%M%S")
#             audio_path = f"response_{lang_code}_{timestamp}.wav"
        
#         result = self.ask(query_text, source_lang=lang_code, return_audio=return_audio, audio_output_path=audio_path)
#         result['query'] = query_text
#         return result


# def main():
#     """Interactive CLI"""
#     print("\n" + "="*60)
#     print(" MeitY RAG Chatbot ")
#     print("="*60)
    
#     print("\nSelect mode:")
#     print("  1) Text-only (English) - Ready in ~1 minute ✓ RECOMMENDED")
#     print("  2) Text + Translation (22 languages) - 10-30 min first run")
#     print("  3) Full (Translation + Speech) - 15-40 min first run")
    
#     mode = input("\nEnter choice (1/2/3) [default: 1]: ").strip()
    
#     if mode == "3":
#         enable_stt, enable_tts, enable_translation = True, True, True
#         print("\n✓ Full mode selected\n")
#     elif mode == "2":
#         enable_stt, enable_tts, enable_translation = False, False, True
#         print("\n✓ Translation mode selected\n")
#     else:
#         enable_stt, enable_tts, enable_translation = False, False, False
#         print("\n✓ Text-only mode selected (fastest!)\n")
    
#     chatbot = None
#     try:
#         chatbot = IntegratedMeityRAGChatbot(
#             index_path=FAISS_INDEX_PATH,
#             chunks_path=CHUNKS_PATH,
#             llm_path=LLM_MODEL_PATH,
#             embed_model_name=EMBEDDING_MODEL_NAME,
#             reranker_model_name=RERANKER_MODEL_NAME,
#             enable_stt=enable_stt,
#             enable_tts=enable_tts,
#             enable_translation=enable_translation,
#             skip_translation=False
#         )
#     except KeyboardInterrupt:
#         print("\n\n👋 Setup cancelled. Exiting...")
#         return
#     except Exception as e:
#         print(f"\n❌ FATAL ERROR during initialization: {e}")
#         import traceback
#         traceback.print_exc()
#         input("\nPress Enter to exit...")
#         return
    
#     # Show instructions
#     print("\n" + "="*60)
#     print(" COMMANDS ")
#     print("="*60)
#     print("📝 TEXT:      Just type your question")
#     if chatbot.translator:
#         print("🌐 LANGUAGE:  Type 'lang:hi' to switch language")
#     if chatbot.stt_model:
#         print("🎤 VOICE:     Type 'voice' to record question")
#     if chatbot.tts_model:
#         print("🔊 AUDIO:     Type 'audio:on' for speech output")
#     print("❌ EXIT:      Type 'quit' to exit")
#     print("="*60 + "\n")
    
#     current_lang = "en"
#     audio_output = False
    
#     # Main loop
#     while True:
#         try:
#             user_input = input(f"\n[{current_lang}] Your Question: ").strip()
            
#             if not user_input:
#                 continue
            
#             if user_input.lower() == 'quit':
#                 print("\n👋 Goodbye!")
#                 break
            
#             # Language change
#             if user_input.lower().startswith('lang:'):
#                 if not chatbot.translator:
#                     print("⚠️  Translation not available")
#                     continue
#                 current_lang = user_input.split(':', 1)[1].strip()
#                 print(f"✓ Language: {current_lang}")
#                 continue
            
#             # Audio toggle
#             if user_input.lower() in ['audio:on', 'audio:off']:
#                 if not chatbot.tts_model:
#                     print("⚠️  TTS not available")
#                     continue
#                 audio_output = (user_input.lower() == 'audio:on')
#                 print(f"✓ Audio output: {'ON' if audio_output else 'OFF'}")
#                 continue
            
#             # Voice input
#             if user_input.lower().startswith('voice'):
#                 if not chatbot.stt_model:
#                     print("⚠️  STT not available")
#                     continue
                
#                 lang = user_input.split(':', 1)[1].strip() if ':' in user_input else current_lang
#                 result = chatbot.voice_query(lang, duration=5, return_audio=audio_output)
                
#                 print(f"\n{'='*60}")
#                 print(f"✅ ANSWER:\n{result['text']}")
#                 print(f"{'='*60}")
#                 continue
            
#             # Text query
#             result = chatbot.ask(user_input, source_lang=current_lang, return_audio=audio_output)
            
#             print(f"\n{'='*60}")
#             print(f"✅ ANSWER:\n{result['text']}")
#             print(f"{'='*60}")
            
#         except KeyboardInterrupt:
#             print("\n\n👋 Interrupted. Goodbye!")
#             break
#         except Exception as e:
#             print(f"\n❌ Error: {e}")
#             import traceback
#             traceback.print_exc()


# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\n\n👋 Goodbye!")
#     except Exception as e:
#         print(f"\n❌ Fatal error: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         print("\nExiting...")
#         sys.exit(0)




###shi haiiiii
# import json
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer, CrossEncoder
# from llama_cpp import Llama
# import time
# import os
# import torch
# import sounddevice as sd
# import soundfile as sf
# from transformers import AutoModel, AutoTokenizer
# from parler_tts import ParlerTTSForConditionalGeneration
# import sys

# # Import translation module with error handling
# try:
#     from translation_module import IndicTranslator
#     TRANSLATION_AVAILABLE = True
# except ImportError as e:
#     print(f"Warning: Could not import translation module: {e}")
#     TRANSLATION_AVAILABLE = False

# # --- CONFIGURATION ---
# FAISS_INDEX_PATH = 'meity_faiss.index'
# CHUNKS_PATH = 'meity_chunks.json'
# LLM_MODEL_PATH = 'Meta-Llama-3-8B-Instruct.Q4_K_M.gguf'
# EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
# RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# # STT and TTS Models
# STT_MODEL_NAME = "ai4bharat/indic-conformer-600m-multilingual"
# TTS_MODEL_NAME = "ai4bharat/indic-parler-tts"

# class IntegratedMeityRAGChatbot:
#     def __init__(self, index_path, chunks_path, llm_path, embed_model_name, 
#                  reranker_model_name, enable_stt=False, enable_tts=False, 
#                  enable_translation=True, skip_translation=False):
#         print("Initializing MeitY RAG Chatbot")
        
#         self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         print(f"Device: {self.device}")
        
#         # 1. Load FAISS index and text chunks
#         print("\n[1/7] Loading FAISS index and text chunks...")
#         try:
#             self.index = faiss.read_index(index_path)
#             with open(chunks_path, 'r', encoding='utf-8') as f:
#                 self.chunks = json.load(f)
#             print(f"Loaded {len(self.chunks)} chunks")
#         except Exception as e:
#             print(f"Failed to load FAISS/chunks: {e}")
#             raise
        
#         # 2. Load Embedding Model
#         print("\n[2/7] Loading embedding model...")
#         try:
#             self.embedding_model = SentenceTransformer(embed_model_name)
#             print("Embedding model loaded")
#         except Exception as e:
#             print(f"Failed to load embedding model: {e}")
#             raise
        
#         # 3. Load Reranker Model
#         print("\n[3/7] Loading reranker model...")
#         try:
#             self.reranker_model = CrossEncoder(reranker_model_name)
#             print("Reranker model loaded")
#         except Exception as e:
#             print(f"Failed to load reranker: {e}")
#             raise
        
#         # 4. Load LLM
#         print(f"\n[4/7] Loading LLM from: {llm_path}...")
#         try:
#             n_threads = os.cpu_count() or 4
#             self.llm = Llama(
#                 model_path=llm_path,
#                 n_ctx=4096,
#                 n_threads=n_threads,
#                 verbose=False
#             )
#             print(f"LLM loaded (using {n_threads} threads)")
#         except Exception as e:
#             print(f"Failed to load LLM: {e}")
#             raise
        
#         # 5. Load Translation Module
#         self.translator = None
#         if skip_translation:
#             print("\n[5/7] Translation skipped by user - English-only mode")
#         elif enable_translation and TRANSLATION_AVAILABLE:
#             print("\n[5/7] Loading IndicTrans2 translation module...")
#             print("Note: Translation models are large (approximately 8.5GB total)")
#             print("First-time download may take 10-30 minutes")
#             print("Loading translation models, please wait...")
            
#             try:
#                 self.translator = IndicTranslator(device=self.device)
#                 print("Translation module loaded successfully")
#             except KeyboardInterrupt:
#                 print("Translation loading interrupted by user")
#                 print("Continuing without translation support (English-only mode)")
#                 self.translator = None
#             except Exception as e:
#                 print(f"Translation failed: {e}")
#                 print("Continuing without translation support (English-only mode)")
#                 self.translator = None
#         elif enable_translation and not TRANSLATION_AVAILABLE:
#             print("\n[5/7] Translation module not available (import failed)")
#             print("Continuing in English-only mode")
#         else:
#             print("\n[5/7] Translation disabled - English-only mode")
        
#         # 6. Load STT Model (Optional)
#         self.stt_model = None
#         if enable_stt:
#             print("\n[6/7] Loading Speech-to-Text model...")
#             try:
#                 self.stt_model = AutoModel.from_pretrained(
#                     STT_MODEL_NAME, 
#                     trust_remote_code=True
#                 ).to(self.device)
#                 print("STT model loaded")
#             except Exception as e:
#                 print(f"Failed to load STT model: {e}")
#                 print("Continuing without STT support")
#         else:
#             print("\n[6/7] STT disabled - skipping")
        
#         # 7. Load TTS Model (Optional)
#         self.tts_model = None
#         self.tts_tokenizer = None
#         self.tts_desc_tokenizer = None
#         if enable_tts:
#             print("\n[7/7] Loading Text-to-Speech model...")
#             try:
#                 self.tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
#                     TTS_MODEL_NAME
#                 ).to(self.device)
#                 self.tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_NAME)
#                 self.tts_desc_tokenizer = AutoTokenizer.from_pretrained(
#                     self.tts_model.config.text_encoder._name_or_path
#                 )
#                 print("TTS model loaded")
#             except Exception as e:
#                 print(f"Failed to load TTS model: {e}")
#                 print("Continuing without TTS support")
#         else:
#             print("\n[7/7] TTS disabled - skipping")
        
#         # Speaker mapping for TTS
#         self.speakers = {
#             "as": "Amit", "bn": "Arjun", "brx": "Bikram", "doi": "Karan",
#             "en": "Thoma", "gu": "Yash", "hi": "Divya", "kn": "Suresh",
#             "ks": "FemaleSpeaker", "ml": "Anjali", "mni": "Laishram",
#             "mr": "Sanjay", "ne": "Amrita", "or": "Manas", "pa": "Divjot",
#             "sa": "Aryan", "sd": "FemaleSpeaker", "ta": "Jaya",
#             "te": "Prakash", "ur": "FemaleSpeaker", "mai": "FemaleSpeaker"
#         }
        
#         print("\nChatbot initialization complete")
#         self._print_features()
    
#     def _print_features(self):
#         """Print available features"""
#         print("\nAvailable Features:")
#         print(f"  Text Query (RAG): Enabled")
#         print(f"  Translation (22 Indian languages): {'Enabled' if self.translator else 'Disabled'}")
#         print(f"  Speech Input (STT): {'Enabled' if self.stt_model else 'Disabled'}")
#         print(f"  Speech Output (TTS): {'Enabled' if self.tts_model else 'Disabled'}")
    
#     def speech_to_text(self, audio_file_or_array, lang_code, sample_rate=16000):
#         """Convert speech to text using Indic Conformer"""
#         if self.stt_model is None:
#             raise RuntimeError("STT model not loaded. Initialize with enable_stt=True")
        
#         if isinstance(audio_file_or_array, str):
#             audio_arr, sr = sf.read(audio_file_or_array)
#             if sr != 16000:
#                 import torchaudio
#                 resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
#                 audio_arr = resampler(torch.tensor(audio_arr)).numpy()
#         else:
#             audio_arr = audio_file_or_array
        
#         wav = torch.tensor(audio_arr).unsqueeze(0)
#         print("Transcribing audio...")
#         transcription = self.stt_model(wav, lang_code, "rnnt")
        
#         return transcription
    
#     def record_audio(self, duration=5, sample_rate=16000):
#         """Record audio from microphone"""
#         print(f"\nRecording {duration} seconds of audio...")
#         print("Starting in 1 second... Get ready to speak!")
#         time.sleep(1)
        
#         audio = sd.rec(
#             int(duration * sample_rate), 
#             samplerate=sample_rate, 
#             channels=1, 
#             dtype='float32'
#         )
        
#         for i in range(duration, 0, -1):
#             print(f"   Recording... {i} seconds remaining", end='\r', flush=True)
#             time.sleep(1)
        
#         sd.wait()
#         print("\nRecording complete                    ")
#         return np.squeeze(audio)
    
#     def record_and_transcribe(self, lang_code, duration=5, save_audio=False):
#         """Record audio and transcribe it"""
#         if self.stt_model is None:
#             raise RuntimeError("STT model not loaded")
        
#         audio_array = self.record_audio(duration=duration)
        
#         if save_audio:
#             timestamp = time.strftime("%Y%m%d_%H%M%S")
#             audio_file = f"recorded_query_{lang_code}_{timestamp}.wav"
#             sf.write(audio_file, audio_array, 16000)
#             print(f"Audio saved to: {audio_file}")
        
#         transcription = self.speech_to_text(audio_array, lang_code)
#         return transcription
    
#     def text_to_speech(self, text, lang_code, output_path=None):
#         """Convert text to speech"""
#         if self.tts_model is None:
#             raise RuntimeError("TTS model not loaded")
        
#         speaker = self.speakers.get(lang_code, "FemaleSpeaker")
#         description = f"{speaker}'s voice, clear and natural."
        
#         desc_inputs = self.tts_desc_tokenizer(description, return_tensors="pt").to(self.device)
#         prompt_inputs = self.tts_tokenizer(text, return_tensors="pt").to(self.device)
        
#         print("Generating speech...")
#         with torch.no_grad():
#             generation = self.tts_model.generate(
#                 input_ids=desc_inputs.input_ids,
#                 attention_mask=desc_inputs.attention_mask,
#                 prompt_input_ids=prompt_inputs.input_ids,
#                 prompt_attention_mask=prompt_inputs.attention_mask
#             )
        
#         audio_arr = generation.cpu().numpy().squeeze()
        
#         if output_path:
#             os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
#             sf.write(output_path, audio_arr, self.tts_model.config.sampling_rate)
#             print(f"Audio saved to: {output_path}")
        
#         return audio_arr
    
#     def _retrieve_context(self, query, k_initial=10):
#         """Retrieve contexts"""
#         query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
#         query_embedding = np.array(query_embedding).astype('float32')
#         distances, indices = self.index.search(query_embedding, k_initial)
#         return [self.chunks[i] for i in indices[0] if i != -1]
    
#     def _rerank_context(self, query, chunks, k_rerank=3):
#         """Rerank chunks"""
#         if not chunks:
#             return ""
#         pairs = [[query, chunk] for chunk in chunks]
#         scores = self.reranker_model.predict(pairs)
#         scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
#         return "\n\n---\n\n".join([c for c, s in scored[:k_rerank]])
    
#     def _generate_response(self, query, context):
#         """Generate response"""
#         prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a helpful assistant for MeitY, India. Answer based ONLY on the context provided.

# <|eot_id|><|start_header_id|>user<|end_header_id|>
# CONTEXT:
# {context}

# QUESTION:
# {query}

# ANSWER:
# <|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """
#         response = self.llm(prompt, max_tokens=500, stop=["<|eot_id|>"], echo=False)
#         return response['choices'][0]['text'].strip()
    
#     def ask(self, query, source_lang="en", return_audio=False, audio_output_path=None):
#         """Main ask method"""
#         print(f"\nQuery ({source_lang}): {query}\n")
        
#         # Translate to English if needed
#         if source_lang != "en" and self.translator:
#             try:
#                 query_en = self.translator.translate_to_english(query, source_lang)
#                 print(f"English: {query_en}\n")
#             except Exception as e:
#                 print(f"Translation failed: {e}. Using original query.")
#                 query_en = query
#         else:
#             query_en = query
#             if source_lang != "en" and not self.translator:
#                 print("Translation not available. Processing as English.\n")
        
#         # RAG pipeline
#         print("Searching knowledge base...")
#         chunks = self._retrieve_context(query_en)
#         context = self._rerank_context(query_en, chunks)
        
#         if not context:
#             response_en = "I don't have enough information to answer this question."
#         else:
#             print("Generating answer...")
#             response_en = self._generate_response(query_en, context)
        
#         # Translate back if needed
#         if source_lang != "en" and self.translator:
#             try:
#                 response_local = self.translator.translate_from_english(response_en, source_lang)
#             except Exception as e:
#                 print(f"Back-translation failed: {e}")
#                 response_local = response_en
#         else:
#             response_local = response_en
        
#         result = {"text": response_local}
        
#         # TTS if requested
#         if return_audio and self.tts_model:
#             try:
#                 audio_arr = self.text_to_speech(response_local, source_lang, audio_output_path)
#                 result["audio"] = audio_arr
#             except Exception as e:
#                 print(f"TTS failed: {e}")
        
#         return result
    
#     def voice_query(self, lang_code, duration=5, return_audio=False, save_recordings=False):
#         """Complete voice interaction"""
#         query_text = self.record_and_transcribe(lang_code, duration, save_audio=save_recordings)
#         print(f"\nYou said: {query_text}\n")
        
#         audio_path = None
#         if return_audio and save_recordings:
#             timestamp = time.strftime("%Y%m%d_%H%M%S")
#             audio_path = f"response_{lang_code}_{timestamp}.wav"
        
#         result = self.ask(query_text, source_lang=lang_code, return_audio=return_audio, audio_output_path=audio_path)
#         result['query'] = query_text
#         return result


# def main():
#     """Interactive CLI"""
#     print("\nMeitY RAG Chatbot\n")
    
#     print("Select mode:")
#     print("  1) Text-only (English) - Ready in ~1 minute (RECOMMENDED)")
#     print("  2) Text + Translation (22 languages) - 10-30 min first run")
#     print("  3) Full (Translation + Speech) - 15-40 min first run")
    
#     mode = input("\nEnter choice (1/2/3) [default: 1]: ").strip()
    
#     if mode == "3":
#         enable_stt, enable_tts, enable_translation = True, True, True
#         print("\nFull mode selected\n")
#     elif mode == "2":
#         enable_stt, enable_tts, enable_translation = False, False, True
#         print("\nTranslation mode selected\n")
#     else:
#         enable_stt, enable_tts, enable_translation = False, False, False
#         print("\nText-only mode selected (fastest)\n")
    
#     chatbot = None
#     try:
#         chatbot = IntegratedMeityRAGChatbot(
#             index_path=FAISS_INDEX_PATH,
#             chunks_path=CHUNKS_PATH,
#             llm_path=LLM_MODEL_PATH,
#             embed_model_name=EMBEDDING_MODEL_NAME,
#             reranker_model_name=RERANKER_MODEL_NAME,
#             enable_stt=enable_stt,
#             enable_tts=enable_tts,
#             enable_translation=enable_translation,
#             skip_translation=False
#         )
#     except KeyboardInterrupt:
#         print("\nSetup cancelled. Exiting...")
#         return
#     except Exception as e:
#         print(f"\nFATAL ERROR during initialization: {e}")
#         import traceback
#         traceback.print_exc()
#         input("\nPress Enter to exit...")
#         return
    
#     # Show instructions
#     print("\nCOMMANDS")
#     print("TEXT:      Just type your question")
#     if chatbot.translator:
#         print("LANGUAGE:  Type 'lang:hi' to switch language")
#     if chatbot.stt_model:
#         print("VOICE:     Type 'voice' to record question")
#     if chatbot.tts_model:
#         print("AUDIO:     Type 'audio:on' for speech output")
#     print("EXIT:      Type 'quit' to exit\n")
    
#     current_lang = "en"
#     audio_output = False
    
#     # Main loop
#     while True:
#         try:
#             user_input = input(f"\n[{current_lang}] Your Question: ").strip()
            
#             if not user_input:
#                 continue
            
#             if user_input.lower() == 'quit':
#                 print("\nGoodbye!")
#                 break
            
#             # Language change
#             if user_input.lower().startswith('lang:'):
#                 if not chatbot.translator:
#                     print("Translation not available")
#                     continue
#                 current_lang = user_input.split(':', 1)[1].strip()
#                 print(f"Language: {current_lang}")
#                 continue
            
#             # Audio toggle
#             if user_input.lower() in ['audio:on', 'audio:off']:
#                 if not chatbot.tts_model:
#                     print("TTS not available")
#                     continue
#                 audio_output = (user_input.lower() == 'audio:on')
#                 print(f"Audio output: {'ON' if audio_output else 'OFF'}")
#                 continue
            
#             # Voice input
#             if user_input.lower().startswith('voice'):
#                 if not chatbot.stt_model:
#                     print("STT not available")
#                     continue
                
#                 lang = user_input.split(':', 1)[1].strip() if ':' in user_input else current_lang
#                 result = chatbot.voice_query(lang, duration=5, return_audio=audio_output)
                
#                 print(f"\nANSWER:\n{result['text']}\n")
#                 continue
            
#             # Text query
#             result = chatbot.ask(user_input, source_lang=current_lang, return_audio=audio_output)
            
#             print(f"\nANSWER:\n{result['text']}\n")
            
#         except KeyboardInterrupt:
#             print("\nInterrupted. Goodbye!")
#             break
#         except Exception as e:
#             print(f"\nError: {e}")
#             import traceback
#             traceback.print_exc()


# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\nGoodbye!")
#     except Exception as e:
#         print(f"\nFatal error: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         print("\nExiting...")
#         sys.exit(0)



import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from llama_cpp import Llama
import time
import os
import torch
import sounddevice as sd
import soundfile as sf
from transformers import AutoModel, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import sys

# Import translation module with error handling
try:
    from translation_module import IndicTranslator
    TRANSLATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import translation module: {e}")
    TRANSLATION_AVAILABLE = False

# --- CONFIGURATION ---
FAISS_INDEX_PATH = 'meity_faiss.index'
CHUNKS_PATH = 'meity_chunks.json'
LLM_MODEL_PATH = 'Meta-Llama-3-8B-Instruct.Q4_K_M.gguf'
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# STT and TTS Models
STT_MODEL_NAME = "ai4bharat/indic-conformer-600m-multilingual"
TTS_MODEL_NAME = "ai4bharat/indic-parler-tts"

class IntegratedMeityRAGChatbot:
    def __init__(self, index_path, chunks_path, llm_path, embed_model_name, 
                 reranker_model_name, enable_stt=False, enable_tts=False, 
                 enable_translation=True, skip_translation=False):
        print("Initializing MeitY RAG Chatbot")
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")
        
        # 1. Load FAISS index and text chunks
        print("\n[1/7] Loading FAISS index and text chunks...")
        try:
            self.index = faiss.read_index(index_path)
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            print(f"Loaded {len(self.chunks)} chunks")
        except Exception as e:
            print(f"Failed to load FAISS/chunks: {e}")
            raise
        
        # 2. Load Embedding Model
        print("\n[2/7] Loading embedding model...")
        try:
            self.embedding_model = SentenceTransformer(embed_model_name)
            print("Embedding model loaded")
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            raise
        
        # 3. Load Reranker Model
        print("\n[3/7] Loading reranker model...")
        try:
            self.reranker_model = CrossEncoder(reranker_model_name)
            print("Reranker model loaded")
        except Exception as e:
            print(f"Failed to load reranker: {e}")
            raise
        
        # 4. Load LLM
        print(f"\n[4/7] Loading LLM from: {llm_path}...")
        try:
            n_threads = os.cpu_count() or 4
            self.llm = Llama(
                model_path=llm_path,
                n_ctx=4096,
                n_threads=n_threads,
                verbose=False
            )
            print(f"LLM loaded (using {n_threads} threads)")
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            raise
        
        # 5. Load Translation Module
        self.translator = None
        if skip_translation:
            print("\n[5/7] Translation skipped by user - English-only mode")
        elif enable_translation and TRANSLATION_AVAILABLE:
            print("\n[5/7] Loading IndicTrans2 translation module...")
            print("Note: Translation models are large (approximately 8.5GB total)")
            print("First-time download may take 10-30 minutes")
            print("Loading translation models, please wait...")
            
            try:
                self.translator = IndicTranslator(device=self.device)
                print("Translation module loaded successfully")
            except KeyboardInterrupt:
                print("Translation loading interrupted by user")
                print("Continuing without translation support (English-only mode)")
                self.translator = None
            except Exception as e:
                print(f"Translation failed: {e}")
                print("Continuing without translation support (English-only mode)")
                self.translator = None
        elif enable_translation and not TRANSLATION_AVAILABLE:
            print("\n[5/7] Translation module not available (import failed)")
            print("Continuing in English-only mode")
        else:
            print("\n[5/7] Translation disabled - English-only mode")
        
        # 6. Load STT Model (Optional)
        self.stt_model = None
        if enable_stt:
            print("\n[6/7] Loading Speech-to-Text model...")
            try:
                self.stt_model = AutoModel.from_pretrained(
                    STT_MODEL_NAME, 
                    trust_remote_code=True
                ).to(self.device)
                print("STT model loaded")
            except Exception as e:
                print(f"Failed to load STT model: {e}")
                print("Continuing without STT support")
        else:
            print("\n[6/7] STT disabled - skipping")
        
        # 7. Load TTS Model (Optional)
        self.tts_model = None
        self.tts_tokenizer = None
        self.tts_desc_tokenizer = None
        if enable_tts:
            print("\n[7/7] Loading Text-to-Speech model...")
            try:
                self.tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
                    TTS_MODEL_NAME
                ).to(self.device)
                self.tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_NAME)
                self.tts_desc_tokenizer = AutoTokenizer.from_pretrained(
                    self.tts_model.config.text_encoder._name_or_path
                )
                print("TTS model loaded")
            except Exception as e:
                print(f"Failed to load TTS model: {e}")
                print("Continuing without TTS support")
        else:
            print("\n[7/7] TTS disabled - skipping")
        
        # Speaker mapping for TTS
        self.speakers = {
            "as": "Amit", "bn": "Arjun", "brx": "Bikram", "doi": "Karan",
            "en": "Thoma", "gu": "Yash", "hi": "Divya", "kn": "Suresh",
            "ks": "FemaleSpeaker", "ml": "Anjali", "mni": "Laishram",
            "mr": "Sanjay", "ne": "Amrita", "or": "Manas", "pa": "Divjot",
            "sa": "Aryan", "sd": "FemaleSpeaker", "ta": "Jaya",
            "te": "Prakash", "ur": "FemaleSpeaker", "mai": "FemaleSpeaker"
        }
        
        print("\nChatbot initialization complete")
        self._print_features()
    
    def _print_features(self):
        """Print available features"""
        print("\nAvailable Features:")
        print(f"  Text Query (RAG): Enabled")
        print(f"  Translation (22 Indian languages): {'Enabled' if self.translator else 'Disabled'}")
        print(f"  Speech Input (STT): {'Enabled' if self.stt_model else 'Disabled'}")
        print(f"  Speech Output (TTS): {'Enabled' if self.tts_model else 'Disabled'}")
    
    def speech_to_text(self, audio_file_or_array, lang_code, sample_rate=16000):
        """Convert speech to text using Indic Conformer"""
        if self.stt_model is None:
            raise RuntimeError("STT model not loaded. Initialize with enable_stt=True")
        
        if isinstance(audio_file_or_array, str):
            audio_arr, sr = sf.read(audio_file_or_array)
            if sr != 16000:
                import torchaudio
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                audio_arr = resampler(torch.tensor(audio_arr)).numpy()
        else:
            audio_arr = audio_file_or_array
        
        wav = torch.tensor(audio_arr).unsqueeze(0)
        print("Transcribing audio...")
        transcription = self.stt_model(wav, lang_code, "rnnt")
        
        return transcription
    
    def record_audio(self, duration=5, sample_rate=16000):
        """Record audio from microphone"""
        print(f"\nRecording {duration} seconds of audio...")
        print("Starting in 1 second... Get ready to speak!")
        time.sleep(1)
        
        audio = sd.rec(
            int(duration * sample_rate), 
            samplerate=sample_rate, 
            channels=1, 
            dtype='float32'
        )
        
        for i in range(duration, 0, -1):
            print(f"   Recording... {i} seconds remaining", end='\r', flush=True)
            time.sleep(1)
        
        sd.wait()
        print("\nRecording complete                    ")
        return np.squeeze(audio)
    
    def record_and_transcribe(self, lang_code, duration=5, save_audio=False):
        """Record audio and transcribe it"""
        if self.stt_model is None:
            raise RuntimeError("STT model not loaded")
        
        audio_array = self.record_audio(duration=duration)
        
        if save_audio:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            audio_file = f"recorded_query_{lang_code}_{timestamp}.wav"
            sf.write(audio_file, audio_array, 16000)
            print(f"Audio saved to: {audio_file}")
        
        transcription = self.speech_to_text(audio_array, lang_code)
        return transcription
    
    def text_to_speech(self, text, lang_code, output_path=None):
        """Convert text to speech"""
        if self.tts_model is None:
            raise RuntimeError("TTS model not loaded")
        
        speaker = self.speakers.get(lang_code, "FemaleSpeaker")
        description = f"{speaker}'s voice, clear and natural."
        
        desc_inputs = self.tts_desc_tokenizer(description, return_tensors="pt").to(self.device)
        prompt_inputs = self.tts_tokenizer(text, return_tensors="pt").to(self.device)
        
        print("Generating speech...")
        with torch.no_grad():
            generation = self.tts_model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask
            )
        
        audio_arr = generation.cpu().numpy().squeeze()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            sf.write(output_path, audio_arr, self.tts_model.config.sampling_rate)
            print(f"Audio saved to: {output_path}")
        
        return audio_arr
    
    def _retrieve_context(self, query, k_initial=10):
        """Retrieve contexts"""
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding).astype('float32')
        distances, indices = self.index.search(query_embedding, k_initial)
        return [self.chunks[i] for i in indices[0] if i != -1]
    
    def _rerank_context(self, query, chunks, k_rerank=3):
        """Rerank chunks"""
        if not chunks:
            return ""
        pairs = [[query, chunk] for chunk in chunks]
        scores = self.reranker_model.predict(pairs)
        scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return "\n\n---\n\n".join([c for c, s in scored[:k_rerank]])
    
    def _generate_response(self, query, context):
        """Generate response"""
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant for MeitY, India. Answer based ONLY on the context provided.

<|eot_id|><|start_header_id|>user<|end_header_id|>
CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        response = self.llm(prompt, max_tokens=500, stop=["<|eot_id|>"], echo=False)
        return response['choices'][0]['text'].strip()
    
    def ask(self, query, source_lang="en", return_audio=False, audio_output_path=None):
        """Main ask method - NOW PRINTS BOTH ENGLISH AND TRANSLATED RESPONSES"""
        print(f"\nQuery ({source_lang}): {query}\n")
        
        # Translate to English if needed
        if source_lang != "en" and self.translator:
            try:
                query_en = self.translator.translate_to_english(query, source_lang)
                print(f"English Query: {query_en}\n")
            except Exception as e:
                print(f"Translation failed: {e}. Using original query.")
                query_en = query
        else:
            query_en = query
            if source_lang != "en" and not self.translator:
                print("Translation not available. Processing as English.\n")
        
        # RAG pipeline
        print("Searching knowledge base...")
        chunks = self._retrieve_context(query_en)
        context = self._rerank_context(query_en, chunks)
        
        if not context:
            response_en = "I don't have enough information to answer this question."
        else:
            print("Generating answer...")
            response_en = self._generate_response(query_en, context)
        
        # ✨ ALWAYS PRINT ENGLISH RESPONSE FIRST
        print(f"\n{'='*60}")
        print(f"ENGLISH RESPONSE:")
        print(f"{'='*60}")
        print(response_en)
        print(f"{'='*60}\n")
        
        # Translate back if needed
        if source_lang != "en" and self.translator:
            try:
                response_local = self.translator.translate_from_english(response_en, source_lang)
                
                # Print translated response
                print(f"{'='*60}")
                print(f"TRANSLATED RESPONSE ({source_lang.upper()}):")
                print(f"{'='*60}")
                print(response_local)
                print(f"{'='*60}\n")
            except Exception as e:
                print(f"Back-translation failed: {e}")
                response_local = response_en
        else:
            response_local = response_en
        
        result = {"text": response_local, "text_en": response_en}
        
        # TTS if requested
        if return_audio and self.tts_model:
            try:
                audio_arr = self.text_to_speech(response_local, source_lang, audio_output_path)
                result["audio"] = audio_arr
            except Exception as e:
                print(f"TTS failed: {e}")
        
        return result
    
    def voice_query(self, lang_code, duration=5, return_audio=False, save_recordings=False):
        """Complete voice interaction"""
        query_text = self.record_and_transcribe(lang_code, duration, save_audio=save_recordings)
        print(f"\nYou said: {query_text}\n")
        
        audio_path = None
        if return_audio and save_recordings:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            audio_path = f"response_{lang_code}_{timestamp}.wav"
        
        result = self.ask(query_text, source_lang=lang_code, return_audio=return_audio, audio_output_path=audio_path)
        result['query'] = query_text
        return result


def main():
    """Interactive CLI"""
    print("\nMeitY RAG Chatbot\n")
    
    print("Select mode:")
    print("  1) Text-only (English) - Ready in ~1 minute (RECOMMENDED)")
    print("  2) Text + Translation (22 languages) - 10-30 min first run")
    print("  3) Full (Translation + Speech) - 15-40 min first run")
    
    mode = input("\nEnter choice (1/2/3) [default: 1]: ").strip()
    
    if mode == "3":
        enable_stt, enable_tts, enable_translation = True, True, True
        print("\nFull mode selected\n")
    elif mode == "2":
        enable_stt, enable_tts, enable_translation = False, False, True
        print("\nTranslation mode selected\n")
    else:
        enable_stt, enable_tts, enable_translation = False, False, False
        print("\nText-only mode selected (fastest)\n")
    
    chatbot = None
    try:
        chatbot = IntegratedMeityRAGChatbot(
            index_path=FAISS_INDEX_PATH,
            chunks_path=CHUNKS_PATH,
            llm_path=LLM_MODEL_PATH,
            embed_model_name=EMBEDDING_MODEL_NAME,
            reranker_model_name=RERANKER_MODEL_NAME,
            enable_stt=enable_stt,
            enable_tts=enable_tts,
            enable_translation=enable_translation,
            skip_translation=False
        )
    except KeyboardInterrupt:
        print("\nSetup cancelled. Exiting...")
        return
    except Exception as e:
        print(f"\nFATAL ERROR during initialization: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        return
    
    # Show instructions
    print("\nCOMMANDS")
    print("TEXT:      Just type your question")
    if chatbot.translator:
        print("LANGUAGE:  Type 'lang:hi' to switch language")
    if chatbot.stt_model:
        print("VOICE:     Type 'voice' to record question")
    if chatbot.tts_model:
        print("AUDIO:     Type 'audio:on' for speech output")
    print("EXIT:      Type 'quit' to exit\n")
    
    current_lang = "en"
    audio_output = False
    
    # Main loop
    while True:
        try:
            user_input = input(f"\n[{current_lang}] Your Question: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("\nGoodbye!")
                break
            
            # Language change
            if user_input.lower().startswith('lang:'):
                if not chatbot.translator:
                    print("Translation not available")
                    continue
                current_lang = user_input.split(':', 1)[1].strip()
                print(f"Language: {current_lang}")
                continue
            
            # Audio toggle
            if user_input.lower() in ['audio:on', 'audio:off']:
                if not chatbot.tts_model:
                    print("TTS not available")
                    continue
                audio_output = (user_input.lower() == 'audio:on')
                print(f"Audio output: {'ON' if audio_output else 'OFF'}")
                continue
            
            # Voice input
            if user_input.lower().startswith('voice'):
                if not chatbot.stt_model:
                    print("STT not available")
                    continue
                
                lang = user_input.split(':', 1)[1].strip() if ':' in user_input else current_lang
                result = chatbot.voice_query(lang, duration=5, return_audio=audio_output)
                
                # No need to print again, already printed in ask() method
                continue
            
            # Text query
            result = chatbot.ask(user_input, source_lang=current_lang, return_audio=audio_output)
            
            # No need to print again, already printed in ask() method
            
        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nExiting...")
        sys.exit(0)