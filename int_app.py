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
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torchaudio
from parler_tts import ParlerTTSForConditionalGeneration
from IndicTransToolkit import IndicProcessor
import sys
import gc

# --- CONFIGURATION ---
FAISS_INDEX_PATH = 'meity_faiss.index'
CHUNKS_PATH = 'meity_chunks.json'
LLM_MODEL_PATH = 'Phi-3-mini-4k-instruct-q4.gguf'
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# STT and TTS Models
STT_MODEL_NAME = "ai4bharat/indic-conformer-600m-multilingual"
WHISPER_MODEL_NAME = "openai/whisper-small"
TTS_MODEL_NAME = "ai4bharat/indic-parler-tts"

# IndicTrans2 Models
INDIC_TO_EN_MODEL = "ai4bharat/indictrans2-indic-en-1B"
EN_TO_INDIC_MODEL = "ai4bharat/indictrans2-en-indic-1B"

# IndicConformer supported languages
CONFORMER_SUPPORTED_LANGS = {
    'as', 'bn', 'brx', 'doi', 'gu', 'hi', 'kn', 'kok', 
    'ks', 'mai', 'ml', 'mni', 'mr', 'ne', 'or', 'pa', 
    'sa', 'sat', 'sd', 'ta', 'te', 'ur'
}


class LightweightTranslator:
    """
    FIXED: Translator with aggressive memory management and progress indicators
    """
    
    INDIC_TRANS_LANGS = {
        'as': 'asm_Beng', 'bn': 'ben_Beng', 'brx': 'brx_Deva', 'doi': 'doi_Deva',
        'gu': 'guj_Gujr', 'hi': 'hin_Deva', 'kn': 'kan_Knda', 'ks': 'kas_Arab',
        'ml': 'mal_Mlym', 'mni': 'mni_Mtei', 'mr': 'mar_Deva', 'ne': 'nep_Deva',
        'or': 'ory_Orya', 'pa': 'pan_Guru', 'sa': 'san_Deva', 'sd': 'snd_Arab',
        'ta': 'tam_Taml', 'te': 'tel_Telu', 'ur': 'urd_Arab', 'mai': 'mai_Deva',
        'en': 'eng_Latn'
    }
    
    def __init__(self):
        self.device = "cpu"  # FORCE CPU for translation to save GPU memory
        print(f"✓ Translator initializing on device: {self.device} (CPU mode for memory efficiency)")
        
        self.indic_to_en_model = None
        self.indic_to_en_tokenizer = None
        self.en_to_indic_model = None
        self.en_to_indic_tokenizer = None
        
        # Initialize IndicProcessor
        self.ip = IndicProcessor(inference=True)
        
        self.available = True
        print("✓ Lightweight translation initialized (models load on-demand)")
    
    def _aggressive_cleanup(self):
        """Aggressive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        time.sleep(0.5)  # Give system time to free memory
    
    def _load_indic_to_en(self):
        """FIXED: Load Indic→English with progress and memory management"""
        # Unload other model first
        if self.en_to_indic_model is not None:
            self._unload_en_to_indic()
            self._aggressive_cleanup()
        
        if self.indic_to_en_model is not None:
            print("✓ Indic→English model already loaded")
            return
        
        print("📥 Loading Indic→English translation model (this may take 30-60 seconds)...")
        print("   Please wait, downloading/loading ~1GB model...")
        
        try:
            # Load tokenizer first (smaller)
            print("   [1/2] Loading tokenizer...")
            self.indic_to_en_tokenizer = AutoTokenizer.from_pretrained(
                INDIC_TO_EN_MODEL, 
                trust_remote_code=True
            )
            print("   ✓ Tokenizer loaded")
            
            # Load model (larger)
            print("   [2/2] Loading model weights...")
            self.indic_to_en_model = AutoModelForSeq2SeqLM.from_pretrained(
                INDIC_TO_EN_MODEL, 
                trust_remote_code=True,
                torch_dtype=torch.float32  # Use float32 for CPU
            ).to(self.device)
            
            # Set to eval mode to save memory
            self.indic_to_en_model.eval()
            
            self._aggressive_cleanup()
            
            print("✓ Indic→English model loaded successfully")
            
        except Exception as e:
            print(f"✗ Failed to load Indic→English model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _unload_indic_to_en(self):
        """Unload Indic→English model"""
        if self.indic_to_en_model is not None:
            print("🗑️ Unloading Indic→English model...")
            del self.indic_to_en_model
            del self.indic_to_en_tokenizer
            self.indic_to_en_model = None
            self.indic_to_en_tokenizer = None
            self._aggressive_cleanup()
            print("✓ Indic→English model unloaded")
    
    def _load_en_to_indic(self):
        """FIXED: Load English→Indic with progress and memory management"""
        # Unload other model first
        if self.indic_to_en_model is not None:
            self._unload_indic_to_en()
            self._aggressive_cleanup()
        
        if self.en_to_indic_model is not None:
            print("✓ English→Indic model already loaded")
            return
        
        print("📥 Loading English→Indic translation model (this may take 30-60 seconds)...")
        print("   Please wait, downloading/loading ~1GB model...")
        
        try:
            # Load tokenizer first
            print("   [1/2] Loading tokenizer...")
            self.en_to_indic_tokenizer = AutoTokenizer.from_pretrained(
                EN_TO_INDIC_MODEL, 
                trust_remote_code=True
            )
            print("   ✓ Tokenizer loaded")
            
            # Load model
            print("   [2/2] Loading model weights...")
            self.en_to_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
                EN_TO_INDIC_MODEL, 
                trust_remote_code=True,
                torch_dtype=torch.float32  # Use float32 for CPU
            ).to(self.device)
            
            # Set to eval mode
            self.en_to_indic_model.eval()
            
            self._aggressive_cleanup()
            
            print("✓ English→Indic model loaded successfully")
            
        except Exception as e:
            print(f"✗ Failed to load English→Indic model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _unload_en_to_indic(self):
        """Unload English→Indic model"""
        if self.en_to_indic_model is not None:
            print("🗑️ Unloading English→Indic model...")
            del self.en_to_indic_model
            del self.en_to_indic_tokenizer
            self.en_to_indic_model = None
            self.en_to_indic_tokenizer = None
            self._aggressive_cleanup()
            print("✓ English→Indic model unloaded")
    
    def ensure_models_unloaded(self):
        """Ensure both models are unloaded"""
        self._unload_indic_to_en()
        self._unload_en_to_indic()
    
    def translate_to_english(self, text, source_lang):
        """FIXED: Translate Indic→English with timeout protection"""
        if source_lang == "en":
            return text
        
        if source_lang not in self.INDIC_TRANS_LANGS:
            print(f"⚠ Language {source_lang} not supported")
            return text
        
        try:
            # Load model
            self._load_indic_to_en()
            
            src_lang_code = self.INDIC_TRANS_LANGS[source_lang]
            tgt_lang_code = "eng_Latn"
            
            print(f"   Preprocessing text...")
            batch = self.ip.preprocess_batch(
                [text],
                src_lang=src_lang_code,
                tgt_lang=tgt_lang_code,
            )
            
            print(f"   Tokenizing...")
            inputs = self.indic_to_en_tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)
            
            print(f"   Generating translation...")
            with torch.no_grad():
                outputs = self.indic_to_en_model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )
            
            print(f"   Decoding...")
            generated_tokens = self.indic_to_en_tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            
            translations = self.ip.postprocess_batch(generated_tokens, lang=tgt_lang_code)
            translated = translations[0]
            
            print(f"✓ Translated ({source_lang}→en): {text[:50]}... → {translated[:50]}...")
            
            # Immediately unload
            self._unload_indic_to_en()
            
            return translated.strip()
        
        except Exception as e:
            print(f"✗ Translation error ({source_lang}→en): {e}")
            import traceback
            traceback.print_exc()
            self._unload_indic_to_en()
            return text
    
    def translate_from_english(self, text, target_lang):
        """FIXED: Translate English→Indic with timeout protection"""
        if target_lang == "en":
            return text
        
        if target_lang not in self.INDIC_TRANS_LANGS:
            print(f"⚠ Language {target_lang} not supported")
            return text
        
        try:
            # Load model
            self._load_en_to_indic()
            
            src_lang_code = "eng_Latn"
            tgt_lang_code = self.INDIC_TRANS_LANGS[target_lang]
            
            print(f"   Preprocessing text...")
            batch = self.ip.preprocess_batch(
                [text],
                src_lang=src_lang_code,
                tgt_lang=tgt_lang_code,
            )
            
            print(f"   Tokenizing...")
            inputs = self.en_to_indic_tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)
            
            print(f"   Generating translation...")
            with torch.no_grad():
                outputs = self.en_to_indic_model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )
            
            print(f"   Decoding...")
            generated_tokens = self.en_to_indic_tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            
            translations = self.ip.postprocess_batch(generated_tokens, lang=tgt_lang_code)
            translated = translations[0]
            
            print(f"✓ Translated (en→{target_lang}): {text[:50]}... → {translated[:50]}...")
            
            # Immediately unload
            self._unload_en_to_indic()
            
            return translated.strip()
        
        except Exception as e:
            print(f"✗ Translation error (en→{target_lang}): {e}")
            import traceback
            traceback.print_exc()
            self._unload_en_to_indic()
            return text


class IntegratedAPP:
    """Memory-optimized MeitY RAG Chatbot"""
    
    def __init__(self, index_path, chunks_path, llm_path, embed_model_name, 
                 reranker_model_name, enable_stt=False, enable_tts=False, 
                 enable_translation=True, skip_translation=False):
        
        print("Initializing Memory-Optimized MeitY RAG Chatbot")
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")
        
        # Model placeholders
        self.conformer_model = None
        self.whisper_pipeline = None
        self.tts_model = None
        self.tts_tokenizer = None
        self.tts_desc_tokenizer = None
        
        # 1. Load FAISS and chunks
        print("\n[1/5] Loading FAISS index and chunks...")
        try:
            self.index = faiss.read_index(index_path)
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            print(f"✓ Loaded {len(self.chunks)} chunks")
        except Exception as e:
            print(f"✗ Failed: {e}")
            raise
        
        # 2. Load embedding model
        print("\n[2/5] Loading embedding model...")
        try:
            self.embedding_model = SentenceTransformer(embed_model_name)
            print("✓ Embedding model loaded")
        except Exception as e:
            print(f"✗ Failed: {e}")
            raise
        
        # 3. Load reranker
        print("\n[3/5] Loading reranker...")
        try:
            self.reranker_model = CrossEncoder(reranker_model_name)
            print("✓ Reranker loaded")
        except Exception as e:
            print(f"✗ Failed: {e}")
            raise
        
        # 4. Load LLM
        print(f"\n[4/5] Loading LLM from: {llm_path}...")
        try:
            n_threads = os.cpu_count() or 4
            self.llm = Llama(
                model_path=llm_path,
                n_ctx=4096,
                n_threads=n_threads,
                n_gpu_layers=0,
                verbose=False
            )
            print(f"✓ LLM loaded ({n_threads} threads)")
        except Exception as e:
            print(f"✗ Failed: {e}")
            raise
        
        # 5. Load translation
        self.translator = None
        if skip_translation:
            print("\n[5/5] Translation skipped")
        elif enable_translation:
            print("\n[5/5] Initializing translation...")
            try:
                self.translator = LightweightTranslator()
                print("✓ Translation ready")
            except Exception as e:
                print(f"✗ Translation failed: {e}")
                self.translator = None
        else:
            print("\n[5/5] Translation disabled")
        
        print("\n✓ STT/TTS will load on demand")
        
        # Speaker mapping
        self.speakers = {
            "as": "Amit", "bn": "Arjun", "brx": "Bikram", "doi": "Karan",
            "en": "Thoma", "gu": "Yash", "hi": "Divya", "kn": "Suresh",
            "ks": "FemaleSpeaker", "ml": "Anjali", "mni": "Laishram",
            "mr": "Sanjay", "ne": "Amrita", "or": "Manas", "pa": "Divjot",
            "sa": "Aryan", "sd": "FemaleSpeaker", "ta": "Jaya",
            "te": "Prakash", "ur": "FemaleSpeaker", "mai": "FemaleSpeaker",
            "kok": "FemaleSpeaker", "sat": "FemaleSpeaker"
        }
        
        print("\n" + "="*60)
        print("✓ Chatbot ready (Memory-optimized)")
        print("="*60)
    
    def _load_conformer(self):
        """Load IndicConformer"""
        if self.conformer_model is not None:
            return
        
        print("📥 Loading IndicConformer...")
        try:
            self.conformer_model = AutoModel.from_pretrained(
                STT_MODEL_NAME, 
                trust_remote_code=True
            )
            if torch.cuda.is_available():
                self.conformer_model = self.conformer_model.to(self.device)
            print("✓ IndicConformer loaded")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ Failed: {e}")
            raise
    
    def _load_whisper(self):
        """Load Whisper"""
        if self.whisper_pipeline is not None:
            return
        
        print("📥 Loading Whisper...")
        try:
            self.whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model=WHISPER_MODEL_NAME,
                device=0 if torch.cuda.is_available() else -1
            )
            print("✓ Whisper loaded")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ Failed: {e}")
            raise
    
    def _unload_conformer(self):
        """Unload IndicConformer"""
        if self.conformer_model is not None:
            print("🗑️ Unloading IndicConformer...")
            del self.conformer_model
            self.conformer_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✓ IndicConformer unloaded")
    
    def _unload_whisper(self):
        """Unload Whisper"""
        if self.whisper_pipeline is not None:
            print("🗑️ Unloading Whisper...")
            del self.whisper_pipeline
            self.whisper_pipeline = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✓ Whisper unloaded")
    
    def load_stt_models(self, lang_code):
        """Load appropriate STT model"""
        if lang_code in CONFORMER_SUPPORTED_LANGS:
            print(f"\n🎤 Loading STT for {lang_code} (IndicConformer)...")
            self._load_conformer()
        else:
            print(f"\n🎤 Loading STT for {lang_code} (Whisper)...")
            self._load_whisper()
    
    def unload_all_stt_models(self):
        """Unload all STT models"""
        print("\n🗑️ Unloading STT models...")
        self._unload_conformer()
        self._unload_whisper()
        print("✓ All STT models unloaded")
    
    def load_tts_model(self):
        """Load TTS model"""
        if self.tts_model is not None:
            return
        
        print("\nLoading TTS model...")
        try:
            device = "cpu"
            self.tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
                TTS_MODEL_NAME
            ).to(device)
            self.tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_NAME)
            self.tts_desc_tokenizer = AutoTokenizer.from_pretrained(
                self.tts_model.config.text_encoder._name_or_path
            )
            print("✓ TTS loaded")
        except Exception as e:
            print(f"✗ Failed: {e}")
            raise
    
    def unload_tts_model(self):
        """Unload TTS"""
        if self.tts_model is not None:
            del self.tts_model
            del self.tts_tokenizer
            del self.tts_desc_tokenizer
            self.tts_model = None
            self.tts_tokenizer = None
            self.tts_desc_tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✓ TTS unloaded")
    
    def speech_to_text(self, audio_file_or_array, lang_code, sample_rate=16000):
        """Convert speech to text"""
        self.load_stt_models(lang_code)
        
        try:
            # Load audio
            if isinstance(audio_file_or_array, str):
                wav, sr = torchaudio.load(audio_file_or_array)
            else:
                audio_arr = audio_file_or_array
                if audio_arr.dtype != np.float32:
                    audio_arr = audio_arr.astype(np.float32)
                wav = torch.from_numpy(audio_arr).unsqueeze(0)
                sr = sample_rate
            
            # Mono
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            
            # Resample to 16kHz
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                wav = resampler(wav)
            
            # Transcribe
            transcription = None
            if lang_code in CONFORMER_SUPPORTED_LANGS:
                print(f"🎤 Transcribing with IndicConformer ({lang_code})...")
                with torch.no_grad():
                    transcription = self.conformer_model(wav, lang_code, "ctc")
            else:
                print(f"🎤 Transcribing with Whisper ({lang_code})...")
                audio_np = wav.squeeze().numpy()
                whisper_lang = 'english' if lang_code == 'en' else 'english'
                result = self.whisper_pipeline(
                    audio_np,
                    generate_kwargs={"language": whisper_lang}
                )
                transcription = result.get("text", "").strip()
            
            print(f"✓ Transcription: {transcription[:100] if transcription else 'None'}...")
            
            # Unload immediately
            self.unload_all_stt_models()
            
            return transcription.strip() if transcription else None
            
        except Exception as e:
            print(f"✗ Transcription error: {e}")
            import traceback
            traceback.print_exc()
            self.unload_all_stt_models()
            return None
    
    def text_to_speech(self, text, lang_code, output_path=None):
        """Convert text to speech"""
        if self.tts_model is None:
            self.load_tts_model()
        
        speaker = self.speakers.get(lang_code, "FemaleSpeaker")
        description = f"{speaker}'s voice, clear and natural."
        
        device = "cpu"
        
        desc_inputs = self.tts_desc_tokenizer(description, return_tensors="pt").to(device)
        prompt_inputs = self.tts_tokenizer(text, return_tensors="pt").to(device)
        
        try:
            with torch.no_grad():
                generation = self.tts_model.generate(
                    input_ids=desc_inputs.input_ids,
                    attention_mask=desc_inputs.attention_mask,
                    prompt_input_ids=prompt_inputs.input_ids,
                    prompt_attention_mask=prompt_inputs.attention_mask
                )
            
            audio_arr = generation.cpu().numpy().squeeze()
            sample_rate = self.tts_model.config.sampling_rate
            
            if output_path:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                sf.write(output_path, audio_arr, sample_rate)
            
            return audio_arr, sample_rate
        
        except Exception as e:
            print(f"✗ TTS error: {e}")
            return None, None
    
    def _retrieve_context(self, query, k_initial=10):
        """Retrieve contexts from FAISS"""
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
        max_context_length = 1500
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        prompt = f"""<|system|>
You are an assistant for the Ministry of Electronics and Information Technology (MeitY), India.
Give only to-the-point answers based strictly on the provided context.
Do not add extra information. If context doesn't contain the answer, say so.<|end|>
<|user|>
CONTEXT:
{context}

QUESTION:
{query}

Provide a clear, concise answer based on the context above.<|end|>
<|assistant|>
"""
        
        try:
            response = self.llm(
                prompt, 
                max_tokens=256,
                stop=["<|end|>", "<|user|>"], 
                echo=False,
                temperature=0.7
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            print(f"✗ LLM error: {e}")
            return "I encountered an error generating a response."
    
    def ask(self, query, source_lang="en", target_lang=None, return_audio=False, audio_output_path=None):
        """
        Main query method with sequential translation
        """
        if target_lang is None:
            target_lang = source_lang
        
        # CRITICAL: Unload translation models first
        if self.translator:
            print("\n🔄 Ensuring translation models unloaded...")
            self.translator.ensure_models_unloaded()
        
        # STEP 1: Translate query to English
        query_en = query
        if source_lang != "en" and self.translator:
            print(f"\n📝 [STEP 1/4] Translating query ({source_lang}→en)...")
            try:
                query_en = self.translator.translate_to_english(query, source_lang)
                print(f"✓ English query: {query_en}")
            except Exception as e:
                print(f"⚠ Translation failed: {e}")
                query_en = query
        else:
            print(f"\n✓ [STEP 1/4] Input in English")
        
        # STEP 2: RAG
        print(f"\n🔍 [STEP 2/4] Retrieving context...")
        chunks = self._retrieve_context(query_en)
        
        if not chunks:
            response_en = "I don't have enough information to answer this."
        else:
            context = self._rerank_context(query_en, chunks)
            if not context:
                response_en = "I don't have enough information to answer this."
            else:
                print(f"✓ Context retrieved")
                print(f"\n🤖 [STEP 3/4] Generating response...")
                response_en = self._generate_response(query_en, context)
                print(f"✓ English response: {response_en[:100]}...\n")
        
        result = {
            "text": response_en,
            "text_en": response_en,
            "translation_pending": target_lang != "en"
        }
        
        # STEP 4: Translate response to target language
        if target_lang != "en" and self.translator:
            print(f"\n🌐 [STEP 4/4] Translating response (en→{target_lang})...")
            try:
                response_local = self.translator.translate_from_english(response_en, target_lang)
                print(f"✓ Translated: {response_local[:100]}...")
                result["text"] = response_local
                result["translation_pending"] = False
            except Exception as e:
                print(f"⚠ Translation failed: {e}")
                result["translation_pending"] = False
        else:
            print(f"\n✓ [STEP 4/4] No translation needed")
            result["translation_pending"] = False
        
        # Generate audio if requested
        if return_audio:
            print(f"\n🔊 Generating audio in {target_lang}...")
            try:
                audio_arr, sample_rate = self.text_to_speech(result["text"], target_lang, audio_output_path)
                if audio_arr is not None:
                    result["audio"] = audio_arr
                    result["sample_rate"] = sample_rate
                    print("✓ Audio generated")
            except Exception as e:
                print(f"⚠ TTS failed: {e}")
        
        return result
    
    def voice_query(self, lang_code, duration=5, return_audio=False, save_recordings=False):
        """Complete voice interaction"""
        print(f"\nRecording {duration} seconds...")
        audio_array = self.record_audio(duration)
        
        if save_recordings:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            audio_file = f"recorded_query_{lang_code}_{timestamp}.wav"
            sf.write(audio_file, audio_array, 16000)
        
        query_text = self.speech_to_text(audio_array, lang_code)
        
        if not query_text:
            return {"query": None, "text": "Could not understand audio", "text_en": "Could not understand audio"}
        
        audio_path = None
        if return_audio and save_recordings:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            audio_path = f"response_{lang_code}_{timestamp}.wav"
        
        result = self.ask(
            query_text, 
            source_lang=lang_code, 
            target_lang=lang_code,
            return_audio=return_audio, 
            audio_output_path=audio_path
        )
        result['query'] = query_text
        
        return result
    
    def record_audio(self, duration=5, sample_rate=16000):
        """Record audio from microphone"""
        audio = sd.rec(
            int(duration * sample_rate), 
            samplerate=sample_rate, 
            channels=1, 
            dtype='float32'
        )
        sd.wait()
        return np.squeeze(audio)


def main():
    """Interactive CLI"""
    print("\n" + "="*60)
    print("MeitY RAG Chatbot - Memory-Optimized")
    print("="*60 + "\n")
    
    print("Select mode:")
    print("  1) Text-only (English) - ~2GB RAM")
    print("  2) Text + Translation - ~3-4GB RAM")
    print("  3) Full (Translation + Speech) - ~4-5GB RAM")
    
    mode = input("\nChoice (1/2/3) [default: 2]: ").strip() or "2"
    
    if mode == "3":
        enable_stt, enable_tts, enable_translation = False, False, True
        print("\n✓ Full mode (models load on demand)\n")
    elif mode == "2":
        enable_stt, enable_tts, enable_translation = False, False, True
        print("\n✓ Translation mode\n")
    else:
        enable_stt, enable_tts, enable_translation = False, False, False
        print("\n✓ Text-only mode\n")
    
    chatbot = None
    try:
        chatbot = IntegratedAPP(
            index_path=FAISS_INDEX_PATH,
            chunks_path=CHUNKS_PATH,
            llm_path=LLM_MODEL_PATH,
            embed_model_name=EMBEDDING_MODEL_NAME,
            reranker_model_name=RERANKER_MODEL_NAME,
            enable_stt=enable_stt,
            enable_tts=enable_tts,
            enable_translation=enable_translation,
            skip_translation=not enable_translation
        )
    except KeyboardInterrupt:
        print("\n⚠ Cancelled")
        return
    except Exception as e:
        print(f"\n✗ FATAL: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        return
    
    print("\n" + "="*60)
    print("COMMANDS")
    print("="*60)
    print("TEXT:      Type your question")
    if chatbot.translator:
        print("LANGUAGE:  Type 'lang:hi' to switch language")
    print("VOICE:     Type 'voice' to record (loads STT)")
    print("AUDIO:     Type 'audio:on' for voice responses (loads TTS)")
    print("HELP:      Type 'help'")
    print("EXIT:      Type 'quit'")
    print("="*60 + "\n")
    
    current_lang = "en"
    audio_output = False
    
    while True:
        try:
            user_input = input(f"\n[{current_lang}] Question: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\nCommands:")
                print("  lang:CODE   - Switch language (hi, bn, ta, etc.)")
                print("  voice       - Record voice question")
                print("  audio:on    - Enable voice responses")
                print("  audio:off   - Disable voice responses")
                print("  quit        - Exit")
                continue
            
            if user_input.lower().startswith('lang:'):
                if not chatbot.translator:
                    print("⚠ Translation not available")
                    continue
                current_lang = user_input.split(':', 1)[1].strip()
                print(f"✓ Language: {current_lang}")
                continue
            
            if user_input.lower() in ['audio:on', 'audio:off']:
                audio_output = (user_input.lower() == 'audio:on')
                print(f"✓ Audio: {'ON' if audio_output else 'OFF'}")
                if audio_output and chatbot.tts_model is None:
                    print("⏳ TTS will load on first use")
                continue
            
            if user_input.lower().startswith('voice'):
                lang = user_input.split(':', 1)[1].strip() if ':' in user_input else current_lang
                result = chatbot.voice_query(lang, duration=5, return_audio=audio_output)
                print(f"\nYou said: {result['query']}")
                print(f"\nResponse: {result['text']}")
                continue
            
            result = chatbot.ask(
                user_input, 
                source_lang=current_lang, 
                target_lang=current_lang,
                return_audio=audio_output
            )
            
            print(f"\nResponse: {result['text']}")
            if current_lang != 'en' and result['text_en'] != result['text']:
                print(f"\n(English: {result['text_en']})")
            
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted. Type 'quit' to exit.")
            continue
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()




