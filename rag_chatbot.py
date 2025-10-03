# import json
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from llama_cpp import Llama
# import time

# # --- 1. CONFIGURATION ---
# # Make sure these paths are correct
# FAISS_INDEX_PATH = 'meity_faiss.index'
# CHUNKS_PATH = 'meity_chunks.json'
# # Update this to the exact name of the GGUF model file you downloaded
# LLM_MODEL_PATH = 'Meta-Llama-3-8B-Instruct.Q4_K_M.gguf' 
# EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

# class MeityRAGChatbot:
#     def __init__(self, index_path, chunks_path, llm_path, embed_model_name):
#         """
#         Initializes the chatbot by loading all necessary models and data.
#         """
#         print("Initializing the MeitY RAG Chatbot...")
        
#         # --- Load Retrieval Components ---
#         print("Loading FAISS index and text chunks...")
#         self.index = faiss.read_index(index_path)
#         with open(chunks_path, 'r', encoding='utf-8') as f:
#             self.chunks = json.load(f)
        
#         print("Loading embedding model...")
#         self.embedding_model = SentenceTransformer(embed_model_name)
        
#         # --- Load Generation Component (LLM) ---
#         print(f"Loading LLM from: {llm_path}...")
#         # n_ctx is the context window size. 2048 is a safe default.
#         self.llm = Llama(model_path=llm_path, n_ctx=2048, verbose=False)
        
#         print("\n✅ Chatbot is ready to answer questions!")
#         print("-----------------------------------------")

#     def _retrieve_context(self, query: str, k: int = 5):
#         """
#         Retrieves the top k most relevant document chunks from the vector store.
#         """
#         query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
#         query_embedding = np.array(query_embedding).astype('float32')
        
#         # Search the FAISS index
#         distances, indices = self.index.search(query_embedding, k)
        
#         # Get the corresponding text chunks
#         retrieved_docs = [self.chunks[i] for i in indices[0] if i != -1]
#         return "\n\n".join(retrieved_docs)

#     def _generate_response(self, query: str, context: str):
#         """
#         Generates a response using the LLM based on the query and retrieved context.
#         """
#         # This prompt template is crucial for RAG. It instructs the LLM on how to behave.
#         prompt = f"""
#         <|begin_of_text|><|start_header_id|>system<|end_header_id|>
#         You are a helpful and respectful assistant for the Ministry of Electronics and Information Technology (MeitY), India.
#         Your task is to answer user questions based ONLY on the provided context.
#         - If the answer is available in the context, provide a clear and concise answer.
#         - If the answer is not found in the context, you MUST say "I do not have enough information to answer this question."
#         - Do not add any information that is not present in the context.
#         - Answer in the same language as the user's question.
#         <|eot_id|><|start_header_id|>user<|end_header_id|>
#         CONTEXT:
#         {context}

#         QUESTION:
#         {query}

#         ANSWER:
#         <|eot_id|><|start_header_id|>assistant<|end_header_id|>
#         """
        
#         print("Generating response...")
#         start_time = time.time()
        
#         # Call the LLM to generate text
#         response = self.llm(
#             prompt,
#             max_tokens=350,  # Max length of the generated answer
#             stop=["<|eot_id|>"], # Stop generation at the end of assistant's turn
#             echo=False       # Do not repeat the prompt in the output
#         )
        
#         end_time = time.time()
#         print(f"LLM generation took {end_time - start_time:.2f} seconds.")
        
#         # The actual text is in the 'choices'[0]['text'] field
#         return response['choices'][0]['text'].strip()

#     def ask(self, query: str):
#         """
#         The main method to ask a question to the chatbot.
#         It orchestrates the retrieval and generation steps.
#         """
#         # 1. Retrieve relevant context
#         context = self._retrieve_context(query)
        
#         if not context:
#             return "I could not find any relevant information for your query."
        
#         # 2. Generate a response based on the context
#         answer = self._generate_response(query, context)
#         return answer

# if __name__ == "__main__":
#     # Ensure you have the required files in the same directory:
#     # - meity_faiss.index
#     # - meity_chunks.json
#     # - The downloaded .gguf model file
    
#     # Initialize the chatbot
#     chatbot = MeityRAGChatbot(
#         index_path=FAISS_INDEX_PATH,
#         chunks_path=CHUNKS_PATH,
#         llm_path=LLM_MODEL_PATH,
#         embed_model_name=EMBEDDING_MODEL_NAME
#     )
    
#     # Start an interactive chat loop
#     print("Enter 'quit' to exit the chat.")
#     while True:
#         user_query = input("\nYour Question: ")
#         if user_query.lower() == 'quit':
#             break
        
#         response = chatbot.ask(user_query)
#         print("\nMeitY Assistant:", response)


# import json
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from llama_cpp import Llama
# import time
# import os # For getting thread count

# # --- 1. CONFIGURATION ---
# FAISS_INDEX_PATH = 'meity_faiss.index'
# CHUNKS_PATH = 'meity_chunks.json'
# LLM_MODEL_PATH = 'Meta-Llama-3-8B-Instruct.Q4_K_M.gguf' 
# EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

# class MeityRAGChatbot:
#     def __init__(self, index_path, chunks_path, llm_path, embed_model_name):
#         print("Initializing the MeitY RAG Chatbot...")
        
#         print("Loading FAISS index and text chunks...")
#         self.index = faiss.read_index(index_path)
#         with open(chunks_path, 'r', encoding='utf-8') as f:
#             self.chunks = json.load(f)
        
#         print("Loading embedding model...")
#         self.embedding_model = SentenceTransformer(embed_model_name)
        
#         print(f"Loading LLM from: {llm_path}...")
#         # Get the number of physical CPU cores for better performance
#         # Fallback to 4 if not detectable
#         n_threads = os.cpu_count() or 4
#         print(f"Using {n_threads} threads for LLM.")
        
#         self.llm = Llama(
#             model_path=llm_path, 
#             n_ctx=2048, 
#             n_threads=n_threads, # Use detected number of threads
#             verbose=False
#         )
        
#         print("\n✅ Chatbot is ready to answer questions!")
#         print("-----------------------------------------")

#     def _retrieve_context(self, query: str, k: int = 5):
#         """
#         Retrieves the top k most relevant document chunks from the vector store.
#         """
#         print(f"\n1. Retrieving context for query: '{query}'")
#         query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
#         query_embedding = np.array(query_embedding).astype('float32')
        
#         distances, indices = self.index.search(query_embedding, k)
        
#         retrieved_docs = [self.chunks[i] for i in indices[0] if i != -1]
#         return "\n\n".join(retrieved_docs)

#     def _generate_response(self, query: str, context: str):
#         """
#         Generates a response using the LLM based on the query and retrieved context.
#         """
#         print("3. Generating response with LLM...")
#         prompt = f"""
#         <|begin_of_text|><|start_header_id|>system<|end_header_id|>
#         You are a helpful and respectful assistant for the Ministry of Electronics and Information Technology (MeitY), India.
#         Your task is to answer user questions based ONLY on the provided context.
#         - If the answer is available in the context, provide a clear and concise answer.
#         - If the answer is not found in the context, you MUST say "I do not have enough information to answer this question."
#         - Do not add any information that is not present in the context.
#         - Answer in the same language as the user's question.
#         <|eot_id|><|start_header_id|>user<|end_header_id|>
#         CONTEXT:
#         {context}

#         QUESTION:
#         {query}

#         ANSWER:
#         <|eot_id|><|start_header_id|>assistant<|end_header_id|>
#         """
        
#         start_time = time.time()
#         response = self.llm(
#             prompt,
#             max_tokens=350,
#             stop=["<|eot_id|>"],
#             echo=False
#         )
#         end_time = time.time()
#         print(f"LLM generation took {end_time - start_time:.2f} seconds.")
#         return response['choices'][0]['text'].strip()

#     def ask(self, query: str):
#         """
#         The main method to ask a question to the chatbot.
#         """
#         # Step 1: Retrieve context
#         context = self._retrieve_context(query)
        
#         # --- THIS IS THE CRUCIAL DEBUGGING STEP ---
#         print("\n2. Displaying Retrieved Context:")
#         if not context or context.isspace():
#             print("--- CONTEXT IS EMPTY ---")
#             # If context is empty, we don't need to ask the LLM. We already know the answer.
#             return "I do not have enough information to answer this question because no relevant documents were found in the database."
#         else:
#             print("--- START OF CONTEXT ---")
#             print(context)
#             print("--- END OF CONTEXT ---")
#         # ------------------------------------------

#         # Step 2: Generate a response based on the context
#         answer = self._generate_response(query, context)
#         return answer

# if __name__ == "__main__":
#     chatbot = MeityRAGChatbot(
#         index_path=FAISS_INDEX_PATH,
#         chunks_path=CHUNKS_PATH,
#         llm_path=LLM_MODEL_PATH,
#         embed_model_name=EMBEDDING_MODEL_NAME
#     )
    
#     print("Enter 'quit' to exit the chat.")
#     while True:
#         user_query = input("\nYour Question: ")
#         if user_query.lower() == 'quit':
#             break
        
#         response = chatbot.ask(user_query)
#         print("\nMeitY Assistant:", response)



import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from sentence_transformers import CrossEncoder
import time
import os

# --- 1. CONFIGURATION ---
FAISS_INDEX_PATH = 'meity_faiss.index'
CHUNKS_PATH = 'meity_chunks.json'
LLM_MODEL_PATH = 'Meta-Llama-3-8B-Instruct.Q4_K_M.gguf' 
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2' # New addition for improved retrieval

class MeityRAGChatbot:
    def __init__(self, index_path, chunks_path, llm_path, embed_model_name, reranker_model_name):
        print("Initializing the MeitY RAG Chatbot...")
        
        # 1. Load FAISS index and text chunks
        print("Loading FAISS index and text chunks...")
        self.index = faiss.read_index(index_path)
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        # 2. Load Embedding Model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embed_model_name)
        
        # 3. Load Reranker Model (New)
        print("Loading Reranker model...")
        self.reranker_model = CrossEncoder(reranker_model_name)

        # 4. Load LLM
        print(f"Loading LLM from: {llm_path}...")
        n_threads = os.cpu_count() or 4
        print(f"Using {n_threads} threads for LLM.")
        
        self.llm = Llama(
            model_path=llm_path, 
            # Increased context size to 4096 to accommodate more relevant chunks
            n_ctx=4096, 
            n_threads=n_threads, 
            verbose=False
        )
        
        print("\n Chatbot is ready to answer questions!")
        print("-----------------------------------------")

    def _retrieve_context(self, query: str, k_initial: int = 10):
        """
        Retrieves an initial set of chunks (k_initial) using vector similarity.
        """
        print(f"\n1. Retrieving initial {k_initial} contexts for query: '{query}'")
        
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding).astype('float32')
        
        distances, indices = self.index.search(query_embedding, k_initial)
        
        # Only include valid indices
        retrieved_chunks = [self.chunks[i] for i in indices[0] if i != -1]
        
        # We also need the original indices for logging/debugging if necessary, 
        # but for reranking, we just need the text.
        return retrieved_chunks

    def _rerank_context(self, query: str, chunks: list, k_rerank: int = 3):
        """
        Reranks the retrieved chunks using a CrossEncoder for precise relevance.
        """
        if not chunks:
            return ""

        print(f"1.5. Reranking contexts and keeping top {k_rerank}...")
        
        # Format the pairs for the cross-encoder
        pairs = [[query, chunk] for chunk in chunks]
        
        # Predict the relevance score for each pair
        scores = self.reranker_model.predict(pairs)
        
        # Combine chunks and scores, then sort by score (descending)
        scored_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        
        # Take the top k_rerank chunks and join them
        top_chunks = [chunk for chunk, score in scored_chunks[:k_rerank]]
        
        return "\n\n---\n\n".join(top_chunks)


    def _generate_response(self, query: str, context: str):
        """
        Generates a response using the LLM based on the query and RERANKED context.
        """
        print("3. Generating response with LLM...")
        
        # IMPROVED SYSTEM PROMPT: More aggressive on extraction, clearer role, and strict instruction to extract facts.
        prompt = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a highly capable and precise factual extraction assistant for the Ministry of Electronics and Information Technology (MeitY), India.
        Your primary task is to **extract the most direct and concise answer** to the user's QUESTION **strictly and only** from the provided CONTEXT.

        **Instructions for Extraction:**
        1. **PRIORITIZE EXTRACTION:** Look hard for the specific fact (e.g., an address, a name, a definition) and extract it verbatim or paraphrase only to make it a direct answer.
        2. **STRICT CONTEXT RULE:** If the exact fact (like an address or definition) is NOT present in the CONTEXT, you **MUST** respond with: "I do not have enough information to answer this question from the provided text."
        3. **FORMATTING:** Present the final answer clearly and directly. Do not add introductory or concluding sentences like "Based on the context..."
        
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        CONTEXT:
        {context}

        QUESTION:
        {query}

        ANSWER:
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        
        start_time = time.time()
        response = self.llm(
            prompt,
            max_tokens=500, # Increased max_tokens slightly for safety
            stop=["<|eot_id|>"],
            echo=False
        )
        end_time = time.time()
        print(f"LLM generation took {end_time - start_time:.2f} seconds.")
        return response['choices'][0]['text'].strip()

    def ask(self, query: str):
        """
        The main method to ask a question to the chatbot.
        """
        # Step 1: Retrieve a larger set of context
        initial_chunks = self._retrieve_context(query)
        
        # Step 1.5: Rerank and aggregate the best context
        context = self._rerank_context(query, initial_chunks)
        
        # Step 2: Display and check context
        print("\n2. Displaying Retrieved Context:")
        if not context or context.isspace():
            print("--- CONTEXT IS EMPTY ---")
            return "I do not have enough information to answer this question because no relevant documents were found in the database."
        else:
            print("--- START OF RERANKED CONTEXT ---")
            print(context)
            print("--- END OF RERANKED CONTEXT ---")

        # Step 3: Generate a response based on the context
        answer = self._generate_response(query, context)
        return answer

if __name__ == "__main__":
    try:
        chatbot = MeityRAGChatbot(
            index_path=FAISS_INDEX_PATH,
            chunks_path=CHUNKS_PATH,
            llm_path=LLM_MODEL_PATH,
            embed_model_name=EMBEDDING_MODEL_NAME,
            reranker_model_name=RERANKER_MODEL_NAME
        )
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: A required file was not found: {e}")
        print("Please ensure FAISS_INDEX_PATH, CHUNKS_PATH, and LLM_MODEL_PATH are correct.")
        exit()
    except Exception as e:
        print(f"\nFATAL ERROR during initialization: {e}")
        exit()
    
    print("Enter 'quit' to exit the chat.")
    while True:
        user_query = input("\nYour Question: ")
        if user_query.lower() == 'quit':
            break
        
        response = chatbot.ask(user_query)
        print("\nMeitY Assistant:", response)