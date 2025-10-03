import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time

# --- Configuration ---
# This is a powerful, open-source multilingual model that supports many Indian languages.
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
CHUNKED_DATA_PATH = 'meity_chunked_data.json'
FAISS_INDEX_PATH = 'meity_faiss.index'
CHUNKS_PATH = 'meity_chunks.json'

def load_chunked_data(file_path):
    """
    Loads the chunked data from a JSON file.
    Assumes the JSON file is a list of objects, each with a 'content' key.
    """
    print(f"Loading data from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Assuming each item in the list is a dictionary with a 'content' field
        # Modify this line if your JSON structure is different
        content_chunks = [item['content'] for item in data if 'content' in item and item['content']]
        print(f"Successfully loaded {len(content_chunks)} text chunks.")
        return content_chunks
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON file.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

def create_and_save_embeddings():
    """
    Main function to generate embeddings and create a FAISS vector store.
    """
    # 1. Load the text data
    chunks = load_chunked_data(CHUNKED_DATA_PATH)
    if not chunks:
        print("No data loaded. Exiting.")
        return

    # 2. Load the pre-trained Sentence Transformer model
    print(f"Loading sentence transformer model: {MODEL_NAME}...")
    # This model will be downloaded automatically on the first run
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully.")

    # 3. Create text embeddings
    print("Generating embeddings for all text chunks. This may take a while...")
    start_time = time.time()
    embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
    end_time = time.time()
    print(f"Embeddings generated in {end_time - start_time:.2f} seconds.")
    print(f"Shape of the embedding matrix: {embeddings.shape}")

    # Ensure embeddings are in float32 format for FAISS
    embeddings = np.array(embeddings).astype('float32')

    # 4. Store embeddings in a FAISS Vector Database
    embedding_dimension = embeddings.shape[1]
    # We use IndexFlatL2, a simple but effective index for dense vectors.
    # It performs a brute-force L2 distance search.
    index = faiss.IndexFlatL2(embedding_dimension)
    
    # For faster search, especially with millions of vectors, you could use more advanced indexes
    # like IndexIVFFlat. For now, IndexFlatL2 is perfect.
    # index = faiss.IndexIDMap(index) # Optional: to map to original IDs

    print("Building FAISS index...")
    index.add(embeddings)
    print(f"FAISS index built successfully. Total vectors in index: {index.ntotal}")

    # 5. Save the FAISS index and the text chunks
    print(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("Index saved.")

    print(f"Saving text chunks to {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    print("Chunks saved.")
    
    print("\nAll steps completed successfully!")

if __name__ == "__main__":
    create_and_save_embeddings()