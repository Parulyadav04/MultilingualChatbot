# import json
# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss
# import re
# import pickle
# from typing import List, Dict

# class DocumentProcessor:
#     def __init__(self):
#         self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
#     def clean_text(self, text: str) -> str:
#         """Clean and normalize text"""
#         if not text:
#             return ""
        
#         # Remove extra whitespace and newlines
#         text = re.sub(r'\s+', ' ', text)
#         text = re.sub(r'\n+', '\n', text)
        
#         # Remove special characters but keep basic punctuation
#         text = re.sub(r'[^\w\s\.\,\?\!\-\(\)\[\]]', ' ', text)
        
#         return text.strip()
    
#     def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
#         """Split text into overlapping chunks"""
#         words = text.split()
#         chunks = []
        
#         for i in range(0, len(words), chunk_size - overlap):
#             chunk = ' '.join(words[i:i + chunk_size])
#             if len(chunk.strip()) > 50:  # Only keep meaningful chunks
#                 chunks.append(chunk.strip())
        
#         return chunks
    
#     def process_documents(self, json_file: str) -> tuple:
#         """Process JSON documents and create embeddings"""
#         with open(json_file, 'r', encoding='utf-8') as f:
#             documents = json.load(f)
        
#         processed_docs = []
#         doc_id = 0
        
#         for doc in documents:
#             title = doc.get('title', '')
#             content = doc.get('content', '')
#             url = doc.get('url', '')
            
#             # Clean content
#             clean_content = self.clean_text(content)
            
#             # Combine title and content
#             full_text = f"Title: {title}\n\nContent: {clean_content}"
            
#             # Create chunks
#             chunks = self.chunk_text(full_text, chunk_size=400, overlap=50)
            
#             for chunk in chunks:
#                 processed_docs.append({
#                     'id': doc_id,
#                     'text': chunk,
#                     'title': title,
#                     'url': url,
#                     'original_doc_id': len(processed_docs)
#                 })
#                 doc_id += 1
        
#         # Generate embeddings
#         print("Generating embeddings...")
#         texts = [doc['text'] for doc in processed_docs]
#         embeddings = self.model.encode(texts, show_progress_bar=True)
        
#         # Create FAISS index
#         dimension = embeddings.shape[1]
#         index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
#         # Normalize embeddings for cosine similarity
#         faiss.normalize_L2(embeddings)
#         index.add(embeddings.astype('float32'))
        
#         return processed_docs, index, embeddings
    
#     def save_data(self, processed_docs: List[Dict], index, embeddings, 
#                   docs_file: str = 'documents.pkl', 
#                   index_file: str = 'faiss_index.bin',
#                   embeddings_file: str = 'embeddings.npy'):
#         """Save processed data"""
        
#         # Save documents
#         with open(docs_file, 'wb') as f:
#             pickle.dump(processed_docs, f)
        
#         # Save FAISS index
#         faiss.write_index(index, index_file)
        
#         # Save embeddings
#         np.save(embeddings_file, embeddings)
        
#         print(f"Saved {len(processed_docs)} document chunks")

# # Run processing
# if __name__ == "__main__":
#     processor = DocumentProcessor()
#     docs, index, embeddings = processor.process_documents('cleaned.json')
#     processor.save_data(docs, index, embeddings)
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
import pickle
from typing import List, Dict

class DocumentProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove special characters but keep essential punctuation
        text = re.sub(r'[^\w\s\.\,\?\!\-\(\)\[\]:@/]', ' ', text)
        
        return text.strip()
    
    def smart_chunk_text(self, text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
        """Smarter chunking that preserves sentence boundaries"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short fragments
                continue
                
            # If adding this sentence would exceed chunk size, start new chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                if len(current_chunk.strip()) > 50:  # Only add meaningful chunks
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_text = ' '.join(words[-overlap//5:]) if len(words) > overlap//5 else ""
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence
        
        # Add the last chunk
        if len(current_chunk.strip()) > 50:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_documents(self, json_file: str) -> tuple:
        with open(json_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        processed_docs = []
        doc_id = 0
        
        for doc in documents:
            title = doc.get('title', '')
            content = doc.get('content', '')
            url = doc.get('url', '')
            
            if not content or len(content.strip()) < 100:
                continue
            
            clean_content = self.clean_text(content)
            full_text = f"{title}\n\n{clean_content}"
            
            # Better chunking
            chunks = self.smart_chunk_text(full_text, chunk_size=500, overlap=80)
            
            for chunk in chunks:
                processed_docs.append({
                    'id': doc_id,
                    'text': chunk,
                    'title': title,
                    'url': url
                })
                doc_id += 1
        
        print(f"Processing {len(processed_docs)} document chunks...")
        texts = [doc['text'] for doc in processed_docs]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        return processed_docs, index, embeddings
    
    def save_data(self, processed_docs, index, embeddings):
        with open('documents.pkl', 'wb') as f:
            pickle.dump(processed_docs, f)
        faiss.write_index(index, 'faiss_index.bin')
        np.save('embeddings.npy', embeddings)
        print(f"Saved {len(processed_docs)} document chunks")

if __name__ == "__main__":
    processor = DocumentProcessor()
    docs, index, embeddings = processor.process_documents('cleaned.json')
    processor.save_data(docs, index, embeddings)