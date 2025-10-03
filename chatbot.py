# import pickle
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# import torch
# from typing import List, Dict, Tuple
# import re

# class MeityChatbot:
#     def __init__(self):
#         # Load embedding model
#         self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
#         # Load LLM (using a free model)
#         model_name = "microsoft/DialoGPT-medium"
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
#         # Alternative: Use a text generation pipeline with a smaller model
#         # self.llm = pipeline("text-generation", 
#         #                     model="distilgpt2", 
#         #                     max_length=512, 
#         #                     temperature=0.7,
#         #                     do_sample=True)
        
#         # Load processed data
#         self.load_data()
        
#     def load_data(self):
#         """Load processed documents and FAISS index"""
#         try:
#             with open('documents.pkl', 'rb') as f:
#                 self.documents = pickle.load(f)
            
#             self.index = faiss.read_index('faiss_index.bin')
#             self.embeddings = np.load('embeddings.npy')
            
#             print(f"Loaded {len(self.documents)} document chunks")
#         except FileNotFoundError:
#             print("Data files not found. Please run process_data.py first.")
#             self.documents = []
#             self.index = None
    
#     def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
#         """Search for relevant documents"""
#         if not self.documents or self.index is None:
#             return []
        
#         # Generate query embedding
#         query_embedding = self.embedding_model.encode([query])
#         faiss.normalize_L2(query_embedding)
        
#         # Search
#         scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
#         results = []
#         for score, idx in zip(scores[0], indices[0]):
#             if idx < len(self.documents):
#                 doc = self.documents[idx].copy()
#                 doc['score'] = float(score)
#                 results.append(doc)
        
#         return results
    
#     def generate_response_simple(self, query: str, context_docs: List[Dict]) -> str:
#         """Generate response using simple template (fallback method)"""
#         if not context_docs:
#             return "I couldn't find relevant information about your query. Please try rephrasing your question about MeitY policies, schemes, or notifications."
        
#         # Create context from top documents
#         context = ""
#         sources = []
        
#         for i, doc in enumerate(context_docs[:3]):  # Use top 3 docs
#             context += f"Document {i+1}: {doc['text'][:300]}...\n\n"
#             if doc['url'] and doc['url'] not in sources:
#                 sources.append(doc['url'])
        
#         # Simple template-based response
#         response = f"Based on the MeitY documents, here's what I found:\n\n"
        
#         # Extract key information
#         if "scheme" in query.lower() or "yojana" in query.lower():
#             response += "This appears to be related to a government scheme. "
#         elif "policy" in query.lower() or "notification" in query.lower():
#             response += "This relates to a policy or notification. "
        
#         response += f"Key information from the documents:\n\n{context_docs[0]['text'][:400]}..."
        
#         if sources:
#             response += f"\n\nSources:\n" + "\n".join(sources[:2])
        
#         return response
    
#     def generate_response_llm(self, query: str, context_docs: List[Dict]) -> str:
#         """Generate response using LLM"""
#         if not context_docs:
#             return self.generate_response_simple(query, context_docs)
        
#         # Prepare context
#         context = "\n".join([doc['text'][:200] for doc in context_docs[:2]])
        
#         # Create prompt
#         prompt = f"""Context from MeitY documents:
# {context}

# Question: {query}

# Based on the provided context about Ministry of Electronics and Information Technology, please provide a helpful answer:"""
        
#         try:
#             # Using DialoGPT (conversation model)
#             inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
            
#             with torch.no_grad():
#                 outputs = self.model.generate(
#                     inputs, 
#                     max_length=inputs.shape[1] + 150,
#                     num_return_sequences=1,
#                     temperature=0.7,
#                     do_sample=True,
#                     pad_token_id=self.tokenizer.eos_token_id
#                 )
            
#             response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
#             # Add sources
#             sources = [doc['url'] for doc in context_docs[:2] if doc['url']]
#             if sources:
#                 response += f"\n\nSources: {', '.join(sources)}"
            
#             return response.strip()
            
#         except Exception as e:
#             print(f"LLM generation failed: {e}")
#             return self.generate_response_simple(query, context_docs)
    
#     def chat(self, query: str) -> Dict:
#         """Main chat function"""
#         # Search for relevant documents
#         relevant_docs = self.search_documents(query, top_k=5)
        
#         # Generate response
#         response = self.generate_response_llm(query, relevant_docs)
        
#         return {
#             'response': response,
#             'sources': [{'title': doc['title'], 'url': doc['url'], 'score': doc['score']} 
#                        for doc in relevant_docs[:3]],
#             'query': query
#         }

# # Initialize chatbot
# chatbot = MeityChatbot()
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import re

class MeityChatbot:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.load_data()
        
    def load_data(self):
        try:
            with open('documents.pkl', 'rb') as f:
                self.documents = pickle.load(f)
            self.index = faiss.read_index('faiss_index.bin')
            print(f"Loaded {len(self.documents)} document chunks")
        except FileNotFoundError:
            print("Data files not found. Please run process_data.py first.")
            self.documents = []
            self.index = None
    
    def search_documents(self, query: str, top_k: int = 10) -> List[Dict]:
        if not self.documents or self.index is None:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents) and score > 0.3:  # Filter low relevance
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                results.append(doc)
        
        return results
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        if not context_docs:
            return "I couldn't find relevant information in the MeitY documents. Please try asking about specific MeitY schemes, policies, or contact information."
        
        # Combine all relevant context
        full_context = " ".join([doc['text'] for doc in context_docs])
        query_lower = query.lower()
        
        # Determine query type and generate appropriate response
        if self.is_definition_query(query_lower):
            return self.answer_definition(query, full_context)
        elif self.is_contact_query(query_lower):
            return self.answer_contact(full_context)
        elif self.is_scheme_query(query_lower):
            return self.answer_scheme(full_context)
        elif self.is_vacancy_query(query_lower):
            return self.answer_vacancy(full_context)
        else:
            return self.answer_general(query, full_context)
    
    def is_definition_query(self, query: str) -> bool:
        return any(phrase in query for phrase in ['what is', 'what are', 'define', 'explain'])
    
    def is_contact_query(self, query: str) -> bool:
        return any(word in query for word in ['contact', 'phone', 'email', 'address', 'number'])
    
    def is_scheme_query(self, query: str) -> bool:
        return any(word in query for word in ['scheme', 'policy', 'initiative', 'program', 'yojana'])
    
    def is_vacancy_query(self, query: str) -> bool:
        return any(word in query for word in ['vacancy', 'job', 'recruitment', 'hiring', 'career'])
    
    def answer_definition(self, query: str, context: str) -> str:
        # Extract the term being defined
        query_lower = query.lower()
        if 'what is' in query_lower:
            term = query_lower.split('what is')[1].strip().rstrip('?').strip()
        elif 'what are' in query_lower:
            term = query_lower.split('what are')[1].strip().rstrip('?').strip()
        else:
            term = query.strip()
        
        # Handle specific known entities
        if 'c-dac' in term or 'cdac' in term:
            return self.define_cdac(context)
        elif 'meity' in term:
            return self.define_meity(context)
        elif 'digital india' in term:
            return self.define_digital_india(context)
        else:
            return self.generic_definition(term, context)
    
    def define_cdac(self, context: str) -> str:
        # Look for C-DAC definition in context
        sentences = self.split_into_sentences(context)
        definition_sentences = []
        
        for sentence in sentences:
            if ('c-dac' in sentence.lower() or 'cdac' in sentence.lower()) and \
               any(word in sentence.lower() for word in ['centre', 'development', 'advanced', 'computing', 'scientific']):
                if len(sentence.strip()) > 30:
                    definition_sentences.append(sentence.strip())
        
        if definition_sentences:
            return f"C-DAC (Centre for Development of Advanced Computing) is {definition_sentences[0].lower()}. " + \
                   f"It is a premier R&D organization under the Ministry of Electronics and Information Technology (MeitY), " + \
                   f"Government of India, working on advanced computing technologies and solutions."
        
        return "C-DAC (Centre for Development of Advanced Computing) is a Scientific Society under the Ministry of Electronics and Information Technology (MeitY), Government of India. It is involved in research and development of advanced computing technologies, software development, and technology transfer activities."
    
    def define_meity(self, context: str) -> str:
        return "MeitY stands for Ministry of Electronics and Information Technology. It is a ministry of the Government of India responsible for formulating and implementing policies related to electronics, information technology, internet governance, and cyber security. The ministry works on promoting digitalization, supporting IT industry growth, and implementing various digital initiatives across India."
    
    def define_digital_india(self, context: str) -> str:
        sentences = self.split_into_sentences(context)
        for sentence in sentences:
            if 'digital india' in sentence.lower() and any(word in sentence.lower() for word in ['initiative', 'program', 'campaign']):
                if len(sentence.strip()) > 50:
                    return f"Digital India is {sentence.strip()}"
        
        return "Digital India is a flagship initiative by the Government of India launched to transform the country into a digitally empowered society and knowledge economy. It aims to provide digital infrastructure, deliver government services digitally, and promote digital literacy among citizens."
    
    def answer_contact(self, context: str) -> str:
        # Extract contact information
        phone_matches = re.findall(r'(?:Phone|Tel|Contact)[\s:]*(\+?[\d\s\-\(\)]{10,})', context, re.IGNORECASE)
        email_matches = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', context)
        
        response = "MeitY Contact Information:\n\n"
        
        if phone_matches:
            clean_phone = phone_matches[0].strip()
            response += f"📞 Phone: {clean_phone}\n"
        
        if email_matches:
            response += f"📧 Email: {email_matches[0]}\n"
        
        # Add standard information
        response += "📍 Address: Electronics Niketan, 6, CGO Complex, Lodhi Road, New Delhi - 110003\n"
        response += "🌐 Website: www.meity.gov.in"
        
        return response
    
    def answer_scheme(self, context: str) -> str:
        sentences = self.split_into_sentences(context)
        scheme_info = []
        
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['scheme', 'policy', 'initiative']) and \
               len(sentence.strip()) > 40:
                scheme_info.append(sentence.strip())
        
        if scheme_info:
            return f"Here are some key MeitY schemes and policies:\n\n{'. '.join(scheme_info[:3])}."
        
        return "MeitY implements various schemes including Digital India, Production Linked Incentive (PLI) schemes for electronics manufacturing, and initiatives for promoting startups and innovation in the technology sector."
    
    def answer_vacancy(self, context: str) -> str:
        sentences = self.split_into_sentences(context)
        vacancy_info = []
        
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['vacancy', 'recruitment', 'job', 'position', 'employment']):
                if len(sentence.strip()) > 30:
                    vacancy_info.append(sentence.strip())
        
        if vacancy_info:
            return f"Regarding employment opportunities: {'. '.join(vacancy_info[:2])}. For the latest job openings, please check the official MeitY website at www.meity.gov.in"
        
        return "For current job vacancies in MeitY and its organizations, please visit the official website at www.meity.gov.in or check the employment section. Recruitment notifications are regularly published for various technical and administrative positions."
    
    def generic_definition(self, term: str, context: str) -> str:
        sentences = self.split_into_sentences(context)
        relevant_sentences = []
        
        for sentence in sentences:
            if term.lower() in sentence.lower() and len(sentence.strip()) > 30:
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return relevant_sentences[0] + (f" {relevant_sentences[1]}" if len(relevant_sentences) > 1 else "")
        
        return f"I found some information about {term} in the documents, but couldn't extract a clear definition. Please try asking more specific questions about this topic."
    
    def answer_general(self, query: str, context: str) -> str:
        # Find most relevant sentences based on query keywords
        query_words = [word.lower() for word in query.split() if len(word) > 3]
        sentences = self.split_into_sentences(context)
        
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 30:
                continue
            
            score = sum(1 for word in query_words if word in sentence.lower())
            if score > 0:
                scored_sentences.append((score, sentence.strip()))
        
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        
        if scored_sentences:
            top_sentences = [s[1] for s in scored_sentences[:2]]
            return ". ".join(top_sentences) + "."
        
        return "I found some related information in the documents, but couldn't provide a specific answer to your question. Please try rephrasing your query or ask about specific MeitY policies, schemes, or services."
    
    def split_into_sentences(self, text: str) -> List[str]:
        # Better sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    def chat(self, query: str) -> Dict:
        relevant_docs = self.search_documents(query, top_k=8)
        response = self.generate_answer(query, relevant_docs)
        
        return {
            'response': response,
            'sources': [{'title': doc['title'][:100], 'score': doc['score']} 
                       for doc in relevant_docs[:3]],
            'query': query
        }