import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

def chunk_data(input_filepath: str, 
               output_filepath: str, 
               chunk_size: int = 1000, 
               chunk_overlap: int = 200):
    """
    Loads cleaned data, splits the content into chunks, and saves the
    chunked data with its original metadata.

    Args:
        input_filepath (str): Path to the cleaned JSON data file.
        output_filepath (str): Path to save the chunked JSON data.
        chunk_size (int): The maximum number of characters in a chunk.
        chunk_overlap (int): The number of characters to overlap between chunks
                             to maintain context.
    """
    # --- Step 1: Load the cleaned data ---
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} documents from '{input_filepath}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
        print("Please make sure you have run the cleaning script first.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_filepath}'.")
        return

    # --- Step 2: Initialize the Text Splitter ---
    # The RecursiveCharacterTextSplitter is recommended for generic text.
    # It tries to split on a hierarchy of separators ["\n\n", "\n", " ", ""].
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len # This is the default, but it's good to be explicit
    )

    all_chunks = []
    print("\nStarting the chunking process...")
    
    # --- Step 3: Iterate through documents and create chunks ---
    for i, doc in enumerate(data):
        # The create_documents method is powerful because it keeps the metadata
        # associated with the original document for each new chunk.
        # We pass the content as a list of one document.
        # We pass the metadata (title and url) for that document.
        
        content = doc.get("content", "")
        metadata = {
            "title": doc.get("title", "No Title"),
            "url": doc.get("url", "No URL")
        }
        
        # Create the chunks from the content
        chunks = text_splitter.create_documents([content], metadatas=[metadata])
        
        # The output `chunks` is a list of LangChain Document objects.
        # We'll convert them to a more standard dictionary format for saving.
        for chunk in chunks:
            all_chunks.append({
                "content": chunk.page_content,
                "metadata": chunk.metadata
            })

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(data)} documents...")

    # --- Step 4: Save the chunked data ---
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)
        
    print(f"\nChunking complete!")
    print(f"Total documents processed: {len(data)}")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Chunked data has been saved to '{output_filepath}'.")

# --- Main execution block ---
if __name__ == "__main__":
    # Use the output from the previous cleaning step as input here
    INPUT_FILE = "meity_cleaned_data.json"
    OUTPUT_FILE = "meity_chunked_data.json"
    
    # You can experiment with these values
    # A smaller chunk_size might be better for very specific Q&A,
    # while a larger size retains more context.
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    chunk_data(INPUT_FILE, OUTPUT_FILE, CHUNK_SIZE, CHUNK_OVERLAP)