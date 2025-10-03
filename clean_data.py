import json
import re
import unicodedata

def clean_text(text: str) -> str:
    """
    Applies a series of cleaning steps to a single string of text.

    Args:
        text: The raw string to be cleaned.

    Returns:
        A cleaned version of the string.
    """
    if not isinstance(text, str):
        return ""

    # --- Step 1: Normalize Unicode Characters ---
    # This ensures that characters that look the same but have different underlying
    # representations are treated as one (e.g., different types of dashes).
    # 'NFKC' stands for Normalization Form Compatibility Composition.
    text = unicodedata.normalize('NFKC', text)

    # --- Step 2: Remove any remaining HTML tags ---
    # Uses a regular expression to find and remove anything that looks like an HTML tag.
    text = re.sub(r'<[^>]+>', '', text)

    # --- Step 3: Remove URLs ---
    # Finds and removes web addresses. The content should be about the text itself,
    # not the links within it.
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # --- Step 4: Remove extra whitespace, newlines, and tabs ---
    # This is a multi-part process to make the text clean and readable.
    text = text.replace('\\n', ' ').replace('\n', ' ').replace('\t', ' ') # Replace escaped and regular newlines/tabs
    text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with a single space
    text = text.strip() # Remove leading/trailing whitespace

    # --- Step 5: Remove specific patterns (optional but recommended) ---
    # For instance, the "##" markdown-style headers from your data.
    text = re.sub(r'##\s*', '', text)

    # --- Step 6: Remove non-ASCII characters (optional) ---
    # If you want to keep only English text, you can uncomment the line below.
    # For your project with 22 Indian languages, YOU SHOULD NOT DO THIS.
    # text = re.sub(r'[^\x00-\x7F]+', '', text)

    return text

def process_json_file(input_filepath: str, output_filepath: str):
    """
    Loads a JSON file, cleans the 'content' field for each entry,
    and saves the result to a new JSON file.
    """
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{input_filepath}' is not a valid JSON file.")
        return

    cleaned_data = []
    print(f"Starting to process {len(data)} entries...")

    for i, item in enumerate(data):
        # Ensure the 'content' key exists before trying to clean it.
        if 'content' in item:
            original_content = item['content']
            cleaned_content = clean_text(original_content)

            # Keep other data like title and URL associated with the content
            cleaned_item = {
                "title": item.get("title", ""), # Use .get for safety
                "url": item.get("url", ""),
                "content": cleaned_content
            }
            cleaned_data.append(cleaned_item)
        
        # Print progress
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(data)} entries.")

    # Save the cleaned data to a new file
    with open(output_filepath, 'w', encoding='utf-8') as f:
        # indent=4 makes the JSON file human-readable
        # ensure_ascii=False is important for saving multilingual characters correctly
        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

    print(f"\nProcessing complete!")
    print(f"Cleaned data has been saved to '{output_filepath}'.")


# --- Main execution ---
if __name__ == "__main__":
    INPUT_FILE = "meity_complete.json"
    OUTPUT_FILE = "meity_cleaned_data.json"
    process_json_file(INPUT_FILE, OUTPUT_FILE)