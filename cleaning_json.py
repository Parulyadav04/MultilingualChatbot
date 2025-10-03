import json, hashlib, os

# ---------- CONFIG ----------
INPUT_FILE = "meity_data1.json"       # your raw scraped JSON
OUTPUT_FILE = "cleaned.json"  # cleaned output
MIN_CHARS = 10               # skip docs shorter than this many characters
MIN_WORDS = 2                # skip docs with very few words
PREVIEW_COUNT = 2             # preview first N docs

# ---------- NOISE FILTER ----------
def is_noise(line: str) -> bool:
    junk = [
        "chevron_right", "arrow_forward_ios", "arrow_back_ios",
        "arrow_drop_down", "pause", "View more", "Rate this translation"
    ]
    return any(j in line for j in junk)

def clean_text(text: str) -> str:
    """Cleans one document's text."""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not is_noise(ln)]

    seen, out = set(), []
    for ln in lines:
        if ln in seen:
            continue
        seen.add(ln)
        out.append(ln)

    return " ".join(out)

# ---------- VALIDATION FILTER ----------
def is_valid_content(text: str) -> bool:
    if "Application error" in text:
        return False
    if len(text) < MIN_CHARS:
        return False
    if len(text.split()) < MIN_WORDS:
        return False
    return True

# ---------- PREVIEW ----------
def preview_before_after(docs, cleaned_docs, n=PREVIEW_COUNT):
    print("\n=== PREVIEW BEFORE & AFTER CLEANING ===\n")
    for i, (raw, clean) in enumerate(zip(docs, cleaned_docs)):
        if i >= n:
            break
        print(f"--- Document {i+1} ---")
        print(f"Title   : {raw.get('title','')}")
        print(f"URL     : {raw.get('url','')}")
        print(f"Lang    : {raw.get('language','unknown')}")
        print("\n[BEFORE CLEANING] (truncated to 500 chars)")
        print(raw.get("content","")[:500], "...\n")
        print("[AFTER CLEANING] (truncated to 500 chars)")
        print(clean.get("content","")[:500], "...\n")
        print("="*60)

# ---------- MAIN PIPELINE ----------
def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)

    cleaned_docs = []
    seen_hash = set()

    for d in docs:
        title = d.get("title", "")
        url = d.get("url", "")
        content = d.get("content", "")
        lang = d.get("language", "unknown")

        text = clean_text(content)

        # skip invalid/noisy docs
        if not is_valid_content(text):
            continue

        # deduplicate across documents
        h = hashlib.md5((url + text).encode("utf-8")).hexdigest()
        if h in seen_hash:
            continue
        seen_hash.add(h)

        cleaned_docs.append({
            "title": title,
            "url": url,
            "content": text,
            "language": lang
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_docs, f, ensure_ascii=False, indent=2)

    print(f"Cleaned {len(cleaned_docs)} documents saved to {OUTPUT_FILE}")

    preview_before_after(docs, cleaned_docs)

# ---------- RUN ----------
if __name__ == "__main__":
    main()
