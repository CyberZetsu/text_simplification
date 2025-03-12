# Required libraries
from transformers import pipeline  # Hugging Face Transformers pipeline for summarization
import difflib                     # Standard library for computing differences between texts
import requests                    # For calling the Thesaurus API
import json                        # For handling JSON responses from the API (if needed)
from APIs import API_KEYS

# Load the two text simplification models using Hugging Face pipelines.
# Use the "summarization" task for both models.
# BART Large CNN is a high-quality summarization .
bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# T5-small can perform summarization.
t5_summarizer   = pipeline("summarization", model="t5-small")

# Function to get synonyms from a Thesaurus API.
def get_synonyms(word):
    """Lookup synonyms for a given word using a Thesaurus API and return a list of synonyms."""
    # Here we use the API Ninjas Thesaurus API.
    api_url = f"https://api.api-ninjas.com/v1/thesaurus?word={word}"
    headers = {"X-Api-Key": "API_KEYS['ninja]"}  # Replace with your actual API key.
    try:
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # The API returns a JSON with possibly 'synonyms' and 'antonyms'.
            synonyms = data.get("synonyms", [])
            return synonyms
        else:
            # If the API call fails or no synonyms found, return an empty list.
            return []
    except Exception as e:
        # Handle any exceptions (e.g., network issues)
        return []

# Main script functionality
def main():
    # 1. Get user input for the text to simplify
    text = input("Enter the text to be simplified:\n")

    # 2. Generate simplified text using BART summarization model
    # We use max_length and min_length to control summary length; adjust as needed.
    bart_result = bart_summarizer(text, max_length=120, min_length=30, do_sample=False)
    simplified_bart = bart_result[0]['summary_text']
    
    # 3. Generate simplified text using T5 model.
    # Important: prefix the text with "summarize: " for T5 to signal the task.
    t5_input = "summarize: " + text
    t5_result = t5_summarizer(t5_input, max_length=120, min_length=30, do_sample=False)
    simplified_t5 = t5_result[0]['summary_text']
    
    # 4. Print the original and simplified texts
    print("\nOriginal Text:\n" + text)
    print("\nSimplified Text (BART):\n" + simplified_bart)
    print("\nSimplified Text (T5):\n" + simplified_t5)
    
    # 5. Highlight differences between original and simplified texts
    print("\nDifferences between original and BART-simplified text:")
    # Use difflib to compare word by word
    orig_tokens = text.split()
    bart_tokens = simplified_bart.split()
    matcher = difflib.SequenceMatcher(None, orig_tokens, bart_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            orig_segment = " ".join(orig_tokens[i1:i2])
            new_segment = " ".join(bart_tokens[j1:j2])
            print(f"  Replaced \"{orig_segment}\" ➜ \"{new_segment}\"")
        elif tag == 'delete':
            orig_segment = " ".join(orig_tokens[i1:i2])
            print(f"  Removed \"{orig_segment}\"")
        elif tag == 'insert':
            new_segment = " ".join(bart_tokens[j1:j2])
            print(f"  Added \"{new_segment}\"")
    # Repeat for T5
    print("\nDifferences between original and T5-simplified text:")
    t5_tokens = simplified_t5.split()
    matcher = difflib.SequenceMatcher(None, orig_tokens, t5_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            orig_segment = " ".join(orig_tokens[i1:i2])
            new_segment = " ".join(t5_tokens[j1:j2])
            print(f"  Replaced \"{orig_segment}\" ➜ \"{new_segment}\"")
        elif tag == 'delete':
            orig_segment = " ".join(orig_tokens[i1:i2])
            print(f"  Removed \"{orig_segment}\"")
        elif tag == 'insert':
            new_segment = " ".join(t5_tokens[j1:j2])
            print(f"  Added \"{new_segment}\"")
    
    # 6. Suggest synonyms for words in the simplified texts that are different from the original
    # Identify "new" words in the BART simplified text
    bart_new_words = set([w.strip(".,!?").lower() for w in bart_tokens if w.lower() not in [ow.lower() for ow in orig_tokens]])
    # Identify new words in the T5 simplified text
    t5_new_words = set([w.strip(".,!?").lower() for w in t5_tokens if w.lower() not in [ow.lower() for ow in orig_tokens]])
    
    new_words = bart_new_words.union(t5_new_words)
    if new_words:
        print("\nSynonym suggestions for simplified words:")
        for word in sorted(new_words):
            if not word:  # skip empty strings if any
                continue
            synonyms = get_synonyms(word)
            if synonyms:
                print(f"  {word} ➜ {', '.join(synonyms[:5])}")  # show up to 5 synonyms
            else:
                print(f"  {word} ➜ (no synonyms found)")
    else:
        print("\nNo new words to suggest synonyms for.")
        

# Entry point for script execution
if __name__ == "__main__":
    main()
