## Nettskjema and Google Forms

# Required libraries
import spacy
from transformers import pipeline  # Hugging Face Transformers pipeline for summarization
import difflib                     # Standard library for computing differences between texts
import requests                    # For calling the Thesaurus API
import json                        # For handling JSON responses from the API (if needed)
from APIs import API_KEYS
import textstat
# from datasets import AllCombined



# Load the two text simplification models using Hugging Face pipelines.
# Use the "summarization" task for both models.
# BART Large CNN is a high-quality summarization .


# Spacy english model for passive voice detection
nlp = spacy.load("en_core_web_sm")


bart_summarizer = pipeline("sumarization", model="facebook/bart-large-cnn")
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
    


def convert_passive_to_active(text):
        # 1. Get user input for the text to simplify
    doc = nlp(text)
    new_sentences = []

    for sent in doc.sents:
        words = [token.text for token in sent]
        if any(token.dep_ == "agent" for token in sent): # Passive detection
            # Try to  rephrase 
            subj = None
            verb = None
            obj = None
            for token in sent:
                if token.dep == "nsjubjpass":
                    obj = token.text
                elif token.dep_ == "aux" or token.dep_ == "auxpass":
                    verb = token.head.text
                elif token.dep =="agent":
                    subj = token.text

            if subj and verb and obj:
                active_sentence = f"{subj} {verb} {obj}."
                new_sentences.append(active_sentence)
            

        new_sentences.append(" ".join(words))

    return " ".join(new_sentences)






# Main script functionality
def main():
    # 1. Get user input for the text to simplify
    text = input("Enter the text to be simplified:\n")
    #Convert passive to active voice
    active_text=convert_passive_to_active(text)


    # Define the rules but make sure they aren't included in the summary itself.
    prompt = (
        "Summarize this text for a general audience while following these rules:\n"
        "1. Replace written-out numbers with numerical values (e.g., 'ten thousand' → '10,000').\n"
        "2. Group similar descriptive words into a single simplified term (e.g., 'black, blue, green, white, red, neon' → 'a collection of colors').\n"
        "3. Convert casual language into a formal tone.\n"
        "4. Shorten verbal expressions using more direct wording.\n"
        "5. Adjust verb sentence structures for better readability.\n"
        "6. Simplify noun structures to make the text clearer and more concise.\n"
        "7. Avoid redundancy in phrasing.\n"
        "8. DO NOT WRITE THE RULES IN THE SIMPLIFIED TEXT RESPONSE.\n\n"
        f"Text: {text}"
    )

    

    # Summarize the input text using BART model.
    bart_result = bart_summarizer(prompt, max_length=120, min_length=30, do_sample=False)
    simplified_bart = bart_result[0]['summary_text']

    # Summarize the input text using T5 model.
    t5_result = t5_summarizer(prompt, max_length=120, min_length=30, do_sample=False)
    simplified_t5 = t5_result[0]['summary_text']

    # Print the readability statistics for the original text.
    print(f"Flesch Reading Ease: {textstat.flesch_reading_ease(text)}")
    print(f"Flesch-Kincaid Grade Level: {textstat.flesch_kincaid_grade(text)}")
    print(f"Gunning Fog Score: {textstat.gunning_fog(text)}")
    print(f"Automated readability index: {textstat.automated_readability_index(text)}")
    print(f"Coleman Liau index score: {textstat.coleman_liau_index(text)}")
    print(f"Linsar write Formula score: {textstat.linsear_write_formula(text)}")
    print(f"Dale Chall readability Grade: {textstat.dale_chall_readability_score(text)}")
    print(f"Readability consensus level: {textstat.text_standard(text, float_output=False)}")
    print(f"Spache Readability Formula Level: {textstat.spache_readability(text)}")
    print(f"Reading time: {textstat.reading_time(text, ms_per_char=14.69)}")
    print(f"Amount of syllables: {textstat.syllable_count(text)}")
    print(f"Lexicon count: {textstat.lexicon_count(text, removepunct=True)}")

    # Print the original and simplified texts
    print("\nOriginal Text:\n" + text)
    
    print("\nSimplified Text (BART):\n" + simplified_bart)
    print(f"Simplified Bart Reading ease: {textstat.flesch_reading_ease(simplified_bart)}")
    print(f"Simplified Bart Flesch-Kincaid Grade Level: {textstat.flesch_kincaid_grade(simplified_bart)}")
    print(f"Simplified Bart Gunning Fog Score: {textstat.gunning_fog(simplified_bart)}")
    print(f"Simplified Bart Automated readability index: {textstat.automated_readability_index(simplified_bart)}")
    print(f"Simplified Bart Coleman Liau index score: {textstat.coleman_liau_index(simplified_bart)}")
    print(f"Simplified Bart Linsar write Formula score: {textstat.linsear_write_formula(simplified_bart)}")
    print(f"Simplified Bart Dale Chall readability Grade: {textstat.dale_chall_readability_score(simplified_bart)}")
    print(f"Simplified Bart Readability consensus level: {textstat.text_standard(simplified_bart, float_output=False)}")
    print(f"Simplified Bart Spache Readability Formula Level: {textstat.spache_readability(simplified_bart)}")
    print(f"Simplified Bart Reading time: {textstat.reading_time(simplified_bart, ms_per_char=14.69)}")
    print(f"Simplified Bart Amount of syllables: {textstat.syllable_count(simplified_bart)}")
    print(f"Simplified Bart Lexicon count: {textstat.lexicon_count(simplified_bart, removepunct=True)}")
    
    print("\nSimplified Text (T5):\n" + simplified_t5)
    print(f"Simplified T5 Reading ease: {textstat.flesch_reading_ease(simplified_t5)}")
    print(f"Simplified T5 Flesch-Kincaid Grade Level: {textstat.flesch_kincaid_grade(simplified_t5)}")
    print(f"Simplified T5 Gunning Fog Score: {textstat.gunning_fog(simplified_t5)}")
    print(f"Simplified T5 Automated readability index: {textstat.automated_readability_index(simplified_t5)}")
    print(f"Simplified T5 Coleman Liau index score: {textstat.coleman_liau_index(simplified_t5)}")
    print(f"Simplified T5 Linsar write Formula score: {textstat.linsear_write_formula(simplified_t5)}")
    print(f"Simplified T5 Dale Chall readability Grade: {textstat.dale_chall_readability_score(simplified_t5)}")
    print(f"Simplified T5 Readability consensus level: {textstat.text_standard(simplified_t5, float_output=False)}")
    print(f"Simplified T5 Spache Readability Formula Level: {textstat.spache_readability(simplified_t5)}")
    print(f"Simplified T5 Reading time: {textstat.reading_time(simplified_t5, ms_per_char=14.69)}")
    print(f"Simplified T5 Amount of syllables: {textstat.syllable_count(simplified_t5)}")
    print(f"Simplified T5 Lexicon count: {textstat.lexicon_count(simplified_t5, removepunct=True)}")

    # Highlight differences
    print("\nDifferences between original and BART-simplified text:")
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

    # Suggest synonyms
    bart_new_words = set([w.strip(".,!?").lower() for w in bart_tokens if w.lower() not in [ow.lower() for ow in orig_tokens]])
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
    