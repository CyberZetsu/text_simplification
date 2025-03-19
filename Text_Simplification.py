# Required libraries
from transformers import pipeline  # Hugging Face Transformers pipeline for summarization
import difflib                     # Standard library for computing differences between texts
import requests                    # For calling the Thesaurus API
import json                        # For handling JSON responses from the API (if needed)
from APIs import API_KEYS
import textstat




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

    print(f"Flesch Reading Ease: {textstat.flesch_reading_ease(text)}")
    print(f"Flesch-Kincaid Grade Level: {textstat.flesch_kincaid_grade(text)}") #a score of 9.3 means that a ninth grader would be able to understand that sentence
    print(f"Gunning Fog Score:: {textstat.gunning_fog(text)}") # returns the FOG index, a 9.3 means that a ninth grader will understand the paragraph
#    print(f"Smog Index Score: {textstat.smog_index(text)}") #needs 30 sentences to give a valid score. 9.3 means a ninth grader will understand
    print(f"Automated readability index: {textstat.automated_readability_index(text)}") # gives a number that approximates the grade level needed to understand a 9.5 means between a 9th and 10th grader
    print(f"Coleman Liau index score: {textstat.coleman_liau_index(text)}") # same as the other a 9.3 means a 9th grader will understand
    print(f"Linsar write Formula score: {textstat.linsear_write_formula(text)}") # same as the above a 9.3 means a 9th grader will understand
    print(f"Dale Chall readability Grade: {textstat.dale_chall_readability_score(text)}") # Looks up the most commonly used 3000 english words, and returns a grade using the Dale-Chall formula.
    print(f" Readability consensus level: {textstat.text_standard(text, float_output=False)}") # sums up the above tests and returns the estimated grade
    print(f"Spache Readability Formula Level: {textstat.spache_readability(text)}") # Returns a grade level of the text
    print(f"Reading time: {textstat.reading_time(text, ms_per_char=14.69)}") # Returns the reading time in ms 
    print(f"Amount of syllables: {textstat.syllable_count(text)}") # Syllable count
    print(f"Lexicon count: {textstat.lexicon_count(text, removepunct=True)}") # amount of words in the sentence/text




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
    print(f"Simplified Bart Reading ease: {textstat.flesch_reading_ease(simplified_bart)}")
    print(f"Simplified Bart Flesch-Kincaid Grade Level: {textstat.flesch_kincaid_grade(simplified_bart)}") #a score of 9.3 means that a ninth grader would be able to understand that sentence
    print(f"Simplified Bart Gunning Fog Score:: {textstat.gunning_fog(simplified_bart)}") # returns the FOG index, a 9.3 means that a ninth grader will understand the paragraph
#    print(f"Smog Index Score: {textstat.smog_index(text)}") #needs 30 sentences to give a valid score. 9.3 means a ninth grader will understand
    print(f"Simplified Bart Automated readability index: {textstat.automated_readability_index(simplified_bart)}") # gives a number that approximates the grade level needed to understand a 9.5 means between a 9th and 10th grader
    print(f"Simplified Bart Coleman Liau index score: {textstat.coleman_liau_index(simplified_bart)}") # same as the other a 9.3 means a 9th grader will understand
    print(f"Simplified Bart Linsar write Formula score: {textstat.linsear_write_formula(simplified_bart)}") # same as the above a 9.3 means a 9th grader will understand
    print(f"Simplified Bart Dale Chall readability Grade: {textstat.dale_chall_readability_score(simplified_bart)}") # Looks up the most commonly used 3000 english words, and returns a grade using the Dale-Chall formula.
    print(f"Simplified Bart Readability consensus level: {textstat.text_standard(simplified_bart, float_output=False)}") # sums up the above tests and returns the estimated grade
    print(f"Simplified Bart Spache Readability Formula Level: {textstat.spache_readability(simplified_bart)}") # Returns a grade level of the text
    print(f"Simplified Bart Reading time: {textstat.reading_time(simplified_bart, ms_per_char=14.69)}") # Returns the reading time in ms 
    print(f"Simplified Bart Amount of syllables: {textstat.syllable_count(simplified_bart)}") # Syllable count
    print(f"Simplified Bart Lexicon count: {textstat.lexicon_count(simplified_bart, removepunct=True)}") # amount of words in the sentence/text
    
    print("\nSimplified Text (T5):\n" + simplified_t5)
    print(f"Simplified T5 Reading ease: {textstat.flesch_reading_ease(simplified_t5)}")
    print(f"Simplified T5 Flesch-Kincaid Grade Level: {textstat.flesch_kincaid_grade(simplified_t5)}") #a score of 9.3 means that a ninth grader would be able to understand that sentence
    print(f"Simplified T5 Gunning Fog Score:: {textstat.gunning_fog(simplified_t5)}") # returns the FOG index, a 9.3 means that a ninth grader will understand the paragraph
#    print(f"Smog Index Score: {textstat.smog_index(text)}") #needs 30 sentences to give a valid score. 9.3 means a ninth grader will understand
    print(f"Simplified T5 Automated readability index: {textstat.automated_readability_index(simplified_t5)}") # gives a number that approximates the grade level needed to understand a 9.5 means between a 9th and 10th grader
    print(f"Simplified T5 Coleman Liau index score: {textstat.coleman_liau_index(simplified_t5)}") # same as the other a 9.3 means a 9th grader will understand
    print(f"Simplified T5 Linsar write Formula score: {textstat.linsear_write_formula(simplified_t5)}") # same as the above a 9.3 means a 9th grader will understand
    print(f"Simplified T5 Dale Chall readability Grade: {textstat.dale_chall_readability_score(simplified_t5)}") # Looks up the most commonly used 3000 english words, and returns a grade using the Dale-Chall formula.
    print(f"Simplified T5 Readability consensus level: {textstat.text_standard(simplified_t5, float_output=False)}") # sums up the above tests and returns the estimated grade
    print(f"Simplified T5 Spache Readability Formula Level: {textstat.spache_readability(simplified_t5)}") # Returns a grade level of the text
    print(f"Simplified T5 Reading time: {textstat.reading_time(simplified_t5, ms_per_char=14.69)}") # Returns the reading time in ms 
    print(f"Simplified T5 Amount of syllables: {textstat.syllable_count(simplified_t5)}") # Syllable count
    print(f"Simplified T5 Lexicon count: {textstat.lexicon_count(simplified_t5, removepunct=True)}") # amount of words in the sentence/text
    
    
    
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
