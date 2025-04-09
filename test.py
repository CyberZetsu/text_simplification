## Nettskjema and Google Forms

# Required libraries
import spacy
import difflib # Standard library for computing differences between texts
import requests # For calling Thesaurus API
import json # For handling JSON responses from the API
import textstat # For metrics
import streamlit as st

from transformers import pipeline  # Hugging Face Transformers pipeline for summarization
from langchain.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from APIs import API_KEYS # Private keys for thesaurus and huggingface api
# from datasets import AllCombined



# Load the two text simplification models using Hugging Face pipelines.
# Use the "summarization" task for both models.
# BART Large CNN is a high-quality summarization .

HF_API_KEY= API_KEYS['HF'] #Huggingface private key if you wish to run it via huggingface and not locally

# Spacy english model for passive voice detection
nlp = spacy.load("en_core_web_sm")

# HuggingFace models for simplification

# Bart large CNN
bart_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", temperature = 0.3) # added lower temperature for less randomness
# T5-large 
t5_pipeline   = pipeline("summarization", model="t5-large")


#Wrapping models with Langchain

bart_llm=HuggingFacePipeline(pipeline=bart_pipeline) # for using langchain with bart-large-cnn
t5_llm = HuggingFacePipeline(pipeline=t5_pipeline)

#Langchain prompt
"""
prompt_template = PromptTemplate(
    input_variable=["text"],
    template=(
        "Summarize this text for a general audience while following these rules:\n"
        "1. Replace written-out numbers with numerical values (e.g., 'ten thousand' → '10,000').\n"
        "2. Group similar descriptive words into a single simplified term (e.g., 'black, blue, green, white, red, neon' → 'a collection of colors').\n"
        "3. Convert casual language into a formal tone.\n"
        "4. Shorten verbal expressions using more direct wording.\n"
        "5. Adjust verb sentence structures for better readability.\n"
        "6. Simplify noun structures to make the text clearer and more concise.\n"
        "7. Avoid redundancy in phrasing.\n"
        "8. DO NOT WRITE THE RULES IN THE SIMPLIFIED TEXT RESPONSE.\n\n"
        "Text: {text}"
    )
)
"""

"""
prompt_template = PromptTemplate(
    input_variables=["text"],
    template=(
        "Please summarize the following text while applying these rules:\n"
        "- Replace written-out numbers with numerical values.\n"
        "- Group similar descriptive words into a single simplified term.\n"
        "- Convert casual language into a formal tone.\n"
        "- Shorten verbal expressions using more direct wording.\n"
        "- Adjust verb sentence structures for better readability.\n"
        "- Simplify noun structures to make the text clearer.\n"
        "- Avoid redundancy in phrasing.\n\n"
        "Text to summarize:\n{text}\n\n"
        "Simplified Summary:"
    )
)
"""


#Baseline template1 for BART siden den er ikke god med abstrakte regler. så eksempler må være tilstedet og regler må bli føyd inn i et og ikke i numerisk rekker.

""" 
prompt_template1 = PromptTemplate( #Garantert å fungere om man fjerner reglene
    input_variables=["text"],
    template=(
        "Rewrite the following text for clarity and conciseness, ensuring that you replace written-out numbers with numerical values. change from casual to formal.:\n\n"
        "Example:\n"
        "Original: 'There were a lot of different kinds of colors, including red, blue, and green.'\n"
        "Simplified: 'There were many colors, including red, blue, and green.'\n\n"
        "{text}\n\n"
        "Simplified Summary:"
    )
)
"""


prompt_template1 = PromptTemplate( #Garantert å fungere om man fjerner reglene
    input_variables=["text"],
    template=(
        "Rewrite the following text for clarity and conciseness:\n\n"
        "Example:\n"
        "Original: 'There were a lot of different kinds of colors, including red, blue, and green.'\n"
        "Simplified: 'There were many colors, including red, blue, and green.'\n\n"
        "{text}\n\n"
        "Simplified Summary:"
    )
)



prompt_template = PromptTemplate(
    input_variables=["text"],
    template=(
        "Summarize the following text while following these rules:\n\n"
        "1. Replace written-out numbers with numerical values.\n"
        "2. Group similar descriptive words into a single simplified term.\n"
        "3. Convert casual language into a formal tone.\n"
        "4. Shorten verbal expressions using more direct wording.\n"
        "5. Adjust verb sentence structures for better readability.\n"
        "6. Simplify noun structures to make the text clearer.\n"
        "7. Avoid redundancy in phrasing.\n\n"
        "Text:\n{text}\n\n"
        "**Simplified Summary:**"
    )
)




#langchain simplification chains
bart_chain = LLMChain(llm=bart_llm, prompt=prompt_template1)
t5_chain = LLMChain(llm=t5_llm, prompt=prompt_template)


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
    


# Streamlit UI

st.set_page_config(page_title ="Text simplifier", layout ="wide")
st.title ("Opensourced Text simplifier")

text_input = st.text_area("Enter the text that is to be simplified", height = 250)

if st.button("simplify text"):
    if not text_input.strip():
        st.warning("Please enter some text to simplify.")
    else: 
        active_text = convert_passive_to_active(text_input)

        with st.spinner("Simplifying with BART and T5:"):
            simplified_bart = bart_chain.run(text=active_text)
            simplified_t5 = t5_chain.run(text=active_text)
        
        #Outputs

        st.subheader("original text")
        st.write(text_input)

        st.subheader("Simplified: BART")
        st.write(simplified_bart)

        st.subheader("Simplified: T5")
        st.write(simplified_t5)

        st.subheader("Readability metrics")
        col1, col2, col3 = st.columns(3)
        for col, label, text in zip([col1, col2, col3], ["original", "BART", "T5"],
                                    [text_input, simplified_bart, simplified_t5]):
            with col:
                st.markdown(f"**{label}**")
                st.write({
                    "Flesch Reading Ease": textstat.flesch_reading_ease(text),
                    "Grade Level": textstat.flesch_kincaid_grade(text),
                    "Gunning Fog": textstat.gunning_fog(text),
                    "Readability Index": textstat.automated_readability_index(text),
                    "Consensus": textstat.text_standard(text, float_output=False)
                })

        def get_diff(original, simplified):
            orig_tokens = original.split()
            simp_tokens= simplified.split()
            matcher=difflib.SequenceMatcher(None, orig_tokens, simp_tokens)
            diffs = []
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'replace':
                    diffs.append(f"Replaced '{' '.join(orig_tokens[i1:i2])}' ➜ '{' '.join(simp_tokens[j1:j2])}'")
                elif tag == 'delete':
                    diffs.append(f"Removed '{' '.join(orig_tokens[i1:i2])}'")
                elif tag == 'insert':
                    diffs.append(f"Added '{' '.join(simp_tokens[j1:j2])}'")
            return diffs
        
        st.subheader("Differences from original")
        st.markdown("**BART Differences**")
        st.code("\n".join(get_diff(text_input, simplified_bart)) or "No differences found.")

        st.markdown("**T5 Differences**")
        st.code("\n".join(get_diff(text_input, simplified_t5)) or "No differences found.")

        #synonyms

        st.subheader(" Synonym Suggestions")
        new_words = set(simplified_bart.lower().split()) | set(simplified_t5.lower().split())
        new_words = new_words - set(text_input.lower().split())
        synonym_dict = {w: get_synonyms(w) for w in new_words if w.isalpha()}
        for word, synonyms in synonym_dict.items():
            st.write(f"**{word}** ➜ {', '.join(synonyms[:5]) if synonyms else 'No synonyms found'}")

"""
# Main script functionality
def main():
    # 1. Get user input for the text to simplify
    text1 = input("Enter the text to be simplified:\n")
    #Convert passive to active voice
    active_text=convert_passive_to_active(text1)


    # Summarize the input text using BART model.
    simplified_bart = bart_chain.run(text=active_text)
    # Summarize the input text using T5 model.
    simplified_t5 = t5_chain.run(text=f"summarize: {active_text}")

    # Print the readability statistics for the original text.
    print(f"Flesch Reading Ease: {textstat.flesch_reading_ease(text1)}")
    print(f"Flesch-Kincaid Grade Level: {textstat.flesch_kincaid_grade(text1)}")
    print(f"Gunning Fog Score: {textstat.gunning_fog(text1)}")
    print(f"Automated readability index: {textstat.automated_readability_index(text1)}")
    print(f"Coleman Liau index score: {textstat.coleman_liau_index(text1)}")
    print(f"Linsar write Formula score: {textstat.linsear_write_formula(text1)}")
    print(f"Dale Chall readability Grade: {textstat.dale_chall_readability_score(text1)}")
    print(f"Readability consensus level: {textstat.text_standard(text1, float_output=False)}")
    print(f"Spache Readability Formula Level: {textstat.spache_readability(text1)}")
    print(f"Reading time: {textstat.reading_time(text1, ms_per_char=14.69)}")
    print(f"Amount of syllables: {textstat.syllable_count(text1)}")
    print(f"Lexicon count: {textstat.lexicon_count(text1, removepunct=True)}")

    # Print the original and simplified texts
    print("\nOriginal Text:\n" + text1)
    
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
    orig_tokens = text1.split()
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
"""
"""
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
"""

# Entry point for script execution
#if __name__ == "__main__":
#    main()
    