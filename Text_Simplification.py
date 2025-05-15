import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, BitsAndBytesConfig
import torch  # For PyTorch model loading and tensor operations
 # Hugging Face Transformers pipeline for summarization
import difflib  # Standard library for computing differences between texts
import requests  # For calling the Thesaurus API
import json  # For handling JSON responses from the API (
import textstat
from APIs import API_KEYS


# Load the text simplification models using Hugging Face pipelines.
# Use the "summarization" task for the BART and T5 models.
# BART Large CNN is a high-quality summarization .
bart_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", token= API_KEYS['HF'])

# T5large summarization due to it not having a "simplification".
t5_large_pipeline = pipeline("summarization", model="t5-large", token = API_KEYS['HF'])

# FLAN-T5-large for more powerful summarization, not having "simplification".
flan_t5_large_pipeline = pipeline("summarization", model="google/flan-t5-large", token = API_KEYS['HF'])

# Load Qwen-7B-Chat with int4 quantization using bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # or "fp4"
    bnb_4bit_compute_dtype=torch.float16
)

qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", token=API_KEYS['HF'], trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat",
    token=API_KEYS['HF'],
    quantization_config=bnb_config,
    trust_remote_code=True,  # Trust the remote code for model loading
)

qwen_pipeline = TextGenerationPipeline(model=qwen_model, tokenizer=qwen_tokenizer)

# Function to get synonyms from a Thesaurus API.
def get_synonyms(word):
    """Lookup synonyms for a given word using a Thesaurus API and return a list of synonyms."""
    # Here we use the API Ninjas Thesaurus API.
    api_url = f"https://api.api-ninjas.com/v1/thesaurus?word={word}"
    headers = {"X-Api-Key": API_KEYS['ninja']}  # Replace with your API key. look at the GitHub repo for the readme.md for how to create an API key.
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


# Streamlit App
st.title("Text Simplification and Readability Analysis")

# Get user input for the text to simplify
text = st.text_area("Enter the text to be simplified:")

if text:
    # Display readability scores for the original text
    st.write("### Original Text Readability Scores")
    st.write(f"Flesch Reading Ease: {textstat.flesch_reading_ease(text)}")
    st.write(f"Flesch-Kincaid Grade Level: {textstat.flesch_kincaid_grade(text)}")
    st.write(f"Gunning Fog Score: {textstat.gunning_fog(text)}")
    st.write(f"Automated Readability Index: {textstat.automated_readability_index(text)}")
    st.write(f"Coleman Liau Index: {textstat.coleman_liau_index(text)}")
    st.write(f"Linsar Write Formula: {textstat.linsear_write_formula(text)}")
    st.write(f"Dale Chall Readability Grade: {textstat.dale_chall_readability_score(text)}")
    st.write(f"Readability Consensus Level: {textstat.text_standard(text, float_output=False)}")
    st.write(f"Spache Readability Formula Level: {textstat.spache_readability(text)}")
    st.write(f"Reading Time: {textstat.reading_time(text, ms_per_char=14.69)}")
    st.write(f"Syllable Count: {textstat.syllable_count(text)}")
    st.write(f"Lexicon Count: {textstat.lexicon_count(text, removepunct=True)}")

    # 2. Generate simplified text using BART summarization model
    bart_result = bart_pipeline(text, max_length=120, min_length=30, do_sample=False)
    simplified_bart = bart_result[0]['summary_text']

    # 3. Generate simplified text using T5 model.
    t5_large_input = "summarize: " + text
    t5_large_result = t5_large_pipeline(t5_large_input, max_length=120, min_length=30, do_sample=False)
    simplified_t5_large = t5_large_result[0]['summary_text']

    # 4. Generate simplified text using FLAN-T5-Large model.
    flan_t5_large_input = "summarize: " + text
    flan_t5_large_result = flan_t5_large_pipeline(flan_t5_large_input, max_length=120, min_length=30, do_sample=False)
    simplified_flan_t5_large = flan_t5_large_result[0]['summary_text']

    # 5. Generate simplified text using Qwen-7B-Chat
    prompt = f"Please provide a simplification of the following text:\n{text}\n\nSummary:"
    qwen_result = qwen_pipeline(prompt, max_new_tokens=150, do_sample=False, truncation=True)
    simplified_qwen = qwen_result[0]['generated_text'].replace(prompt, "").strip()

    bart_results = bart_pipeline(prompt, max_new_tokens=150, do_sample=False, truncation=True)
    simplified_bart = bart_results[0]['generated_text'].replace(prompt, "").strip()

    t5_large_results = t5_large_pipeline(prompt, max_new_tokens=150, do_sample=False, truncation=True)
    simplified_t5 = t5_large_results[0]['generated_text'].replace(prompt, "").strip()


    flan_t5_large_results= flan_t5_large_pipeline(prompt, max_new_tokens=150, do_sample=False, truncation=True)
    simplified_flan_t5_large = flan_t5_large_results[0]['generated_text'].replace(prompt, "").strip()

#    Display simplified texts and their readability scores
    st.write("### Simplified Text (BART)")
    st.write(simplified_bart)
    st.write(f"Reading Ease: {textstat.flesch_reading_ease(simplified_bart)}")
    st.write(f"Flesch-Kincaid Grade Level: {textstat.flesch_kincaid_grade(simplified_bart)}")
    st.write(f"Gunning Fog Score: {textstat.gunning_fog(simplified_bart)}")
    st.write(f"Automated Readability Index: {textstat.automated_readability_index(simplified_bart)}")
    st.write(f"Coleman Liau Index: {textstat.coleman_liau_index(simplified_bart)}")
    st.write(f"Linsar Write Formula: {textstat.linsear_write_formula(simplified_bart)}")
    st.write(f"Dale Chall Readability Grade: {textstat.dale_chall_readability_score(simplified_bart)}")
    st.write(f"Readability Consensus Level: {textstat.text_standard(simplified_bart, float_output=False)}")
    st.write(f"Spache Readability Formula Level: {textstat.spache_readability(simplified_bart)}")
    st.write(f"Reading Time: {textstat.reading_time(simplified_bart, ms_per_char=14.69)}")
    st.write(f"Syllable Count: {textstat.syllable_count(simplified_bart)}")
    st.write(f"Lexicon Count: {textstat.lexicon_count(simplified_bart, removepunct=True)}")

    st.write("### Simplified Text (T5)")
    st.write(simplified_t5)
    st.write(f"Reading Ease: {textstat.flesch_reading_ease(simplified_t5)}")
    st.write(f"Flesch-Kincaid Grade Level: {textstat.flesch_kincaid_grade(simplified_t5)}")
    st.write(f"Gunning Fog Score: {textstat.gunning_fog(simplified_t5)}")
    st.write(f"Automated Readability Index: {textstat.automated_readability_index(simplified_t5)}")
    st.write(f"Coleman Liau Index: {textstat.coleman_liau_index(simplified_t5)}")
    st.write(f"Linsar Write Formula: {textstat.linsear_write_formula(simplified_t5)}")
    st.write(f"Dale Chall Readability Grade: {textstat.dale_chall_readability_score(simplified_t5)}")
    st.write(f"Readability Consensus Level: {textstat.text_standard(simplified_t5, float_output=False)}")
    st.write(f"Spache Readability Formula Level: {textstat.spache_readability(simplified_t5)}")
    st.write(f"Reading Time: {textstat.reading_time(simplified_t5, ms_per_char=14.69)}")
    st.write(f"Syllable Count: {textstat.syllable_count(simplified_t5)}")
    st.write(f"Lexicon Count: {textstat.lexicon_count(simplified_t5, removepunct=True)}")

    st.write("### Simplified Text (FLAN-T5-Large)")
    st.write(simplified_flan_t5_large)
    st.write(f"Reading Ease: {textstat.flesch_reading_ease(simplified_flan_t5_large)}")
    st.write(f"Flesch-Kincaid Grade Level: {textstat.flesch_kincaid_grade(simplified_flan_t5_large)}")
    st.write(f"Gunning Fog Score: {textstat.gunning_fog(simplified_flan_t5_large)}")
    st.write(f"Automated Readability Index: {textstat.automated_readability_index(simplified_flan_t5_large)}")
    st.write(f"Coleman Liau Index: {textstat.coleman_liau_index(simplified_flan_t5_large)}")
    st.write(f"Linsar Write Formula: {textstat.linsear_write_formula(simplified_flan_t5_large)}")
    st.write(f"Dale Chall Readability Grade: {textstat.dale_chall_readability_score(simplified_flan_t5_large)}")
    st.write(f"Readability Consensus Level: {textstat.text_standard(simplified_flan_t5_large, float_output=False)}")
    st.write(f"Spache Readability Formula Level: {textstat.spache_readability(simplified_flan_t5_large)}")
    st.write(f"Reading Time: {textstat.reading_time(simplified_flan_t5_large, ms_per_char=14.69)}")
    st.write(f"Syllable Count: {textstat.syllable_count(simplified_flan_t5_large)}")
    st.write(f"Lexicon Count: {textstat.lexicon_count(simplified_flan_t5_large, removepunct=True)}")


    st.write("### Simplified Text (Qwen-7B-Chat)")
    st.write(simplified_qwen)
    st.write(f"Reading Ease: {textstat.flesch_reading_ease(simplified_qwen)}")
    st.write(f"Flesch-Kincaid Grade Level: {textstat.flesch_kincaid_grade(simplified_qwen)}")
    st.write(f"Gunning Fog Score: {textstat.gunning_fog(simplified_qwen)}")
    st.write(f"Automated Readability Index: {textstat.automated_readability_index(simplified_qwen)}")
    st.write(f"Coleman Liau Index: {textstat.coleman_liau_index(simplified_qwen)}")
    st.write(f"Linsar Write Formula: {textstat.linsear_write_formula(simplified_qwen)}")
    st.write(f"Dale Chall Readability Grade: {textstat.dale_chall_readability_score(simplified_qwen)}")
    st.write(f"Readability Consensus Level: {textstat.text_standard(simplified_qwen, float_output=False)}")
    st.write(f"Spache Readability Formula Level: {textstat.spache_readability(simplified_qwen)}")
    st.write(f"Reading Time: {textstat.reading_time(simplified_qwen, ms_per_char=14.69)}")
    st.write(f"Syllable Count: {textstat.syllable_count(simplified_qwen)}")
    st.write(f"Lexicon Count: {textstat.lexicon_count(simplified_qwen, removepunct=True)}")

    # 5. Highlight differences between original and simplified texts
    st.write("### Differences between Original and Simplified Texts")

    # Compare BART output
    st.write("#### BART-simplified Text Differences:")
    orig_tokens = text.split()
    bart_tokens = simplified_bart.split()
    matcher = difflib.SequenceMatcher(None, orig_tokens, bart_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            orig_segment = " ".join(orig_tokens[i1:i2])
            new_segment = " ".join(bart_tokens[j1:j2])
            st.write(f"Replaced: \"{orig_segment}\" ➜ \"{new_segment}\"")
        elif tag == 'delete':
            orig_segment = " ".join(orig_tokens[i1:i2])
            st.write(f"Removed: \"{orig_segment}\"")
        elif tag == 'insert':
            new_segment = " ".join(bart_tokens[j1:j2])
            st.write(f"Added: \"{new_segment}\"")



#     Compare T5-Large output
    st.write("#### T5-Large-simplified Text Differences:")
    t5_large_tokens = simplified_t5_large.split()
    matcher = difflib.SequenceMatcher(None, orig_tokens, t5_large_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            orig_segment = " ".join(orig_tokens[i1:i2])
            new_segment = " ".join(t5_large_tokens[j1:j2])
            st.write(f"Replaced: \"{orig_segment}\" ➜ \"{new_segment}\"")
        elif tag == 'delete':
            orig_segment = " ".join(orig_tokens[i1:i2])
            st.write(f"Removed: \"{orig_segment}\"")
        elif tag == 'insert':
            new_segment = " ".join(t5_large_tokens[j1:j2])
            st.write(f"Added: \"{new_segment}\"")





#     Compare FLAN-T5-Large output
    st.write("#### FLAN-T5-Large-simplified Text Differences:")
    flan_t5_large_tokens = simplified_flan_t5_large.split()
    matcher = difflib.SequenceMatcher(None, orig_tokens, flan_t5_large_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            orig_segment = " ".join(orig_tokens[i1:i2])
            new_segment = " ".join(flan_t5_large_tokens[j1:j2])
            st.write(f"Replaced: \"{orig_segment}\" ➜ \"{new_segment}\"")
        elif tag == 'delete':
            orig_segment = " ".join(orig_tokens[i1:i2])
            st.write(f"Removed: \"{orig_segment}\"")
        elif tag == 'insert':
            new_segment = " ".join(flan_t5_large_tokens[j1:j2])
            st.write(f"Added: \"{new_segment}\"")


#     Compare Qwen-7B-Chat output
    st.write("#### Qwen-7B-Chat-simplified Text Differences:")
    qwen_tokens = simplified_qwen.split()
    matcher = difflib.SequenceMatcher(None, orig_tokens, qwen_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            orig_segment = " ".join(orig_tokens[i1:i2])
            new_segment = " ".join(qwen_tokens[j1:j2])
            st.write(f"Replaced: \"{orig_segment}\" ➜ \"{new_segment}\"")
        elif tag == 'delete':
            orig_segment = " ".join(orig_tokens[i1:i2])
            st.write(f"Removed: \"{orig_segment}\"")
        elif tag == 'insert':
            new_segment = " ".join(qwen_tokens[j1:j2])
            st.write(f"Added: \"{new_segment}\"")

#     Suggest synonyms for words in the simplified texts
    st.write("### Synonym Suggestions for New Words in Simplified Texts")

    bart_new_words = set([w.strip(".,!?").lower() for w in bart_tokens if w.lower() not in [ow.lower() for ow in orig_tokens]])
    t5_large_new_words = set([w.strip(".,!?").lower() for w in t5_large_tokens if w.lower() not in [ow.lower() for ow in orig_tokens]])
    flan_t5_large_new_words = set([w.strip(".,!?").lower() for w in flan_t5_large_tokens if w.lower() not in [ow.lower() for ow in orig_tokens]])
    qwen_new_words = set([w.strip(".,!?").lower() for w in qwen_tokens if w.lower() not in [ow.lower() for ow in orig_tokens]])

    # Combine new words from all models
    new_words = bart_new_words.union(flan_t5_large_new_words).union(t5_large_new_words).union(qwen_new_words)
    if new_words:
        for word in sorted(new_words):
            if not word:  # skip empty strings if any
                continue
            synonyms = get_synonyms(word)
            if synonyms:
                st.write(f"{word} ➜ {', '.join(synonyms[:5])}")  # show up to 5 synonyms
            else:
                st.write(f"{word} ➜ (no synonyms found)")
    else:
        st.write("No new words to suggest synonyms for.")
