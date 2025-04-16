import streamlit as st
import spacy
import difflib
import requests
import textstat
import json
import torch
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from APIs import API_KEYS  # Custom module to securely store your keys

# Load models and NLP pipeline
nlp = spacy.load("en_core_web_sm")


HF_API_KEY = API_KEYS["HF"]  # Assuming it's in your APIs.py



combined_prompt_template = PromptTemplate(
    input_variables=["text"],
    template=(
        "Simplify the following text so that it is easy to understand for a high school-level student or someone without a technical background.\n\n"
        "Please follow these rules:\n"
        "1. Use plain, everyday language. Avoid technical or academic terms unless you briefly explain them.\n"
        "2. Break down long or complex sentences into shorter, clearer ones.\n"
        "3. Rephrase concepts using simpler wording and relatable examples if needed.\n"
        "4. Do NOT copy or repeat the original text. Rewrite it in your own words while keeping the original meaning.\n"
        "5. Keep all important details, but make the writing much easier to understand.\n\n"
        "- Completely rephrase the sentence using simpler words.\n"
        "- Avoid using any of the original phrasing unless absolutely necessary.\n"
        "- Break long sentences into shorter ones.\n"
        "- Explain technical terms in plain language.\n"
        "- Keep all the important ideas.\n\n"
        "Examples:\n"
        "Original: 'There were a lot of different kinds of colors, including red, blue, and green.'\n"
        "Simplified: 'There were many colors, like red, blue, and green.'\n\n"
        "Original: 'The proliferation of autonomous transportation systems necessitates robust regulatory oversight.'\n"
        "Simplified: 'Self-driving cars need strong government rules.'\n\n"
        "Original: 'Urban planners must incorporate resilient and adaptive infrastructure strategies to address climate-related risks.'\n"
        "Simplified: 'City planners should build stronger and more flexible systems to handle climate change.'\n\n"
        "Original: 'Photosynthesis is a process through which green plants use sunlight to synthesize foods from carbon dioxide and water.'\n"
        "Simplified: 'Plants make their own food using sunlight, water, and air.'\n\n"
        "Now simplify this:\n\n{text}\n\n"
        "**Simplified Version:**"
    )
)



st.set_page_config(page_title="Text Simplifier", layout="wide")

st.title("AI-Powered Text Simplifier")


inference_mode = st.sidebar.selectbox("Choose Inference Mode", ["Use Local Models", "Use Hugging Face API"])

if inference_mode == "Use Hugging Face API":
    st.sidebar.info("You're using the Hugging Face Inference API. Models run on Hugging Face's servers.")
    bart_pipeline = pipeline(
        "text2text-generation",
        model="facebook/bart-large-cnn",
        tokenizer="facebook/bart-large-cnn",
        token=HF_API_KEY
    )
    t5_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        tokenizer="google/flan-t5-large",
        token=HF_API_KEY,
        max_length=512,
        min_length=80,
        do_sample=False
    )

else:
    st.sidebar.info("You're using local models. Make sure your machine can handle them.")
    bart_pipeline = pipeline(
        "text2text-generation",
        model="facebook/bart-large-cnn"
    )
    t5_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        tokenizer="google/flan-t5-large",
        max_length=512,
        min_length=80,
        do_sample=False
    )



#result = t5_pipeline( # Teste uten å bruke langchain
 #   prompt,
  #  max_length=512,
   # min_length=80,
    #do_sample=False
#)[0]["generated_text"]

# LangChain
bart_llm = HuggingFacePipeline(pipeline=bart_pipeline)
t5_llm = HuggingFacePipeline(pipeline=t5_pipeline)


bart_chain = LLMChain(llm=bart_llm, prompt=combined_prompt_template)
t5_chain = LLMChain(llm=t5_llm, prompt=combined_prompt_template)

def convert_passive_to_active(text):
    doc = nlp(text)
    new_sentences = []
    for sent in doc.sents:
        words = [token.text for token in sent]
        new_sentences.append(" ".join(words))
    return " ".join(new_sentences)


def is_too_similar(original, simplified, threshold=0.95):
    return difflib.SequenceMatcher(None, original.strip().lower(), simplified.strip().lower()).ratio() >= threshold



def get_synonyms(word):
    api_url = f"https://api.api-ninjas.com/v1/thesaurus?word={word}"
    headers = {"X-Api-Key": API_KEYS['ninja']}
    try:
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get("synonyms", [])
        return []
    except Exception:
        return []


# --- Streamlit UI ---
text_input = st.text_area("Enter text to simplify:", height=250)
if st.button("Simplify Text"):
    try:
        if not text_input.strip():
            st.warning("Please enter some text to simplify.")
        else:
            active_text = convert_passive_to_active(text_input)

            with st.spinner("Simplifying using BART and T5..."):
                simplified_bart = bart_chain.run(text=active_text)
                st.write("BART Output:", simplified_bart)

                simplified_t5 = t5_chain.run(text=f"summarize: {active_text}")
                st.write("T5 Output:", simplified_t5)



            # Display results...
            st.success("Simplification complete!")
            ...
    except Exception as e:
        st.error(f"An error occurred: {e}")

        with st.spinner("Simplifying using BART and T5..."):
            simplified_bart = bart_chain.run(text=active_text)
            simplified_t5 = t5_chain.run(text=f"summarize: {active_text}")
            # Similarity Warnings
        if is_too_similar(text_input, simplified_bart):
            st.warning("⚠️ BART output is very similar to the original. It may not be properly simplified.")

        if is_too_similar(text_input, simplified_t5):
            st.warning("⚠️ T5 output is very similar to the original. It may not be properly simplified.")


        # Show outputs
        st.subheader("Original Text")
        st.write(text_input)

        st.subheader("Simplified (BART)")
        st.write(simplified_bart)

        st.subheader("Simplified (T5)")
        st.write(simplified_t5)


        st.subheader(" Readability Metrics")
        col1, col2, col3, col4 = st.columns(4)
        for col, label, text in zip([col1, col2, col3, col4], ["Original", "BART", "T5"],
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

        # Differences
        def get_diff(original, simplified):
            orig_tokens = original.split()
            simp_tokens = simplified.split()
            matcher = difflib.SequenceMatcher(None, orig_tokens, simp_tokens)
            diffs = []
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'replace':
                    diffs.append(f"Replaced '{' '.join(orig_tokens[i1:i2])}' ➜ '{' '.join(simp_tokens[j1:j2])}'")
                elif tag == 'delete':
                    diffs.append(f"Removed '{' '.join(orig_tokens[i1:i2])}'")
                elif tag == 'insert':
                    diffs.append(f"Added '{' '.join(simp_tokens[j1:j2])}'")
            return diffs

        st.subheader("Differences from Original")
        st.markdown("**BART Differences**")
        st.code("\n".join(get_diff(text_input, simplified_bart)) or "No differences found.")

        st.markdown("**T5 Differences**")
        st.code("\n".join(get_diff(text_input, simplified_t5)) or "No differences found.")

        # Synonyms
        st.subheader(" Synonym Suggestions")
        new_words = set(simplified_bart.lower().split()) | set(simplified_t5.lower().split())
        new_words = new_words - set(text_input.lower().split())
        synonym_dict = {w: get_synonyms(w) for w in new_words if w.isalpha()}
        for word, synonyms in synonym_dict.items():
            st.write(f"**{word}** ➜ {', '.join(synonyms[:5]) if synonyms else 'No synonyms found'}")


