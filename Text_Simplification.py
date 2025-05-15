import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import textstat
import streamlit as st
from HOVED_APIS import API_KEYS




# Streamlit page config
st.set_page_config(page_title="Sentence Simplifier", layout="centered")
st.title(" Sentence Simplifier")
st.markdown("Type in a complex sentence, and get a simpler, more understandable version.")

HF_API_KEY = API_KEYS["HF"]  # From the API keys file, if youre using HuggingFace API


@st.cache_resource
def load_model_and_tokenizer():
    model_path= "Zetsu00/Qwen-7B-QLoRA-simplifier"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-7B-Chat",
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    return model, tokenizer, 


model, tokenizer = load_model_and_tokenizer()


def simplify(text):
    prompt = (
        "You are an assistant that rewrites technical sentences in plain English for students and non-experts. From this sentence, identify all the nouns, and then provide one overarching common noun that could represent the whole group. Use this common noun in the simplified sentence\n\n"
       
        f"Sentence: {text.strip()}\n"
        "Simplified:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            temperature=0.2, # originally 0.7
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    simplified = output.split("Simplified:")[-1].strip()
    return simplified


# User Input
user_input = st.text_area("Enter a complex sentence below:", height=150)

if st.button(" Simplify"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence to simplify.")
    else:
        with st.spinner("Simplifying..."):
            simplified = simplify(user_input)

        st.subheader(" output:")
        st.success(simplified)

        # Readability Metrics
        st.subheader(" Readability Metrics")
        st.markdown(f"- **Flesch Reading Ease**: {textstat.flesch_reading_ease(user_input):.2f}")
        st.markdown(f"- **Flesch-Kincaid Grade Level**: {textstat.flesch_kincaid_grade(user_input):.2f}")
        st.markdown(f"- **Gunning Fog Index**: {textstat.gunning_fog(user_input):.2f}")
        st.markdown(f"- **Overall Consensus**: {textstat.text_standard(user_input)}")
