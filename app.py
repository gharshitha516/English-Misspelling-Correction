import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model + tokenizer from Hugging Face Hub
@st.cache_resource
def load_model():
    model_name = "harshhitha/Misspelling_Correction"  # your HF repo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

st.title("📝 English Spelling Corrector")

user_input = st.text_area("Enter your text:", height=150)

if st.button("Correct Text"):
    if user_input.strip():
        inputs = tokenizer([user_input], return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs, max_length=128, num_beams=4)
        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("✅ Corrected Text")
        st.success(corrected_text)
    else:
        st.warning("Please enter some text.")
