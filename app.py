import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load model + tokenizer
@st.cache_resource
def load_model():
    model_name = "./spell_model"  # local path
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

st.title("📝 English Spelling Corrector")

user_input = st.text_area("Enter your text:", height=150)

if st.button("Correct Text"):
    if user_input.strip():
        inputs = tokenizer([user_input], return_tensors="pt", padding=True)
        outputs = model.generate(**inputs, max_length=128, num_beams=4)
        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success(corrected_text)
    else:
        st.warning("Please enter some text.")
