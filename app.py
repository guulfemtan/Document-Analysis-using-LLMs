import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

st.title("Document Summarizer")
st.subheader("Analyze and summarize documents using LLMs")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
    else:
        import pdfplumber
        with pdfplumber.open(uploaded_file) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
        text = "\n".join(pages)

    st.write("Document Preview:")
    st.write(text[:500])

    model = T5ForConditionalGeneration.from_pretrained("saved_summarizer")
    tokenizer = T5Tokenizer.from_pretrained("saved_summarizer")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    summary = summarizer(text[:1000], max_length=150, min_length=30, do_sample=False)
    st.subheader("Summary:")
    st.write(summary[0]['summary_text'])