import streamlit as st
from transformers import pipeline

st.title("🧠 BERT NLP Tasks Demo")

task = st.selectbox("Choose NLP Task", ["Sentiment Analysis", "Named Entity Recognition", "Question Answering"])

if task == "Sentiment Analysis":
    classifier = pipeline("sentiment-analysis")
    text = st.text_area("Enter text:")
    if st.button("Analyze"):
        st.write(classifier(text))

elif task == "Named Entity Recognition":
    ner = pipeline("ner", aggregation_strategy="simple")
    text = st.text_area("Enter text:")
    if st.button("Extract Entities"):
        st.json(ner(text))

elif task == "Question Answering":
    qa = pipeline("question-answering")
    context = st.text_area("Enter context passage:")
    question = st.text_input("Enter question:")
    if st.button("Get Answer"):
        st.write(qa(question=question, context=context))
