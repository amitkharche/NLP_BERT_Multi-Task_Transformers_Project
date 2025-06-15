import streamlit as st
from transformers import pipeline
import plotly.express as px
import pandas as pd
import time

# Page configuration
st.set_page_config(
    page_title="BERT NLP Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .task-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .result-container {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #28a745;
        margin: 1rem 0;
    }
    
    .error-container {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dc3545;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #f8f9fa;
    }
    
    .stTextInput > div > div > input {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 Advanced BERT NLP Suite</h1>
    <p>Powered by Transformers | State-of-the-art Natural Language Processing</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for task selection and info
with st.sidebar:
    st.markdown("### 🎯 Select NLP Task")
    task = st.selectbox(
        "Choose your task:",
        ["Sentiment Analysis", "Named Entity Recognition", "Question Answering"],
        help="Select the NLP task you want to perform"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About This Tool")
    
    task_descriptions = {
        "Sentiment Analysis": "Analyze the emotional tone and sentiment of text (positive, negative, neutral)",
        "Named Entity Recognition": "Identify and classify named entities like persons, organizations, locations",
        "Question Answering": "Extract answers from a given context passage based on your questions"
    }
    
    st.info(task_descriptions[task])
    
    st.markdown("---")
    st.markdown("### 🔧 Model Info")
    st.markdown("**Framework:** Hugging Face Transformers")
    st.markdown("**Model:** BERT-based pre-trained models")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    if task == "Sentiment Analysis":
        st.markdown('<div class="task-card">', unsafe_allow_html=True)
        st.markdown("## 😊 Sentiment Analysis")
        st.markdown("Analyze the emotional tone of your text")
        
        text = st.text_area(
            "Enter your text:",
            height=150,
            placeholder="Type or paste your text here for sentiment analysis...",
            help="Enter any text to analyze its sentiment"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_btn:
            st.rerun()
            
        if analyze_btn and text:
            with st.spinner("🤖 Analyzing sentiment..."):
                try:
                    classifier = pipeline("sentiment-analysis")
                    result = classifier(text)
                    
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown("### 📊 Analysis Results")
                    
                    # Display result with enhanced formatting
                    sentiment = result[0]['label']
                    confidence = result[0]['score']
                    
                    # Color coding for sentiment
                    color = "#28a745" if sentiment == "POSITIVE" else "#dc3545"
                    emoji = "😊" if sentiment == "POSITIVE" else "😔"
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem;">
                        <h3 style="color: {color};">{emoji} {sentiment}</h3>
                        <p style="font-size: 1.2em;">Confidence: <strong>{confidence:.2%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown('<div class="error-container">', unsafe_allow_html=True)
                    st.error(f"❌ Error: {str(e)}")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        elif analyze_btn and not text:
            st.warning("⚠️ Please enter some text to analyze")
        
        st.markdown('</div>', unsafe_allow_html=True)

    elif task == "Named Entity Recognition":
        st.markdown('<div class="task-card">', unsafe_allow_html=True)
        st.markdown("## 🏷️ Named Entity Recognition")
        st.markdown("Identify and classify entities in your text")
        
        text = st.text_area(
            "Enter your text:",
            height=150,
            placeholder="Enter text containing names, places, organizations, etc.",
            help="The model will identify entities like PERSON, ORG, LOC, MISC"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            extract_btn = st.button("🎯 Extract Entities", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
            
        if clear_btn:
            st.rerun()
        
        if extract_btn and text:
            with st.spinner("🔍 Extracting entities..."):
                try:
                    ner = pipeline("ner", aggregation_strategy="simple")
                    entities = ner(text)
                    
                    if entities:
                        st.markdown('<div class="result-container">', unsafe_allow_html=True)
                        st.markdown("### 🎯 Detected Entities")
                        
                        # Create a dataframe for better display
                        df = pd.DataFrame(entities)
                        df['confidence'] = df['score'].apply(lambda x: f"{x:.2%}")
                        df = df[['word', 'entity_group', 'confidence']].rename(columns={
                            'word': 'Entity',
                            'entity_group': 'Type',
                            'confidence': 'Confidence'
                        })
                        
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        # Entity type distribution
                        entity_counts = df['Type'].value_counts()
                        if len(entity_counts) > 1:
                            fig = px.pie(
                                values=entity_counts.values,
                                names=entity_counts.index,
                                title="Entity Distribution"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("🔍 No entities detected in the provided text")
                        
                except Exception as e:
                    st.markdown('<div class="error-container">', unsafe_allow_html=True)
                    st.error(f"❌ Error: {str(e)}")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        elif extract_btn and not text:
            st.warning("⚠️ Please enter some text to analyze")
            
        st.markdown('</div>', unsafe_allow_html=True)

    elif task == "Question Answering":
        st.markdown('<div class="task-card">', unsafe_allow_html=True)
        st.markdown("## ❓ Question Answering")
        st.markdown("Get answers from your context passage")
        
        context = st.text_area(
            "Context Passage:",
            height=200,
            placeholder="Paste the text passage that contains the information...",
            help="Provide the context from which you want to extract answers"
        )
        
        question = st.text_input(
            "Your Question:",
            placeholder="What do you want to know about the context?",
            help="Ask a specific question about the context passage"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            answer_btn = st.button("💡 Get Answer", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
            
        if clear_btn:
            st.rerun()
        
        if answer_btn and context and question:
            with st.spinner("🤔 Finding the answer..."):
                try:
                    qa = pipeline("question-answering")
                    result = qa(question=question, context=context)
                    
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown("### 💡 Answer Found")
                    
                    answer = result['answer']
                    confidence = result['score']
                    start = result['start']
                    end = result['end']
                    
                    st.markdown(f"""
                    <div style="background: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                        <h4 style="color: #333;">📝 Answer:</h4>
                        <p style="font-size: 1.1em; color: #2c3e50; font-weight: 500;">"{answer}"</p>
                        <p style="color: #7f8c8d;">Confidence: <strong>{confidence:.2%}</strong></p>
                        <p style="color: #7f8c8d; font-size: 0.9em;">Position in text: {start}-{end}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Highlight the answer in context
                    highlighted_context = (
                        context[:start] + 
                        f"**{context[start:end]}**" + 
                        context[end:]
                    )
                    
                    with st.expander("📄 View Answer in Context"):
                        st.markdown(highlighted_context)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown('<div class="error-container">', unsafe_allow_html=True)
                    st.error(f"❌ Error: {str(e)}")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        elif answer_btn and (not context or not question):
            st.warning("⚠️ Please provide both context and question")
            
        st.markdown('</div>', unsafe_allow_html=True)

# Right column - Tips and examples
with col2:
    st.markdown("### 💡 Tips & Examples")
    
    if task == "Sentiment Analysis":
        st.markdown("""
        **Try these examples:**
        - "I love this product! It works perfectly."
        - "This movie was terrible and boring."
        - "The weather is okay today."
        
        **Best practices:**
        - Use complete sentences
        - Include context when possible
        - Try different text lengths
        """)
    
    elif task == "Named Entity Recognition":
        st.markdown("""
        **Try this example:**
        "Apple Inc. was founded by Steve Jobs in Cupertino, California. The company is now headquartered in Apple Park."
        
        **Entity types detected:**
        - **PER**: Person names
        - **ORG**: Organizations
        - **LOC**: Locations
        - **MISC**: Miscellaneous entities
        """)
    
    elif task == "Question Answering":
        st.markdown("""
        **Example Context:**
        "The Eiffel Tower is located in Paris, France. It was built in 1889 and stands 324 meters tall."
        
        **Example Questions:**
        - "Where is the Eiffel Tower?"
        - "When was it built?"
        - "How tall is the tower?"
        
        **Tips:**
        - Ask specific questions
        - Ensure context contains the answer
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 1rem;">
    Built with ❤️ using Streamlit and Hugging Face Transformers
</div>
""", unsafe_allow_html=True)