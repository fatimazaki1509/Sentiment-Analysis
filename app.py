import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained sentiment model
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Define sentiment categories
sentiment_classes = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
sentiment_emojis = ["😢", "😞", "😐", "😊", "🥰"]
sentiment_messages = [
    "It looks like you're really struggling today. Consider reaching out to someone. 💜",
    "Feeling low? Remember, you're not alone. 💙",
    "A calm and reflective day. Keep journaling! 📝",
    "You seem happy today! Keep it up! 😊",
    "You're in a great mood today! Stay positive! 🥰"
]

# Streamlit UI
st.title("📝 Diary Sentiment Analysis")
st.write("Enter your diary entry below and check your mood!")

# Text input
text = st.text_area("Write your diary entry here:", "")

if st.button("Analyze Sentiment"):
    if text.strip():
        # Process text
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        output = model(**tokens)
        scores = F.softmax(output.logits, dim=1)
        sentiment_score = torch.argmax(scores).item()

        # Show results
        st.write(f"**Sentiment:** {sentiment_classes[sentiment_score]} {sentiment_emojis[sentiment_score]}")
        st.write(f"**Message:** {sentiment_messages[sentiment_score]}")
    else:
        st.write("❌ Please enter some text to analyze.")
