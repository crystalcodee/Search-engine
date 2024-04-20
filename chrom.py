import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Load the Word2Vec model
word2vec_model = Word2Vec.load("skipgram_model1.bin")

# Load your dataset of documents
df = pd.read_csv("D:/search_engine/preprocessed_data.csv")

# Define the path to your background image
background_image_path = r"D:/search_engine/Untitled design.jpg"  # Replace "path_to_your_image.jpg" with the path to your image file

# Emoji
loading_emoji = "⌛"
success_emoji = "✅"

# Streamlit UI with custom styling and emojis
st.markdown(
    """
    <style>
    .title {
        color: #4CAF50;
        text-align: center;
        font-family: Arial, sans-serif;
        font-size: 36px;
        padding-top: 50px;
        padding-bottom: 50px;
    }
    .query-input {
        margin-bottom: 20px;
        padding: 15px;
        border: 1px solid #ccc;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        font-family: Arial, sans-serif;
        font-size: 24px;
        width: 100%;
        box-sizing: border-box;
    }
    .query-input:focus {
        margin-bottom: 20px;
        padding: 15px;
        border: 1px solid #ccc;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        font-family: Arial, sans-serif;
        font-size: 24px;
        width: 100%;
        box-sizing: border-box;
        outline: none;
        border-color: #4CAF50;
        font-size: 24px;
        box-shadow: 0 0 5px rgba(31, 119, 180, 0.5);
    }
    </style>
    """
    , unsafe_allow_html=True
)

# Streamlit UI
st.title("🔍 Document Similarity Finder")

# Display background image
st.image(background_image_path, use_column_width=True)

# Input query from user
query = st.text_input("Enter your search query:", "", key="query_input", help="Type your query here")

# Add Enter button
if st.button("Enter"):
    # Display loading message while processing
    with st.spinner(f"Searching for similar documents {loading_emoji}"):
        if query:
            # Vectorize the user query
            query_vector = np.mean([word2vec_model.wv[word] for word in query.lower().split() if word in word2vec_model.wv], axis=0)
            
            # Calculate cosine similarity with each document in the dataset
            similarity_scores = []
            for idx, row in df.iterrows():
                doc_vector = np.mean([word2vec_model.wv[word] for word in row["Tokenized Text"].lower().split() if word in word2vec_model.wv], axis=0)
                similarity_score = cosine_similarity([query_vector], [doc_vector])[0][0]
                similarity_scores.append((idx, similarity_score, row["Tokenized Text"]))  # Include the text of each document
            
            # Sort similarity scores in descending order
            similarity_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Display top similar documents
            st.write("Top Similar Documents:")
            for idx, similarity_score, text in similarity_scores[:5]:  # Adjust the number of top documents to display as needed
                # Referencing the similarity score ranges from the first code
                similarity_range = int(similarity_score * 100) // 20
                st.write(f"- Document Index: {idx}, Similarity Score: {similarity_score:.2f}, Similarity Range: {similarity_range*20}-{(similarity_range+1)*20}")
                
                # Add an expander for each document with the index as its name
                with st.expander(f"Document {idx}"):
                    st.write(f"Text: {text}")
        else:
            st.warning("Please enter a search query.")
