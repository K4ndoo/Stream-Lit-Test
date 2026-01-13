import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import Normalizer

@st.cache_resource
def load_assets():
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    model = tf.keras.models.load_model("word2vec_full.keras")
    
    vectors = model.layers[0].get_weights()[0]
    
    return tokenizer, model, vectors

tokenizer, model, vectors = load_assets()

word2idx = tokenizer.word_index
idx2word = tokenizer.index_word

def cosine_similarity(vec1, vec2):
    dot = np.sum(vec1 * vec2)
    norm = np.sqrt(np.sum(vec1**2)) * np.sqrt(np.sum(vec2**2))
    return dot / norm

def find_closest(word, vectors, number_closest=10):
    if word not in word2idx:
        return None
    
    query_index = word2idx[word]
    query_vector = vectors[query_index]
    
    results = []
    for index, vector in enumerate(vectors):
        if index != query_index and index != 0: # 0 est souvent le padding
            dist = cosine_similarity(vector, query_vector)
            results.append([dist, index])
            
    return sorted(results, key=lambda x: x[0], reverse=True)[:number_closest]

st.title("🎬 Movie Review Word2Vec Explorer")
st.subheader("Trouvez les mots sémantiquement proches")

user_word = st.text_input("Entrez un mot :", "zombie")

if st.button("Analyser"):
    word_to_search = user_word.lower().strip()
    
    with st.spinner('Recherche en cours...'):
        closest_words = find_closest(word_to_search, vectors)
    
    if closest_words:
        st.success(f"Mots les plus proches de **{word_to_search}** :")
        
        for score, index in closest_words:
            st.write(f"- **{idx2word[index]}** (score : {score:.4f})")
    else:
        st.error(f"Désolé, le mot **{word_to_search}** n'est pas dans le dictionnaire.")

st.divider()
st.subheader("Analogie (Roi - Homme + Femme)")
col1, col2, col3 = st.columns(3)
w1 = col1.text_input("Mot A", "king")
w2 = col2.text_input("Mot B", "man")
w3 = col3.text_input("Mot C", "woman")