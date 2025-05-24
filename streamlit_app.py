import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

# Load data
@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    movies = movies.head(10000).copy()  # reduce size for performance
    movies['genres'] = movies['genres'].replace("(no genres listed)", "")
    return movies

movies = load_data()

# TF-IDF and similarity matrix
@st.cache_data
def compute_similarity(movies):
    tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = compute_similarity(movies)

# Map titles to indices
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices.get(title)
    if idx is None:
        return ["Movie not found. Please check the title."]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Streamlit UI
st.title("🎬 Movie Recommender")
st.write("Get content-based recommendations based on genres.")

user_input = st.selectbox("Choose a movie:", movies['title'].sort_values().tolist())

if st.button("Recommend"):
    st.subheader("Top 10 Similar Movies:")
    recommendations = get_recommendations(user_input)
    for i, movie in enumerate(recommendations, 1):
        st.write(f"{i}. {movie}")
