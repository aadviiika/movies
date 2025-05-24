# 🎬 Movie Recommendation System

A simple content-based movie recommender app using Streamlit and TF-IDF genre vectors.

## 📦 Dataset
- Source: [Kaggle - Movie Recommendation System](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system)
- Used only: `movies.csv`

## ⚙️ How It Works
- Uses TF-IDF to vectorize movie genres
- Computes cosine similarity between movies
- Recommends top 10 movies similar to a selected one

## 🚀 Run Locally
```bash
pip install streamlit pandas scikit-learn
streamlit run streamlit_app.py
