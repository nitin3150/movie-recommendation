# test.py
from recommender.models import SimpleCollaborativeFilter, ContentBasedFilter
import pickle

# Load models
with open('trained_models/collaborative_model.pkl', 'rb') as f:
    cf_model = pickle.load(f)

with open('trained_models/content_based_model.pkl', 'rb') as f:
    cb_model = pickle.load(f)

# Test collaborative filtering
user_id = 1  # Use a user ID that exists in your dataset
recommendations = cf_model.recommend_items(user_id=user_id, n=5)
print("\nCollaborative Filtering Recommendations for user", user_id)
print(recommendations)

# Test content-based filtering
movie_id = 1  # Use a movie ID that exists in your dataset
similar_movies = cb_model.recommend_similar(movie_id=movie_id, n=5)
print("\nContent-Based Recommendations for movie", movie_id)
print(similar_movies)