# train.py
from recommender.models import SimpleCollaborativeFilter, ContentBasedFilter
import pandas as pd
import pickle
import os

def train():
    print("Loading data...")
    ratings_df = pd.read_csv("data/ratings.csv")
    movies_df = pd.read_csv("data/movies.csv")
    
    print("Training collaborative filtering model...")
    cf_model = SimpleCollaborativeFilter()
    cf_model.train(ratings_df)
    
    print("Training content-based model...")
    cb_model = ContentBasedFilter()
    cb_model.train(movies_df)
    
    print("Saving models...")
    os.makedirs('trained_models', exist_ok=True)
    
    with open('trained_models/collaborative_model.pkl', 'wb') as f:
        pickle.dump(cf_model, f)
    
    with open('trained_models/content_based_model.pkl', 'wb') as f:
        pickle.dump(cb_model, f)
        
    print("Models trained and saved successfully!")

if __name__ == "__main__":
    train()