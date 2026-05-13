"""
Simple Movie Recommendation System
MovieLens Dataset - Item-Based Collaborative Filtering
WITH PROPER TRAIN/TEST SPLIT
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# ============================================
# SIMPLE RECOMMENDER CLASS
# ============================================

class SimpleRecommender:
    """Simple movie recommender using item-based collaborative filtering"""
    
    def __init__(self, ratings_df, movies_df):
        print("Building recommender system...")
        self.movies = movies_df
        
        # Create user-movie matrix (users as rows, movies as columns)
        user_movie_matrix = ratings_df.pivot(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        # Calculate movie similarity (transpose so movies are rows)
        print("Calculating movie similarities...")
        movie_matrix = user_movie_matrix.T  # Movies as rows, users as columns
        self.similarity = cosine_similarity(movie_matrix)
        self.similarity_df = pd.DataFrame(
            self.similarity,
            index=movie_matrix.index,
            columns=movie_matrix.index
        )
        
        # Store user ratings (from training data only)
        self.user_ratings = {}
        for user_id in user_movie_matrix.index:
            user_data = user_movie_matrix.loc[user_id]
            # Only store ratings > 0 (actual ratings, not zeros)
            self.user_ratings[user_id] = user_data[user_data > 0]
        
        print(f"Ready! {len(self.user_ratings)} users, {len(self.similarity_df)} movies")
    
    def predict(self, user_id, movie_id):
        """Predict rating for a user on a movie"""
        if user_id not in self.user_ratings:
            return None
        if movie_id not in self.similarity_df.index:
            return None
        
        total_sim = 0
        weighted_sum = 0
        
        for rated_movie, rating in self.user_ratings[user_id].items():
            if rated_movie in self.similarity_df.index:
                sim = self.similarity_df.loc[movie_id, rated_movie]
                if sim > 0:
                    total_sim += sim
                    weighted_sum += sim * rating
        
        if total_sim > 0:
            return weighted_sum / total_sim
        return None
    
    def recommend(self, user_id, n=10):
        """Get top N recommendations for a user"""
        if user_id not in self.user_ratings:
            return []
        
        rated = set(self.user_ratings[user_id].index)
        predictions = []
        
        for movie_id in self.movies['movieId']:
            if movie_id not in rated:
                pred = self.predict(user_id, movie_id)
                if pred:
                    predictions.append((movie_id, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_movies = predictions[:n]
        
        # Get movie details
        results = []
        for movie_id, rating in top_movies:
            movie = self.movies[self.movies['movieId'] == movie_id].iloc[0]
            results.append({
                'title': movie['title'],
                'genres': movie['genres'],
                'predicted_rating': rating
            })
        
        return results
    
    def similar_movies(self, movie_id, n=10):
        """Find movies similar to a given movie"""
        if movie_id not in self.similarity_df.index:
            return []
        
        similar = self.similarity_df[movie_id].sort_values(ascending=False)
        similar = similar[similar.index != movie_id].head(n)
        
        results = []
        for movie_id, score in similar.items():
            movie = self.movies[self.movies['movieId'] == movie_id].iloc[0]
            results.append({
                'title': movie['title'],
                'genres': movie['genres'],
                'similarity': score
            })
        
        return results
    
    def search_movie(self, term):
        """Search for movies by title"""
        result = self.movies[
            self.movies['title'].str.contains(term, case=False, na=False)
        ]
        return result[['movieId', 'title', 'genres']].head(10)


# ============================================
# EVALUATION WITH PROPER TRAIN/TEST SPLIT
# ============================================

def evaluate_recommender(train_ratings, test_ratings, movies_df, num_samples=1000):
    """
    Train on train_ratings, evaluate on test_ratings
    This is the CORRECT way to evaluate
    """
    print("\n" + "="*50)
    print("EVALUATING WITH TRAIN/TEST SPLIT")
    print("="*50)
    
    # Build recommender using ONLY training data
    print(f"\nTraining on {len(train_ratings)} ratings...")
    recommender = SimpleRecommender(train_ratings, movies_df)
    
    # Test on unseen test data
    print(f"Testing on {len(test_ratings)} ratings...")
    sample = test_ratings.sample(min(num_samples, len(test_ratings)))
    
    predictions = []
    actuals = []
    skipped = 0
    
    for _, row in sample.iterrows():
        pred = recommender.predict(row['userId'], row['movieId'])
        if pred is not None:
            predictions.append(pred)
            actuals.append(row['rating'])
        else:
            skipped += 1
    
    if predictions:
        mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
        
        print(f"\n📊 EVALUATION RESULTS:")
        print(f"   Test samples tested: {len(predictions)}")
        print(f"   Samples skipped (no prediction): {skipped}")
        print(f"   MAE:  {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        
        return mae, rmse, recommender
    else:
        print("No predictions could be made!")
        return None, None, None


def evaluate_with_cross_validation(ratings_df, movies_df, k=5):
    """
    Perform k-fold cross-validation
    More robust evaluation than single train/test split
    """
    print("\n" + "="*50)
    print(f"{k}-FOLD CROSS-VALIDATION")
    print("="*50)
    
    # Create folds
    ratings_shuffled = ratings_df.sample(frac=1, random_state=42).reset_index(drop=True)
    fold_size = len(ratings_shuffled) // k
    
    mae_scores = []
    rmse_scores = []
    
    for fold in range(k):
        print(f"\n--- Fold {fold+1}/{k} ---")
        
        # Split into train and validation for this fold
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < k-1 else len(ratings_shuffled)
        
        val_fold = ratings_shuffled.iloc[start_idx:end_idx]
        train_fold = pd.concat([ratings_shuffled.iloc[:start_idx], 
                                ratings_shuffled.iloc[end_idx:]])
        
        # Train on training fold
        recommender = SimpleRecommender(train_fold, movies_df)
        
        # Evaluate on validation fold
        predictions = []
        actuals = []
        
        for _, row in val_fold.iterrows():
            pred = recommender.predict(row['userId'], row['movieId'])
            if pred is not None:
                predictions.append(pred)
                actuals.append(row['rating'])
        
        if predictions:
            mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
            rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
            
            mae_scores.append(mae)
            rmse_scores.append(rmse)
            
            print(f"   MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    # Average across folds
    print(f"\n{'='*50}")
    print(f"CROSS-VALIDATION RESULTS ({k}-fold):")
    print(f"   Average MAE:  {np.mean(mae_scores):.4f} (±{np.std(mae_scores):.4f})")
    print(f"   Average RMSE: {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})")
    
    return mae_scores, rmse_scores


# ============================================
# MAIN PROGRAM WITH PROPER SPLITS
# ============================================

def main():
    print("\n" + "="*50)
    print("SIMPLE MOVIE RECOMMENDER")
    print("WITH PROPER TRAIN/TEST SPLIT")
    print("="*50)
    
    # Load data
    try:
        ratings = pd.read_csv('ratings.csv')
        movies = pd.read_csv('movies.csv')
        print(f"\nLoaded {len(ratings)} ratings, {len(movies)} movies")
    except FileNotFoundError:
        print("Error: Please put ratings.csv and movies.csv in the same folder")
        return
    
    # ============================================
    # PROPER TRAIN/TEST SPLIT
    # ============================================
    print("\n" + "-"*50)
    print("CREATING TRAIN/TEST SPLIT")
    print("-"*50)
    
    # Split data into training (80%) and testing (20%)
    train_ratings, test_ratings = train_test_split(
        ratings, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"Training set: {len(train_ratings)} ratings ({len(train_ratings)/len(ratings)*100:.1f}%)")
    print(f"Test set:     {len(test_ratings)} ratings ({len(test_ratings)/len(ratings)*100:.1f}%)")
    
    # Build recommender on training data only
    print("\n" + "-"*50)
    print("BUILDING RECOMMENDER ON TRAINING DATA")
    print("-"*50)
    recommender = SimpleRecommender(train_ratings, movies)
    
    # Main menu
    while True:
        print("\n" + "-"*50)
        print("MENU")
        print("-"*50)
        print("1. Get recommendations for a user")
        print("2. Predict rating for a movie")
        print("3. Find similar movies")
        print("4. Search for a movie")
        print("5. Evaluate on test data (single split)")
        print("6. Cross-validation evaluation (5-fold)")
        print("7. Show user's rated movies")
        print("0. Exit")
        
        choice = input("\nYour choice: ")
        
        if choice == '0':
            print("Goodbye!")
            break
        
        elif choice == '1':
            user = int(input("Enter user ID: "))
            results = recommender.recommend(user, 10)
            
            if results:
                print(f"\n🎬 Top recommendations for User {user}:")
                for i, movie in enumerate(results, 1):
                    print(f"\n{i}. {movie['title']}")
                    print(f"   Genres: {movie['genres']}")
                    print(f"   Predicted rating: {movie['predicted_rating']:.2f}")
            else:
                print("No recommendations found")
        
        elif choice == '2':
            user = int(input("Enter user ID: "))
            
            search = input("Search for movie: ")
            movies_found = recommender.search_movie(search)
            
            if movies_found.empty:
                print("No movies found")
                continue
            
            print("\nFound movies:")
            for _, row in movies_found.iterrows():
                print(f"  {row['movieId']}: {row['title']}")
            
            movie_id = int(input("\nEnter movie ID: "))
            pred = recommender.predict(user, movie_id)
            
            if pred:
                movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
                print(f"\nPredicted rating for '{movie_title}': {pred:.2f}/5.0")
            else:
                print("Can't predict (user has no ratings or movie not in training)")
        
        elif choice == '3':
            search = input("Search for movie: ")
            movies_found = recommender.search_movie(search)
            
            if movies_found.empty:
                print("No movies found")
                continue
            
            print("\nFound movies:")
            for _, row in movies_found.iterrows():
                print(f"  {row['movieId']}: {row['title']}")
            
            movie_id = int(input("\nEnter movie ID: "))
            similar = recommender.similar_movies(movie_id, 10)
            
            if similar:
                movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
                print(f"\n🎬 Movies similar to '{movie_title}':")
                for i, movie in enumerate(similar, 1):
                    print(f"\n{i}. {movie['title']}")
                    print(f"   Genres: {movie['genres']}")
                    print(f"   Similarity: {movie['similarity']:.3f}")
            else:
                print("No similar movies found")
        
        elif choice == '4':
            search = input("Search for movie: ")
            results = recommender.search_movie(search)
            
            if results.empty:
                print("No movies found")
            else:
                print(f"\nFound {len(results)} movies:")
                for _, row in results.iterrows():
                    print(f"\n  ID: {row['movieId']}")
                    print(f"  Title: {row['title']}")
                    print(f"  Genres: {row['genres']}")
        
        elif choice == '5':
            # Evaluate on held-out test data
            evaluate_recommender(train_ratings, test_ratings, movies)
        
        elif choice == '6':
            # Perform cross-validation
            evaluate_with_cross_validation(ratings, movies, k=5)
        
        elif choice == '7':
            user = int(input("Enter user ID: "))
            
            if user in recommender.user_ratings:
                ratings_list = recommender.user_ratings[user]
                top_rated = ratings_list.nlargest(10)
                
                print(f"\n📊 User {user}'s top rated movies (from training data):")
                for movie_id, rating in top_rated.items():
                    movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
                    stars = "⭐" * int(rating)
                    print(f"  {stars} {rating}/5 - {movie_title}")
                
                print(f"\nTotal ratings: {len(ratings_list)}")
                print(f"Average rating: {ratings_list.mean():.2f}")
            else:
                print("User not found in training data")
        
        else:
            print("Invalid choice")


# ============================================
# RUN THE PROGRAM
# ============================================

if __name__ == "__main__":
    main()