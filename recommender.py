"""
Simple Movie Recommendation System
MovieLens Dataset - Item-Based Collaborative Filtering
WITH PROPER TRAIN/TEST SPLIT AND RECALL METRIC
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
                'predicted_rating': rating,
                'movieId': movie_id
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
# EVALUATION WITH RECALL METRIC
# ============================================

def evaluate_recommender(train_ratings, test_ratings, movies_df, k=10, rating_threshold=4.0):
    """
    Train on train_ratings, evaluate on test_ratings
    Includes Recall@k and Precision@k metrics
    """
    print("\n" + "="*50)
    print("EVALUATING WITH TRAIN/TEST SPLIT")
    print("="*50)
    
    # Build recommender using ONLY training data
    print(f"\nTraining on {len(train_ratings)} ratings...")
    recommender = SimpleRecommender(train_ratings, movies_df)
    
    # Group test ratings by user
    test_user_ratings = {}
    for _, row in test_ratings.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        rating = row['rating']
        
        if user_id not in test_user_ratings:
            test_user_ratings[user_id] = {}
        test_user_ratings[user_id][movie_id] = rating
    
    # Evaluate recommendations
    print(f"Testing on {len(test_user_ratings)} users...")
    
    precision_scores = []
    recall_scores = []
    mae_scores = []
    rmse_scores = []
    
    users_evaluated = 0
    
    for user_id, test_items in test_user_ratings.items():
        # Only evaluate users that are in training data
        if user_id not in recommender.user_ratings:
            continue
        
        # Get recommendations for this user
        recommendations = recommender.recommend(user_id, n=k)
        recommended_movie_ids = [rec['movieId'] for rec in recommendations]
        
        # Find relevant items from test data (ratings >= threshold)
        relevant_items = [movie_id for movie_id, rating in test_items.items() 
                         if rating >= rating_threshold]
        
        if not relevant_items:
            continue
        
        # Calculate hits (recommended AND relevant)
        hits = len(set(recommended_movie_ids) & set(relevant_items))
        
        # Precision@k = hits / k
        precision = hits / k
        
        # Recall@k = hits / total relevant items
        recall = hits / len(relevant_items)
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        
        # Also calculate MAE/RMSE for rating predictions
        for movie_id, actual_rating in test_items.items():
            pred_rating = recommender.predict(user_id, movie_id)
            if pred_rating is not None:
                mae_scores.append(abs(pred_rating - actual_rating))
                rmse_scores.append((pred_rating - actual_rating) ** 2)
        
        users_evaluated += 1
        
        # Limit number of users for speed (remove this line if you want all users)
        if users_evaluated >= 500:
            break
    
    # Calculate metrics
    print(f"\n📊 EVALUATION RESULTS (k={k}, threshold={rating_threshold}):")
    print(f"   Users evaluated: {users_evaluated}")
    
    if precision_scores:
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        print(f"   Precision@{k}: {avg_precision:.4f}")
        print(f"   Recall@{k}:    {avg_recall:.4f}")
        print(f"   F1-Score:      {2 * (avg_precision * avg_recall) / (avg_precision + avg_recall):.4f}")
    
    if mae_scores:
        avg_mae = np.mean(mae_scores)
        avg_rmse = np.sqrt(np.mean(rmse_scores))
        print(f"   MAE:  {avg_mae:.4f}")
        print(f"   RMSE: {avg_rmse:.4f}")
    
    return recommender, {
        'precision': np.mean(precision_scores) if precision_scores else 0,
        'recall': np.mean(recall_scores) if recall_scores else 0,
        'mae': np.mean(mae_scores) if mae_scores else 0,
        'rmse': np.sqrt(np.mean(rmse_scores)) if rmse_scores else 0
    }


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
        print("5. Evaluate on test data (with Recall & Precision)")
        print("6. Show user's rated movies")
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
            # Evaluate on held-out test data with recall
            k = input("Enter k for Recall@k (default 10): ")
            k = int(k) if k.strip() else 10
            
            threshold = input("Enter rating threshold for relevance (default 4.0): ")
            threshold = float(threshold) if threshold.strip() else 4.0
            
            evaluate_recommender(train_ratings, test_ratings, movies, k=k, rating_threshold=threshold)
        
        elif choice == '6':
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
