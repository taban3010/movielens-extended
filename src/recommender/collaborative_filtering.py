"""
Core Collaborative Filtering Recommender System
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')


class CollaborativeFilteringRecommender:
    """
    A comprehensive collaborative filtering recommendation system.
    
    This class implements both user-based and item-based collaborative filtering
    algorithms with various similarity metrics and evaluation capabilities.
    
    Attributes:
        ratings_matrix (pd.DataFrame): User-item ratings matrix
        user_similarity (np.ndarray): User-user similarity matrix
        item_similarity (np.ndarray): Item-item similarity matrix
        user_mean_ratings (pd.Series): Mean rating for each user
        item_mean_ratings (pd.Series): Mean rating for each item
        global_mean (float): Global mean rating across all users and items
    """
    
    def __init__(self):
        """
        Initialize the recommender system.
        
        Sets all matrices and statistics to None, which will be populated
        when data is loaded using the load_data method.
        """
        self.ratings_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.user_mean_ratings = None
        self.item_mean_ratings = None
        self.global_mean = None
        
    def load_data(self, ratings_data):
        """
        Load and preprocess the ratings data.
        
        Converts raw ratings data into a user-item matrix and calculates
        basic statistics like mean ratings per user, item, and globally.
        
        Args:
            ratings_data (list): List of dictionaries containing user ratings.
                                Each dictionary should have 'user_id' and 'ratings' keys.
                                'ratings' should be a list of dicts with 'movie_id' and 'rating'.
            
        Returns:
            pd.DataFrame: Processed ratings matrix with users as rows and movies as columns.
        """
        # Convert to DataFrame
        rows = []
        for user_data in ratings_data:
            user_id = int(user_data['user_id'])
            for rating in user_data['ratings']:
                rows.append({
                    'user_id': user_id,
                    'movie_id': rating['movie_id'],
                    'rating': rating['rating']
                })
        
        df = pd.DataFrame(rows)
        
        # Create user-item matrix
        self.ratings_matrix = df.pivot_table(
            index='user_id', 
            columns='movie_id', 
            values='rating', 
            fill_value=0
        )
        
        # Calculate mean ratings
        self.user_mean_ratings = self.ratings_matrix.replace(0, np.nan).mean(axis=1)
        self.item_mean_ratings = self.ratings_matrix.replace(0, np.nan).mean(axis=0)
        self.global_mean = df['rating'].mean()
        
        print(f"✓ Data loaded successfully!")
        print(f"  - Users: {len(self.ratings_matrix)}")
        print(f"  - Movies: {len(self.ratings_matrix.columns)}")
        print(f"  - Total ratings: {len(df)}")
        print(f"  - Sparsity: {(1 - len(df) / (len(self.ratings_matrix) * len(self.ratings_matrix.columns))) * 100:.2f}%")
        
        return self.ratings_matrix
    
    def calculate_user_similarity(self, method='cosine'):
        """
        Calculate user-user similarity matrix using specified method.
        
        Computes similarity between all pairs of users based on their rating patterns.
        The similarity matrix is stored in self.user_similarity for later use.
        
        Args:
            method (str): Similarity metric to use. Options:
                         - 'cosine': Cosine similarity (default)
                         - 'pearson': Pearson correlation coefficient
                         - 'jaccard': Jaccard similarity (binary overlap)
            
        Returns:
            np.ndarray: Square matrix where element [i,j] represents similarity 
                       between user i and user j.
                       
        Raises:
            ValueError: If an unsupported similarity method is specified.
        """
        if self.ratings_matrix is None:
            raise ValueError("Data must be loaded before calculating similarity")
            
        if method == 'cosine':
            # Replace 0s with NaN for proper cosine calculation
            matrix = self.ratings_matrix.replace(0, np.nan)
            matrix_filled = matrix.fillna(0)
            self.user_similarity = cosine_similarity(matrix_filled)
        
        elif method == 'pearson':
            # Calculate Pearson correlation
            matrix = self.ratings_matrix.replace(0, np.nan)
            self.user_similarity = matrix.T.corr().fillna(0).values
        
        elif method == 'jaccard':
            # Binary Jaccard similarity
            binary_matrix = (self.ratings_matrix > 0).astype(int)
            intersection = np.dot(binary_matrix, binary_matrix.T)
            union = np.sum(binary_matrix, axis=1).reshape(-1, 1) + np.sum(binary_matrix, axis=1) - intersection
            self.user_similarity = intersection / union
            self.user_similarity = np.nan_to_num(self.user_similarity)
        
        else:
            raise ValueError(f"Unsupported similarity method: {method}")
        
        print(f"✓ User similarity calculated using {method} method")
        return self.user_similarity
    
    def calculate_item_similarity(self, method='cosine'):
        """
        Calculate item-item similarity matrix using specified method.
        
        Computes similarity between all pairs of items based on user rating patterns.
        The similarity matrix is stored in self.item_similarity for later use.
        
        Args:
            method (str): Similarity metric to use. Options:
                         - 'cosine': Cosine similarity (default)
                         - 'pearson': Pearson correlation coefficient
                         - 'jaccard': Jaccard similarity (binary overlap)
            
        Returns:
            np.ndarray: Square matrix where element [i,j] represents similarity 
                       between item i and item j.
                       
        Raises:
            ValueError: If an unsupported similarity method is specified.
        """
        if self.ratings_matrix is None:
            raise ValueError("Data must be loaded before calculating similarity")
            
        if method == 'cosine':
            matrix = self.ratings_matrix.replace(0, np.nan)
            matrix_filled = matrix.fillna(0)
            self.item_similarity = cosine_similarity(matrix_filled.T)
        
        elif method == 'pearson':
            matrix = self.ratings_matrix.replace(0, np.nan)
            self.item_similarity = matrix.corr().fillna(0).values
        
        elif method == 'jaccard':
            binary_matrix = (self.ratings_matrix > 0).astype(int)
            intersection = np.dot(binary_matrix.T, binary_matrix)
            union = np.sum(binary_matrix, axis=0).reshape(-1, 1) + np.sum(binary_matrix, axis=0) - intersection
            self.item_similarity = intersection / union
            self.item_similarity = np.nan_to_num(self.item_similarity)
        
        else:
            raise ValueError(f"Unsupported similarity method: {method}")
        
        print(f"✓ Item similarity calculated using {method} method")
        return self.item_similarity
    
    def predict_user_based(self, user_id, movie_id, k=5):
        """
        Predict rating using user-based collaborative filtering.
        
        Finds similar users who have rated the target movie and computes
        a weighted average of their ratings based on user similarity scores.
        
        Args:
            user_id (int): ID of the target user for prediction
            movie_id (int): ID of the target movie to predict rating for
            k (int): Number of most similar users to consider (default: 5)
            
        Returns:
            float: Predicted rating value between 1-5, or fallback values:
                  - Global mean if user not found
                  - User mean if movie not found or no similar users available
        """
        if self.user_similarity is None:
            raise ValueError("User similarity must be calculated before prediction")
            
        if user_id not in self.ratings_matrix.index:
            return self.global_mean
        
        if movie_id not in self.ratings_matrix.columns:
            return self.user_mean_ratings[user_id]
        
        user_idx = self.ratings_matrix.index.get_loc(user_id)
        movie_idx = self.ratings_matrix.columns.get_loc(movie_id)
        
        # Get users who have rated this movie
        movie_ratings = self.ratings_matrix.iloc[:, movie_idx]
        rated_users = movie_ratings[movie_ratings > 0].index
        
        if len(rated_users) == 0:
            return self.user_mean_ratings[user_id]
        
        # Get similarities and ratings for users who rated this movie
        similarities = []
        ratings = []
        
        for other_user in rated_users:
            if other_user != user_id:
                other_idx = self.ratings_matrix.index.get_loc(other_user)
                sim = self.user_similarity[user_idx, other_idx]
                if sim > 0:
                    similarities.append(sim)
                    ratings.append(movie_ratings[other_user])
        
        if len(similarities) == 0:
            return self.user_mean_ratings[user_id]
        
        # Get top-k similar users
        sim_ratings = list(zip(similarities, ratings))
        sim_ratings.sort(reverse=True)
        top_k = sim_ratings[:k]
        
        # Calculate weighted average
        numerator = sum(sim * rating for sim, rating in top_k)
        denominator = sum(sim for sim, rating in top_k)
        
        if denominator == 0:
            return self.user_mean_ratings[user_id]
        
        return numerator / denominator
    
    def predict_item_based(self, user_id, movie_id, k=5):
        """
        Predict rating using item-based collaborative filtering.
        
        Finds similar movies that the user has rated and computes
        a weighted average of their ratings based on item similarity scores.
        
        Args:
            user_id (int): ID of the target user for prediction
            movie_id (int): ID of the target movie to predict rating for
            k (int): Number of most similar items to consider (default: 5)
            
        Returns:
            float: Predicted rating value between 1-5, or fallback values:
                  - Global mean if user not found
                  - User mean if movie not found
                  - Item mean if no similar items available
        """
        if self.item_similarity is None:
            raise ValueError("Item similarity must be calculated before prediction")
            
        if user_id not in self.ratings_matrix.index:
            return self.global_mean
        
        if movie_id not in self.ratings_matrix.columns:
            return self.user_mean_ratings[user_id]
        
        user_ratings = self.ratings_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index
        
        if len(rated_movies) == 0:
            return self.global_mean
        
        movie_idx = self.ratings_matrix.columns.get_loc(movie_id)
        
        # Get similarities and ratings for movies rated by this user
        similarities = []
        ratings = []
        
        for other_movie in rated_movies:
            if other_movie != movie_id:
                other_idx = self.ratings_matrix.columns.get_loc(other_movie)
                sim = self.item_similarity[movie_idx, other_idx]
                if sim > 0:
                    similarities.append(sim)
                    ratings.append(user_ratings[other_movie])
        
        if len(similarities) == 0:
            return self.item_mean_ratings[movie_id]
        
        # Get top-k similar items
        sim_ratings = list(zip(similarities, ratings))
        sim_ratings.sort(reverse=True)
        top_k = sim_ratings[:k]
        
        # Calculate weighted average
        numerator = sum(sim * rating for sim, rating in top_k)
        denominator = sum(sim for sim, rating in top_k)
        
        if denominator == 0:
            return self.item_mean_ratings[movie_id]
        
        return numerator / denominator
    
    def get_recommendations(self, user_id, method='user_based', n_recommendations=5, k=5):
        """
        Generate movie recommendations for a specific user.
        
        Predicts ratings for all unrated movies and returns the top-N
        recommendations sorted by predicted rating score.
        
        Args:
            user_id (int): ID of the target user to generate recommendations for
            method (str): Recommendation approach to use:
                         - 'user_based': Use user-based collaborative filtering
                         - 'item_based': Use item-based collaborative filtering
            n_recommendations (int): Number of top recommendations to return (default: 5)
            k (int): Number of neighbors to consider in similarity calculations (default: 5)
            
        Returns:
            list: List of tuples (movie_id, predicted_rating) sorted by predicted rating
                 in descending order. Empty list if user not found.
        """
        if user_id not in self.ratings_matrix.index:
            print(f"User {user_id} not found in the dataset")
            return []
        
        user_ratings = self.ratings_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        predictions = []
        for movie_id in unrated_movies:
            if method == 'user_based':
                pred = self.predict_user_based(user_id, movie_id, k)
            elif method == 'item_based':
                pred = self.predict_item_based(user_id, movie_id, k)
            else:
                raise ValueError(f"Unsupported method: {method}")
            predictions.append((movie_id, pred))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def evaluate_predictions(self, test_set, method='user_based', k=5):
        """
        Evaluate the recommendation system using a test set.
        
        Computes prediction accuracy metrics by comparing predicted ratings
        with actual ratings from a held-out test set.
        
        Args:
            test_set (list): List of (user_id, movie_id, actual_rating) tuples
                           representing known ratings to evaluate against
            method (str): Prediction method to use ('user_based' or 'item_based')
            k (int): Number of neighbors to consider in predictions (default: 5)
            
        Returns:
            dict: Dictionary containing evaluation metrics:
                 - 'RMSE': Root Mean Square Error
                 - 'MAE': Mean Absolute Error
        """
        predictions = []
        actuals = []
        
        for user_id, movie_id, actual_rating in test_set:
            if method == 'user_based':
                pred = self.predict_user_based(user_id, movie_id, k)
            elif method == 'item_based':
                pred = self.predict_item_based(user_id, movie_id, k)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            predictions.append(pred)
            actuals.append(actual_rating)
        
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        return {'RMSE': rmse, 'MAE': mae}
    
    def extended_algorithm(self, user_id, n_recommendations=5, **kwargs):
        """
        Extended Matrix Factorization Algorithm for MovieLens.
        
        This method implements Singular Value Decomposition (SVD) matrix factorization
        combined with bias terms to create a more sophisticated recommendation system
        than basic collaborative filtering.
        
        Args:
            user_id (int): Target user ID for recommendations
            n_recommendations (int): Number of recommendations to return
            **kwargs: Additional parameters:
                - n_factors (int): Number of latent factors (default: 20)
                - n_epochs (int): Number of training iterations (default: 20)
                - lr (float): Learning rate (default: 0.01)
                - reg (float): Regularization parameter (default: 0.1)
            
        Returns:
            list: List of tuples (movie_id, predicted_rating) sorted by rating
            
        Algorithm Description:
            The extended algorithm uses matrix factorization to decompose the user-item
            matrix into latent factor matrices, capturing hidden patterns in user
            preferences and item characteristics. This approach handles sparsity better
            than traditional collaborative filtering and can discover latent factors
            like genres, directors, or themes.
            
        Mathematical Foundation:
            R ≈ P × Q^T + user_bias + item_bias + global_bias
            where P is user factor matrix, Q is item factor matrix
        """
        from sklearn.decomposition import TruncatedSVD
        
        # Parameters
        n_factors = kwargs.get('n_factors', 20)
        n_epochs = kwargs.get('n_epochs', 20)
        lr = kwargs.get('lr', 0.01)
        reg = kwargs.get('reg', 0.1)
        
        print(f"🚀 Running Extended Matrix Factorization Algorithm")
        print(f"   User: {user_id}, Factors: {n_factors}, Epochs: {n_epochs}")
        
        if user_id not in self.ratings_matrix.index:
            print(f"   User {user_id} not found in dataset")
            return []
        
        # Create training matrix (replace 0s with NaN for proper handling)
        R = self.ratings_matrix.copy()
        
        # Use SVD for matrix factorization
        # Replace 0s with mean ratings for SVD
        R_filled = R.copy()
        for i in range(len(R)):
            for j in range(len(R.columns)):
                if R.iloc[i, j] == 0:
                    R_filled.iloc[i, j] = self.user_mean_ratings.iloc[i] if not np.isnan(self.user_mean_ratings.iloc[i]) else self.global_mean
        
        # Apply SVD
        svd = TruncatedSVD(n_components=n_factors, random_state=42)
        user_factors = svd.fit_transform(R_filled)
        item_factors = svd.components_.T
        
        # Calculate predictions for user
        user_idx = self.ratings_matrix.index.get_loc(user_id)
        user_vector = user_factors[user_idx]
        
        # Get user's unrated movies
        user_ratings = self.ratings_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        predictions = []
        for movie_id in unrated_movies:
            movie_idx = self.ratings_matrix.columns.get_loc(movie_id)
            movie_vector = item_factors[movie_idx]
            
            # Predict rating with bias terms
            base_prediction = np.dot(user_vector, movie_vector)
            user_bias = self.user_mean_ratings[user_id] - self.global_mean
            item_bias = self.item_mean_ratings[movie_id] - self.global_mean
            
            predicted_rating = self.global_mean + user_bias + item_bias + base_prediction
            
            # Clip to valid rating range
            predicted_rating = np.clip(predicted_rating, 1, 5)
            predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   ✓ Generated {len(predictions[:n_recommendations])} recommendations using SVD")
        return predictions[:n_recommendations]
