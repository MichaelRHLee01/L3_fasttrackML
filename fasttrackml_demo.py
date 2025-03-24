import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

# Path to your CSV file - update this to your actual file path
CSV_FILE_PATH = "data/movie_ratings.csv"  # Update this to match your file path

# Set the tracking URI to the FastTrackML server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("movie-recommendation-csv")

def load_data(file_path):
    """Load movie ratings data from CSV file"""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} ratings")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Preprocess the movie ratings data"""
    print("Preprocessing data...")
    
    # Convert timestamp to datetime if needed
    if 'timestamp' in df.columns:
        # Use a more flexible parsing approach for timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format='mixed')
    
    # Create user and movie indices for matrix factorization
    user_ids = df['user_id'].unique()
    movie_ids = df['movie_id'].unique()
    
    user_id_to_index = {user_id: i for i, user_id in enumerate(user_ids)}
    movie_id_to_index = {movie_id: i for i, movie_id in enumerate(movie_ids)}
    
    df['user_idx'] = df['user_id'].map(user_id_to_index)
    df['movie_idx'] = df['movie_id'].map(movie_id_to_index)
    
    return df, user_ids, movie_ids, user_id_to_index, movie_id_to_index

def simple_matrix_factorization(ratings, n_users, n_movies, n_factors=20, learning_rate=0.01, 
                              regularization=0.01, n_epochs=20):
    """Train a simple matrix factorization model and log metrics with FastTrackML"""
    
    # Start an MLflow run
    with mlflow.start_run(run_name=f"matrix-factorization-{time.strftime('%Y%m%d-%H%M%S')}"):
        # Log parameters
        mlflow.log_param("n_factors", n_factors)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("regularization", regularization)
        mlflow.log_param("n_epochs", n_epochs)
        mlflow.log_param("n_users", n_users)
        mlflow.log_param("n_movies", n_movies)
        mlflow.log_param("ratings_count", len(ratings))
        
        # Initialize user and item latent factors
        user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
        movie_factors = np.random.normal(0, 0.1, (n_movies, n_factors))
        
        # Split into train/validation
        train_data, val_data = train_test_split(ratings, test_size=0.2, random_state=42)
        
        # Training loop
        for epoch in range(n_epochs):
            # Shuffle training data
            train_data = train_data.sample(frac=1).reset_index(drop=True)
            
            # Training metrics
            train_predictions = []
            train_true_ratings = []
            
            # SGD update
            for _, row in train_data.iterrows():
                u = row['user_idx']
                i = row['movie_idx']
                r = row['rate']
                
                # Predict rating
                pred = np.dot(user_factors[u], movie_factors[i])
                train_predictions.append(pred)
                train_true_ratings.append(r)
                
                # Error
                err = r - pred
                
                # Update factors
                user_factors[u] += learning_rate * (err * movie_factors[i] - regularization * user_factors[u])
                movie_factors[i] += learning_rate * (err * user_factors[u] - regularization * movie_factors[i])
            
            # Calculate training metrics
            train_rmse = np.sqrt(mean_squared_error(train_true_ratings, train_predictions))
            train_mae = mean_absolute_error(train_true_ratings, train_predictions)
            
            # Validation metrics
            val_pred = []
            val_true = []
            for _, row in val_data.iterrows():
                u = row['user_idx']
                i = row['movie_idx']
                val_pred.append(np.dot(user_factors[u], movie_factors[i]))
                val_true.append(row['rate'])
            
            val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
            val_mae = mean_absolute_error(val_true, val_pred)
            
            # Log metrics
            mlflow.log_metrics({
                "train_rmse": train_rmse,
                "train_mae": train_mae,
                "val_rmse": val_rmse,
                "val_mae": val_mae
            }, step=epoch)
            
            print(f"Epoch {epoch+1}/{n_epochs}: Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}")
        
        # Log the final model as an artifact
        model_info = {
            "user_factors": user_factors,
            "movie_factors": movie_factors,
            "user_id_to_index": user_id_to_index,
            "movie_id_to_index": movie_id_to_index
        }
        
        # In a real implementation, you would use mlflow.sklearn.log_model() or similar
        # Here we just log the metrics for demonstration
        
        mlflow.log_metric("final_train_rmse", train_rmse)
        mlflow.log_metric("final_val_rmse", val_rmse)
        mlflow.log_metric("final_train_mae", train_mae)
        mlflow.log_metric("final_val_mae", val_mae)
        
        return user_factors, movie_factors, model_info

def train_model_with_different_hyperparams():
    """Train multiple models with different hyperparameters to demonstrate FastTrackML tracking"""
    
    # Load and preprocess data
    ratings_df = load_data(CSV_FILE_PATH)
    
    if ratings_df is None:
        return
    
    processed_df, user_ids, movie_ids, user_id_to_index, movie_id_to_index = preprocess_data(ratings_df)
    n_users = len(user_ids)
    n_movies = len(movie_ids)
    
    print(f"Training models with {n_users} users and {n_movies} movies")
    
    # Try different hyperparameter combinations
    for n_factors in [10, 20, 30]:
        for learning_rate in [0.01, 0.05]:
            for reg in [0.01, 0.1]:
                print(f"\nTraining model with n_factors={n_factors}, lr={learning_rate}, reg={reg}")
                user_factors, movie_factors, model_info = simple_matrix_factorization(
                    processed_df, n_users, n_movies,
                    n_factors=n_factors, 
                    learning_rate=learning_rate,
                    regularization=reg,
                    n_epochs=10  # Using fewer epochs for demonstration
                )
                
                # Here, you would typically save the model
                print("Model training complete")

if __name__ == "__main__":
    print("Starting FastTrackML demo with movie ratings CSV data")
    train_model_with_different_hyperparams()
    print("Demo completed. Check the FastTrackML UI at http://localhost:5000")