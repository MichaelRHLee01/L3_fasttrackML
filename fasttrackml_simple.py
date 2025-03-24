import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import time

# Path to your CSV file
CSV_FILE_PATH = "data/movie_ratings.csv"

# Set the tracking URI to the FastTrackML server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("movie-recommendation-sklearn")

def load_data(file_path):
    """Load movie ratings data from CSV file"""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} ratings from {df['user_id'].nunique()} users on {df['movie_id'].nunique()} movies")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def log_data_stats(df):
    """Log dataset statistics to FastTrackML"""
    with mlflow.start_run(run_name="data-analysis"):
        # Log dataset parameters
        mlflow.log_param("num_ratings", len(df))
        mlflow.log_param("num_users", df['user_id'].nunique())
        mlflow.log_param("num_movies", df['movie_id'].nunique())
        
        # Log rating statistics
        mlflow.log_metrics({
            "avg_rating": df['rate'].mean(),
            "rating_std": df['rate'].std(),
            "min_rating": df['rate'].min(),
            "max_rating": df['rate'].max(),
            "median_rating": df['rate'].median()
        })
        
        # Log rating distribution
        for rating in sorted(df['rate'].unique()):
            count = (df['rate'] == rating).sum()
            percentage = count / len(df) * 100
            mlflow.log_metric(f"rating_{rating}_count", count)
            mlflow.log_metric(f"rating_{rating}_percentage", percentage)
        
        print("Data statistics logged to FastTrackML")

def create_user_item_matrix(df, max_users=5000, max_items=5000):
    """Convert ratings dataframe to a user-item matrix"""
    # Get the most active users and most rated items to keep the matrix size manageable
    top_users = df['user_id'].value_counts().nlargest(max_users).index
    top_items = df['movie_id'].value_counts().nlargest(max_items).index
    
    # Filter the dataframe
    df_filtered = df[df['user_id'].isin(top_users) & df['movie_id'].isin(top_items)]
    
    # Create a mapping from user_id to index and movie_id to index
    user_mapping = {uid: i for i, uid in enumerate(top_users)}
    movie_mapping = {mid: i for i, mid in enumerate(top_items)}
    
    # Create the user-item matrix
    user_item_matrix = np.zeros((len(user_mapping), len(movie_mapping)))
    
    # Fill the matrix with ratings
    for _, row in df_filtered.iterrows():
        if row['user_id'] in user_mapping and row['movie_id'] in movie_mapping:
            user_idx = user_mapping[row['user_id']]
            movie_idx = movie_mapping[row['movie_id']]
            user_item_matrix[user_idx, movie_idx] = row['rate']
    
    return user_item_matrix, df_filtered, user_mapping, movie_mapping

def train_svd_models(df):
    """Train truncated SVD recommendation models with different hyperparameters"""
    # Create a smaller user-item matrix for demonstration
    print("Creating user-item matrix...")
    user_item_matrix, df_filtered, user_mapping, movie_mapping = create_user_item_matrix(df)
    print(f"Created matrix of shape {user_item_matrix.shape}")
    
    # Get non-zero elements for train/test split
    user_indices, movie_indices = np.nonzero(user_item_matrix)
    ratings = user_item_matrix[user_indices, movie_indices]
    
    # Create train/test indices
    indices = np.arange(len(user_indices))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Create train/test matrices
    train_matrix = user_item_matrix.copy()
    test_ratings = []
    
    # Remove test ratings from train matrix and save them
    for i in test_indices:
        u, m = user_indices[i], movie_indices[i]
        test_ratings.append((u, m, train_matrix[u, m]))
        train_matrix[u, m] = 0
    
    # Try different hyperparameters
    for n_components in [20, 50, 100]:
        run_name = f"svd-components{n_components}"
        print(f"\nTraining model: {run_name}")
        
        with mlflow.start_run(run_name=run_name):
            # Log hyperparameters
            mlflow.log_param("n_components", n_components)
            mlflow.log_param("matrix_shape", user_item_matrix.shape)
            mlflow.log_param("train_ratings", len(train_indices))
            mlflow.log_param("test_ratings", len(test_indices))
            
            # Create and train the model
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            
            # Track training time
            start_time = time.time()
            user_factors = svd.fit_transform(train_matrix)
            item_factors = svd.components_.T
            training_time = time.time() - start_time
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Get explained variance
            explained_var = svd.explained_variance_ratio_.sum()
            mlflow.log_metric("explained_variance", explained_var)
            
            # Make predictions on test set
            true_ratings = []
            predicted_ratings = []
            
            for u, m, r in test_ratings:
                true_ratings.append(r)
                pred = np.dot(user_factors[u], item_factors[m])
                predicted_ratings.append(pred)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
            mae = mean_absolute_error(true_ratings, predicted_ratings)
            
            # Log evaluation metrics
            mlflow.log_metrics({
                "test_rmse": rmse,
                "test_mae": mae
            })
            
            print(f"Model {run_name}: RMSE = {rmse:.4f}, MAE = {mae:.4f}, Explained Variance = {explained_var:.4f}")

def main():
    """Main function to run the FastTrackML demo"""
    print("Starting FastTrackML movie recommendation demo")
    
    # Load data
    ratings_df = load_data(CSV_FILE_PATH)
    if ratings_df is None:
        return
    
    # Log data statistics
    log_data_stats(ratings_df)
    
    # Train SVD models
    train_svd_models(ratings_df)
    
    print("\nDemo completed! Check the FastTrackML UI at http://localhost:5000")

if __name__ == "__main__":
    main()