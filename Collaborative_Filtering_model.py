# Collaborative_Filtering_model.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import json
import os


# =======================
# Load User-Item Ratings
# =======================
def load_user_ratings():
    """
    Load ratings từ nhiều nguồn:
    1. User comments (restaurant_comments.json)
    2. Foody reviews (restaurants_reviews_new.json)
    3. User preferences (user_preferences.json)
    """
    ratings_data = []

    # 1. Load từ user comments
    if os.path.exists("restaurant_comments.json"):
        try:
            with open("restaurant_comments.json", 'r', encoding='utf-8') as f:
                comments = json.load(f)

            for res_id, comments_list in comments.items():
                for comment in comments_list:
                    ratings_data.append({
                        'user_id': f"user_{comment.get('user', 'anonymous')}",
                        'restaurant_id': int(res_id),
                        'rating': comment.get('rating', 5),
                        'source': 'user_comment'
                    })
        except:
            pass

    # 2. Load từ Foody reviews
    if os.path.exists("restaurants_reviews_new.json"):
        try:
            with open("restaurants_reviews_new.json", 'r', encoding='utf-8') as f:
                reviews = json.load(f)

            for review in reviews:
                ratings_data.append({
                    'user_id': review.get('user_id', 'anonymous'),
                    'restaurant_id': review.get('res_id', 0),
                    'rating': review.get('rating', 5),
                    'source': 'foody'
                })
        except:
            pass

    # 3. Load từ user preferences (liked restaurants)
    if os.path.exists("user_preferences.json"):
        try:
            with open("user_preferences.json", 'r', encoding='utf-8') as f:
                prefs = json.load(f)

            # Liked restaurants = rating 9
            for res_id in prefs.get('liked_restaurants', []):
                ratings_data.append({
                    'user_id': 'current_user',
                    'restaurant_id': res_id,
                    'rating': 9,
                    'source': 'preference'
                })

            # Viewed restaurants = rating 6
            for res_id in prefs.get('viewed_restaurants', [])[-10:]:  # 10 gần nhất
                if res_id not in prefs.get('liked_restaurants', []):
                    ratings_data.append({
                        'user_id': 'current_user',
                        'restaurant_id': res_id,
                        'rating': 6,
                        'source': 'preference'
                    })
        except:
            pass

    if not ratings_data:
        return pd.DataFrame(columns=['user_id', 'restaurant_id', 'rating'])

    df = pd.DataFrame(ratings_data)

    # Normalize ratings về scale 1-10
    df['rating'] = df['rating'].clip(1, 10)

    return df


# =======================
# Build User-Item Matrix
# =======================
def build_user_item_matrix(ratings_df):
    """
    Tạo ma trận User-Item từ ratings
    Rows: Users, Columns: Restaurants
    """
    if ratings_df.empty:
        return None, None, None

    # Tạo pivot table
    user_item_matrix = ratings_df.pivot_table(
        index='user_id',
        columns='restaurant_id',
        values='rating',
        fill_value=0
    )

    # Convert to sparse matrix để tiết kiệm memory
    sparse_matrix = csr_matrix(user_item_matrix.values)

    return user_item_matrix, sparse_matrix, user_item_matrix.index, user_item_matrix.columns


# =======================
# User-Based CF
# =======================
def calculate_user_similarity(user_item_matrix):
    """
    Tính similarity giữa các users
    """
    if user_item_matrix is None:
        return None

    # Cosine similarity giữa users
    user_similarity = cosine_similarity(user_item_matrix.values)

    return pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )


# =======================
# Item-Based CF
# =======================
def calculate_item_similarity(user_item_matrix):
    """
    Tính similarity giữa các items (restaurants)
    """
    if user_item_matrix is None:
        return None

    # Transpose để tính similarity giữa restaurants
    # Cosine similarity giữa columns (restaurants)
    item_similarity = cosine_similarity(user_item_matrix.T.values)

    return pd.DataFrame(
        item_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )


# =======================
# Recommendation Functions
# =======================
def get_cf_recommendations(user_id, user_item_matrix, item_similarity_df, n=10):
    """
    Gợi ý dựa trên Item-Based Collaborative Filtering

    Args:
        user_id: ID của user
        user_item_matrix: Ma trận user-item
        item_similarity_df: Ma trận similarity giữa items
        n: Số lượng gợi ý

    Returns:
        List of (restaurant_id, predicted_score)
    """
    if user_item_matrix is None or item_similarity_df is None:
        return []

    # Kiểm tra user có trong matrix không
    if user_id not in user_item_matrix.index:
        # User mới -> gợi ý popular items
        return get_popular_recommendations(user_item_matrix, n)

    # Lấy ratings của user
    user_ratings = user_item_matrix.loc[user_id]

    # Tìm restaurants user chưa rate (rating = 0)
    unrated_items = user_ratings[user_ratings == 0].index.tolist()

    if not unrated_items:
        return []

    # Predict ratings cho unrated items
    predictions = []

    for item_id in unrated_items:
        if item_id not in item_similarity_df.index:
            continue

        # Lấy similarity với các items đã rate
        rated_items = user_ratings[user_ratings > 0].index

        if len(rated_items) == 0:
            continue

        # Lấy similarity scores
        similarities = item_similarity_df.loc[item_id, rated_items]

        # Weighted average
        if similarities.sum() > 0:
            predicted_rating = (similarities * user_ratings[rated_items]).sum() / similarities.sum()
            predictions.append((item_id, predicted_rating))

    # Sort theo predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions[:n]


def get_popular_recommendations(user_item_matrix, n=10):
    """
    Gợi ý dựa trên popularity (cho cold start users)
    """
    if user_item_matrix is None:
        return []

    # Tính average rating cho mỗi restaurant
    avg_ratings = user_item_matrix.mean(axis=0)

    # Sort và lấy top n
    top_items = avg_ratings.nlargest(n)

    return [(item_id, score) for item_id, score in top_items.items()]


# =======================
# Main CF Model Class
# =======================
class CollaborativeFilteringModel:
    def __init__(self):
        self.ratings_df = None
        self.user_item_matrix = None
        self.user_similarity_df = None
        self.item_similarity_df = None
        self.is_trained = False

    def train(self):
        """
        Train CF model
        """
        print("Loading ratings data...")
        self.ratings_df = load_user_ratings()

        if self.ratings_df.empty:
            print("No ratings data found!")
            self.is_trained = False
            return False

        print(f"Loaded {len(self.ratings_df)} ratings")
        print(f"Users: {self.ratings_df['user_id'].nunique()}")
        print(f"Restaurants: {self.ratings_df['restaurant_id'].nunique()}")

        print("Building user-item matrix...")
        self.user_item_matrix, _, _, _ = build_user_item_matrix(self.ratings_df)

        if self.user_item_matrix is None:
            self.is_trained = False
            return False

        print("Calculating item similarity...")
        self.item_similarity_df = calculate_item_similarity(self.user_item_matrix)

        print("CF Model trained successfully!")
        self.is_trained = True
        return True

    def get_recommendations(self, user_id='current_user', n=10):
        """
        Lấy gợi ý cho user
        """
        if not self.is_trained:
            return []

        return get_cf_recommendations(
            user_id,
            self.user_item_matrix,
            self.item_similarity_df,
            n
        )


# =======================
# Utility Functions
# =======================
def load_cf_model():
    """
    Load và train CF model
    """
    model = CollaborativeFilteringModel()
    model.train()
    return model


# Test
if __name__ == "__main__":
    print("Testing Collaborative Filtering Model...")

    model = load_cf_model()

    if model.is_trained:
        print("\n" + "=" * 50)
        print("Getting recommendations for current_user...")
        recommendations = model.get_recommendations('current_user', n=10)

        print(f"\nTop 10 CF recommendations:")
        for i, (res_id, score) in enumerate(recommendations, 1):
            print(f"{i}. Restaurant ID: {res_id}, Score: {score:.3f}")
    else:
        print("Model training failed - not enough data")