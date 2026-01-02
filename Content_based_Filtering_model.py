# Content_based_Filtering_model.py
import itertools
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


# =======================
# Load & xử lý dữ liệu
# =======================
def load_and_prepare_data(json_path="./restaurants_with_coords.json"):
    """
    Load data và chuẩn bị features cho content-based filtering
    """
    data = pd.read_json(json_path)

    # Chọn features cần thiết
    features = [
        'id', 'name', 'address', 'district', 'city', 'category',
        'food_categories', 'main_opening_hour', 'main_closing_hour',
        'style', 'appropriate', 'suitable_time',
        'average_price_min', 'avarage_price_max', 'average_rating',
        'quality_rating', 'service_rating', 'price_rating',
        'location_rating', 'space_rating'
    ]

    # Lọc chỉ lấy các cột tồn tại
    existing_features = [f for f in features if f in data.columns]
    X = data[existing_features].copy()

    # Fill NA
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna('')
        else:
            X[col] = X[col].fillna(0)

    return X


# =======================
# Xây dựng feature vector
# =======================
def build_feature_matrix(X):
    """
    Tạo ma trận đặc trưng từ nhiều features:
    - Food categories (one-hot encoding)
    - Style (one-hot encoding)
    - Appropriate (one-hot encoding)
    - Suitable time (one-hot encoding)
    - District (one-hot encoding)
    - Price range (normalized)
    """

    # 1. Food Categories Matrix
    all_food_cats = list(set(itertools.chain.from_iterable(X['food_categories'])))
    food_matrix = np.zeros((len(X), len(all_food_cats)))

    for i, row in X.iterrows():
        for j, cat in enumerate(all_food_cats):
            if cat in row['food_categories']:
                food_matrix[i, j] = 1

    # 2. Style Matrix
    all_styles = list(set(itertools.chain.from_iterable(X['style'])))
    style_matrix = np.zeros((len(X), len(all_styles)))

    for i, row in X.iterrows():
        for j, style in enumerate(all_styles):
            if style in row['style']:
                style_matrix[i, j] = 1

    # 3. Appropriate Matrix
    all_appropriate = list(set(itertools.chain.from_iterable(X['appropriate'])))
    appropriate_matrix = np.zeros((len(X), len(all_appropriate)))

    for i, row in X.iterrows():
        for j, app in enumerate(all_appropriate):
            if app in row['appropriate']:
                appropriate_matrix[i, j] = 1

    # 4. Suitable Time Matrix
    if 'suitable_time' in X.columns:
        all_times = list(set(itertools.chain.from_iterable(X['suitable_time'])))
        time_matrix = np.zeros((len(X), len(all_times)))

        for i, row in X.iterrows():
            for j, time in enumerate(all_times):
                if time in row['suitable_time']:
                    time_matrix[i, j] = 1
    else:
        time_matrix = np.zeros((len(X), 1))

    # 5. District One-Hot
    districts = X['district'].unique()
    district_matrix = np.zeros((len(X), len(districts)))
    district_dict = {d: i for i, d in enumerate(districts)}

    for i, row in X.iterrows():
        if row['district'] in district_dict:
            district_matrix[i, district_dict[row['district']]] = 1

    # 6. Price Range (normalized)
    scaler = MinMaxScaler()
    price_matrix = scaler.fit_transform(
        X[['average_price_min', 'avarage_price_max']].values
    )

    # 7. Rating (normalized)
    if 'average_rating' in X.columns:
        rating_matrix = scaler.fit_transform(
            X[['average_rating']].values
        )
    else:
        rating_matrix = np.zeros((len(X), 1))

    # Kết hợp tất cả features với trọng số
    # Food categories quan trọng nhất (weight = 3)
    # Style, Appropriate (weight = 2)
    # District, Time (weight = 1.5)
    # Price, Rating (weight = 1)

    combined_matrix = np.hstack([
        food_matrix * 3,  # Món ăn quan trọng nhất
        style_matrix * 2,  # Phong cách
        appropriate_matrix * 2,  # Phù hợp với
        time_matrix * 1.5,  # Thời gian
        district_matrix * 1.5,  # Khu vực
        price_matrix * 1,  # Giá
        rating_matrix * 1  # Đánh giá
    ])

    return combined_matrix


# =======================
# Build similarity model
# =======================
def build_similarity_model(X):
    """
    Xây dựng ma trận cosine similarity
    """
    feature_matrix = build_feature_matrix(X)
    cosine_sim = cosine_similarity(feature_matrix, feature_matrix)
    return cosine_sim


# =======================
# Recommendation functions
# =======================
def recommend_restaurants(restaurant_id, X, cosine_sim, n=10):
    """
    Gợi ý n quán tương tự dựa trên restaurant_id

    Args:
        restaurant_id: ID của quán gốc
        X: DataFrame chứa data
        cosine_sim: Ma trận cosine similarity
        n: Số lượng gợi ý

    Returns:
        List of restaurant IDs (không bao gồm quán gốc)
    """
    try:
        # Lấy index của restaurant_id trong DataFrame
        idx = X[X['id'] == restaurant_id].index[0]

        # Lấy similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort theo điểm similarity (cao → thấp)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Bỏ chính nó (index 0) và lấy top n
        sim_scores = sim_scores[1:n + 1]

        # Lấy indices
        restaurant_indices = [i[0] for i in sim_scores]

        # Trả về list IDs
        return X.iloc[restaurant_indices]['id'].tolist()

    except IndexError:
        return []


def get_recommendations(name, X, cosine_sim, top_n=5):
    """
    Gợi ý quán dựa trên tên quán

    Args:
        name: Tên quán
        X: DataFrame
        cosine_sim: Ma trận similarity
        top_n: Số lượng gợi ý

    Returns:
        DataFrame chứa thông tin các quán gợi ý
    """
    # Tìm quán theo tên (case-insensitive)
    matches = X[X['name'].str.lower() == name.lower()]

    if len(matches) == 0:
        return pd.DataFrame()

    idx = matches.index[0]
    restaurant_id = X.loc[idx, 'id']

    # Dùng recommend_restaurants
    recommended_ids = recommend_restaurants(restaurant_id, X, cosine_sim, top_n)

    # Lấy thông tin các quán
    recommendations = X[X['id'].isin(recommended_ids)].copy()

    # Thêm cột similarity score
    for i, rec_id in enumerate(recommended_ids):
        rec_idx = X[X['id'] == rec_id].index[0]
        recommendations.loc[recommendations['id'] == rec_id, 'similarity'] = cosine_sim[idx][rec_idx]

    # Sort theo similarity
    recommendations = recommendations.sort_values('similarity', ascending=False)

    return recommendations[['name', 'district', 'address', 'category', 'food_categories', 'similarity']]


def get_recommendations_by_preferences(food_cats, districts, X, cosine_sim, top_n=10):
    """
    Gợi ý quán dựa trên preferences của user

    Args:
        food_cats: List các loại món ăn
        districts: List các quận
        X: DataFrame
        cosine_sim: Ma trận similarity
        top_n: Số lượng gợi ý

    Returns:
        List of restaurant IDs
    """
    # Filter quán match preferences
    filtered = X.copy()

    if food_cats:
        filtered = filtered[
            filtered['food_categories'].apply(
                lambda cats: any(cat in food_cats for cat in cats)
            )
        ]

    if districts:
        filtered = filtered[filtered['district'].isin(districts)]

    # Sort theo rating và lấy top
    filtered = filtered.sort_values('average_rating', ascending=False)

    return filtered.head(top_n)['id'].tolist()


# =======================
# Main function để test
# =======================
def load_data(json_path="./restaurants_with_coords.json"):
    """
    Load data và build model
    """
    X = load_and_prepare_data(json_path)
    cosine_sim = build_similarity_model(X)
    return X, cosine_sim


# Test
if __name__ == "__main__":
    X, cosine_sim = load_data()
    print(f"Loaded {len(X)} restaurants")
    print(f"Similarity matrix shape: {cosine_sim.shape}")

    # Test recommendation
    if len(X) > 0:
        test_id = X.iloc[0]['id']
        test_name = X.iloc[0]['name']
        print(f"\nTest với quán: {test_name}")

        recs = recommend_restaurants(test_id, X, cosine_sim, n=5)
        print(f"Recommended IDs: {recs}")

        recs_df = get_recommendations(test_name, X, cosine_sim, top_n=5)
        print("\nRecommendations:")
        print(recs_df[['name', 'district', 'similarity']])
