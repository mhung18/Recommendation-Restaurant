# Content_based_Filtering_model.py
import itertools

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# =======================
# Load & xử lý dữ liệu
# =======================
def load_and_prepare_data(json_path="./restaurants.json"):
    data = pd.read_json(json_path)
    features = [
        'id', 'name', 'address', 'district', 'city', 'category',
        'food_categories', 'main_opening_hour', 'main_closing_hour',
        'style', 'appropriate', 'average_price_min', 'avarage_price_max', 'average_rating'
    ]
    X = data[features].copy()
    X = X.fillna('')
    # "mở phẳng" danh sách con trong mỗi hàng
    all_items = list(itertools.chain.from_iterable(data["food_categories"]))

    # lấy danh sách các giá trị duy nhất
    categories = list(set(all_items))

    # Tạo cột kết hợp đặc trưng
    def combine_features(row):
        # return f"{row['name']} {row['district']} {row['category']} {row['food_categories']} {row['style']} {row['appropriate']}"
        resCatMatrix = []
        food_list = row['food_categories']  # Lấy danh sách món ăn của quán đó
        for x in categories:
            if (x in food_list):
                resCatMatrix.append(1)
            else:
                resCatMatrix.append(0)
        return resCatMatrix

    X.loc[:, 'combined'] = X.apply(combine_features, axis=1)
    return X


# =======================
# Huấn luyện mô hình TF-IDF
# =======================
# def build_similarity_model(X):
#     tfidf = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = tfidf.fit_transform(X['combined'])
#     cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
#     return tfidf, cosine_sim

def build_similarity_model(X):
    # chuyển list thành ma trận numpy
    food_matrix = np.array(X['combined'].tolist())
    # tính cosine similarity giữa các quán ăn
    cosine_sim = cosine_similarity(food_matrix, food_matrix)
    return cosine_sim


# =======================
# Hàm gợi ý quán tương tự
# =======================
# def recommend_similar_places(X, cosine_sim, place_name, top_n=5):
#     idx = X[X['name'].str.lower() == place_name.lower()].index
#     if len(idx) == 0:
#         return pd.DataFrame(columns=['name', 'district', 'category', 'food_categories', 'style'])
#
#     idx = idx[0]
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:top_n + 1]
#     place_indices = [i[0] for i in sim_scores]
#     return X.iloc[place_indices][['name', 'district', 'category', 'food_categories', 'style']]

def get_recommendations(name, cosine_sim, X, top_n=5):
    indices = pd.Series(X.index, index=X['name']).drop_duplicates()

    if name not in indices:
        return pd.DataFrame()

    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    restaurant_indices = [i[0] for i in sim_scores]

    # ⚠️ Thêm luôn các cột bạn cần hiển thị:
    recommendations = X.iloc[restaurant_indices][['name', 'district', 'address', 'category', 'food_categories']]
    recommendations = recommendations.copy()
    recommendations['similarity'] = [i[1] for i in sim_scores]

    return recommendations



def load_data():
    X = load_and_prepare_data("./restaurants.json")
    cosine_sim = build_similarity_model(X)
    return X, cosine_sim

load_data()