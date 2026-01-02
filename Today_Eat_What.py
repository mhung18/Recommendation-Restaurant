import streamlit as st
import pandas as pd
import json
import os
import re
import time
from datetime import datetime

from Content_based_Filtering_model import (
    load_and_prepare_data,
    build_similarity_model,
    recommend_restaurants
)
from Collaborative_Filtering_model import load_cf_model

st.set_page_config(
    page_title="Hôm nay ăn gì?",
    page_icon="🍽️",
    layout="wide"
)


# ----------------------
# LOAD DATA & MODEL
# ----------------------
@st.cache_data
def load_data():
    X = load_and_prepare_data("./restaurants_with_coords.json")
    cosine_sim = build_similarity_model(X)
    return X, cosine_sim


@st.cache_resource
def load_cf():
    """Load Collaborative Filtering model"""
    cf_model = load_cf_model()
    return cf_model


@st.cache_data
def load_full_data():
    """Load file JSON gốc để có đầy đủ thông tin"""
    with open("./restaurants_with_coords.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)


X, cosine_sim = load_data()
cf_model = load_cf()
full_df = load_full_data()

# ----------------------
# USER PREFERENCE FUNCTIONS
# ----------------------
USER_PREFS_FILE = "user_preferences.json"

# Initialize session state for preferences
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = None


def load_user_preferences():
    """Load preferences từ file hoặc session state"""

    # Ưu tiên session state (trong memory)
    if st.session_state.user_preferences is not None:
        return st.session_state.user_preferences

    # Nếu không có, load từ file
    if not os.path.exists(USER_PREFS_FILE):
        default_prefs = {
            "favorite_categories": [],
            "favorite_districts": [],
            "price_range": [0, 500000],
            "viewed_restaurants": [],
            "liked_restaurants": []
        }
        st.session_state.user_preferences = default_prefs
        return default_prefs

    try:
        with open(USER_PREFS_FILE, 'r', encoding='utf-8') as f:
            prefs = json.load(f)
            st.session_state.user_preferences = prefs
            return prefs
    except:
        default_prefs = {
            "favorite_categories": [],
            "favorite_districts": [],
            "price_range": [0, 500000],
            "viewed_restaurants": [],
            "liked_restaurants": []
        }
        st.session_state.user_preferences = default_prefs
        return default_prefs


def save_user_preferences(prefs):
    """Lưu preferences với dual storage: session state + file"""

    # Convert tất cả int64 sang int trước khi lưu
    def convert_to_native_types(obj):
        """Convert numpy/pandas types sang native Python types"""
        if isinstance(obj, dict):
            return {k: convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        else:
            return obj

    prefs = convert_to_native_types(prefs)

    # 1. Lưu vào session state (LUÔN THÀNH CÔNG)
    st.session_state.user_preferences = prefs

    # 2. Thử lưu vào file (không bắt buộc)
    try:
        # Kiểm tra thư mục tồn tại
        directory = os.path.dirname(USER_PREFS_FILE) or '.'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Thử ghi file
        with open(USER_PREFS_FILE, 'w', encoding='utf-8') as f:
            json.dump(prefs, f, ensure_ascii=False, indent=2)

        return True

    except Exception as e:
        # Silent fail - session state vẫn work
        return True


def add_to_history(restaurant_id, action="viewed"):
    """Thêm quán vào lịch sử"""
    prefs = load_user_preferences()

    # Convert sang int chuẩn (tránh int64 từ pandas)
    restaurant_id = int(restaurant_id)

    if action == "viewed":
        if restaurant_id not in prefs["viewed_restaurants"]:
            prefs["viewed_restaurants"].append(restaurant_id)
            # Giữ tối đa 50 quán gần nhất
            prefs["viewed_restaurants"] = prefs["viewed_restaurants"][-50:]

    elif action == "liked":
        if restaurant_id not in prefs["liked_restaurants"]:
            prefs["liked_restaurants"].append(restaurant_id)
            # Đồng thời xóa khỏi viewed nếu có
            if restaurant_id in prefs["viewed_restaurants"]:
                prefs["viewed_restaurants"].remove(restaurant_id)

    return save_user_preferences(prefs)


def district_sort_key(name):
    if name.startswith("Quận"):
        match = re.search(r"\d+", name)
        if match:
            return (0, int(match.group()))
        else:
            return (0, 999)  # fallback nếu có Quận nhưng không có số
    else:
        return (1, name)


# ----------------------
# HYBRID RECOMMENDATION ENGINE
# ----------------------
def get_hybrid_recommendations(user_prefs, X, full_df, cosine_sim, cf_model, n=12, cf_weight=0.4, cb_weight=0.6):
    """
    Hybrid Recommendation: 40% CF + 60% CB
    """
    hybrid_scores = {}

    print(f"[DEBUG] CF Model trained: {cf_model.is_trained}")

    # ==================
    # 1. GET CF RECOMMENDATIONS (40%)
    # ==================
    if cf_model.is_trained:
        print("[DEBUG] Getting CF recommendations...")
        cf_recs = cf_model.get_recommendations('current_user', n=n * 2)
        print(f"[DEBUG] CF returned {len(cf_recs)} recommendations")

        # Normalize CF scores to 0-1
        if cf_recs:
            max_cf_score = max([score for _, score in cf_recs])
            min_cf_score = min([score for _, score in cf_recs])

            print(f"[DEBUG] CF score range: {min_cf_score:.3f} - {max_cf_score:.3f}")

            if max_cf_score > min_cf_score:
                for res_id, score in cf_recs:
                    normalized_score = (score - min_cf_score) / (max_cf_score - min_cf_score)
                    hybrid_scores[res_id] = {
                        'cf_score': normalized_score * cf_weight,
                        'cb_score': 0,
                        'reason_cf': 'Dựa trên sở thích người dùng tương tự',
                        'type': 'cf'
                    }
                    print(f"[DEBUG] CF: Restaurant {res_id}, score: {normalized_score:.3f}")
    else:
        print("[DEBUG] CF Model not trained - skipping CF recommendations")

    # ==================
    # 2. GET CB RECOMMENDATIONS (60%)
    # ==================
    cb_candidates = []

    print(f"[DEBUG] Getting CB recommendations...")
    print(f"[DEBUG] User liked restaurants: {user_prefs['liked_restaurants']}")

    # Strategy A: Content-Based từ quán đã thích
    if user_prefs["liked_restaurants"]:
        for rest_id in user_prefs["liked_restaurants"][-3:]:
            if rest_id in X.index:
                similar = recommend_restaurants(rest_id, X, cosine_sim, n=10)
                print(f"[DEBUG] CB from liked {rest_id}: found {len(similar)} similar")
                for idx in similar:
                    if idx not in user_prefs["viewed_restaurants"]:
                        cb_candidates.append({
                            'id': idx,
                            'score': 0.95,
                            'reason': f"Tương tự quán bạn đã thích"
                        })

    # Strategy B: Filter theo sở thích
    if user_prefs["favorite_categories"]:
        filtered_df = full_df[
            full_df['food_categories'].apply(
                lambda cats: any(cat in user_prefs["favorite_categories"] for cat in cats)
            )
        ]

        for idx, row in filtered_df.head(15).iterrows():
            if idx not in user_prefs["viewed_restaurants"]:
                matched_cats = [cat for cat in row['food_categories']
                                if cat in user_prefs["favorite_categories"]]
                cb_candidates.append({
                    'id': idx,
                    'score': 0.85,
                    'reason': f"Phù hợp với sở thích: {', '.join(matched_cats[:2])}"
                })

    # Strategy C: Top rated
    top_rated = full_df.nlargest(15, 'average_rating')
    for idx, row in top_rated.iterrows():
        if idx not in user_prefs["viewed_restaurants"]:
            cb_candidates.append({
                'id': idx,
                'score': 0.75,
                'reason': f"Đánh giá cao ({row['average_rating']}/10)"
            })

    print(f"[DEBUG] CB candidates: {len(cb_candidates)}")

    # Normalize CB scores
    for candidate in cb_candidates:
        res_id = candidate['id']
        if res_id in hybrid_scores:
            # Cộng điểm CB vào
            hybrid_scores[res_id]['cb_score'] = candidate['score'] * cb_weight
            hybrid_scores[res_id]['reason_cb'] = candidate['reason']
            hybrid_scores[res_id]['type'] = 'hybrid'
        else:
            # Chỉ có CB
            hybrid_scores[res_id] = {
                'cf_score': 0,
                'cb_score': candidate['score'] * cb_weight,
                'reason_cb': candidate['reason'],
                'type': 'cb'
            }

    print(f"[DEBUG] Total hybrid scores: {len(hybrid_scores)}")
    print(f"[DEBUG] Types: CF={sum(1 for s in hybrid_scores.values() if s['type'] == 'cf')}, "
          f"CB={sum(1 for s in hybrid_scores.values() if s['type'] == 'cb')}, "
          f"Hybrid={sum(1 for s in hybrid_scores.values() if s['type'] == 'hybrid')}")

    # ==================
    # 3. CALCULATE HYBRID SCORES
    # ==================
    recommendations = []

    for res_id, scores in hybrid_scores.items():
        if res_id not in full_df.index:
            continue

        restaurant = full_df.loc[res_id]

        # Tính tổng điểm
        total_score = scores['cf_score'] + scores['cb_score']

        # Tạo reason message
        if scores['type'] == 'hybrid':
            reason = f"🤖 Hybrid: {scores.get('reason_cb', '')} & {scores.get('reason_cf', '')}"
        elif scores['type'] == 'cf':
            reason = f"👥 CF: {scores.get('reason_cf', '')}"
        else:
            reason = f"🎯 CB: {scores.get('reason_cb', '')}"

        recommendations.append({
            'restaurant': restaurant,
            'reason': reason,
            'score': total_score,
            'cf_score': scores['cf_score'],
            'cb_score': scores['cb_score'],
            'type': scores['type']
        })

    # Sort theo hybrid score
    recommendations.sort(key=lambda x: x['score'], reverse=True)

    return recommendations[:n]


# ----------------------
# MAIN UI
# ----------------------
st.title("🍽️ Hôm nay ăn gì?")
st.caption("Khám phá những gợi ý cá nhân hóa dành riêng cho bạn")

# Load user preferences
user_prefs = load_user_preferences()

# ----------------------
# SIDEBAR - User Preferences
# ----------------------
st.sidebar.header("⚙️ Tùy chọn của bạn")

# Categories preference
all_categories = list(set([cat for cats in X['food_categories'] for cat in cats]))
selected_categories = st.sidebar.multiselect(
    "🍜 Món ăn yêu thích",
    options=sorted(all_categories),
    default=user_prefs["favorite_categories"],
    help="Chọn các loại món bạn thích"
)

# Districts preference
all_districts = sorted(
    X['district'].dropna().unique().tolist(),
    key=district_sort_key
)
selected_districts = st.sidebar.multiselect(
    "📍 Khu vực quan tâm",
    options=all_districts,
    default=user_prefs["favorite_districts"],
    help="Chọn các quận bạn muốn tìm quán"
)

# Price range
price_range = st.sidebar.slider(
    "💰 Khoảng giá mong muốn (VNĐ)",
    min_value=0,
    max_value=500000,
    value=(user_prefs["price_range"][0], user_prefs["price_range"][1]),
    step=10000,
    format="%d đ"
)

# Save preferences button
if st.sidebar.button("💾 Lưu sở thích", type="primary", use_container_width=True):
    user_prefs["favorite_categories"] = selected_categories
    user_prefs["favorite_districts"] = selected_districts
    user_prefs["price_range"] = list(price_range)

    if save_user_preferences(user_prefs):
        st.sidebar.success("✅ Đã lưu sở thích!")
        st.rerun()

# Stats
st.sidebar.write("---")
st.sidebar.write("📊 **Thống kê của bạn:**")

# Reload prefs để hiển thị realtime
current_prefs = load_user_preferences()
st.sidebar.metric("Quán đã xem", len(current_prefs.get("viewed_restaurants", [])))
st.sidebar.metric("Quán yêu thích", len(current_prefs.get("liked_restaurants", [])))

# Debug button
if st.sidebar.button("🔄 Refresh Stats"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# Show liked restaurants
if current_prefs.get("liked_restaurants"):
    with st.sidebar.expander("❤️ Quán đã thích"):
        for res_id in current_prefs["liked_restaurants"]:
            try:
                # Tìm restaurant theo ID
                matching = full_df[full_df['id'] == res_id]
                if not matching.empty:
                    restaurant_name = matching.iloc[0]['name']
                    st.write(f"• {restaurant_name} (ID: {res_id})")
                else:
                    st.write(f"• ID: {res_id} (không tìm thấy)")
            except:
                st.write(f"• ID: {res_id}")

# ----------------------
# GET RECOMMENDATIONS
# ----------------------
with st.spinner("🔍 Đang tìm kiếm gợi ý cho bạn..."):
    recommendations = get_hybrid_recommendations(
        user_prefs, X, full_df, cosine_sim, cf_model, n=12
    )

# Debug: Show recommendation sources
if recommendations:
    cf_count = sum(1 for r in recommendations if r['type'] in ['cf', 'hybrid'])
    cb_count = sum(1 for r in recommendations if r['type'] == 'cb')
    hybrid_count = sum(1 for r in recommendations if r['type'] == 'hybrid')

    st.caption(f"📊 Nguồn gợi ý: {cf_count} CF, {cb_count} CB, {hybrid_count} Hybrid")

# ----------------------
# DISPLAY RECOMMENDATIONS
# ----------------------
if not recommendations:
    st.info("""
    👋 Chào mừng bạn đến với TasteMatch!

    Để nhận được gợi ý cá nhân hóa, hãy:
    1. Chọn **món ăn yêu thích** ở sidebar
    2. Chọn **khu vực** bạn muốn tìm quán
    3. Hoặc **like** một vài quán để hệ thống hiểu sở thích của bạn
    """)
else:
    # Show model info
    col_title, col_info = st.columns([3, 1])
    with col_title:
        st.subheader(f"🎯 {len(recommendations)} gợi ý dành cho bạn")
    with col_info:
        if cf_model.is_trained:
            st.success("🤖 Using Hybrid Model")
        else:
            st.info("🎯 Content-Based Only")

    # Display in grid
    for i in range(0, len(recommendations), 3):
        cols = st.columns(3)

        for j in range(3):
            if i + j < len(recommendations):
                rec = recommendations[i + j]
                restaurant = rec['restaurant']

                with cols[j]:
                    with st.container(border=True):
                        # Image
                        st.image(
                            "https://images.unsplash.com/photo-1555992336-cbfad6d9c7b0",
                            use_container_width=True
                        )

                        # Restaurant name
                        st.markdown(f"### {restaurant['name']}")

                        # Rating
                        stars = "⭐" * int(restaurant['average_rating'])
                        st.write(f"{stars} {restaurant['average_rating']}/10")

                        # Info
                        st.write(f"📍 {restaurant['district']}")
                        st.write(
                            f"💰 {int(restaurant['average_price_min']):,}đ - {int(restaurant['avarage_price_max']):,}đ")

                        # Categories
                        categories_str = ", ".join(restaurant['food_categories'][:3])
                        st.caption(f"🍜 {categories_str}")

                        # Reason with score breakdown
                        st.info(f"💡 {rec['reason']}")

                        # Score breakdown (optional, for debugging)
                        if rec['type'] == 'hybrid':
                            with st.expander("📊 Chi tiết điểm"):
                                st.write(f"CF Score: {rec['cf_score']:.2f} (40%)")
                                st.write(f"CB Score: {rec['cb_score']:.2f} (60%)")
                                st.write(f"Total: {rec['score']:.2f}")

                        # Actions
                        col_btn1, col_btn2 = st.columns(2)

                        # Lấy restaurant ID chính xác
                        rest_id = int(restaurant['id'])
                        rest_name = restaurant['name']

                        with col_btn1:
                            if st.button("👁️ Xem", key=f"view_{rest_id}_{i}_{j}", use_container_width=True):
                                add_to_history(rest_id, "viewed")
                                # Lưu tên quán vào session state để trang chi tiết hiển thị
                                st.session_state.selected_restaurant = rest_name
                                # Chuyển trang (cần đúng tên file)
                                st.switch_page("pages/Detail_Place.py")

                        with col_btn2:
                            is_liked = rest_id in user_prefs.get("liked_restaurants", [])
                            like_label = "❤️ Đã thích" if is_liked else "🤍 Thích"

                            if st.button(like_label, key=f"like_{rest_id}_{i}_{j}", use_container_width=True,
                                         disabled=is_liked):
                                if not is_liked:
                                    # Debug: Show which restaurant is being liked
                                    with st.spinner(f"Đang thêm '{rest_name}' (ID: {rest_id}) vào yêu thích..."):
                                        success = add_to_history(rest_id, "liked")

                                    if success:
                                        # Clear cache để reload CF model với data mới
                                        st.cache_resource.clear()
                                        st.cache_data.clear()
                                        # Reload trang
                                        st.rerun()
                                    else:
                                        st.error("❌ Lỗi khi lưu. Vui lòng thử lại!")

# ----------------------
# TIPS
# ----------------------
st.write("---")
st.subheader("💡 Mẹo để có gợi ý tốt hơn")

tip_cols = st.columns(3)

with tip_cols[0]:
    st.info("""
    **🍜 Chọn món yêu thích**

    Càng nhiều loại món bạn chọn, 
    gợi ý càng chính xác!
    """)

with tip_cols[1]:
    st.info("""
    **❤️ Like quán bạn thích**

    Hệ thống sẽ tìm các quán 
    tương tự để gợi ý.
    """)

with tip_cols[2]:
    st.info("""
    **📍 Chọn khu vực**

    Tìm quán gần nơi bạn 
    thường xuyên lui tới.
    """)
