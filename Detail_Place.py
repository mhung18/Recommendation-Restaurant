import streamlit as st
import pandas as pd
import json
from datetime import datetime
import os

st.set_page_config(page_title="Chi tiết địa điểm", page_icon="📍", layout="wide")


# ----------------------
# LOAD DATA
# ----------------------
@st.cache_data
def load_restaurants():
    with open("./restaurants_with_coords.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)


df = load_restaurants()

# ----------------------
# COMMENT FUNCTIONS (JSON FILE)
# ----------------------
COMMENTS_FILE = "restaurant_comments.json"
REVIEWS_FILE = "restaurants_reviews_new.json"


def load_all_comments():
    """Load tất cả comments từ file JSON"""
    if not os.path.exists(COMMENTS_FILE):
        return {}

    try:
        with open(COMMENTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}


def load_foody_reviews():
    """Load reviews từ Foody"""
    if not os.path.exists(REVIEWS_FILE):
        return []

    try:
        with open(REVIEWS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []


def save_all_comments(comments_data):
    """Lưu tất cả comments vào file JSON"""
    try:
        with open(COMMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(comments_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Lỗi khi lưu: {str(e)}")
        return False


def get_restaurant_comments(restaurant_id):
    """Lấy comments của một quán cụ thể"""
    all_comments = load_all_comments()
    return all_comments.get(str(restaurant_id), [])


def get_foody_reviews_by_restaurant(restaurant_id):
    """Lấy reviews từ Foody cho một quán (max 10)"""
    all_reviews = load_foody_reviews()
    restaurant_reviews = [r for r in all_reviews if r.get('res_id') == restaurant_id]
    return restaurant_reviews[:10]  # Giới hạn 10 reviews


def add_comment(restaurant_id, rating, comment_text, user_name):
    """Thêm comment mới"""
    all_comments = load_all_comments()
    restaurant_id_str = str(restaurant_id)

    # Lấy comments hiện tại của quán
    if restaurant_id_str not in all_comments:
        all_comments[restaurant_id_str] = []

    # Tạo comment mới
    new_comment = {
        'id': len(all_comments[restaurant_id_str]) + 1,
        'rating': rating,
        'comment': comment_text,
        'user': user_name,
        'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M"),
        'source': 'user'  # Đánh dấu nguồn
    }

    # Thêm vào đầu list (comment mới nhất hiển thị trước)
    all_comments[restaurant_id_str].insert(0, new_comment)

    # Lưu lại
    return save_all_comments(all_comments)


# ----------------------
# SEARCH BAR
# ----------------------
st.title("📍 Chi tiết địa điểm")

# Tạo list tên quán để autocomplete
restaurant_names = df['name'].tolist()

# Search box
search_query = st.selectbox(
    "🔍 Tìm kiếm quán ăn",
    options=[""] + restaurant_names,
    index=0,
    placeholder="Nhập tên quán..."
)

# Session state để lưu quán được chọn
if 'selected_restaurant' not in st.session_state:
    st.session_state.selected_restaurant = None

# Khi chọn từ search box
if search_query and search_query != "":
    st.session_state.selected_restaurant = search_query

# ----------------------
# DISPLAY RESTAURANT CARDS
# ----------------------
if not st.session_state.selected_restaurant:
    st.subheader("📋 Tất cả quán ăn")
    st.caption("Chọn một quán để xem chi tiết")

    # Hiển thị grid các quán
    cols = st.columns(3)

    for idx, row in df.iterrows():
        col_idx = idx % 3
        with cols[col_idx]:
            with st.container(border=True):
                st.markdown(f"### {row['name']}")
                st.write(f"📍 {row['address']}, {row['district']}")
                st.write(f"⭐ Đánh giá: **{row['average_rating']}/10**")

                # Hiển thị số lượng comment
                comment_count = len(get_restaurant_comments(row['id']))
                if comment_count > 0:
                    st.caption(f"💬 {comment_count} bình luận")

                if st.button("Xem chi tiết", key=f"btn_{idx}"):
                    st.session_state.selected_restaurant = row['name']
                    st.rerun()

else:
    # ----------------------
    # CHI TIẾT QUÁN ĂN
    # ----------------------
    restaurant = df[df['name'] == st.session_state.selected_restaurant].iloc[0]

    # Back button
    if st.button("← Quay lại danh sách"):
        st.session_state.selected_restaurant = None
        st.rerun()

    st.title(f"📍 {restaurant['name']}")

    # Layout 2 columns
    col1, col2 = st.columns([2, 1])

    with col1:
        # ----------------------
        # BASIC INFO
        # ----------------------
        st.image(
            "https://images.unsplash.com/photo-1555992336-cbfad6d9c7b0",
            caption=f"Không gian {restaurant['name']}",
            use_container_width=True
        )

        st.markdown(f"""
        **📍 Địa chỉ:** {restaurant['address']}, {restaurant['district']}  
        **🏙️ Thành phố:** {restaurant['city']}  
        **⏰ Giờ mở cửa:** {restaurant['main_opening_hour']} - {restaurant['main_closing_hour']}  
        **🍽️ Loại hình:** {restaurant['category']}  
        **💰 Giá trung bình:** {int(restaurant['average_price_min']):,}đ - {int(restaurant['avarage_price_max']):,}đ
        """)

        # ----------------------
        # MAP (nếu có tọa độ)
        # ----------------------
        if pd.notna(restaurant['latitude']) and pd.notna(restaurant['longitude']):
            st.subheader("🗺️ Vị trí")
            map_data = pd.DataFrame({
                'lat': [restaurant['latitude']],
                'lon': [restaurant['longitude']]
            })
            st.map(map_data, zoom=15)

    with col2:
        # ----------------------
        # RATINGS
        # ----------------------
        st.subheader("⭐ Đánh giá tổng quan")

        st.metric("Điểm trung bình", f"{restaurant['average_rating']}/10")

        # Rating breakdown
        st.write("**Chi tiết đánh giá:**")
        st.progress(restaurant['quality_rating'] / 10, text=f"Chất lượng: {restaurant['quality_rating']}/10")
        st.progress(restaurant['service_rating'] / 10, text=f"Phục vụ: {restaurant['service_rating']}/10")
        st.progress(restaurant['price_rating'] / 10, text=f"Giá cả: {restaurant['price_rating']}/10")
        st.progress(restaurant['location_rating'] / 10, text=f"Vị trí: {restaurant['location_rating']}/10")
        st.progress(restaurant['space_rating'] / 10, text=f"Không gian: {restaurant['space_rating']}/10")

        # Comment stats
        st.write("---")
        st.write(f"**📝 Tổng số bình luận:** {int(restaurant['comment_quantity'])}")
        st.write(f"✨ Tuyệt vời: {int(restaurant['marvelous_comment'])}")
        st.write(f"👍 Tốt: {int(restaurant['good_comment'])}")
        st.write(f"😐 Bình thường: {int(restaurant['ok_comment'])}")
        st.write(f"👎 Tệ: {int(restaurant['awful_comment'])}")

    # ----------------------
    # ADDITIONAL INFO
    # ----------------------
    st.write("---")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.subheader("🍜 Món ăn")
        for food in restaurant['food_categories']:
            st.write(f"• {food}")

    with col4:
        st.subheader("🎨 Phong cách")
        for style in restaurant['style']:
            st.write(f"• {style}")

    with col5:
        st.subheader("⏰ Thời gian phù hợp")
        for time in restaurant['suitable_time']:
            st.write(f"• {time}")

    # ----------------------
    # SUITABLE FOR
    # ----------------------
    appropriate = restaurant.get("appropriate")

    if isinstance(appropriate, list) and len(appropriate) > 0:
        st.write("---")
        st.subheader("👥 Phù hợp với")

        appropriate_cols = st.columns(len(appropriate))
        for idx, app in enumerate(appropriate):
            with appropriate_cols[idx]:
                st.info(app)

    # ----------------------
    # USER RATING SECTION
    # ----------------------
    st.write("---")
    st.subheader("⭐ Viết đánh giá của bạn")

    with st.form(key=f"rating_form_{restaurant['id']}", clear_on_submit=True):
        col_name, col_rating = st.columns([3, 1])

        with col_name:
            user_name = st.text_input("Tên của bạn *", placeholder="Nhập tên của bạn...")

        with col_rating:
            rating = st.slider("Số sao *", 1, 10, 8)

        comment_text = st.text_area(
            "Bình luận *",
            placeholder="Chia sẻ trải nghiệm của bạn về quán này...",
            height=100
        )

        submit_button = st.form_submit_button("📤 Gửi đánh giá", type="primary", use_container_width=True)

        if submit_button:
            if not user_name.strip():
                st.error("⚠️ Vui lòng nhập tên của bạn")
            elif not comment_text.strip():
                st.error("⚠️ Vui lòng nhập bình luận")
            else:
                if add_comment(restaurant['id'], rating, comment_text.strip(), user_name.strip()):
                    st.success("✅ Cảm ơn bạn đã đánh giá!")
                    st.balloons()
                    st.rerun()

    # ----------------------
    # DISPLAY COMMENTS
    # ----------------------
    st.write("---")

    # Lấy comments từ users và reviews từ Foody
    user_comments = get_restaurant_comments(restaurant['id'])
    foody_reviews = get_foody_reviews_by_restaurant(restaurant['id'])

    # Tổng số bình luận
    total_reviews = len(user_comments) + len(foody_reviews)

    # Header với tabs
    st.subheader("💬 Đánh giá & Bình luận")

    if total_reviews > 0:
        st.metric("Tổng đánh giá", total_reviews)

        # Tabs để phân loại
        tab1, tab2 = st.tabs([
            f"👥 Từ người dùng ({len(user_comments)})",
            f"🍴 Từ Foody ({len(foody_reviews)})"
        ])

        # Tab 1: User Comments
        with tab1:
            if user_comments:
                for comment in user_comments:
                    with st.container(border=True):
                        # Header: user và timestamp
                        col_user, col_time = st.columns([2, 1])
                        with col_user:
                            st.markdown(f"**👤 {comment['user']}**")
                        with col_time:
                            st.caption(f"🕒 {comment['timestamp']}")

                        # Rating stars
                        stars = "⭐" * comment['rating']
                        st.markdown(f"### {stars} {comment['rating']}/10")

                        # Comment text
                        st.write(comment['comment'])
            else:
                st.info("📝 Chưa có bình luận từ người dùng. Hãy là người đầu tiên!")

        # Tab 2: Foody Reviews
        with tab2:
            if foody_reviews:
                for review in foody_reviews:
                    with st.container(border=True):
                        # Header: user info
                        col_user, col_time = st.columns([2, 1])
                        with col_user:
                            # Link tới profile Foody
                            profile_url = review.get('profile_url', '#')
                            username = review.get('username', 'Anonymous')
                            st.markdown(f"**👤 [{username}]({profile_url})**")
                        with col_time:
                            timestamp = review.get('timestamp', '')
                            st.caption(f"🕒 {timestamp}")

                        # Rating (Foody dùng scale 10)
                        rating = review.get('rating', 0)
                        stars = "⭐" * int(rating)
                        st.markdown(f"### {stars} {rating}/10")

                        # Review text
                        review_text = review.get('review_text', '')
                        if len(review_text) > 300:
                            # Truncate long reviews với expander
                            st.write(review_text[:300] + "...")
                            with st.expander("Đọc thêm"):
                                st.write(review_text)
                        else:
                            st.write(review_text)

                        # Badge nguồn
                        st.caption("📱 Nguồn: Foody.vn")
            else:
                st.info("📝 Chưa có đánh giá từ Foody cho quán này.")
    else:
        st.info("📝 Chưa có bình luận nào. Hãy là người đầu tiên đánh giá quán này!")