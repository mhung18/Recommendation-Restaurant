import streamlit as st
import pandas as pd
import itertools
import re

from Content_based_Filtering_model import (
    load_and_prepare_data,
    build_similarity_model
)

# =======================
# Page config
# =======================
st.set_page_config(
    page_title="TasteMatch",
    page_icon="🍜",
    layout="wide"
)


# =======================
# Load data & model
# =======================
@st.cache_data
def load_data():
    X = load_and_prepare_data("./restaurants_with_coords.json")
    cosine_sim = build_similarity_model(X)
    return X, cosine_sim


@st.cache_data
def load_full_data():
    """Load file JSON gốc để có đầy đủ thông tin (bao gồm lat/lon)"""
    import json
    with open("./restaurants_with_coords.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def district_sort_key(name):
    if name.startswith("Quận"):
        match = re.search(r"\d+", name)
        if match:
            return (0, int(match.group()))
        else:
            return (0, 999)  # fallback nếu có Quận nhưng không có số
    else:
        return (1, name)


X, cosine_sim = load_data()
full_df = load_full_data()  # DataFrame đầy đủ có lat/lon

# =======================
# Sidebar – Filters
# =======================
st.sidebar.header("🔍 Bộ lọc")

districts = ["Tất cả"] + sorted(
    X[X["district"].notna()]["district"].unique().tolist(),
    key=district_sort_key
)

# flatten food_categories
all_categories = list(
    set(itertools.chain.from_iterable(X["food_categories"]))
)
categories = ["Tất cả"] + sorted(all_categories)

selected_district = st.sidebar.selectbox("Quận", districts)
selected_category = st.sidebar.selectbox("Loại món", categories)

# =======================
# Filter data
# =======================
filtered_df = X.copy()

if selected_district != "Tất cả":
    filtered_df = filtered_df[filtered_df["district"] == selected_district]

if selected_category != "Tất cả":
    filtered_df = filtered_df[
        filtered_df["food_categories"].apply(
            lambda lst: selected_category in lst
        )
    ]

# Filter full_df (để có lat/lon) dựa trên index của filtered_df
filtered_indices = filtered_df.index
filtered_full_df = full_df.loc[filtered_indices]

# =======================
# MAIN UI
# =======================
st.title("🗺️ Khám phá địa điểm ăn uống")
st.caption("Khám phá các quán ăn nổi bật theo khu vực và sở thích")

# =======================
# MAP - Dùng filtered_full_df thay vì filtered_df
# =======================
st.subheader("📍 Bản đồ địa điểm")

if {"latitude", "longitude"}.issubset(filtered_full_df.columns):
    map_df = filtered_full_df.dropna(subset=["latitude", "longitude"]).copy()

    if not map_df.empty:
        st.write(f"Hiển thị **{len(map_df)}** quán trên bản đồ")

        # Chuẩn bị data cho st.map (cần columns: lat, lon)
        map_data = pd.DataFrame({
            'lat': map_df['latitude'],
            'lon': map_df['longitude'],
            'name': map_df['name']
        })

        # Hiển thị map đơn giản
        st.map(map_data, zoom=12)

        # Hiển thị thông tin khi hover (dùng expander)
        with st.expander("📋 Danh sách quán trên bản đồ"):
            st.dataframe(
                map_df[['name', 'address', 'district']].reset_index(drop=True),
                use_container_width=True
            )
    else:
        st.info("📍 Không có quán nào trong bộ lọc hiện tại có tọa độ")
else:
    st.warning("⚠️ Dữ liệu chưa có thông tin tọa độ (latitude/longitude)")

# =======================
# LIST VIEW
# =======================
st.subheader("📋 Danh sách địa điểm")

# Các cột muốn hiển thị (ưu tiên)
preferred_cols = [
    "name",
    "district",
    "address",
    "category",
    "style",
    "average_rating",
    "average_price_min",
    "avarage_price_max"
]

# Chỉ lấy các cột thực sự tồn tại
display_cols = [col for col in preferred_cols if col in filtered_df.columns]

st.dataframe(
    filtered_df[display_cols].reset_index(drop=True),
    use_container_width=True
)

# =======================
# CTA
# =======================
st.info(
    "👉 Chọn một quán để xem **Chi tiết địa điểm**\n\n"
    "👉 Hoặc sang trang **🍽️ Hôm nay ăn gì?** để nhận gợi ý cá nhân hóa"
)
