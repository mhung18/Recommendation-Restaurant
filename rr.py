# app.py
import itertools

import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
from Content_based_Filtering_model import load_and_prepare_data, build_similarity_model,get_recommendations

# =======================
# Cấu hình Streamlit
# =======================
st.set_page_config(
    page_title="TasteMatch",
    page_icon="logo.jpg",
    layout="wide"
)

# =======================
# Tải dữ liệu & mô hình
# =======================
@st.cache_data
def load_data():
    X = load_and_prepare_data("./restaurants.json")
    cosine_sim = build_similarity_model(X)
    return X, cosine_sim

X, cosine_sim = load_data()

# =======================
# Sidebar lựa chọn
# =======================
def custom_sort(name: str):
    if name.startswith("Quận "):
        # Lấy số quận nếu có, còn không thì để 1000 cho các quận đặc biệt
        parts = name.split()
        if parts[1].isdigit():
            return (0, int(parts[1]))   # nhóm 0 = quận số, sort theo số
        else:
            return (1, parts[1])       # nhóm 1 = quận chữ (Bình Thạnh, Tân Bình...)
    elif name.startswith("Thành phố"):
        return (2, name)
    else:  # Huyện
        return (3, name)

restaurants = ['--- Chọn quán yêu thích ---'] + list(X['name'].unique())
districts = ['--- Chọn quận ---'] + sorted(X[X['district'].notna() & (X['district'].str.strip() != '')]['district'].unique(), key=custom_sort)

resCategories = X['food_categories']
all_items = list(itertools.chain.from_iterable(resCategories))

# lấy danh sách các giá trị duy nhất
categories = list(set(all_items))

food_categories = ['--- Chọn món yêu thích ---'] + sorted(list(categories))

st.sidebar.image("logo.svg")
selected_district = st.sidebar.selectbox("Choose your District", districts, index=0)
selected_category = st.sidebar.selectbox("Choose your favorite food", food_categories, index=0)
selected_restaurant = st.sidebar.selectbox("Choose your favorite Restaurant", restaurants, index=0)

# =======================
# Tiêu đề & mô tả
# =======================
st.title("TasteMatch: Discover Your Next Favorite Restaurant")
st.text("Find restaurants similar to your favorites using a content-based recommendation system!")

# =======================
# Thống kê quán ăn theo quận
# =======================
df_counts = X['district'].value_counts().reset_index()
df_counts.columns = ['district', 'count']

# Hiển thị bảng và biểu đồ
show_stats = st.checkbox("Hiển thị thống kê quán ăn theo quận", value=True)

if show_stats:
    st.subheader("📊 Số lượng quán ăn theo từng quận")
    st.dataframe(df_counts)
    fig, ax = plt.subplots()
    df_counts.plot(kind="bar", x="district", y="count", ax=ax, legend=False)
    ax.set_ylabel("Số quán ăn")
    st.pyplot(fig)

# =======================
# Hiển thị gợi ý
# =======================
# Thêm debug code để kiểm tra
if st.sidebar.button("Show recommendations"):
    # TH1: Người dùng chọn món yêu thích
    if selected_category != "--- Chọn món yêu thích ---":
        st.subheader(f"🍜 Các quán có món: **{selected_category}**")

        # Cách 1: Nếu food_categories là list
        res_list_have_selected_category = X[
            X['food_categories'].apply(
                lambda lst: selected_category in lst if isinstance(lst, list) else False
            )
        ]

        # Cách 2: Nếu food_categories là string (dự phòng)
        if res_list_have_selected_category.empty:
            res_list_have_selected_category = X[
                X['food_categories'].astype(str).str.contains(selected_category, case=False, na=False)
            ]

        # Lọc thêm theo quận nếu có
        if selected_district != "--- Chọn quận ---":
            res_list_have_selected_category = res_list_have_selected_category[
                res_list_have_selected_category['district'] == selected_district
                ]

        # Hiển thị kết quả
        if res_list_have_selected_category.empty:
            st.info(f"❌ Không tìm thấy quán có món **{selected_category}**" +
                    (f" tại **{selected_district}**" if selected_district != "--- Chọn quận ---" else ""))

        else:
            st.success(f"✅ Tìm thấy {len(res_list_have_selected_category)} quán")
            res_list_have_selected_category = res_list_have_selected_category.sort_values("average_rating", ascending=False)
            st.dataframe(
                res_list_have_selected_category[['name', 'address', 'district', 'food_categories','average_rating']].reset_index(drop=True))
