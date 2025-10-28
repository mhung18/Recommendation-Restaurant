import time

import streamlit as st
import pandas as pd
import numpy as np

# Quy trình thực hiện
#0. Load model đã train
model = load_model("Duong dan den model")

#1. Nhận file upload lên
file = st.file_uploader("Chon file anh")

if file is not None:
    #Xu ly file anh:
    tensor  = process(file)
    #Dua vao model predict
    result = model.predict(tensor)
    #
    class_id = np.argmax(result) #0: Chó,  1: Mèo
    class_name = ["Chó", "Mèo"]
    st.write("Ket qua nhan dien: ", class_name[class_id])

#
# st.title("Chương trình thử nghiệm Streamlit bằng Python")
# st.header("Chương trình thử nghiệm Streamlit bằng Python")
# st.subheader("Chương trình thử nghiệm Streamlit bằng Python")
# st.write("Chương trình thử nghiệm Streamlit bằng Python")
# st.caption("Chương trình thử nghiệm Streamlit bằng Python")
# st.text("Đây là đoạn text")
#
# st.code("import streamlit")
#
# ten = st.text_input("Nhập vào tên bạn:")
# st.write("Tên bạn là", ten)
#
# click = st.button("Click me")
# if click:
#     st.write("Đã Click")
#
#
# teptin = st.file_uploader("Chọn file đê")
#
# if teptin is not None:
#     st.write("Kích thước tệp tin là: ", teptin.size / 1000, " MB")
#
# text_download = "Tập tin tải về"
# st.download_button("Download", text_download)
#
# with open ("lisa.jpg", "rb") as file:
#     st.download_button(
#         label="Tải ảnh",
#         data=file,
#         file_name="anh_lisa.png",
#         mime="image/png",
#     )
#
# chart_data = pd.DataFrame(
#     np.random.randn(20, 3),
#     columns = ["A", "B", "C"],
# )
#
# st.bar_chart(chart_data)
#
# st.image("lisa.jpg")
#
# st.sidebar.text("Đây là sidebar")
#
# col1, col2, col3 = st.columns(3)
#
# with col1:
#     st.image("lisa.jpg",caption="Ảnh 1")
#
# with col2:
#     st.image("lisa.jpg",caption="Ảnh 2")
#
# with col3:
#     st.image("lisa.jpg",caption="Ảnh 3")

# with st.spinner("Đang chạy nha"):
#     time.sleep(1)
#     st.write("Xong")
#
# percent = 0
# while percent < 100:
#     st.progress(percent)
#     time.sleep(1)
#     percent += 1

# st.info("Đây là thông báo")
# st.warning("Đây là cảnh báo")
# st.error("Đây là thông báo lỗi")
# st.success("Đây là thông báo thành công")
#
# ten = st.text_input("Họ và tên:")
# st.write(ten)
# st.stop()
# with st.form("Form1:"):
#     ten = st.text_input("Họ và tên:")
#     tuoi = st.text_input("Tuổi:")
#     submit = st.form_submit_button("Submit")