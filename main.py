import streamlit as st
import numpy as np
from sklearn.metrics import pairwise_distances
import folium
from streamlit.components.v1 import html
import pandas as pd
import json
import requests
import base64

# Hàm gọi Google Maps Geocoding API để lấy địa chỉ từ tọa độ
def get_address_from_coordinates(lat, lon, api_key):
    try:
        url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={api_key}"
        response = requests.get(url)
        data = response.json()
        if data["status"] == "OK" and len(data["results"]) > 0:
            return data["results"][0]["formatted_address"]
        elif data["status"] == "REQUEST_DENIED":
            return f"Lỗi: API Key không hợp lệ hoặc chưa kích hoạt Geocoding API"
        elif data["status"] == "OVER_QUERY_LIMIT":
            return f"Lỗi: Vượt quá giới hạn yêu cầu của API Key"
        elif data["status"] == "ZERO_RESULTS":
            return f"Không tìm thấy địa chỉ cho tọa độ ({lat}, {lon})"
        else:
            return f"Lỗi API: {data['status']}"
    except Exception as e:
        return f"Lỗi kết nối: {str(e)}"

# Hàm tạo file CSV và trả về link tải
def create_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encode chuỗi thành base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Tải xuống {filename}</a>'
    return href

# Hàm tính khoảng cách Euclidean giữa hai tập hợp điểm
def euclidean_distance(points, centroids):
    return pairwise_distances(points, centroids)

# Thuật toán tham lam (Greedy - dựa trên K-Means đơn giản)
def greedy_clustering(points, k):
    n = len(points)
    centroids = points[np.random.choice(n, k, replace=False)]  # Chọn ngẫu nhiên k trung tâm
    for _ in range(10):  # Lặp 10 lần để cải thiện kết quả
        distances = euclidean_distance(points, centroids)
        labels = np.argmin(distances, axis=1)  # Gán điểm vào cụm gần nhất
        for i in range(k):
            if len(points[labels == i]) > 0:
                centroids[i] = np.mean(points[labels == i], axis=0)  # Cập nhật trung tâm
    distances = euclidean_distance(points, centroids)
    labels = np.argmin(distances, axis=1)
    cost = np.sum([np.min(distances[i]) for i in range(n)])  # Tổng khoảng cách tối thiểu
    return labels, cost

# Thuật toán quy hoạch động (DP - chia thành k đoạn liên tục trên 1D hoặc tối ưu khoảng cách)
def dp_clustering(points, k):
    n = len(points)
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]
    
    dp = np.full((n + 1, k + 1), np.inf)
    dp[0][0] = 0
    cluster = [[[] for _ in range(k + 1)] for _ in range(n + 1)]
    
    def segment_cost(start, end):
        if start >= end:
            return 0
        segment = sorted_points[start:end]
        centroid = np.mean(segment, axis=0)
        return np.sum(np.linalg.norm(segment - centroid, axis=1))
    
    for i in range(1, n + 1):
        for j in range(1, min(i + 1, k + 1)):
            for t in range(j - 1, i):
                cost = dp[t][j - 1] + segment_cost(t, i)
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    cluster[i][j] = cluster[t][j - 1] + [sorted_indices[t:i]]
    
    labels = np.zeros(n, dtype=int)
    for idx, group in enumerate(cluster[n][k]):
        for point_idx in group:
            labels[point_idx] = idx
    
    return labels, dp[n][k]

# Hàm tạo bản đồ Folium
def create_map(points, labels, title):
    center = [10.853550054039346, 106.60066348996047]  
    m = folium.Map(location=center, zoom_start=12)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightblue', 'gray', 'black', 'lightgreen', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'lightgray']
    
    for i, (lat, lon) in enumerate(points):
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=colors[labels[i] % len(colors)],
            fill=True,
            fill_color=colors[labels[i] % len(colors)],
            fill_opacity=0.7,
            popup=f"Cụm {labels[i]}"
        ).add_to(m)
    
    return m._repr_html_()

# Giao diện Streamlit
st.title("Tối ưu hóa giao hàng - Phân cụm điểm giao hàng trên bản đồ")

# Nhập liệu
st.sidebar.header("Nhập dữ liệu")
n_points = st.sidebar.slider("Số điểm giao hàng (nếu tạo ngẫu nhiên)", 5, 100, 10)
k = st.sidebar.slider("Số nhóm (k)", 2, 10, 3)
api_key = st.sidebar.text_input("Google Maps API Key", value="YOUR_API_KEY")  # Thay bằng API Key thật

data_type = st.sidebar.selectbox("Chọn loại tệp dữ liệu", ["CSV", "JSON"])
uploaded_file = st.sidebar.file_uploader(f"Tải lên file {data_type}", type=[data_type.lower()])
generate_data = st.sidebar.button("Tạo dữ liệu ngẫu nhiên")

# Xử lý dữ liệu đầu vào
points = None

if uploaded_file is not None:
    try:
        if data_type == "CSV":
            df = pd.read_csv(uploaded_file)
            if 'lat' in df.columns and 'lon' in df.columns:
                points = df[['lat', 'lon']].to_numpy()
            else:
                st.error("File CSV cần có cột 'lat' và 'lon'.")
        elif data_type == "JSON":
            data = json.load(uploaded_file)
            if isinstance(data, list) and all('lat' in point and 'lon' in point for point in data):
                points = np.array([[point['lat'], point['lon']] for point in data])
            else:
                st.error("File JSON cần là danh sách các đối tượng có 'lat' và 'lon'.")
    except Exception as e:
        st.error(f"Lỗi khi đọc tệp: {str(e)}")
else:
    if generate_data or 'points' not in st.session_state:
        base_lat, base_lon = 10.853550054039346, 106.60066348996047  # Vị trí cơ sở
        lat_noise = np.random.uniform(-0.05, 0.05, n_points)
        lon_noise = np.random.uniform(-0.05, 0.05, n_points)
        st.session_state.points = np.column_stack((base_lat + lat_noise, base_lon + lon_noise))
    points = st.session_state.points

# Kiểm tra dữ liệu hợp lệ
if points is None:
    st.stop()

# Hiển thị dữ liệu đầu vào
st.subheader("Dữ liệu đầu vào")
st.write("Tọa độ các điểm giao hàng (vĩ độ, kinh độ):")
st.write(points)

# Chạy thuật toán
if st.button("Phân cụm"):
    # Thuật toán tham lam
    greedy_labels, greedy_cost = greedy_clustering(points, k)
    # Thuật toán quy hoạch động
    dp_labels, dp_cost = dp_clustering(points, k)

    # Lấy địa chỉ từ Google Maps Geocoding API
    addresses = []
    if api_key and api_key != "YOUR_API_KEY":
        for lat, lon in points:
            address = get_address_from_coordinates(lat, lon, api_key)
            addresses.append(address)
    else:
        addresses = ["Không có địa chỉ (thiếu API Key)" for _ in range(len(points))]
        st.warning("Vui lòng nhập Google Maps API Key để lấy địa chỉ. Hiện tại cột địa chỉ sẽ để trống.")

    # Tạo DataFrame cho kết quả
    greedy_df = pd.DataFrame({
        'lat': points[:, 0],
        'lon': points[:, 1],
        'cluster_label': greedy_labels,
        'address': addresses
    })

    dp_df = pd.DataFrame({
        'lat': points[:, 0],
        'lon': points[:, 1],
        'cluster_label': dp_labels,
        'address': addresses
    })

    # Hiển thị kết quả trên bản đồ
    st.subheader("Kết quả phân cụm trên bản đồ")
    
    # Bản đồ cho thuật toán tham lam
    st.write("**Thuật toán tham lam**")
    greedy_map_html = create_map(points, greedy_labels, f"Thuật toán tham lam (Chi phí: {greedy_cost:.2f})")
    html(greedy_map_html, height=500)
    st.write(f"Chi phí: {greedy_cost:.2f}")
    
    # Xuất file CSV cho thuật toán tham lam
    st.markdown(create_download_link(greedy_df, "greedy_clusters.csv"), unsafe_allow_html=True)

    # Bản đồ cho thuật toán quy hoạch động
    st.write("**Thuật toán quy hoạch động**")
    dp_map_html = create_map(points, dp_labels, f"Thuật toán quy hoạch động (Chi phí: {dp_cost:.2f})")
    html(dp_map_html, height=500)
    st.write(f"Chi phí: {dp_cost:.2f}")
    
    # Xuất file CSV cho thuật toán quy hoạch động
    st.markdown(create_download_link(dp_df, "dp_clusters.csv"), unsafe_allow_html=True)

    # So sánh chi phí
    st.subheader("So sánh chi phí")
    st.write(f"Chi phí thuật toán tham lam: {greedy_cost:.2f}")
    st.write(f"Chi phí thuật toán quy hoạch động: {dp_cost:.2f}")
    if greedy_cost < dp_cost:
        st.write("Thuật toán tham lam tốt hơn trong trường hợp này!")
    else:
        st.write("Thuật toán quy hoạch động tốt hơn trong trường hợp này!")