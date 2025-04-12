import random
import csv

# Tọa độ trung tâm vùng (TP.HCM)
center_lat = 10.85
center_lon = 106.60

# Hàm sinh tọa độ ngẫu nhiên xung quanh tọa độ trung tâm
def random_nearby(base, delta=0.01):
    return round(base + random.uniform(-delta, delta), 6)

# Tạo dữ liệu
rows = [
    {"lat": random_nearby(center_lat), "lon": random_nearby(center_lon)}
    for _ in range(100)
]

# Ghi vào file CSV
with open("random_coordinates.csv", mode="w", newline='', encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["lat", "lon"])
    writer.writeheader()
    writer.writerows(rows)