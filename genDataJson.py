import random
import json

# Trung tâm tọa độ (ví dụ: TP.HCM)
center_lat = 10.85
center_lon = 106.60

def random_nearby(base, delta=0.01):
    return round(base + random.uniform(-delta, delta), 6)

# Sinh 100 dòng JSON
random_coords = [
    {"lat": random_nearby(center_lat), "lon": random_nearby(center_lon)}
    for _ in range(100)
]

# Lưu vào file JSON
with open("random_coordinates.json", "w", encoding="utf-8") as f:
    json.dump(random_coords, f, ensure_ascii=False, indent=2)