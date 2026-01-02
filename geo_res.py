import json
import time
import requests
from urllib.parse import quote


def geocode_address_flexible(name, address, district, city):
    """
    Lấy tọa độ với nhiều chiến lược fallback
    """
    headers = {
        'User-Agent': 'RestaurantGeocoder/1.0'
    }

    # Chiến lược 1: Địa chỉ đầy đủ
    strategies = [
        f"{address}, {district}, {city}, Vietnam",
        f"{address}, {district}, Ho Chi Minh City, Vietnam",
        f"{address}, {city}",
        f"{name}, {district}, {city}",
        f"{name}, Ho Chi Minh City, Vietnam"
    ]

    for i, full_address in enumerate(strategies, 1):
        encoded_address = quote(full_address)
        url = f"https://nominatim.openstreetmap.org/search?q={encoded_address}&format=json&limit=1&countrycodes=vn"

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data and len(data) > 0:
                lat = float(data[0]['lat'])
                lon = float(data[0]['lon'])

                # Kiểm tra xem tọa độ có ở TPHCM không (khoảng 10.7-10.9, 106.6-106.8)
                if 10.6 <= lat <= 11.0 and 106.5 <= lon <= 107.0:
                    print(f"✓ Tìm thấy (cách {i}): {name} - ({lat}, {lon})")
                    return {
                        'latitude': lat,
                        'longitude': lon,
                        'geocoded': True,
                        'method': i
                    }

            # Delay ngắn giữa các thử
            if i < len(strategies):
                time.sleep(0.3)

        except Exception as e:
            continue

    print(f"✗ Không tìm thấy: {name}")
    return {
        'latitude': None,
        'longitude': None,
        'geocoded': False,
        'method': None
    }


def add_coordinates_to_json(input_file, output_file):
    """
    Đọc file JSON, thêm tọa độ và lưu vào file mới
    """
    print(f"Đang đọc file {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        restaurants = json.load(f)

    print(f"Tìm thấy {len(restaurants)} quán ăn")
    print("Bắt đầu geocoding...\n")

    success_count = 0
    for i, restaurant in enumerate(restaurants, 1):
        print(f"[{i}/{len(restaurants)}] ", end="")

        coords = geocode_address_flexible(
            restaurant['name'],
            restaurant['address'],
            restaurant['district'],
            restaurant['city']
        )

        restaurant['latitude'] = coords['latitude']
        restaurant['longitude'] = coords['longitude']

        if coords['geocoded']:
            success_count += 1

        # Delay chính giữa các quán
        if i < len(restaurants):
            time.sleep(1.2)

    print(f"\nĐang lưu vào {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(restaurants, f, ensure_ascii=False, indent=4)

    print(f"\n✓ Hoàn thành!")
    print(f"  - Thành công: {success_count}/{len(restaurants)} quán ({success_count * 100 // len(restaurants)}%)")
    print(f"  - Thất bại: {len(restaurants) - success_count} quán")
    print(f"  - File đã lưu: {output_file}")

    # In danh sách các quán không tìm thấy
    if success_count < len(restaurants):
        print(f"\n❌ Các quán không tìm thấy tọa độ:")
        for r in restaurants:
            if r['latitude'] is None:
                print(f"  - {r['name']}: {r['address']}, {r['district']}")


if __name__ == "__main__":
    INPUT_FILE = "restaurants.json"
    OUTPUT_FILE = "restaurants_with_coords.json"

    add_coordinates_to_json(INPUT_FILE, OUTPUT_FILE)