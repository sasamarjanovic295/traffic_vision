import os
import json

def check_speed_file():
    # Provera da li postoji speed.json datoteka
    if not os.path.exists('speed.json'):
        with open('speed.json', 'w') as f:
            json.dump({}, f)

def log_speed(device, image_size, speed):
    with open('speed.json', 'r') as f:
        data = json.load(f)
    
    device_str = str(device)
    image_size_str = str(image_size)

    if device_str not in data:
        data[device_str] = {}

    if image_size_str not in data[device_str]:
        data[device_str][image_size_str] = []

    data[device_str][image_size_str].append(speed)

    with open('speed.json', 'w') as f:
        json.dump(data, f, indent=4)