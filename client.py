import requests
import json
input = {"features": "normal"}

url = "http://192.168.28.92:8888/predict"
data = json.dumps(input)

repposense = requests.post(url, data)
print(repposense.json())