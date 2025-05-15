import requests
import json
input = {"features": "normal"}

url = "http://127.0.0.1:8888/predict"
data = json.dumps(input)

repposense = requests.post(url, data)
print(repposense.json())