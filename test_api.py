import requests
print("Testing Fixed Server (8001)...")
try:
    r = requests.post("http://127.0.0.1:8001/predict", json={"data": [0.5]*7}, timeout=2)
    if r.status_code == 200:
        print("SUCCESS: Fixed server works!")
        print(r.json())
    else:
        print(f"FAILURE: {r.status_code}")
except Exception as e:
    print(f"FAILURE: {e}")
