import requests
import json

# Flask API endpoint
url = "http://127.0.0.1:5000/api/predict"

# Sample data (sesuai format dari kuisioner)
sample_data = {
    "fs1": "Setuju",  # Sangat setuju = 5, Setuju = 4, Netral = 3, Tidak setuju = 2, Sangat tidak setuju = 1
    "fs2": "Setuju",
    "fs3": "Netral",
    "fs4": "Setuju",
    "fs5": "Setuju",
    "fs6": "Setuju",
    "fs7": "Setuju",
    "fs8": "Setuju",
    "fs9": "Setuju",
    "fs10": "Setuju",
    "fs11": "Setuju",
    "fs12": "Setuju",
    # DASS-21 (0-3 scale: 0 = Tidak pernah, 1 = Kadang-kadang, 2 = Sering, 3 = Sangat sering)
    "das1": 0,
    "das2": 1,
    "das3": 1,
    "das4": 0,
    "das5": 2,
    "das6": 0,
    "das7": 0,
    "das8": 1,
    "das9": 2,
    "das10": 1,
    "das11": 0,
    "das12": 0,
    "das13": 2,
    "das14": 1,
    "das15": 0,
    "das16": 1,
    "das17": 2,
    "das18": 1,
    "das19": 0,
    "das20": 2,
    "das21": 1
}

try:
    response = requests.post(url, json=sample_data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Prediction successful!")
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"Model Prediction: {result['prediction']}")
        print(f"Manual Calculation: {result['manual_calculation']}")
        print("\nConfidence Scores:")
        for label, confidence in result['confidence'].items():
            print(f"  {label}: {confidence}")
        print("\nMSPSS Scores:")
        for key, value in result['scores']['MSPSS'].items():
            print(f"  {key}: {value}")
        print("\nDASS-21 Scores:")
        for key, value in result['scores']['DASS_21'].items():
            print(f"  {key}: {value}")
        print("="*60)
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("❌ Error: Cannot connect to Flask API. Make sure the server is running on http://127.0.0.1:5000")
except Exception as e:
    print(f"❌ Error: {e}")
