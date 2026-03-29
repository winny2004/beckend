import requests
import json

# Endpoint API
url = "http://127.0.0.1:5000/api/predict"

# =========================
# SAMPLE DATA (SESUAI MODEL KAMU)
# =========================
sample_data = {
    # SELF-EFFICACY (boleh angka atau teks)
    "SE01": "Setuju",
    "SE02": "Sangat setuju",
    "SE03": "Setuju",
    "SE04": "Tidak setuju",
    "SE05": "Setuju",
    "SE06": "Sangat setuju",
    "SE07": "Setuju",
    "SE08": "Tidak setuju",
    "SE09": "Setuju",
    "SE10": "Sangat setuju",

    # WELL-BEING (Q1 - Q18)
    "Q1": "Setuju",
    "Q2": "Setuju",
    "Q3": "Netral",
    "Q4": "Setuju",
    "Q5": "Setuju",
    "Q6": "Setuju",
    "Q7": "Setuju",
    "Q8": "Netral",
    "Q9": "Setuju",
    "Q10": "Setuju",
    "Q11": "Setuju",
    "Q12": "Setuju",
    "Q13": "Setuju",
    "Q14": "Setuju",
    "Q15": "Setuju",
    "Q16": "Setuju",
    "Q17": "Setuju",
    "Q18": "Setuju"
}

try:
    response = requests.post(url, json=sample_data)

    print("\n" + "="*60)
    print("STATUS:", response.status_code)
    print("="*60)

    if response.status_code == 200:
        result = response.json()

        print("✅ Prediction Success")
        print("\nPREDICTION RESULT")
        print("="*60)

        print("Label:", result["prediction"])
        print("Explanation:", result["explanation"])

        print("\nConfidence:")
        for key, value in result["confidence"].items():
            print(f"  {key}: {value:.4f}")

        print("="*60)

    else:
        print("❌ Error:")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("❌ Tidak bisa connect ke API.")
    print("Pastikan Flask jalan di http://127.0.0.1:5000")

except Exception as e:
    print("❌ Error:", str(e))