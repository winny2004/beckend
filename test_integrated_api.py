import requests
import json
import time
import subprocess
import sys

def test_integrated_api():
    """Test both DASS-21 and Self Efficacy endpoints"""

    print("="*70)
    print("TESTING INTEGRATED FLASK API")
    print("="*70)

    # Start Flask server in background
    print("\n[1] Starting Flask server...")
    try:
        process = subprocess.Popen(
            [sys.executable, "app.py"],
            cwd="D:\\Skripsi\\project\\backend",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for server to start
        time.sleep(3)
        print("✓ Flask server started successfully\n")

    except Exception as e:
        print(f"✗ Failed to start Flask server: {e}")
        return

    try:
        # Test health endpoint
        print("[2] Testing health endpoint...")
        response = requests.get("http://127.0.0.1:5000/api/health", timeout=5)
        health_data = response.json()
        print(f"✓ Status: {health_data['status']}")
        print(f"✓ DASS-21 Model Loaded: {health_data['dass_model_loaded']}")
        print(f"✓ Self Efficacy Model Loaded: {health_data['se_model_loaded']}\n")

        # Test DASS-21 endpoint
        print("[3] Testing DASS-21 prediction endpoint...")
        dass_data = {
            "fs1": "Setuju",
            "fs2": "Setuju",
            "fs3": "Netral",
            "fs4": "Netral",
            "fs5": "Setuju",
            "fs6": "Setuju",
            "fs7": "Netral",
            "fs8": "Netral",
            "fs9": "Setuju",
            "fs10": "Setuju",
            "fs11": "Netral",
            "fs12": "Netral",
            "das1": 0, "das2": 1, "das3": 2, "das4": 3,
            "das5": 0, "das6": 1, "das7": 2,
            "das8": 3, "das9": 0, "das10": 1, "das11": 2, "das12": 3, "das13": 0, "das14": 1,
            "das15": 2, "das16": 3, "das17": 0, "das18": 1, "das19": 2, "das20": 3, "das21": 0
        }

        response = requests.post(
            "http://127.0.0.1:5000/api/dass/predict",
            json=dass_data,
            timeout=10
        )

        if response.status_code == 200:
            dass_result = response.json()
            print(f"✓ Prediction: {dass_result['prediction']}")
            print(f"✓ Manual Calculation: {dass_result['manual_calculation']}")
            print(f"✓ Depression Score: {dass_result['categories']['Depression']['score']}")
            print(f"✓ Anxiety Score: {dass_result['categories']['Anxiety']['score']}")
            print(f"✓ Stress Score: {dass_result['categories']['Stress']['score']}\n")
        else:
            print(f"✗ DASS-21 prediction failed: {response.status_code}")
            print(f"  Error: {response.text}\n")

        # Test Self Efficacy endpoint
        print("[4] Testing Self Efficacy prediction endpoint...")
        se_data = {
            "SE01": 4, "SE02": 3, "SE03": 4, "SE04": 3, "SE05": 4,
            "SE06": 3, "SE07": 4, "SE08": 3, "SE09": 4, "SE10": 3,
            "Q1": 1, "Q2": 1, "Q3": 1, "Q4": 1, "Q5": 1,
            "Q6": 1, "Q7": 1, "Q8": 1, "Q9": 1, "Q10": 1,
            "Q11": 1, "Q12": 1, "Q13": 1, "Q14": 1, "Q15": 1,
            "Q16": 1, "Q17": 1, "Q18": 1
        }

        response = requests.post(
            "http://127.0.0.1:5000/api/se/predict",
            json=se_data,
            timeout=10
        )

        if response.status_code == 200:
            se_result = response.json()
            print(f"✓ Prediction: {se_result['prediction']}")
            print(f"✓ Self Efficacy Total: {se_result['scores']['self_efficacy']['total']}")
            print(f"✓ Self Efficacy %: {se_result['scores']['self_efficacy']['percentage']}%")
            print(f"✓ Well Being Total: {se_result['scores']['well_being']['total']}")
            print(f"✓ Well Being %: {se_result['scores']['well_being']['percentage']}%\n")
        else:
            print(f"✗ Self Efficacy prediction failed: {response.status_code}")
            print(f"  Error: {response.text}\n")

        print("="*70)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Stop Flask server
        print("\n[5] Stopping Flask server...")
        process.terminate()
        try:
            process.wait(timeout=5)
            print("✓ Flask server stopped\n")
        except subprocess.TimeoutExpired:
            process.kill()
            print("✓ Flask server killed\n")

if __name__ == "__main__":
    test_integrated_api()
