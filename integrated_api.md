# Integrated Flask API - Backend Service

## Overview

Flask API ini sekarang menggabungkan 2 model machine learning dalam satu server dengan endpoint yang berbeda untuk menghindari konflik port dan mempermudah integrasi dengan Laravel.

## Models

1. **DASS-21 Model (Random Forest)** - Untuk quiz `family_social`
   - Predict: Normal/Depression/Anxiety/Stress
   - Data: MSPSS (12 questions) + DASS-21 (21 questions)

2. **Self Efficacy Model (SVM)** - Untuk quiz `self_efficacy`
   - Predict: high_well_being/low_well_being
   - Data: Self Efficacy (10 questions) + Well Being (18 questions)

## Endpoints

### 1. Root Endpoint
**GET** `/`

Response:
```json
{
  "message": "Mental Health Prediction API",
  "version": "1.0",
  "endpoints": {
    "dass_predict": "/api/dass/predict",
    "se_predict": "/api/se/predict",
    "health": "/api/health"
  }
}
```

### 2. Health Check
**GET** `/api/health`

Response:
```json
{
  "status": "healthy",
  "dass_model_loaded": true,
  "se_model_loaded": true
}
```

### 3. DASS-21 Prediction (Family Social Quiz)
**POST** `/api/dass/predict`

**Request Format:**
```json
{
  "fs1": "Sangat setuju",
  "fs2": "Setuju",
  "fs3": "Netral",
  "fs4": "Tidak setuju",
  "fs5": "Sangat tidak setuju",
  "fs6": "Sangat setuju",
  "fs7": "Setuju",
  "fs8": "Netral",
  "fs9": "Tidak setuju",
  "fs10": "Sangat tidak setuju",
  "fs11": "Sangat setuju",
  "fs12": "Setuju",
  "das1": 0,
  "das2": 1,
  "das3": 2,
  "das4": 3,
  "das5": 0,
  "das6": 1,
  "das7": 2,
  "das8": 3,
  "das9": 0,
  "das10": 1,
  "das11": 2,
  "das12": 3,
  "das13": 0,
  "das14": 1,
  "das15": 2,
  "das16": 3,
  "das17": 0,
  "das18": 1,
  "das19": 2,
  "das20": 3,
  "das21": 0
}
```

**Response Format:**
```json
{
  "prediction": "Anxiety",
  "manual_calculation": "Anxiety",
  "explanation": "Berdasarkan jawaban Anda, Anda mengalami Cemas dengan skor 20. Gejala Cemas yang Anda tunjukkan sudah termasuk dalam kategori Sangat Berat dan memerlukan perhatian yang serius. Dukungan sosial Anda tergolong tinggi (rata-rata: 2.8/5). Dukungan baik dari keluarga (2.8), teman (3.0), dan orang terdekat (2.5) dapat menjadi faktor protektif yang membantu mengurangi dampak dari gejala yang Anda alami.",
  "confidence": {
    "Anxiety": "0.6193",
    "Depression": "0.1216",
    "Normal": "0.0000",
    "Stress": "0.2592"
  },
  "categories": {
    "Depression": {
      "score": 18,
      "category_en": "Moderate",
      "category_id": "Sedang"
    },
    "Anxiety": {
      "score": 20,
      "category_en": "Extremely Severe",
      "category_id": "Sangat Berat"
    },
    "Stress": {
      "score": 22,
      "category_en": "Moderate",
      "category_id": "Sedang"
    }
  },
  "scores": {
    "MSPSS": {
      "Significant_Other": "2.50",
      "Family": "2.75",
      "Friends": "3.00",
      "Total": "2.75"
    },
    "DASS_21": {
      "Depression": 18,
      "Anxiety": 20,
      "Stress": 22,
      "Total": 60,
      "Total_Max": 126
    }
  }
}
```

### 4. Self Efficacy Prediction (Self Efficacy Quiz)
**POST** `/api/se/predict`

**Request Format:**
```json
{
  "SE01": 4,
  "SE02": 3,
  "SE03": 4,
  "SE04": 3,
  "SE05": 4,
  "SE06": 3,
  "SE07": 4,
  "SE08": 3,
  "SE09": 4,
  "SE10": 3,
  "Q1": 1,
  "Q2": 1,
  "Q3": 1,
  "Q4": 1,
  "Q5": 1,
  "Q6": 1,
  "Q7": 1,
  "Q8": 1,
  "Q9": 1,
  "Q10": 1,
  "Q11": 1,
  "Q12": 1,
  "Q13": 1,
  "Q14": 1,
  "Q15": 1,
  "Q16": 1,
  "Q17": 1,
  "Q18": 1
}
```

**Response Format:**
```json
{
  "prediction": "high_well_being",
  "explanation": "Berdasarkan jawaban Anda, Anda memiliki kondisi psychological well-being yang baik dengan skor total 108. Aspek terkuat Anda terdapat pada Kemandirian dengan skor 18, sedangkan aspek yang masih bisa ditingkatkan adalah Hubungan positif dengan skor 16. Dari sisi self-efficacy, Anda paling percaya diri dalam Kemampuan menyelesaikan masalah sulit (skor 4), namun masih perlu meningkatkan Keyakinan menyelesaikan masalah melalui usaha (skor 3).",
  "confidence": {
    "high_well_being": 0.95,
    "low_well_being": 0.05
  },
  "top_features": {
    "SE01": 4,
    "SE03": 4,
    "SE05": 4,
    "Autonomy": 18,
    "Personal_Growth": 18
  },
  "scores": {
    "self_efficacy": {
      "total": 35,
      "max": 40,
      "percentage": 87.5,
      "items": {
        "SE01": 4,
        "SE02": 3,
        "SE03": 4,
        "SE04": 3,
        "SE05": 4,
        "SE06": 3,
        "SE07": 4,
        "SE08": 3,
        "SE09": 4,
        "SE10": 3
      }
    },
    "well_being": {
      "total": 108,
      "max": 126,
      "percentage": 85.71
    },
    "subscales": {
      "Autonomy": 18,
      "Environmental_Mastery": 18,
      "Personal_Growth": 18,
      "Positive_Relations": 16,
      "Purpose_in_Life": 18,
      "Self_Acceptance": 18
    }
  }
}
```

## Running the API

### Start Flask Server

1. Navigate to backend directory:
   ```bash
   cd D:\Skripsi\project\backend
   ```

2. Start the server:
   ```bash
   python app.py
   ```

3. Expected output:
   ```
   DASS-21 Model, scaler, and label encoder loaded successfully!
   Self Efficacy Model, scaler, encoder, and features loaded successfully!
    * Serving Flask app 'app'
    * Debug mode: on
    * Running on http://127.0.0.1:5000
   ```

## Laravel Integration

### FlaskApiService

The Laravel `FlaskApiService` automatically calls the correct endpoint based on quiz type:

```php
// For family_social quiz
$flaskService = new FlaskApiService();
$quizData = $flaskService->transformQuizData($answers, $questions);
$result = $flaskService->predictMentalHealth('family_social', $quizData);
// This calls: POST http://127.0.0.1:5000/api/dass/predict

// For self_efficacy quiz
$flaskService = new FlaskApiService();
$quizData = $flaskService->transformSelfEfficacyData($answers, $questions);
$result = $flaskService->predictMentalHealth('self_efficacy', $quizData);
// This calls: POST http://127.0.0.1:5000/api/se/predict
```

### Configuration

In Laravel `.env`:
```
FLASK_API_URL=http://127.0.0.1:5000
```

In `config/services.php`:
```php
'flask' => [
    'url' => env('FLASK_API_URL', 'http://127.0.0.1:5000'),
],
```

## Troubleshooting

### Models not loaded

If you see errors like "DASS-21 Model not loaded" or "Self Efficacy Model not loaded":

1. Check if model files exist:
   - DASS-21: `backend/models/random_forest_model.pkl`
   - Self Efficacy: `backend/ml_api/models/model_svm.pkl`

2. Retrain models if needed:
   - DASS-21: Run `python train_model_fixed.py` in backend directory
   - Self Efficacy: Run `python train_model.py` in ml_api directory

### Port conflict

If port 5000 is already in use:

```bash
# Find process using port 5000 (Windows)
netstat -ano | findstr :5000

# Kill process (replace <PID> with actual PID)
taskkill /PID <PID> /F

# Or change port in app.py:
app.run(debug=True, host='0.0.0.0', port=5001)
```

## Testing

### Test DASS-21 Endpoint

```bash
curl -X POST http://127.0.0.1:5000/api/dass/predict \
  -H "Content-Type: application/json" \
  -d "{\"fs1\": \"Setuju\", \"das1\": 0, ...}"
```

### Test Self Efficacy Endpoint

```bash
curl -X POST http://127.0.0.1:5000/api/se/predict \
  -H "Content-Type: application/json" \
  -d "{\"SE01\": 4, \"Q1\": 1, ...}"
```

### Test Health Check

```bash
curl http://127.0.0.1:5000/api/health
```

## Changes Made

### Backend (Flask)

1. ✅ Combined 2 Flask apps into one (`backend/app.py`)
2. ✅ Changed endpoints to avoid conflicts:
   - `/api/dass/predict` - for DASS-21 prediction
   - `/api/se/predict` - for Self Efficacy prediction
3. ✅ Loads both models simultaneously
4. ✅ Single port (5000) for both services

### Laravel

1. ✅ Updated `FlaskApiService::predictMentalHealth()` to accept quiz type parameter
2. ✅ Automatically routes to correct endpoint based on quiz type
3. ✅ Updated `QuizController::submit()` to pass quiz type
4. ✅ No configuration changes needed in `.env` or `config/services.php`

### Cleanup

1. ✅ Removed duplicate `ml_api` folder from Laravel project
2. ✅ All models now centralized in `backend/` directory

## Benefits

1. **No Port Conflicts**: Only one Flask server running on port 5000
2. **Clear Separation**: Different endpoints for different quiz types
3. **Simplified Deployment**: Only one service to manage
4. **Better Performance**: Both models loaded and ready
5. **Easier Maintenance**: Single codebase for both APIs
