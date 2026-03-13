# Mental Health Prediction API - Flask Backend

## Overview

Backend Flask API untuk sistem prediksi kesehatan mental menggunakan Random Forest classifier. API ini menerima data kuisioner dari Laravel dan mengembalikan prediksi mental health beserta kategori severity.

## Tech Stack

- **Python**: 3.13
- **Flask**: 3.1.3
- **Machine Learning**: scikit-learn 1.7.2
- **Data Processing**: pandas 2.2.3, numpy 2.2.3
- **Balancing**: imbalanced-learn 0.14.1 (SMOTE)

## Features

- ✅ Prediksi mental health (Normal/Depression/Anxiety/Stress)
- ✅ Kategori severity (Normal/Mild/Moderate/Severe/Extremely Severe)
- ✅ Confidence scores per kategori
- ✅ Analisis MSPSS (Multidimensional Scale of Perceived Social Support)
- ✅ Analisis DASS-21 (Depression Anxiety Stress Scales)
- ✅ Feature importance analysis

## Project Structure

```
backend/
├── app.py                      # Flask application & API endpoints
├── train_model_fixed.py        # Training script untuk Random Forest
├── check_dataset.py            # Script untuk inspect dataset
├── test_api.py                 # Script untuk testing API
├── requirements.txt            # Python dependencies
├── models/                     # Saved ML models
│   ├── random_forest_model.pkl # Random Forest classifier
│   ├── scaler.pkl              # StandardScaler
│   └── label_encoder.pkl       # LabelEncoder
└── dataset/                    # Training data
    ├── DATASET TA.xlsx         # Dataset asli (450 samples)
    └── skripsi.csv             # Dummy dataset untuk testing
```

## Installation

### Prerequisites

1. Python 3.13 atau higher
2. pip package manager

### Setup

1. **Navigate to backend directory**:
   ```bash
   cd D:\Skripsi\project\backend
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python --version
   pip list | grep flask
   pip list | grep scikit
   ```

## Model Training

### Dataset Preparation

Dataset harus dalam format Excel dengan kolom:
- `FS1-FS12`: Pertanyaan MSPSS (skala 1-5)
- `DAS1-DAS21`: Pertanyaan DASS-21 (skala 0-3)

### Training Steps

1. **Prepare dataset**:
   - Pastikan file `dataset/DATASET TA.xlsx` ada
   - Dataset minimal 450 samples untuk hasil terbaik

2. **Run training script**:
   ```bash
   python train_model_fixed.py
   ```

3. **Expected output**:
   ```
   Loading data...
   OK DASS-21 scale validated (0-3)
   OK All DASS-21 columns validated (0-3 scale)

   Data shape: (450, 33)

   Distribusi Label:
   Depression    245
   Anxiety       126
   Normal         58
   Stress         21

   Training Random Forest Model...

   RANDOM FOREST MODEL EVALUATION
   ============================================================
   Accuracy: 0.9333

   Classification Report:
                 precision    recall  f1-score   support

        Anxiety       0.95      0.80      0.87        25
     Depression       0.91      0.98      0.94        49
          Normal       1.00      1.00      1.00        12
          Stress       1.00      1.00      1.00         4

   FEATURE IMPORTANCE
   ============================================================
   Depression_Score: 0.2928
   Anxiety_Score: 0.2842
   Stress_Score: 0.2819
   SO_Score: 0.0623
   Family_Score: 0.0481
   Friends_Score: 0.0307

   MODEL TRAINING COMPLETED!
   ```

4. **Model files created**:
   - `models/random_forest_model.pkl`
   - `models/scaler.pkl`
   - `models/label_encoder.pkl`

### Re-training Model

Jika ingin re-train dengan dataset baru:

1. Replace file `dataset/DATASET TA.xlsx`
2. Run training script lagi
3. Model lama akan ditimpa dengan yang baru

## Running the API

### Development Mode

1. **Start Flask server**:
   ```bash
   python app.py
   ```

2. **Expected output**:
   ```
   Model, scaler, and label encoder loaded successfully!
    * Serving Flask app 'app'
    * Debug mode: on
    * Running on http://127.0.0.1:5000
   ```

3. **Server running at**: `http://127.0.0.1:5000`

### Production Mode

Untuk production, gunakan WSGI server seperti Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## API Endpoints

### 1. Root Endpoint
**GET** `/`

Response:
```json
{
  "message": "Mental Health Prediction API",
  "version": "1.0",
  "endpoints": {
    "predict": "/api/predict",
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
  "model_loaded": true
}
```

### 3. Predict Mental Health
**POST** `/api/predict`

**Request Format**:
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

**Field Descriptions**:
- `fs1-fs12`: MSPSS questions (1-5 Likert scale)
  - "Sangat tidak setuju" = 1
  - "Tidak setuju" = 2
  - "Netral" = 3
  - "Setuju" = 4
  - "Sangat setuju" = 5

- `das1-das21`: DASS-21 questions (0-3 scale)
  - 0 = Tidak berlaku bagi saya sama sekali
  - 1 = Berlaku bagi saya sampai tingkat tertentu, atau sebagian waktu
  - 2 = Berlaku bagi saya sampai tingkat tertentu, atau sebagian besar waktu
  - 3 = Berlaku bagi saya sangat banyak, atau sebagian besar waktu

**Response Format**:
```json
{
  "prediction": "Anxiety",
  "manual_calculation": "Anxiety",
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

**Response Fields**:
- `prediction`: Prediksi utama (Normal/Depression/Anxiety/Stress)
- `manual_calculation`: Verifikasi menggunakan rule-based
- `confidence`: Probability score per kategori (0-1)
- `categories`: Skor dan severity level per kategori
- `scores.MSPSS`: Skor dukungan sosial (1-5 scale)
- `scores.DASS_21`: Skor DASS-21 (0-42 per kategori, total 0-126)

## DASS-21 Scoring System

### Score Calculation

1. **Raw Score per Category**:
   - 7 questions × 3 points max = 21

2. **Final Score** (Multiplied by 2):
   - Raw score × 2 = Final score
   - Max per category: 42
   - Total max: 126

### Severity Levels

**Depression**:
- Normal: 0-9
- Mild: 10-13
- Moderate: 14-20
- Severe: 21-27
- Extremely Severe: 28+

**Anxiety**:
- Normal: 0-7
- Mild: 8-9
- Moderate: 10-14
- Severe: 15-19
- Extremely Severe: 20+

**Stress**:
- Normal: 0-14
- Mild: 15-18
- Moderate: 19-25
- Severe: 26-33
- Extremely Severe: 34+

## Testing

### Manual Testing dengan Python Script

1. **Run test script**:
   ```bash
   python test_api.py
   ```

2. **Expected output**:
   ```
   ✅ Prediction successful!

   PREDICTION RESULT
   ============================================================
   Model Prediction: Anxiety
   Manual Calculation: Anxiety

   Confidence Scores:
     Anxiety: 0.6193
     Depression: 0.1216
     Normal: 0.0000
     Stress: 0.2592

   MSPSS Scores:
     Significant Other: 2.50
     Family: 2.75
     Friends: 3.00
     Total: 2.75

   DASS-21 Scores:
     Depression: 18
     Anxiety: 20
     Stress: 22
     Total: 60
   ```

### Testing with cURL

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fs1": "Setuju",
    "fs2": "Setuju",
    ...
    "das1": 0,
    "das2": 1,
    ...
  }'
```

### Testing with Postman

1. Create new POST request
2. URL: `http://127.0.0.1:5000/api/predict`
3. Headers:
   - `Content-Type: application/json`
4. Body: Paste request JSON

## Debugging

### Enable Debug Logging

Debugging sudah otomatis di-enable di development mode:

```python
# Flask akan print:
# - Incoming request data
# - Outgoing response data
# - Exception traceback (jika error)
```

### Common Issues

**1. Model Not Found**
```
Error: Model not loaded
```
**Solution**: Run training script first

**2. Invalid Input Range**
```
Error: DASS score must be between 0-3
```
**Solution**: Validate input before sending

**3. JSON Serialization Error**
```
TypeError: Object of type int64 is not JSON serializable
```
**Solution**: Already fixed in code - convert numpy types to Python native

**4. Connection Refused**
```
Error: Cannot connect to Flask API
```
**Solution**: Ensure Flask server is running on port 5000

## Performance

- **Prediction Time**: < 100ms per request
- **Model Size**: ~2 MB
- **Memory Usage**: ~50 MB
- **Accuracy**: 93.33% (450 training samples)

## Model Information

- **Algorithm**: Random Forest Classifier
- **n_estimators**: 100
- **max_depth**: 10
- **min_samples_split**: 5
- **min_samples_leaf**: 2
- **random_state**: 42
- **Balancing**: SMOTE (Synthetic Minority Over-sampling Technique)

## Feature Importance

1. Depression_Score: 29.28%
2. Anxiety_Score: 28.42%
3. Stress_Score: 28.19%
4. SO_Score: 6.23%
5. Family_Score: 4.81%
6. Friends_Score: 3.07%

## Deployment

### Environment Variables

Create `.env` file:
```
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_PORT=5000
```

### Production Deployment

1. **Use production WSGI server**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Use systemd service** (Linux):
   ```ini
   [Unit]
   Description=Flask Mental Health API
   After=network.target

   [Service]
   User=www-data
   WorkingDirectory=/path/to/backend
   ExecStart=/usr/bin/gunicorn -w 4 -b 127.0.0.1:5000 app:app

   [Install]
   WantedBy=multi-user.target
   ```

3. **Use Nginx as reverse proxy**:
   ```nginx
   server {
       listen 80;
       server_name api.example.com;

       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

## Maintenance

### Regular Tasks

1. **Monitor model performance**
2. **Retrain with new data** (quarterly)
3. **Update dependencies** (monthly)
4. **Check API response time** (daily)

### Backup

Backup files to protect:
- `models/random_forest_model.pkl`
- `models/scaler.pkl`
- `models/label_encoder.pkl`
- `dataset/DATASET TA.xlsx`

## Troubleshooting Guide

### Flask Server Won't Start

**Symptoms**: Port already in use

**Solution**:
```bash
# Find process using port 5000
netstat -ano | findstr :5000

# Kill process
taskkill /PID <PID> /F

# Or use different port
python app.py  # It will automatically find available port
```

### Model Prediction is Wrong

**Symptoms**: Prediction doesn't match input

**Solution**:
1. Check input data format
2. Verify scoring calculation
3. Review model training data
4. Consider retraining with more data

### API Response is Slow

**Symptoms**: Response time > 1 second

**Solution**:
1. Check system resources
2. Optimize model (reduce n_estimators)
3. Use caching for repeated requests
4. Upgrade server hardware

## Contributing

Untuk kontributor yang ingin mengembangkan:

1. Fork repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## License

Proprietary - Untuk keperluan Skripsi

## Contact

- Developer: [Your Name]
- Project: Skripsi Sistem Prediksi Kesehatan Mental
- Date: Maret 2026

## References

- [DASS-21](https://psychology-tools.com/depression-anxiety-stress-scales/)
- [MSPSS](https://www.fostercare.ca/multidimensional-scale-perceived-social-support/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
