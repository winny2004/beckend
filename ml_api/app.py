from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import json

app = Flask(__name__)
CORS(app)

# =========================
# LOAD MODEL
# =========================
model_path = 'models/model_svm.pkl'
scaler_path = 'models/scaler.pkl'
encoder_path = 'models/label_encoder.pkl'
feature_path = 'models/feature_cols.pkl'

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)
    feature_cols = joblib.load(feature_path)

    print("Model, scaler, encoder loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None
    label_encoder = None
    feature_cols = None


# =========================
# HOME
# =========================
@app.route('/')
def home():
    return jsonify({
        'message': 'Mental Health Prediction API (SVM - Self Efficacy & Well Being)',
        'version': '1.0',
        'endpoints': {
            'predict': '/api/predict',
            'health': '/api/health'
        }
    })


# =========================
# HEALTH CHECK
# =========================
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


# =========================
# PREDICT
# =========================
@app.route('/api/predict', methods=['POST'])
def predict():

    if model is None or scaler is None or label_encoder is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()

        print("\n" + "="*50)
        print("REQUEST DATA:")
        print(json.dumps(data, indent=2))
        print("="*50)

        df = pd.DataFrame([data])

        # =========================
        # MAPPING SE
        # =========================
        map_se = {
            'Sangat tidak setuju': 1,
            'Tidak setuju': 2,
            'Setuju': 3,
            'Sangat setuju': 4
        }

        se_cols = ['SE01','SE02','SE03','SE04','SE05','SE06','SE07','SE08','SE09','SE10']

        for col in se_cols:
            val = df.at[0, col]
            if isinstance(val, str):
                df[col] = df[col].map(map_se)
            df[col] = df[col].fillna(0).astype(int)

        # =========================
        # MAPPING Q
        # =========================
        map_q = {
            'Sangat setuju': 1,
            'Setuju': 2,
            'Agak setuju': 3,
            'Netral': 4,
            'Agak tidak setuju': 5,
            'Tidak setuju': 6,
            'Sangat tidak setuju': 7
        }

        for i in range(1, 19):
            col = f"Q{i}"
            val = df.at[0, col]

            if isinstance(val, str):
                df[col] = df[col].map(map_q)

            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        # =========================
        # REVERSE
        # =========================
        rev_items = ["Q1","Q2","Q3","Q8","Q9","Q11","Q12","Q13","Q17","Q18"]
        for q in rev_items:
            df[q] = 8 - df[q]

        # =========================
        # SUBSCALE
        # =========================
        subscales = {
            "Autonomy": ["Q15", "Q17", "Q18"],
            "Environmental_Mastery": ["Q4", "Q8", "Q9"],
            "Personal_Growth": ["Q11", "Q12", "Q14"],
            "Positive_Relations": ["Q6", "Q13", "Q16"],
            "Purpose_in_Life": ["Q3", "Q7", "Q10"],
            "Self_Acceptance": ["Q1", "Q2", "Q5"]
        }

        for s, items in subscales.items():
            df[s] = df[items].sum(axis=1)

        # =========================
        # ANALISIS WELL-BEING
        # =========================

        wb_scores = {
            key: int(df[key].iloc[0]) for key in subscales.keys()
        }

        max_feature_key = max(wb_scores, key=wb_scores.get)
        min_feature_key = min(wb_scores, key=wb_scores.get)

        max_score = wb_scores[max_feature_key]
        min_score = wb_scores[min_feature_key]

        # label biar lebih manusiawi
        wb_label_map = {
            "Autonomy": "Kemandirian",
            "Environmental_Mastery": "Penguasaan lingkungan",
            "Personal_Growth": "Pengembangan diri",
            "Positive_Relations": "Hubungan positif",
            "Purpose_in_Life": "Tujuan hidup",
            "Self_Acceptance": "Penerimaan diri"
        }

        max_feature = wb_label_map.get(max_feature_key, max_feature_key)
        min_feature = wb_label_map.get(min_feature_key, min_feature_key)

        # =========================
        # TOTAL SCORE
        # =========================
        se_total = df[se_cols].sum(axis=1).iloc[0]
        wb_total = df[list(subscales.keys())].sum(axis=1).iloc[0]
        se_items = {col: int(df[col].iloc[0]) for col in se_cols}

        # =========================
        # ANALISIS SELF-EFFICACY
        # =========================
        se_scores = {
            col: int(df[col].iloc[0]) for col in se_cols
        }

        # cari tertinggi & terendah
        max_se = max(se_scores, key=se_scores.get)
        min_se = min(se_scores, key=se_scores.get)

        max_se_score = se_scores[max_se]
        min_se_score = se_scores[min_se]

        se_label_map = {
            "SE01": "Kemampuan menyelesaikan masalah sulit",
            "SE02": "Kemampuan mencari solusi saat menghadapi hambatan",
            "SE03": "Ketekunan dalam mencapai tujuan",
            "SE04": "Kepercayaan diri menghadapi situasi tak terduga",
            "SE05": "Kecerdikan dalam mengatasi situasi baru",
            "SE06": "Keyakinan menyelesaikan masalah melalui usaha",
            "SE07": "Kemampuan tetap tenang saat menghadapi kesulitan",
            "SE08": "Kemampuan menemukan berbagai alternatif solusi",
            "SE09": "Kemampuan berpikir solusi saat dalam masalah",
            "SE10": "Keyakinan menghadapi berbagai situasi kehidupan"
        }

        max_se = se_label_map.get(max_se, max_se)
        min_se = se_label_map.get(min_se, min_se)

        # =========================
        # TOP 5 FITUR PALING BERPENGARUH
        # =========================

        # Self-Efficacy
        se_sorted = sorted(
            {col: int(df[col].iloc[0]) for col in se_cols}.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Well-Being
        wb_sorted = sorted(
            {key: int(df[key].iloc[0]) for key in subscales.keys()}.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Ambil 3 SE + 2 WB
        top_features = dict(se_sorted[:3] + wb_sorted[:2])

        # =========================
        # FEATURE
        # =========================
        X = df[feature_cols]
        X_scaled = scaler.transform(X)

        # =========================
        # PREDICTION
        # =========================
        pred_encoded = model.predict(X_scaled)[0]
        pred_proba = model.predict_proba(X_scaled)[0]

        label = label_encoder.inverse_transform([pred_encoded])[0]

        # =========================
        # PENJELASAN OTOMATIS
        # =========================
        if label == "high_well_being":
            explanation = (
                f"Berdasarkan jawaban Anda, Anda memiliki kondisi psychological well-being yang baik "
                f"dengan skor total {wb_total}. "

                # WELL BEING
                f"Aspek terkuat Anda terdapat pada {max_feature} dengan skor {max_score}, "
                f"sedangkan aspek yang masih bisa ditingkatkan adalah {min_feature} dengan skor {min_score}. "

                # SELF EFFICACY
                f"Dari sisi self-efficacy, Anda paling percaya diri dalam {max_se} (skor {max_se_score}), "
                f"namun masih perlu meningkatkan {min_se} (skor {min_se_score})."
            )

        else:
            explanation = (
                f"Berdasarkan jawaban Anda, Anda berada dalam kategori low well-being "
                f"dengan skor total {wb_total}. "

                # WELL BEING
                f"Aspek yang paling perlu diperhatikan adalah {min_feature} dengan skor {min_score}, "
                f"sementara kekuatan Anda terdapat pada {max_feature} dengan skor {max_score}. "

                # SELF EFFICACY
                f"Dari sisi self-efficacy, Anda cukup baik dalam {max_se} (skor {max_se_score}), "
                f"namun masih perlu meningkatkan {min_se} (skor {min_se_score}) untuk hasil yang lebih optimal."
            )

        # =========================
        # CONFIDENCE
        # =========================
        confidence = {
            label_encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(pred_proba)
        }

        # =========================
        # RESPONSE (SUPER LENGKAP)
        # =========================
        response = {
            "prediction": label,
            "explanation": explanation,
            "confidence": confidence,
            "top_features": top_features,

            "scores": {
                "self_efficacy": {
                    "total": int(se_total),
                    "max": 40,
                    "percentage": round((se_total / 40) * 100, 2),
                    "items": se_items
                },
                "well_being": {
                    "total": int(wb_total),
                    "max": 126,
                    "percentage": round((wb_total / 126) * 100, 2)
                },
                "subscales": {
                    "Autonomy": int(df["Autonomy"].iloc[0]),
                    "Environmental_Mastery": int(df["Environmental_Mastery"].iloc[0]),
                    "Personal_Growth": int(df["Personal_Growth"].iloc[0]),
                    "Positive_Relations": int(df["Positive_Relations"].iloc[0]),
                    "Purpose_in_Life": int(df["Purpose_in_Life"].iloc[0]),
                    "Self_Acceptance": int(df["Self_Acceptance"].iloc[0])
                }
            }
        }

        print("\nRESPONSE:")
        print(json.dumps(response, indent=2))
        print("="*60)

        return jsonify(response)

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({'error': str(e)}), 500


# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)