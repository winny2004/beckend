from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import json
import os

app = Flask(__name__)
CORS(app)

# =========================
# LOAD DASS-21 MODELS (Random Forest)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dass_model_path = os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl')
dass_scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
dass_encoder_path = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')

try:
    dass_model = joblib.load(dass_model_path)
    dass_scaler = joblib.load(dass_scaler_path)
    dass_label_encoder = joblib.load(dass_encoder_path)
    print("DASS-21 Model, scaler, and label encoder loaded successfully!")
except Exception as e:
    print(f"Error loading DASS-21 model: {e}")
    dass_model = None
    dass_scaler = None
    dass_label_encoder = None

# =========================
# LOAD SELF EFFICACY MODELS (SVM)
# =========================
se_model_path = os.path.join(BASE_DIR, 'models', 'model_svm.pkl')
se_scaler_path = os.path.join(BASE_DIR, 'models', 'scaler_se.pkl')
se_encoder_path = os.path.join(BASE_DIR, 'models', 'label_encoder_se.pkl')
se_feature_path = os.path.join(BASE_DIR, 'models', 'feature_cols.pkl')

try:
    se_model = joblib.load(se_model_path)
    se_scaler = joblib.load(se_scaler_path)
    se_label_encoder = joblib.load(se_encoder_path)
    se_feature_cols = joblib.load(se_feature_path)
    print("Self Efficacy Model, scaler, encoder, and features loaded successfully!")
except Exception as e:
    print(f"Error loading Self Efficacy model: {e}")
    se_model = None
    se_scaler = None
    se_label_encoder = None
    se_feature_cols = None

def get_severity_category(score, type):
    """
    Mendapatkan kategori severity berdasarkan skor
    """
    if type == "Depression":
        if score >= 28:
            return "Extremely Severe", "Sangat Berat"
        elif score >= 21:
            return "Severe", "Berat"
        elif score >= 14:
            return "Moderate", "Sedang"
        elif score >= 10:
            return "Mild", "Ringan"
        else:
            return "Normal", "Normal"
    elif type == "Anxiety":
        if score >= 20:
            return "Extremely Severe", "Sangat Berat"
        elif score >= 15:
            return "Severe", "Berat"
        elif score >= 10:
            return "Moderate", "Sedang"
        elif score >= 8:
            return "Mild", "Ringan"
        else:
            return "Normal", "Normal"
    else:  # Stress
        if score >= 34:
            return "Extremely Severe", "Sangat Berat"
        elif score >= 26:
            return "Severe", "Berat"
        elif score >= 19:
            return "Moderate", "Sedang"
        elif score >= 15:
            return "Mild", "Ringan"
        else:
            return "Normal", "Normal"

def generate_explanation(prediction_label, depression_score, anxiety_score, stress_score, 
                         depression_category_id, anxiety_category_id, stress_category_id,
                         so_score, family_score, friends_score):
    """
    Generate penjelasan hasil kuisioner untuk user
    """
    # Mapping kategori ke Bahasa Indonesia
    category_mapping = {
        'Depression': 'Depresi',
        'Anxiety': 'Cemas',
        'Stress': 'Stres',
        'Normal': 'Normal'
    }
    
    prediction_id = category_mapping.get(prediction_label, prediction_label)
    
    # Tentukan kategori mana yang menjadi runner-up (skor tertinggi kedua)
    scores = {
        'Depression': depression_score,
        'Anxiety': anxiety_score,
        'Stress': stress_score
    }
    
    # Sort by score descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    highest_category, highest_score = sorted_scores[0]
    second_highest_category, second_highest_score = sorted_scores[1] if len(sorted_scores) > 1 else (None, 0)
    
    # Mapping severity ke Bahasa Indonesia
    severity_mapping = {
        'Normal': 'Normal',
        'Mild': 'Ringan',
        'Moderate': 'Sedang',
        'Severe': 'Berat',
        'Extremely Severe': 'Sangat Berat'
    }
    
    # Generate penjelasan utama
    if prediction_label == 'Normal':
        main_explanation = f"Berdasarkan jawaban Anda, kondisi kesehatan mental Anda berada dalam kategori Normal. Skor depresi ({depression_score}), cemas ({anxiety_score}), dan stres ({stress_score}) Anda masih dalam rentang yang sehat."
    else:
        # Dapatkan kategori severity dalam Bahasa Indonesia
        categories_id = {
            'Depression': severity_mapping.get(depression_category_id, depression_category_id),
            'Anxiety': severity_mapping.get(anxiety_category_id, anxiety_category_id),
            'Stress': severity_mapping.get(stress_category_id, stress_category_id)
        }
        
        prediction_severity_id = categories_id.get(prediction_label, '')
        
        # Cek apakah ada kategori lain dengan skor mendekati
        if second_highest_category and abs(highest_score - second_highest_score) < 5:
            second_highest_id = category_mapping.get(second_highest_category, second_highest_category)
            second_highest_severity = categories_id.get(second_highest_category, '')
            
            main_explanation = f"Berdasarkan jawaban Anda, Anda mengalami {prediction_id}. Skor {prediction_id} ({highest_score}) dan {second_highest_id} ({second_highest_score}) Anda sama-sama tinggi dan termasuk dalam kategori {prediction_severity_id}. Artinya, gejala {prediction_id} yang Anda alami sudah sangat menonjol dan memerlukan perhatian yang serius."
        else:
            main_explanation = f"Berdasarkan jawaban Anda, Anda mengalasi {prediction_id} dengan skor {highest_score}. Gejala {prediction_id} yang Anda tunjukkan sudah termasuk dalam kategori {prediction_severity_id} dan memerlukan perhatian yang serius."
    
    # Generate penjelasan dukungan sosial
    mspps_avg = (so_score + family_score + friends_score) / 3
    if mspps_avg >= 4:
        support_explanation = f"Dukungan sosial Anda tergolong tinggi (rata-rata: {mspps_avg:.1f}/5). Dukungan baik dari keluarga ({family_score:.1f}), teman ({friends_score:.1f}), dan orang terdekat ({so_score:.1f}) dapat menjadi faktor protektif yang membantu mengurangi dampak dari gejala yang Anda alami."
    elif mspps_avg >= 3:
        support_explanation = f"Dukungan sosial Anda tergolong sedang (rata-rata: {mspps_avg:.1f}/5). Dukungan dari keluarga ({family_score:.1f}), teman ({friends_score:.1f}), dan orang terdekat ({so_score:.1f}) dapat membantu, namun masih dapat ditingkatkan untuk mendukung pemulihan kondisi Anda."
    else:
        support_explanation = f"Dukungan sosial Anda tergolong rendah (rata-rata: {mspps_avg:.1f}/5). Kurangnya dukungan dari keluarga ({family_score:.1f}), teman ({friends_score:.1f}), atau orang terdekat ({so_score:.1f}) dapat menjadi faktor yang memperburuk kondisi kesehatan mental Anda. Mencari dukungan dari orang-orang terdekat dapat membantu proses pemulihan."
    
    # Combine penjelasan
    full_explanation = f"{main_explanation} {support_explanation}"
    
    return full_explanation

def categorize_DAS(row):
    """
    Kategorisasi skor DASS menjadi label
    """
    depression = row["Depression_Score"]
    anxiety = row["Anxiety_Score"]
    stress = row["Stress_Score"]

    def get_severity_level(score, type):
        if type == "Depression":
            if score >= 28:
                return 4
            elif score >= 21:
                return 3
            elif score >= 14:
                return 2
            elif score >= 10:
                return 1
            else:
                return 0
        elif type == "Anxiety":
            if score >= 20:
                return 4
            elif score >= 15:
                return 3
            elif score >= 10:
                return 2
            elif score >= 8:
                return 1
            else:
                return 0
        else:  # stress
            if score >= 34:
                return 4
            elif score >= 26:
                return 3
            elif score >= 19:
                return 2
            elif score >= 15:
                return 1
            else:
                return 0

    if depression <= 9 and anxiety <= 7 and stress <= 14:
        return "Normal"
    else:
        if depression > 9:
            depression_abnormal = True
        else:
            depression_abnormal = False

        if anxiety > 7:
            anxiety_abnormal = True
        else:
            anxiety_abnormal = False

        if stress > 14:
            stress_abnormal = True
        else:
            stress_abnormal = False

        depression_level = get_severity_level(depression, "Depression") if depression_abnormal else 0
        anxiety_level = get_severity_level(anxiety, "Anxiety") if anxiety_abnormal else 0
        stress_level = get_severity_level(stress, "Stress") if stress_abnormal else 0

        if depression_level > anxiety_level and depression_level > stress_level:
            return "Depression"
        elif anxiety_level > depression_level and anxiety_level > stress_level:
            return "Anxiety"
        elif stress_level > depression_level and stress_level > anxiety_level:
            return "Stress"
        else:
            if depression_level == anxiety_level and depression_level == stress_level:
                if depression >= anxiety and depression >= stress:
                    return "Depression"
                elif anxiety >= depression and anxiety >= stress:
                    return "Anxiety"
                else:
                    return "Stress"
            elif depression_level == anxiety_level and depression_level > stress_level:
                if depression >= anxiety:
                    return "Depression"
                else:
                    return "Anxiety"
            elif depression_level == stress_level and depression_level > anxiety_level:
                if depression >= stress:
                    return "Depression"
                else:
                    return "Stress"
            elif anxiety_level == stress_level and anxiety_level > depression_level:
                if anxiety >= stress:
                    return "Anxiety"
                else:
                    return "Stress"
            else:
                if depression >= anxiety and depression >= stress:
                    return "Depression"
                elif anxiety >= depression and anxiety >= stress:
                    return "Anxiety"
                else:
                    return "Stress"

@app.route('/')
def home():
    return jsonify({
        'message': 'Mental Health Prediction API',
        'version': '1.0',
        'endpoints': {
            'dass_predict': '/api/dass/predict',
            'se_predict': '/api/se/predict',
            'health': '/api/health'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'dass_model_loaded': dass_model is not None,
        'se_model_loaded': se_model is not None
    })

# =========================
# DASS-21 PREDICTION ENDPOINT
# =========================
@app.route('/api/dass/predict', methods=['POST'])
def predict_dass():
    if dass_model is None or dass_scaler is None or dass_label_encoder is None:
        return jsonify({'error': 'DASS-21 Model not loaded'}), 500

    try:
        data = request.get_json()
        
        # DEBUG: Print incoming request
        print("\n" + "="*60)
        print("REQUEST DATA:")
        print("="*60)
        import json
        print(json.dumps(data, indent=2))
        print("="*60 + "\n")
        
        # Ekstrak data dari request
        # Expected format: {
        #   "fs1": 4, "fs2": 5, ..., "fs12": 4,
        #   "das1": 0, "das2": 1, ..., "das21": 0
        # }
        
        # Mapping Likert untuk MSPSS
        likert_mapping = {
            'Sangat tidak setuju': 1,
            'Tidak setuju': 2,
            'Netral': 3,
            'Setuju': 4,
            'Sangat setuju': 5
        }
        
        # Proses MSPSS data
        fs_data = []
        for i in range(1, 13):
            key = f'fs{i}'
            value = data.get(key, 3)  # Default Netral
            
            # Convert string to numeric if needed
            if isinstance(value, str):
                value = likert_mapping.get(value, 3)
            
            fs_data.append(int(value))
        
        # Proses DASS-21 data
        dass_data = []
        for i in range(1, 22):
            key = f'das{i}'
            value = data.get(key, 0)
            dass_data.append(int(value))
        
        # Hitung skor MSPSS
        significant_other = [fs_data[0], fs_data[1], fs_data[2], fs_data[3]]  # FS1-FS4
        family = [fs_data[4], fs_data[5], fs_data[6], fs_data[7]]  # FS5-FS8
        friends = [fs_data[8], fs_data[9], fs_data[10], fs_data[11]]  # FS9-FS12
        
        so_score = np.mean(significant_other)
        family_score = np.mean(family)
        friends_score = np.mean(friends)
        
        # Hitung skor DASS-21
        depression_items = dass_data[0:7]  # DAS1-DAS7
        anxiety_items = dass_data[7:14]    # DAS8-DAS14
        stress_items = dass_data[14:21]    # DAS15-DAS21
        
        # Validasi range 0-3 untuk setiap item DASS
        depression_items = [max(0, min(3, int(x))) for x in depression_items]
        anxiety_items = [max(0, min(3, int(x))) for x in anxiety_items]
        stress_items = [max(0, min(3, int(x))) for x in stress_items]
        
        # Skor per kategori (7 items × 0-3 = 0-21, dikali 2 = 0-42)
        depression_score = np.sum(depression_items) * 2
        anxiety_score = np.sum(anxiety_items) * 2
        stress_score = np.sum(stress_items) * 2
        
        # Total DASS (semua items: 21 items × 0-3 = 0-63, dikali 2 = 0-126)
        # ATAU total dari 3 kategori: 0-42 + 0-42 + 0-42 = 0-126
        total_dass_score = depression_score + anxiety_score + stress_score
        
        # Buat dataframe untuk prediksi
        input_data = pd.DataFrame({
            'SO_Score': [so_score],
            'Family_Score': [family_score],
            'Friends_Score': [friends_score],
            'Stress_Score': [stress_score],
            'Anxiety_Score': [anxiety_score],
            'Depression_Score': [depression_score]
        })
        
        # Scale input
        input_scaled = dass_scaler.transform(input_data)

        # Prediksi
        prediction_encoded = dass_model.predict(input_scaled)[0]
        prediction_proba = dass_model.predict_proba(input_scaled)[0]

        # Decode label
        prediction_label = dass_label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Buat manual label untuk verifikasi
        manual_row = {
            'Depression_Score': depression_score,
            'Anxiety_Score': anxiety_score,
            'Stress_Score': stress_score
        }
        manual_label = categorize_DAS(manual_row)
        
        # Get severity categories untuk setiap kondisi
        depression_category_en, depression_category_id = get_severity_category(depression_score, "Depression")
        anxiety_category_en, anxiety_category_id = get_severity_category(anxiety_score, "Anxiety")
        stress_category_en, stress_category_id = get_severity_category(stress_score, "Stress")

        # Generate penjelasan untuk user
        explanation = generate_explanation(
            manual_label, depression_score, anxiety_score, stress_score,
            depression_category_id, anxiety_category_id, stress_category_id,
            so_score, family_score, friends_score
        )

        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Response
        response = {
            'prediction': prediction_label,
            'manual_calculation': manual_label,
            'explanation': explanation,
            'confidence': {
                'Anxiety': f"{float(prediction_proba[dass_label_encoder.transform(['Anxiety'])[0]]):.4f}",
                'Depression': f"{float(prediction_proba[dass_label_encoder.transform(['Depression'])[0]]):.4f}",
                'Normal': f"{float(prediction_proba[dass_label_encoder.transform(['Normal'])[0]]):.4f}",
                'Stress': f"{float(prediction_proba[dass_label_encoder.transform(['Stress'])[0]]):.4f}"
            },
            'categories': {
                'Depression': {
                    'score': int(depression_score),
                    'category_en': depression_category_en,
                    'category_id': depression_category_id
                },
                'Anxiety': {
                    'score': int(anxiety_score),
                    'category_en': anxiety_category_en,
                    'category_id': anxiety_category_id
                },
                'Stress': {
                    'score': int(stress_score),
                    'category_en': stress_category_en,
                    'category_id': stress_category_id
                }
            },
            'scores': {
                'MSPSS': {
                    'Significant_Other': f"{float(so_score):.2f}",
                    'Family': f"{float(family_score):.2f}",
                    'Friends': f"{float(friends_score):.2f}",
                    'Total': f"{float(np.mean([so_score, family_score, friends_score])):.2f}"
                },
                'DASS_21': {
                    'Depression': int(depression_score),
                    'Anxiety': int(anxiety_score),
                    'Stress': int(stress_score),
                    'Total': int(total_dass_score),
                    'Total_Max': 126  # 21 items × max 3 points × 2 = 126
                }
            }
        }
        
        # DEBUG: Print response
        print("\n" + "="*60)
        print("RESPONSE DATA:")
        print("="*60)
        print(json.dumps(response, indent=2))
        print("="*60 + "\n")
        
        return jsonify(response)
        
    except Exception as e:
        # DEBUG: Print exception
        print("\n" + "="*60)
        print("EXCEPTION OCCURRED:")
        print("="*60)
        print(str(e))
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        return jsonify({'error': str(e)}), 500

# =========================
# SELF EFFICACY PREDICTION ENDPOINT
# =========================
@app.route('/api/se/predict', methods=['POST'])
def predict_se():
    if se_model is None or se_scaler is None or se_label_encoder is None:
        return jsonify({'error': 'Self Efficacy Model not loaded'}), 500

    try:
        data = request.get_json()

        print("\n" + "="*50)
        print("SELF EFFICACY REQUEST DATA:")
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
        rev_items = ["Q2","Q3","Q5","Q6","Q7","Q8","Q11","Q13","Q16","Q17"]
        for q in rev_items:
            df[q] = 8 - df[q]

        # =========================
        # SUBSCALE
        # =========================
        subscales = {
            "Autonomy": ["Q1", "Q2", "Q3"],
            "Environmental_Mastery": ["Q4", "Q5", "Q6"],
            "Personal_Growth": ["Q7", "Q8", "Q9"],
            "Positive_Relations": ["Q10", "Q11", "Q12"],
            "Purpose_in_Life": ["Q13", "Q14", "Q15"],
            "Self_Acceptance": ["Q16", "Q17", "Q18"]
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
        X = df[se_feature_cols]
        X_scaled = se_scaler.transform(X)

        # =========================
        # PREDICTION
        # =========================
        pred_encoded = se_model.predict(X_scaled)[0]
        pred_proba = se_model.predict_proba(X_scaled)[0]

        label = se_label_encoder.inverse_transform([pred_encoded])[0]

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
            se_label_encoder.inverse_transform([i])[0]: float(prob)
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

        print("\nSELF EFFICACY RESPONSE:")
        print(json.dumps(response, indent=2))
        print("="*60)

        return jsonify(response)

    except Exception as e:
        print("SELF EFFICACY ERROR:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
