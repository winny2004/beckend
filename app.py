from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
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
    print("DASS-21 Model loaded successfully!")
except Exception as e:
    print(f"Error loading DASS-21 model: {e}")
    dass_model = None
    dass_scaler = None
    dass_label_encoder = None


# =========================
# SEVERITY HELPERS
# =========================
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

    else:  # Stress
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


def get_severity_category(score, type):
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


# =========================
# MAIN CATEGORY RESOLVER
# =========================
def resolve_main_category(depression_score, anxiety_score, stress_score):
    """
    Priority:
    Depression > Anxiety > Stress
    """

    # =========================
    # HANDLE NORMAL
    # =========================
    if (
        depression_score <= 9 and
        anxiety_score <= 7 and
        stress_score <= 14
    ):
        return "Normal"

    levels = {
        'Depression': get_severity_level(depression_score, "Depression"),
        'Anxiety': get_severity_level(anxiety_score, "Anxiety"),
        'Stress': get_severity_level(stress_score, "Stress")
    }

    scores = {
        'Depression': depression_score,
        'Anxiety': anxiety_score,
        'Stress': stress_score
    }

    priority = ['Depression', 'Anxiety', 'Stress']

    max_level = max(levels.values())

    candidates = [k for k, v in levels.items() if v == max_level]

    max_score = max(scores[k] for k in candidates)

    candidates = [k for k in candidates if scores[k] == max_score]

    for p in priority:
        if p in candidates:
            return p

    return "Normal"


# =========================
# CATEGORIZATION
# =========================
def categorize_DAS(row):
    depression = row["Depression_Score"]
    anxiety = row["Anxiety_Score"]
    stress = row["Stress_Score"]

    if depression <= 9 and anxiety <= 7 and stress <= 14:
        return "Normal"

    return resolve_main_category(
        depression,
        anxiety,
        stress
    )


# =========================
# EXPLANATION
# =========================
def generate_explanation(
    prediction_label,
    depression_score,
    anxiety_score,
    stress_score,
    depression_category_id,
    anxiety_category_id,
    stress_category_id,
    so_score,
    family_score,
    friends_score
):

    category_mapping = {
        'Depression': 'Depresi',
        'Anxiety': 'Cemas',
        'Stress': 'Stres',
        'Normal': 'Normal'
    }

    severity_mapping = {
        'Normal': 'Normal',
        'Mild': 'Ringan',
        'Moderate': 'Sedang',
        'Severe': 'Berat',
        'Extremely Severe': 'Sangat Berat'
    }

    scores_dict = {
        'Depression': depression_score,
        'Anxiety': anxiety_score,
        'Stress': stress_score
    }

    highest_category = resolve_main_category(
        depression_score,
        anxiety_score,
        stress_score
    )

    highest_score = scores_dict[highest_category]

    remaining = {
        k: v for k, v in scores_dict.items()
        if k != highest_category
    }

    if remaining:
        second_highest_category = max(remaining, key=remaining.get)
        second_highest_score = remaining[second_highest_category]
    else:
        second_highest_category = None
        second_highest_score = 0

    highest_category_id = category_mapping.get(
        highest_category,
        highest_category
    )

    categories_id = {
        'Depression': severity_mapping.get(
            depression_category_id,
            depression_category_id
        ),
        'Anxiety': severity_mapping.get(
            anxiety_category_id,
            anxiety_category_id
        ),
        'Stress': severity_mapping.get(
            stress_category_id,
            stress_category_id
        )
    }

    highest_severity_id = categories_id.get(
        highest_category,
        ''
    )

    if prediction_label == 'Normal':

        main_explanation = (
            f"Berdasarkan jawaban Anda, kondisi kesehatan mental "
            f"Anda berada dalam kategori Normal. "
            f"Skor depresi ({depression_score}), "
            f"cemas ({anxiety_score}), dan "
            f"stres ({stress_score}) Anda masih "
            f"dalam rentang yang sehat."
        )

    else:

        if (
            second_highest_category and
            abs(highest_score - second_highest_score) < 5
        ):

            second_highest_id = category_mapping.get(
                second_highest_category,
                second_highest_category
            )

            main_explanation = (
                f"Berdasarkan jawaban Anda, Anda mengalami "
                f"{highest_category_id} dengan skor {highest_score}. "
                f"Skor {highest_category_id} dan "
                f"{second_highest_id} Anda sama-sama tinggi "
                f"dan memerlukan perhatian yang serius."
            )

        else:

            main_explanation = (
                f"Berdasarkan jawaban Anda, Anda mengalami "
                f"{highest_category_id} dengan skor "
                f"{highest_score}. "
                f"Gejala {highest_category_id} yang Anda "
                f"tunjukkan sudah termasuk dalam kategori "
                f"{highest_severity_id} dan memerlukan "
                f"perhatian yang serius."
            )

    mspps_avg = (
        so_score +
        family_score +
        friends_score
    ) / 3

    if mspps_avg >= 4:

        support_explanation = (
            f"Dukungan sosial Anda tergolong tinggi "
            f"(rata-rata: {mspps_avg:.1f}/5). "
            f"Dukungan baik dari keluarga "
            f"({family_score:.1f}), teman "
            f"({friends_score:.1f}), dan orang terdekat "
            f"({so_score:.1f}) dapat menjadi faktor "
            f"protektif yang membantu mengurangi dampak "
            f"dari gejala yang Anda alami."
        )

    elif mspps_avg >= 3:

        support_explanation = (
            f"Dukungan sosial Anda tergolong sedang "
            f"(rata-rata: {mspps_avg:.1f}/5). "
            f"Dukungan dari keluarga "
            f"({family_score:.1f}), teman "
            f"({friends_score:.1f}), dan orang terdekat "
            f"({so_score:.1f}) dapat membantu, namun "
            f"masih dapat ditingkatkan untuk mendukung "
            f"pemulihan kondisi Anda."
        )

    else:

        support_explanation = (
            f"Dukungan sosial Anda tergolong rendah "
            f"(rata-rata: {mspps_avg:.1f}/5). "
            f"Kurangnya dukungan dari keluarga "
            f"({family_score:.1f}), teman "
            f"({friends_score:.1f}), atau orang terdekat "
            f"({so_score:.1f}) dapat menjadi faktor "
            f"yang memperburuk kondisi kesehatan mental "
            f"Anda."
        )

    return f"{main_explanation} {support_explanation}"


@app.route('/')
def home():
    return jsonify({
        'message': 'Mental Health Prediction API',
        'version': '2.0',
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'dass_model_loaded': dass_model is not None
    })


@app.route('/api/dass/predict', methods=['POST'])
def predict_dass():

    if dass_model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:

        data = request.get_json()

        likert_mapping = {
            'Sangat tidak setuju': 1,
            'Tidak setuju': 2,
            'Netral': 3,
            'Setuju': 4,
            'Sangat setuju': 5
        }

        # MSPSS
        fs_data = []

        for i in range(1, 13):
            key = f'fs{i}'
            value = data.get(key, 3)

            if isinstance(value, str):
                value = likert_mapping.get(value, 3)

            fs_data.append(int(value))

        # DASS
        dass_data = []

        for i in range(1, 22):
            key = f'das{i}'
            value = data.get(key, 0)
            dass_data.append(int(value))

        significant_other = fs_data[0:4]
        family = fs_data[4:8]
        friends = fs_data[8:12]

        so_score = np.mean(significant_other)
        family_score = np.mean(family)
        friends_score = np.mean(friends)

        depression_items = [max(0, min(3, int(x))) for x in dass_data[0:7]]
        anxiety_items = [max(0, min(3, int(x))) for x in dass_data[7:14]]
        stress_items = [max(0, min(3, int(x))) for x in dass_data[14:21]]

        depression_score = np.sum(depression_items) * 2
        anxiety_score = np.sum(anxiety_items) * 2
        stress_score = np.sum(stress_items) * 2

        total_dass_score = (
            depression_score +
            anxiety_score +
            stress_score
        )

        input_data = pd.DataFrame({
            'SO_Score': [so_score],
            'Family_Score': [family_score],
            'Friends_Score': [friends_score],
            'Stress_Score': [stress_score],
            'Anxiety_Score': [anxiety_score],
            'Depression_Score': [depression_score]
        })

        input_scaled = dass_scaler.transform(input_data)

        # =========================
        # ML Prediction
        # =========================
        prediction_encoded = dass_model.predict(input_scaled)[0]
        prediction_proba = dass_model.predict_proba(input_scaled)[0]

        ml_prediction_label = dass_label_encoder.inverse_transform(
            [prediction_encoded]
        )[0]

        # =========================
        # FINAL PREDICTION
        # =========================
        prediction_label = resolve_main_category(
            depression_score,
            anxiety_score,
            stress_score
        )

        manual_row = {
            'Depression_Score': depression_score,
            'Anxiety_Score': anxiety_score,
            'Stress_Score': stress_score
        }

        manual_label = categorize_DAS(manual_row)

        depression_category_en, depression_category_id = get_severity_category(
            depression_score,
            "Depression"
        )

        anxiety_category_en, anxiety_category_id = get_severity_category(
            anxiety_score,
            "Anxiety"
        )

        stress_category_en, stress_category_id = get_severity_category(
            stress_score,
            "Stress"
        )

        explanation = generate_explanation(
            prediction_label,
            depression_score,
            anxiety_score,
            stress_score,
            depression_category_id,
            anxiety_category_id,
            stress_category_id,
            so_score,
            family_score,
            friends_score
        )

        response = {
            'prediction': prediction_label,
            'ml_prediction': ml_prediction_label,
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
                    'Total_Max': 126
                }
            }
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )
