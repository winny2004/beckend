import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
import os

def train_model(data_path):

    print("Loading data...")
    df = pd.read_csv(data_path, sep=';')

    # =========================
    # MAPPING SELF-EFFICACY
    # =========================
    map_se = {
        'Sangat tidak setuju': 1,
        'Tidak setuju': 2,
        'Setuju': 3,
        'Sangat setuju': 4
    }

    se_cols = ['SE01','SE02','SE03','SE04','SE05','SE06','SE07','SE08','SE09','SE10']

    for col in se_cols:
        df[col] = df[col].map(map_se).fillna(0).astype(int)

    # =========================
    # MAPPING WELL-BEING
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

    for i in range(1,19):
        col = f"Q{i}"
        if col in df.columns:
            df[col] = df[col].map(map_q).fillna(0).astype(int)

    # =========================
    # REVERSE SCORING
    # =========================
    rev_items = ["Q1","Q2","Q3","Q8","Q9","Q11","Q12","Q13","Q17","Q18"]
    for q in rev_items:
        if q in df.columns:
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
        df[s] = df[items].sum(axis=1, skipna=True)

    df["WB_Total"] = df[list(subscales.keys())].sum(axis=1)

    # =========================
    # LABELING
    # =========================
    median_wb = df["WB_Total"].median()
    df["WB_Label"] = df["WB_Total"].apply(
        lambda x: "low_well_being" if x <= median_wb else "high_well_being"
    )

    # =========================
    # FEATURE
    # =========================
    feature_cols = se_cols + list(subscales.keys())
    X = df[feature_cols]
    y = df["WB_Label"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # =========================
    # TRAIN SVM
    # =========================
    print("Training SVM...")
    model = SVC(probability=True, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # =========================
    # SAVE MODEL
    # =========================
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/model_svm.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(le, "models/label_encoder.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")

    print("\nModel saved successfully!")

    return model


if __name__ == "__main__":
    train_model(r"C:\kuisioner_mental-health\ml_api\dataset\DATASET SELF-EFFICACY_WELL-BEING.csv")