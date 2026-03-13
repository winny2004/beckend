import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

def categorize_DAS(row):
    """
    Kategorisasi skor DASS menjadi label
    """
    depression = row["Depression_Score"]
    anxiety = row["Anxiety_Score"]
    stress = row["Stress_Score"]

    # Fungsi untuk mendapatkan level severity
    def get_severity_level(score, type):
        if type == "Depression":
            if score >= 28:
                return 4  # Extremely severe
            elif score >= 21:
                return 3  # Severe
            elif score >= 14:
                return 2  # Moderate
            elif score >= 10:
                return 1  # Mild
            else:
                return 0  # Normal
        elif type == "Anxiety":
            if score >= 20:
                return 4  # Extremely severe
            elif score >= 15:
                return 3  # Severe
            elif score >= 10:
                return 2  # Moderate
            elif score >= 8:
                return 1  # Mild
            else:
                return 0  # Normal
        else:  # stress
            if score >= 34:
                return 4  # Extremely severe
            elif score >= 26:
                return 3  # Severe
            elif score >= 19:
                return 2  # Moderate
            elif score >= 15:
                return 1  # Mild
            else:
                return 0  # Normal

    # Cek apakah semua skor dalam kategori Normal
    if depression <= 9 and anxiety <= 7 and stress <= 14:
        return "Normal"
    else:
        # Cek apakah masuk kategori abnormal
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

        # Hitung level severity untuk yang abnormal
        depression_level = get_severity_level(depression, "Depression") if depression_abnormal else 0
        anxiety_level = get_severity_level(anxiety, "Anxiety") if anxiety_abnormal else 0
        stress_level = get_severity_level(stress, "Stress") if stress_abnormal else 0

        # Pilih berdasarkan level tertinggi
        if depression_level > anxiety_level and depression_level > stress_level:
            return "Depression"
        elif anxiety_level > depression_level and anxiety_level > stress_level:
            return "Anxiety"
        elif stress_level > depression_level and stress_level > anxiety_level:
            return "Stress"
        else:
            # Jika level sama, pilih berdasarkan skor tertinggi
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

def train_random_forest(data_path, model_output_path='models/random_forest_model.pkl', 
                        scaler_output_path='models/scaler.pkl', encoder_output_path='models/label_encoder.pkl'):
    """
    Training Random Forest Model
    """
    print("Loading data...")
    
    # Detect file type and read accordingly
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
        # Convert MSPSS text responses to numeric
        likert_mapping = {
            'Sangat tidak setuju': 1,
            'Tidak setuju': 2,
            'Netral': 3,
            'Setuju': 4,
            'Sangat setuju': 5
        }
        
        # Convert FS columns (MSPSS)
        fs_columns = [col for col in df.columns if col.startswith('FS')]
        for col in fs_columns:
            df[col] = df[col].map(likert_mapping)
            # Fill NaN values with median
            df[col] = df[col].fillna(df[col].median()).astype(int)
        
        # Convert DAS21 columns to numeric
        for i in range(1, 22):
            col = f'DAS{i}'
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    else:
        df = pd.read_csv(data_path)
    
    print("\nData shape:", df.shape)
    print("Columns:", df.columns.tolist())
    
    # Pastikan kolom yang diperlukan ada
    required_columns = [
        'FS1', 'FS2', 'FS3', 'FS4', 'FS5', 'FS6', 'FS7', 'FS8', 'FS9', 'FS10', 'FS11', 'FS12',
        'DAS1', 'DAS2', 'DAS3', 'DAS4', 'DAS5', 'DAS6', 'DAS7', 'DAS8', 'DAS9', 'DAS10',
        'DAS11', 'DAS12', 'DAS13', 'DAS14', 'DAS15', 'DAS16', 'DAS17', 'DAS18', 'DAS19', 'DAS20', 'DAS21'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Kolom {col} tidak ditemukan dalam dataset!")
    
    # Preprocessing MSPSS
    significant_other = ['FS1', 'FS2', 'FS3', 'FS4']
    family = ['FS5', 'FS6', 'FS7', 'FS8']
    friends = ['FS9', 'FS10', 'FS11', 'FS12']
    
    # Hitung skor MSPSS
    df['SO_Score'] = df[significant_other].mean(axis=1)
    df['Family_Score'] = df[family].mean(axis=1)
    df['Friends_Score'] = df[friends].mean(axis=1)
    df['MSPSS'] = df[['SO_Score', 'Family_Score', 'Friends_Score']].mean(axis=1)
    
    # Preprocessing DASS-21
    depression_items = ['DAS1', 'DAS2', 'DAS3', 'DAS4', 'DAS5', 'DAS6', 'DAS7']
    anxiety_items = ['DAS8', 'DAS9', 'DAS10', 'DAS11', 'DAS12', 'DAS13', 'DAS14']
    stress_items = ['DAS15', 'DAS16', 'DAS17', 'DAS18', 'DAS19', 'DAS20', 'DAS21']
    
    # Hitung skor DASS
    df['Depression_Score'] = df[depression_items].sum(axis=1) * 2
    df['Anxiety_Score'] = df[anxiety_items].sum(axis=1) * 2
    df['Stress_Score'] = df[stress_items].sum(axis=1) * 2
    
    # Buat label DASS
    df["DASS_Label"] = df.apply(categorize_DAS, axis=1)
    
    print("\nDistribusi Label:")
    print(df['DASS_Label'].value_counts())
    
    # Tentukan fitur (X) dan target (y)
    X = df[['SO_Score', 'Family_Score', 'Friends_Score', 'Stress_Score', 'Anxiety_Score', 'Depression_Score']]
    y = df[['DASS_Label']]
    
    # Initialize LabelEncoder dan Scaler
    le = LabelEncoder()
    scaler = StandardScaler()
    
    # Encode target
    y_encoded = le.fit_transform(y.values.ravel())
    
    # Scaling fitur
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    
    # Balancing menggunakan SMOTE
    print("\nDistribusi sebelum SMOTE:")
    print(pd.Series(y_train).value_counts())
    
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print("\nDistribusi setelah SMOTE:")
    print(pd.Series(y_train_balanced).value_counts())
    
    # Training Random Forest
    print("\nTraining Random Forest Model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    rf_model.fit(X_train_balanced, y_train_balanced)
    
    # Prediksi
    y_pred = rf_model.predict(X_test)
    
    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "="*60)
    print("RANDOM FOREST MODEL EVALUATION")
    print("="*60)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Feature Importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    feature_names = X.columns
    importances = rf_model.feature_importances_
    
    for feature, importance in zip(feature_names, importances):
        print(f"{feature}: {importance:.4f}")
    
    # Simpan model
    import os
    os.makedirs('models', exist_ok=True)
    
    print(f"\nSaving model to {model_output_path}...")
    joblib.dump(rf_model, model_output_path)
    
    print(f"Saving scaler to {scaler_output_path}...")
    joblib.dump(scaler, scaler_output_path)
    
    print(f"Saving label encoder to {encoder_output_path}...")
    joblib.dump(le, encoder_output_path)
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED!")
    print("="*60)
    print(f"Model saved: {model_output_path}")
    print(f"Scaler saved: {scaler_output_path}")
    print(f"Label encoder saved: {encoder_output_path}")
    
    return rf_model, scaler, le

if __name__ == "__main__":
    # Ganti dengan path dataset Anda
    # Gunakan file Excel untuk dataset asli
    data_path = "dataset/DATASET TA.xlsx"
    
    train_random_forest(data_path)
