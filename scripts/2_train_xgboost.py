import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# 1. Baca data bersih hasil PySpark (dari Sel 2)
df = pd.read_parquet("diabetes_selected.parquet")

features = [
    "GenHlth", "HighBP", "BMI", "DiffWalk", "HighChol",
    "Age", "HeartDiseaseorAttack", "PhysHlth", "Stroke"
]
X = df[features]
y = df["Diabetes_012"]

# 2. Bagi data latih (80%) dan data uji (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Bobot kelas terkalibrasi (tidak ekstrem agar akurasi tetap tinggi & stabil)
class_weights = {0: 1.0, 1: 3.0, 2: 2.5}
sample_weights = y_train.map(class_weights)

# 4. Latih model XGBoost Classifier
model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=250,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss",
    n_jobs=-1
)

print("Sedang melatih model XGBoost terkalibrasi...")
model.fit(X_train, y_train, sample_weight=sample_weights)

# 5. Gunakan predict_proba() untuk mengatur keputusan prediksi berdasarkan probabilitas risiko
y_proba = model.predict_proba(X_test)
# y_proba[:, 0] = Probabilitas Sehat
# y_proba[:, 1] = Probabilitas Prediabetes
# y_proba[:, 2] = Probabilitas Diabetes

y_pred_tuned = []
for p0, p1, p2 in y_proba:
    risk_score = p1 + p2  # Gabungan probabilitas risiko Prediabetes + Diabetes
    if p2 >= 0.42:
        y_pred_tuned.append(2)  # Diabetes
    elif risk_score >= 0.48:
        y_pred_tuned.append(1)  # Prediabetes / Waspada Risiko
    else:
        y_pred_tuned.append(0)  # Tidak Diabetes

y_pred_tuned = np.array(y_pred_tuned)

# 6. Tampilkan Evaluasi Lengkap untuk Laporan / Presentasi Dosen
roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
acc = accuracy_score(y_test, y_pred_tuned)

print(f"\n=== METRIK UTAMA MODEL ===")
print(f"Weighted ROC-AUC Score : {roc_auc:.4f} (Skala 0 - 1)")
print(f"Overall Accuracy       : {acc:.2%}")

print("\n=== Classification Report (Setelah Kalibrasi predict_proba) ===")
print(classification_report(
    y_test, y_pred_tuned,
    target_names=["0: Tidak Diabetes", "1: Prediabetes", "2: Diabetes"]
))

# 7. Evaluasi Efektivitas Deteksi Dini (Sehat vs Berisiko Prediabetes/Diabetes)
y_test_binary = (y_test > 0).astype(int)
y_pred_binary = (y_pred_tuned > 0).astype(int)
print("=== Performa Screening Awal (Sehat vs Berisiko Pre/Diabetes) ===")
print(classification_report(
    y_test_binary, y_pred_binary,
    target_names=["Aman (0)", "Perlu Screening Lanjut (1 & 2)"]
))

# 8. Simpan model & aturan threshold ke file JSON untuk aplikasi Streamlit temanmu
model.save_model("xgboost_diabetes.json")

metadata = {
    "features": features,
    "labels": {
        "0": "Tidak Diabetes",
        "1": "Prediabetes",
        "2": "Diabetes"
    },
    "thresholds": {
        "diabetes_prob": 0.42,
        "combined_risk_prob": 0.48
    }
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Selesai! File 'xgboost_diabetes.json' dan 'model_metadata.json' siap di-download.")