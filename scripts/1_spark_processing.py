from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
import os

# 1. Inisialisasi Spark
spark = SparkSession.builder \
    .appName("Diabetes_Processing_JASHujan") \
    .getOrCreate()

# 2. Baca dataset dari path Google Drive kelompokmu
DATASET_PATH = '/content/drive/MyDrive/Dataset PJBL (Big Data dan VISIKOM Citra)/Diabetes-Health-Indicators-Dataset/diabetes_012_health_indicators_BRFSS2015.csv'
df_raw = spark.read.csv(DATASET_PATH, header=True, inferSchema=True)

# 3. Seleksi 9 fitur utama hasil EDA + target Diabetes_012 & casting tipe data
selected_features = [
    "GenHlth", "HighBP", "BMI", "DiffWalk", "HighChol",
    "Age", "HeartDiseaseorAttack", "PhysHlth", "Stroke"
]

df_processed = df_raw.select(
    col("Diabetes_012").cast("int"),
    col("GenHlth").cast("int"),
    col("HighBP").cast("int"),
    col("BMI").cast("float"),
    col("DiffWalk").cast("int"),
    col("HighChol").cast("int"),
    col("Age").cast("int"),
    col("HeartDiseaseorAttack").cast("int"),
    col("PhysHlth").cast("int"),
    col("Stroke").cast("int")
)

# 4. Simpan ke zona Processed (format Parquet untuk pelatihan model)
df_processed.write.mode("overwrite").parquet("diabetes_selected.parquet")

# 5. Hitung tabel korelasi 9 fitur untuk zona Analytical (Tab 2 Streamlit)
correlations = []
for feat in selected_features:
    corr_val = df_processed.stat.corr("Diabetes_012", feat)
    correlations.append({"Fitur": feat, "Korelasi": round(corr_val, 4)})

df_corr = pd.DataFrame(correlations).sort_values(by="Korelasi", ascending=False)
df_corr.to_csv("top_9_risk_factors.csv", index=False)

print("Pemrosesan PySpark selesai! Jumlah baris:", df_processed.count())
print(df_corr)
spark.stop()