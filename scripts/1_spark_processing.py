from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, round as spark_round

# 1. Inisialisasi SparkSession
spark = SparkSession.builder \
    .appName("Diabetes_Processing_JASHujan") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 2. Baca dataset mentah dari HDFS (/raw)
RAW_PATH = "hdfs://namenode:9000/raw/diabetes_012_health_indicators_BRFSS2015.csv"
print(f"Membaca data mentah dari: {RAW_PATH}")
df_raw = spark.read.csv(RAW_PATH, header=True, inferSchema=True)

total_raw = df_raw.count()
print(f"Total baris data mentah terbaca: {total_raw}")

# 3. Seleksi 9 fitur utama hasil EDA + target Diabetes_012 & pembersihan missing value
selected_features = [
    "GenHlth", "HighBP", "BMI", "DiffWalk", "HighChol",
    "Age", "HeartDiseaseorAttack", "PhysHlth", "Stroke"
]
target_col = "Diabetes_012"

df_clean = df_raw.select([target_col] + selected_features).dropna()

# Pastikan tipe data numerik konsisten
df_processed = df_clean.withColumn(target_col, col(target_col).cast("int"))
for feat in selected_features:
    df_processed = df_processed.withColumn(feat, col(feat).cast("double"))

total_clean = df_processed.count()
print(f"Total baris bersih setelah seleksi fitur: {total_clean}")

# 4. Simpan hasil pemrosesan ke HDFS (/processed) dalam format Parquet & CSV
PROCESSED_PARQUET = "hdfs://namenode:9000/processed/diabetes_selected_parquet"
PROCESSED_CSV = "hdfs://namenode:9000/processed/diabetes_selected_csv"

df_processed.write.mode("overwrite").parquet(PROCESSED_PARQUET)
df_processed.coalesce(1).write.mode("overwrite").option("header", True).csv(PROCESSED_CSV)
print("Berhasil menyimpan data bersih ke HDFS: /processed")

# 5. Buat ringkasan analitik per kelas risiko & simpan ke HDFS (/analytical)
df_analytical = df_processed.groupBy("Diabetes_012").agg(
    count("*").alias("jumlah_responden"),
    spark_round(avg("BMI"), 2).alias("rata_rata_BMI"),
    spark_round(avg("GenHlth"), 2).alias("rata_rata_GenHlth"),
    spark_round(avg("HighBP"), 3).alias("proporsi_HighBP"),
    spark_round(avg("Age"), 2).alias("rata_rata_Age")
).orderBy("Diabetes_012")

print("\nRingkasan Analitik per Kelas Risiko:")
df_analytical.show()

ANALYTICAL_PATH = "hdfs://namenode:9000/analytical/risk_class_summary"
df_analytical.coalesce(1).write.mode("overwrite").option("header", True).csv(ANALYTICAL_PATH)
print("Berhasil menyimpan ringkasan analitik ke HDFS: /analytical")

spark.stop()