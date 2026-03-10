from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, LongType, ArrayType, DoubleType
import pandas as pd
import numpy as np

# 1. Initialize Spark Session (Ensure Arrow is enabled for performance)
spark = SparkSession.builder \
    .appName("LLM_Behavioral_Embeddings") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

# Mock Data (Imagine this is your massive transaction table)
data = [
    (1, "Starbucks", 5.50, "2026-01-01 08:00:00"),
    (1, "Amazon", 120.00, "2026-01-02 14:00:00"),
    (1, "Apple", 999.00, "2026-03-01 10:00:00"),
    (2, "Walmart", 45.00, "2026-02-15 09:00:00"),
    (2, "Target", 30.00, "2026-03-05 18:00:00")
]
columns = ["customer_id", "merchant_name", "amount", "timestamp"]
df = spark.createDataFrame(data, columns)
df = df.withColumn("timestamp", F.to_timestamp("timestamp"))

# 2. Mock Merchant Embeddings (Assuming pre-computed 256-dim PCA vectors)
# In production, load this from a Parquet file and broadcast-join it.
mock_emb = {
    "Starbucks": np.random.rand(256).tolist(),
    "Amazon": np.random.rand(256).tolist(),
    "Apple": np.random.rand(256).tolist(),
    "Walmart": np.random.rand(256).tolist(),
    "Target": np.random.rand(256).tolist()
}
# Convert to Spark DataFrame and explode into separate columns for easier NumPy conversion later
emb_data = [(k, *v) for k, v in mock_emb.items()]
emb_cols = ["merchant_name"] + [f"emb_{i}" for i in range(256)]
df_embeddings = spark.createDataFrame(emb_data, emb_cols)

# Join transactions with embeddings
df_tx = df.join(df_embeddings, on="merchant_name", how="left")

# 3. Native PySpark Feature Engineering (Faster than UDFs)
# Cyclical Time & Log Amount
df_tx = df_tx.withColumn("hour", F.hour("timestamp")) \
    .withColumn("time_sin", F.sin(2 * F.lit(np.pi) * F.col("hour") / 24)) \
    .withColumn("time_cos", F.cos(2 * F.lit(np.pi) * F.col("hour") / 24)) \
    .withColumn("amount_log", F.log1p("amount"))

# To calculate recency, we need the global max timestamp across all data
global_max_time = df_tx.select(F.max("timestamp")).collect()[0][0]
df_tx = df_tx.withColumn("global_max_time", F.lit(global_max_time))

# 4. Multi-Headed Aggregation via Pandas UDF
# Define the output schema for the final customer vector
output_schema = StructType([
    StructField("customer_id", LongType(), True),
    StructField("customer_vector", ArrayType(DoubleType()), True) # The final 1036-dim array
])

def aggregate_habits(pdf: pd.DataFrame) -> pd.DataFrame:
    """Pandas function applied to each customer's group of transactions."""
    customer_id = pdf['customer_id'].iloc[0]
    
    # Calculate Recency Weights
    delta_t = (pdf['global_max_time'] - pdf['timestamp']).dt.days
    lambda_decay = 0.01 
    weights = np.exp(-lambda_decay * delta_t).values.reshape(-1, 1)
    
    # Extract just the features for math (256 emb cols + 3 numerical cols)
    feature_cols = [c for c in pdf.columns if c.startswith('emb_')] + ['amount_log', 'time_sin', 'time_cos']
    feats = pdf[feature_cols].values # Convert to NumPy array
    
    # Calculate the 4 Statistical Heads
    mean_head = np.mean(feats, axis=0)
    # ddof=0 handles cases where a customer only has 1 transaction (prevents NaN)
    std_head = np.std(feats, axis=0, ddof=0) 
    max_head = np.max(feats, axis=0)
    
    # Avoid division by zero if weights sum to 0
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        recency_head = np.sum(feats * weights, axis=0) / weight_sum
    else:
        recency_head = mean_head
        
    # Concatenate into final vector (259 * 4 = 1036 dimensions)
    final_vector = np.concatenate([mean_head, std_head, max_head, recency_head]).tolist()
    
    # Return as a Pandas DataFrame matching the output_schema
    return pd.DataFrame({
        "customer_id": [customer_id],
        "customer_vector": [final_vector]
    })

# 5. Apply the Grouped Pandas UDF
df_customer_features = df_tx.groupBy("customer_id").applyInPandas(
    aggregate_habits, 
    schema=output_schema
)

df_customer_features.show(truncate=80)
