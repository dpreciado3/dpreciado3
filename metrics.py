 
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def calculate_ks(df, num_buckets=10):
    """
    Calculate the Kolmogorov-Smirnov (KS) statistic for binary classification using deciles.
    
    Parameters:
    df (DataFrame): Input DataFrame with 'target' (0/1) and 'probability' (vector of size 2)
    num_buckets (int): Number of buckets to divide the data into (default 10)
    
    Returns:
    ks_value (float): The maximum KS value
    ks_table (DataFrame): Table showing metrics per bucket
    """
    # Extract the probability of class 1
    df = df.withColumn("prob_class_1", F.col("probability")[1])

    # Create a window specification for deciles (descending prob_class_1)
    quantile_window = Window.orderBy(F.desc("prob_class_1"))
    
    # Add a row number and compute decile bucket
    df = df.withColumn("row_num", F.row_number().over(quantile_window))
    
    total_count = df.count()
    bucket_size = total_count // num_buckets
    
    df = df.withColumn("bucket", ((F.col("row_num") - 1) / bucket_size).cast("int"))
    
    # Fix any overflow into an 11th bucket
    df = df.withColumn("bucket", F.when(F.col("bucket") >= num_buckets, num_buckets - 1).otherwise(F.col("bucket")))

    # Group by bucket and compute stats
    bucket_stats = df.groupBy("bucket").agg(
        F.count("*").alias("total"),
        F.sum("target").alias("positives"),
        F.sum(1 - F.col("target")).alias("negatives")
    ).orderBy("bucket")
    
    # Get total positives and negatives
    totals = df.agg(
        F.sum("target").alias("total_positives"),
        F.sum(1 - F.col("target")).alias("total_negatives")
    ).collect()[0]
    
    total_positives = totals["total_positives"]
    total_negatives = totals["total_negatives"]
    
    # Compute cumulative metrics and KS
    window_bucket = Window.orderBy("bucket").rowsBetween(Window.unboundedPreceding, 0)
    
    bucket_stats = bucket_stats.withColumn("cum_positives", F.sum("positives").over(window_bucket))
    bucket_stats = bucket_stats.withColumn("cum_negatives", F.sum("negatives").over(window_bucket))
    
    bucket_stats = bucket_stats.withColumn("tpr", F.col("cum_positives") / total_positives)
    bucket_stats = bucket_stats.withColumn("fpr", F.col("cum_negatives") / total_negatives)
    
    bucket_stats = bucket_stats.withColumn("ks", F.abs(F.col("tpr") - F.col("fpr")))
    
    ks_value = bucket_stats.agg(F.max("ks")).collect()[0][0]

    return ks_value, bucket_stats