import csv
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

DATA = "data/processed"
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

spark = (
    SparkSession.builder
    .appName("export")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

interactions = spark.read.parquet(f"{DATA}/interactions")
product_features = spark.read.parquet(f"{DATA}/product_features")
skin_stats = spark.read.parquet(f"{DATA}/product_skin_stats")


def save_csv(df, name):
    rows = df.collect()
    with open(REPORTS / f"{name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(df.columns)
        writer.writerows(rows)
    print(f"wrote reports/{name}.csv ({len(rows)} rows)")

# 1. Top 20 products for each skin type
names = product_features.select(
    "product_id", "brand_name", "product_name", "secondary_category", "price_usd"
)
top_by_skin = (
    skin_stats.filter(F.col("popularity_rank") <= 20)
    .join(F.broadcast(names), "product_id")
    .select(
        F.col("user_skin_type").alias("skin_type"),
        "popularity_rank", "brand_name", "product_name",
        "secondary_category", "price_usd", "n_reviews", "avg_rating",
    )
    .orderBy("skin_type", "popularity_rank")
)
save_csv(top_by_skin, "top_products_by_skin_type")

# 2. Reviews per month
by_month = (
    interactions
    .withColumn("month", F.trunc("review_date", "month"))
    .groupBy("month")
    .agg(
        F.count("*").alias("n_reviews"),
        F.round(F.avg("rating"), 3).alias("avg_rating"),
    )
    .orderBy("month")
)
save_csv(by_month, "reviews_by_month")

# 3. Rating distribution
ratings = (
    interactions.groupBy("rating")
    .agg(F.count("*").alias("n_reviews"))
    .orderBy("rating")
)
save_csv(ratings, "rating_distribution")

# 4. Customer activity
users = interactions.select("author_id", "user_review_count").distinct()
bucket = (
    F.when(F.col("user_review_count") >= 10, "10+")
    .when(F.col("user_review_count") >= 5, "5-9")
    .when(F.col("user_review_count") >= 3, "3-4")
    .when(F.col("user_review_count") == 2, "2")
    .otherwise("1")
)
activity = (
    users.withColumn("reviews_per_customer", bucket)
    .groupBy("reviews_per_customer")
    .agg(
        F.count("*").alias("n_customers"),
        F.min("user_review_count").alias("sort_order"),
    )
    .orderBy("sort_order")
)
save_csv(activity, "customer_activity")

# 5. One row per product
product_summary = product_features.select(
    "product_id", "brand_name", "product_name", "secondary_category",
    "price_usd", "n_reviews", "avg_rating", "pct_recommended",
).orderBy(F.desc("n_reviews"))
save_csv(product_summary, "product_summary")

spark.stop()