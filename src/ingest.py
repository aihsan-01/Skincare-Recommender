from pyspark.sql import SparkSession
from pyspark.sql import functions as F

RAW = "data/raw"
OUT = "data/processed"

spark = (
    SparkSession.builder
    .appName("ingest")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# Review text contains commas, quotes, and line breaks,
# so tell Spark how to parse them correctly.
csv_options = dict(header=True, multiLine=True, quote='"', escape='"')

reviews_raw = spark.read.options(**csv_options).csv(f"{RAW}/reviews_*.csv")
products_raw = spark.read.options(**csv_options).csv(f"{RAW}/product_info.csv")

reviews_raw.printSchema()

reviews = (
    reviews_raw
    .select(
        "author_id",
        "product_id",
        F.col("rating").try_cast("int").alias("rating"),
        F.col("is_recommended").try_cast("double").alias("is_recommended"),
        F.col("submission_time").try_cast("date").alias("review_date"),
        "skin_type",
        "skin_tone",
    )
    .dropna(subset=["author_id", "product_id", "rating", "review_date"])
)

products = products_raw.select(
    "product_id",
    "product_name",
    "brand_name",
    "primary_category",
    "secondary_category",
    F.col("price_usd").try_cast("double").alias("price_usd"),
    F.col("loves_count").try_cast("int").alias("loves_count"),
)

reviews.write.mode("overwrite").parquet(f"{OUT}/reviews")
products.write.mode("overwrite").parquet(f"{OUT}/products")

# Read back what we saved and sanity-check it.
reviews = spark.read.parquet(f"{OUT}/reviews")
products = spark.read.parquet(f"{OUT}/products")

print("raw review rows:  ", reviews_raw.count())
print("clean review rows:", reviews.count())
print("products:         ", products.count())

reviews.show(5, truncate=False)
reviews.groupBy("rating").count().orderBy("rating").show()
reviews.groupBy("skin_type").count().orderBy(F.desc("count")).show()

spark.stop()