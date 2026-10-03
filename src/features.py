from pyspark.sql import SparkSession, Window
#Sparksession is the connection to the spark engine, and window lets use define groups or rows
from pyspark.sql import functions as F
# functions as F allows us to do column operations, such as f.avg


DATA = "data/processed"

#starting spark
spark = (
    SparkSession.builder
    .appName("features") # label of this run
    .master("local[*]") #set to run locally on my laptop
    .config("spark.driver.memory", "4g") # here we set a cap on memory usage
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate() # start the engine
)

spark.sparkContext.setLogLevel("WARN") #only print out warning messages (spark prints out a lot of other messages)

# load the dataset, and create two dataFrames, one for reviews, and another for product information
reviews = spark.read.parquet(f"{DATA}/reviews")
products = spark.read.parquet(f"{DATA}/products")


# Data Cleaning/Organization part!

# 1. Deduplicate: keep each customer's most recent review of each product.
# some customers in this dataset have reviwed the same product more than once, we will keep the newest.
latest_first = Window.partitionBy("author_id", "product_id").orderBy(F.desc("review_date"))
interactions = (
    reviews
    .withColumn("rn", F.row_number().over(latest_first))
    .filter(F.col("rn") == 1)
    .drop("rn")
)

# 2. Per-customer features, added to every row without collapsing rows.
# add 2 column to each row describing how many reviews the customer wrote, and their most common skin type. 

per_user = Window.partitionBy("author_id")
interactions = (
    interactions
    .withColumn("user_review_count", F.count("*").over(per_user))
    #F.mode picks the most frequent value and ignores blanks, so any empty fields get filled with previous reviews where
    #the field has answered.
    .withColumn("user_skin_type", F.mode("skin_type").over(per_user))
)

# This table is reused several times below, so keep it in memory.
interactions.cache()

# 3. Per-product statistics, joined to the product catalog.
product_stats = interactions.groupBy("product_id").agg(
    F.count("*").alias("n_reviews"),
    F.round(F.avg("rating"), 3).alias("avg_rating"),
    F.round(F.avg("is_recommended"), 3).alias("pct_recommended"),
)
product_features = product_stats.join(F.broadcast(products), on="product_id", how="left")

# 4. Popularity of each product within each skin type, with a rank.
skin_stats = (
    interactions
    .filter(F.col("user_skin_type").isNotNull())
    .groupBy("user_skin_type", "product_id")
    .agg(
        F.count("*").alias("n_reviews"),
        F.round(F.avg("rating"), 3).alias("avg_rating"),
    )
)
by_skin = Window.partitionBy("user_skin_type").orderBy(F.desc("n_reviews"))
skin_stats = skin_stats.withColumn("popularity_rank", F.row_number().over(by_skin))

# 5. Save.
interactions.write.mode("overwrite").parquet(f"{DATA}/interactions")
product_features.write.mode("overwrite").parquet(f"{DATA}/product_features")
skin_stats.write.mode("overwrite").parquet(f"{DATA}/product_skin_stats")

# 6. Diagnostics.
users = interactions.select("author_id", "user_review_count").distinct()
print("interactions after dedup:", interactions.count())
print("customers:               ", users.count())
print("products with reviews:   ", product_stats.count())

bucket = (
    F.when(F.col("user_review_count") >= 10, "10+")
    .when(F.col("user_review_count") >= 5, "5-9")
    .when(F.col("user_review_count") >= 3, "3-4")
    .when(F.col("user_review_count") == 2, "2")
    .otherwise("1")
)
(
    users.withColumn("reviews_per_customer", bucket)
    .groupBy("reviews_per_customer")
    .count()
    .orderBy(F.desc("count"))
    .show()
)

interactions.groupBy("user_skin_type").count().orderBy(F.desc("count")).show()

names = products.select("product_id", "brand_name", "product_name")
(
    skin_stats.filter(F.col("popularity_rank") <= 3)
    .join(F.broadcast(names), "product_id")
    .orderBy("user_skin_type", "popularity_rank")
    .select("user_skin_type", "popularity_rank", "brand_name", "product_name", "n_reviews", "avg_rating")
    .show(12, truncate=45)
)

spark.stop()