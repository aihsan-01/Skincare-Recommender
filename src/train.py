from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
import time
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS

DATA = "data/processed"
MIN_REVIEWS = 5

spark = (
    SparkSession.builder
    .appName("train")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

interactions = spark.read.parquet(f"{DATA}/interactions")

# Number each customer's reviews from newest (1) to oldest.
newest_first = Window.partitionBy("author_id").orderBy(F.desc("review_date"), "product_id")
numbered = interactions.withColumn("recency", F.row_number().over(newest_first))

# Test set: the most recent review of each customer with enough history.
is_test = (F.col("user_review_count") >= MIN_REVIEWS) & (F.col("recency") == 1)
test = numbered.filter(is_test)
train = numbered.filter(~is_test)


# Turn IDs into Numbers
# Sparks ALS can only accept integer ids, but ours are strings.
indexer = StringIndexer( #StringIndexer scans the data abd builds a lookup table from each distinct id to a number
    inputCols=["author_id", "product_id"],
    outputCols=["user_idx", "item_idx"],
).fit(interactions)

def add_ids(df): # this helper function applies that lookup to any table and converts the results to whole numbers.
    return ( 
        indexer.transform(df)
        .withColumn("user_idx", F.col("user_idx").cast("int"))
        .withColumn("item_idx", F.col("item_idx").cast("int"))
    )

train = add_ids(train).cache() #.cache : the model reads train many times during training, so we keep it in memory
test = add_ids(test)

#Train the model and save
"""
ALS is a matrix factorization. How it works is the model learns a short like of numbers for every customer, and product.
When a customer's vector is multiplied by a products vector and results in a high score, that represents how much of a good
fit that product is for that customer.

"Alternating least squares" means it fixes the product vectors and solves for the customers, then swaps, and repeats. 
Each of those solves can be split across cores, which is why it's the standard recommender in Spark.
"""
als = ALS(
    userCol="user_idx",
    itemCol="item_idx",
    ratingCol="rating",
    implicitPrefs=True,
    rank=32,
    maxIter=10,
    regParam=0.1,
    alpha=10.0,
    coldStartStrategy="drop",
    seed=42,
)

start = time.time()
model = als.fit(train)
print(f"trained ALS in {time.time() - start:.0f} seconds")

train.write.mode("overwrite").parquet(f"{DATA}/train")
test.write.mode("overwrite").parquet(f"{DATA}/test")
model.write().overwrite().save("models/als")
print("saved train, test, and model")

spark.stop()