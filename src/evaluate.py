from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel

DATA = "data/processed"
K = 10
N_CANDIDATES = 100

spark = (
    SparkSession.builder
    .appName("evaluate")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

train = spark.read.parquet(f"{DATA}/train")
test = spark.read.parquet(f"{DATA}/test")
model = ALSModel.load("models/als")

test_users = test.select("user_idx").distinct()
seen = train.join(test_users, "user_idx", "left_semi").select("user_idx", "item_idx")


def top_k_unseen(candidates):
    unseen = candidates.join(seen, ["user_idx", "item_idx"], "left_anti")
    best_first = Window.partitionBy("user_idx").orderBy(F.desc("score"), "item_idx")
    return (
        unseen
        .withColumn("rank", F.row_number().over(best_first))
        .filter(F.col("rank") <= K)
        .groupBy("user_idx")
        .agg(F.sort_array(F.collect_list(F.struct("rank", "item_idx"))).alias("ranked"))
        .select("user_idx", F.col("ranked.item_idx").alias("recs"))
    )

# ALS model
als_candidates = (
    model.recommendForUserSubset(test_users, N_CANDIDATES)
    .select("user_idx", F.explode("recommendations").alias("rec"))
    .select("user_idx", F.col("rec.item_idx").alias("item_idx"), F.col("rec.rating").alias("score"))
)
als_recs = top_k_unseen(als_candidates)

# Baseline 1: most popular products overall
popular = (
    train.groupBy("item_idx").agg(F.count("*").alias("score"))
    .orderBy(F.desc("score")).limit(N_CANDIDATES)
)
pop_candidates = test_users.crossJoin(F.broadcast(popular))
pop_recs = top_k_unseen(pop_candidates)

# Baseline 2: most popular products for the customer's skin type
by_skin = Window.partitionBy("user_skin_type").orderBy(F.desc("score"))
popular_by_skin = (
    train.filter(F.col("user_skin_type").isNotNull())
    .groupBy("user_skin_type", "item_idx").agg(F.count("*").alias("score"))
    .withColumn("r", F.row_number().over(by_skin))
    .filter(F.col("r") <= N_CANDIDATES).drop("r")
)
user_skin = test.select("user_idx", "user_skin_type")
with_skin = (
    user_skin.filter(F.col("user_skin_type").isNotNull())
    .join(F.broadcast(popular_by_skin), "user_skin_type")
    .select("user_idx", "item_idx", "score")
)
without_skin = (
    user_skin.filter(F.col("user_skin_type").isNull()).select("user_idx")
    .crossJoin(F.broadcast(popular))
    .select("user_idx", "item_idx", "score")
)
skin_recs = top_k_unseen(with_skin.unionByName(without_skin))

def score(name, recs):
    joined = test.select("user_idx", "item_idx").join(recs, "user_idx", "left")
    pos = F.coalesce(F.expr("array_position(recs, item_idx)"), F.lit(0))
    gain = 1.0 / F.log2(F.greatest(pos, F.lit(1)) + 1)
    row = joined.select(
        F.avg((pos > 0).cast("double")).alias("recall"),
        F.avg(F.when(pos > 0, gain).otherwise(0.0)).alias("ndcg"),
    ).first()
    return (name, round(row["recall"], 4), round(row["ndcg"], 4))


results = [
    score("Most popular overall", pop_recs),
    score("Most popular for skin type", skin_recs),
    score("ALS", als_recs),
]
print(f"{'method':<28} {'recall@10':>10} {'ndcg@10':>10}")
for name, recall, ndcg in results:
    print(f"{name:<28} {recall:>10.4f} {ndcg:>10.4f}")
spark.stop()