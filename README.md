# Skincare Recommender

A skin-type-aware skincare recommender built with PySpark and Spark MLlib on 1.09M Sephora reviews. Given a customer's past reviews, it recommends the 10 skincare products they are most likely to choose next.

## Results

Evaluated on 40,433 customers with at least five reviews. For each one, their most recent review was hidden and the model had to rank that product in its top 10 out of about 2,350 products.

| Method | Recall@10 | NDCG@10 |
|---|---|---|
| Most popular overall | 0.0437 | 0.0224 |
| Most popular for the customer's skin type | 0.0503 | 0.0252 |
| **ALS (this project)** | **0.2790** | **0.1872** |

ALS places the hidden product in the top 10 for 27.9% of customers, about 6.4x the popularity baseline.

## Pipeline

Four PySpark scripts, each reading the previous one's Parquet output:

| Script | What it does |
|---|---|
| `src/ingest.py` | Reads the raw CSVs, selects and type-casts columns, writes Parquet |
| `src/features.py` | Deduplicates reviews with a window function, adds per-customer features (review count, most common skin type), builds per-product and per-skin-type statistics with a broadcast join to the product catalog |
| `src/train.py` | Holds out each qualifying customer's most recent review, indexes IDs, trains an implicit-feedback ALS model |
| `src/evaluate.py` | Generates top-10 unseen recommendations for ALS and two popularity baselines, computes recall@10 and NDCG@10 |

## Data

[Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews) from Kaggle (not included in this repo).

- 1,094,411 raw reviews and 8,494 products
- 1,088,886 customer-product interactions after deduplication
- 503,216 customers and 2,351 reviewed products
- 58% of customers wrote only one review, so evaluation uses the 40,433 customers with five or more

## Method

- **Split:** leave-last-out. The most recent review of each customer with at least five reviews is the test set (40,433 rows); everything else is training (1,048,453 rows).
- **Model:** Spark MLlib ALS with implicit feedback, using the star rating as confidence. Rank 32, 10 iterations, regParam 0.1, alpha 10. Trains in about 80 seconds in local mode on an 8-core laptop.
- **Baselines:** the most-reviewed products overall, and the most-reviewed products among customers with the same skin type. Both are computed from the training data only.
- **Fair comparison:** products a customer already reviewed are removed from every method's recommendations before taking the top 10.

## Limitations

- Leave-last-out means the training data includes other customers' reviews written after a test customer's hidden review. A global time-based split would be stricter.
- Customers often review several products on the same day, so some of the model's lift comes from learning which products are commonly reviewed together.
- Low-star reviews still count as weak positive signals under implicit feedback.
- Hyperparameters are reasonable defaults, not tuned.
- The data fits on one machine. Spark runs in local mode here; the same code runs on a cluster by changing the master setting.

## Run it

Requires conda. From the repo root:

```bash
conda create -n skincare -c conda-forge python=3.12 openjdk=17 -y
conda activate skincare
python -m pip install -r requirements.txt
```

Download the dataset from Kaggle and put the six CSV files in `data/raw/`, then:

```bash
python src/ingest.py
python src/features.py
python src/train.py
python src/evaluate.py
```

## Tech

Python 3.12, PySpark 4.2.0 (DataFrames, Spark SQL window functions, MLlib ALS), Java 17, Parquet.
