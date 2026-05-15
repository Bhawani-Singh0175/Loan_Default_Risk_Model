import numpy as np
import pandas as pd


# ============================================================
# Load Data
# ============================================================

DATA_FILE = "Task 3 and 4_Loan_Data.csv"

df = pd.read_csv(DATA_FILE)
df = df[["fico_score", "default"]].copy()
df = df.sort_values("fico_score").reset_index(drop=True)


# ============================================================
# Bucket Log-Likelihood
# ============================================================

def bucket_log_likelihood(defaults):
    """
    Calculate log-likelihood for one bucket.

    defaults contains 0/1 values:
    0 = non-default
    1 = default
    """

    n = len(defaults)

    if n == 0:
        return -np.inf

    k = np.sum(defaults)

    if k == 0 or k == n:
        return 0.0

    p = k / n

    return k * np.log(p) + (n - k) * np.log(1 - p)


# ============================================================
# Dynamic Programming Quantization
# ============================================================

def create_fico_rating_map(data, num_buckets=10):
    """
    Create optimal FICO score buckets using dynamic programming.

    Objective:
    Maximize total bucket log-likelihood.

    Rating rule:
    Lower rating = better credit score.
    Higher FICO score receives lower rating.
    """

    data = data.sort_values("fico_score").reset_index(drop=True)

    fico_scores = data["fico_score"].values
    defaults = data["default"].values

    n = len(data)

    # dp[i][j] = best score using i buckets for first j records
    dp = np.full((num_buckets + 1, n + 1), -np.inf)
    split = np.zeros((num_buckets + 1, n + 1), dtype=int)

    dp[0][0] = 0.0

    # Precompute likelihood for all possible segments
    likelihood = np.full((n, n + 1), -np.inf)

    for start in range(n):
        for end in range(start + 1, n + 1):
            likelihood[start][end] = bucket_log_likelihood(defaults[start:end])

    # Dynamic programming
    for b in range(1, num_buckets + 1):
        for end in range(b, n + 1):
            for start in range(b - 1, end):
                value = dp[b - 1][start] + likelihood[start][end]

                if value > dp[b][end]:
                    dp[b][end] = value
                    split[b][end] = start

    # Backtrack bucket boundaries
    boundaries = []
    end = n

    for b in range(num_buckets, 0, -1):
        start = split[b][end]

        min_fico = fico_scores[start]
        max_fico = fico_scores[end - 1]

        bucket_defaults = defaults[start:end]
        default_rate = np.mean(bucket_defaults)

        boundaries.append({
            "min_fico": int(min_fico),
            "max_fico": int(max_fico),
            "num_records": int(end - start),
            "num_defaults": int(np.sum(bucket_defaults)),
            "default_rate": round(float(default_rate), 4)
        })

        end = start

    # Reverse because backtracking gives high index first
    boundaries = boundaries[::-1]

    # Higher FICO = better rating = lower number
    rating_map = []

    for i, bucket in enumerate(boundaries):
        rating = num_buckets - i

        rating_map.append({
            "rating": rating,
            "min_fico": bucket["min_fico"],
            "max_fico": bucket["max_fico"],
            "num_records": bucket["num_records"],
            "num_defaults": bucket["num_defaults"],
            "default_rate": bucket["default_rate"]
        })

    # Sort by rating: 1 is best, num_buckets is worst
    rating_map = sorted(rating_map, key=lambda x: x["rating"])

    return rating_map


# ============================================================
# Rating Function
# ============================================================

def get_fico_rating(fico_score, rating_map):
    """
    Return rating for a given FICO score.

    Lower rating means better credit score.
    """

    for bucket in rating_map:
        if bucket["min_fico"] <= fico_score <= bucket["max_fico"]:
            return bucket["rating"]

    # Handle scores outside training range
    highest_fico_bucket = min(rating_map, key=lambda x: x["rating"])
    lowest_fico_bucket = max(rating_map, key=lambda x: x["rating"])

    if fico_score > highest_fico_bucket["max_fico"]:
        return highest_fico_bucket["rating"]

    if fico_score < lowest_fico_bucket["min_fico"]:
        return lowest_fico_bucket["rating"]

    return None


def get_default_rate_from_rating(fico_score, rating_map):
    """
    Return estimated default rate based on FICO rating bucket.
    """

    rating = get_fico_rating(fico_score, rating_map)

    for bucket in rating_map:
        if bucket["rating"] == rating:
            return bucket["default_rate"]

    return None


# ============================================================
# Test / Demo
# ============================================================

if __name__ == "__main__":

    NUM_BUCKETS = 10

    rating_map = create_fico_rating_map(df, num_buckets=NUM_BUCKETS)

    print("\n==============================================")
    print(" FICO Score Rating Map")
    print(" Lower rating = better credit score")
    print("==============================================\n")

    rating_df = pd.DataFrame(rating_map)
    print(rating_df.to_string(index=False))

    print("\n==============================================")
    print(" Sample Rating Tests")
    print("==============================================")

    sample_scores = [500, 550, 600, 650, 700, 750, 800]

    for score in sample_scores:
        rating = get_fico_rating(score, rating_map)
        default_rate = get_default_rate_from_rating(score, rating_map)

        print(
            f"FICO Score: {score} | "
            f"Rating: {rating} | "
            f"Estimated Default Rate: {default_rate}"
        )