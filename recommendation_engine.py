import pandas as pd
import numpy as np
import pickle
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DATASET")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)


def load_unified_data():
    path = os.path.join(DATASET_DIR, "unified_courses.csv")
    df = pd.read_csv(path)
    return df


def build_content_features(df):
    df = df.copy()
    df["description"] = df["description"].fillna("")
    df["skills"] = df["skills"].fillna("")
    df["category"] = df["category"].fillna("")

    df["text_blob"] = (
        df["title"] + " " +
        df["description"] + " " +
        df["skills"] + " " +
        df["category"] + " " +
        df["level"]
    )
    df["text_blob"] = df["text_blob"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    return df


def train_content_model(df):
    df = build_content_features(df)
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    tfidf_matrix = tfidf.fit_transform(df["text_blob"])
    return tfidf, tfidf_matrix, df


def get_content_recommendations(query_text, tfidf, tfidf_matrix, df, top_n=10):
    query_vec = tfidf.transform([query_text.lower()])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_n]
    results = df.iloc[top_indices][["course_id", "title", "platform", "category", "level", "rating"]].copy()
    results["content_score"] = scores[top_indices]
    return results


USER_PROFILES = {
    "data_science": ["data science", "machine learning", "python", "statistics", "deep learning", "data analysis", "r programming"],
    "web_dev": ["web development", "javascript", "react", "html", "css", "node.js", "frontend", "backend", "django", "flask"],
    "mobile_dev": ["android", "ios", "flutter", "react native", "swift", "kotlin", "mobile"],
    "cloud": ["aws", "azure", "docker", "kubernetes", "devops", "cloud computing", "terraform", "ci/cd"],
    "design": ["graphic design", "ui/ux", "figma", "photoshop", "illustrator", "adobe", "animation"],
    "business": ["business", "marketing", "finance", "management", "excel", "accounting", "project management"],
    "cybersec": ["cybersecurity", "ethical hacking", "network security", "penetration testing", "information security"]
}


def generate_synthetic_ratings(df, n_users=2000):
    print("generating synthetic user ratings...")
    profile_courses = defaultdict(list)
    for idx, course in df.iterrows():
        combined = (str(course["title"]) + " " + str(course["category"]) + " " + str(course["skills"])).lower()
        for prof, keywords in USER_PROFILES.items():
            if any(kw in combined for kw in keywords):
                profile_courses[prof].append(course["course_id"])

    ratings_list = []
    for user_id in range(1, n_users + 1):
        profs = list(USER_PROFILES.keys())
        primary = random.choice(profs)
        secondary = random.choice([p for p in profs if p != primary])

        matched = profile_courses[primary] + profile_courses[secondary]
        if len(matched) < 10:
            matched = df["course_id"].tolist()

        # users rate matched courses highly
        n_match = random.randint(15, 25)
        for cid in random.sample(matched, min(n_match, len(matched))):
            ratings_list.append({
                "user_id": user_id,
                "course_id": cid,
                "rating": random.choice([4, 4, 5, 5])
            })

        # and a few random non-matched courses lowly
        non_matched = list(set(df["course_id"].tolist()) - set(matched))
        n_non_match = random.randint(2, 5)
        for cid in random.sample(non_matched, min(n_non_match, len(non_matched))):
            ratings_list.append({
                "user_id": user_id,
                "course_id": cid,
                "rating": random.choice([1, 2, 2, 3])
            })

    ratings_df = pd.DataFrame(ratings_list)
    return ratings_df


def train_collab_model(ratings_df):
    print("training collaborative SVD model...")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[["user_id", "course_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    svd.fit(trainset)
    return svd, trainset


def get_collab_predictions(user_id, svd, course_ids):
    predictions = []
    for cid in course_ids:
        pred = svd.predict(user_id, cid)
        predictions.append({"course_id": cid, "predicted_rating": pred.est})
    return pd.DataFrame(predictions)


def hybrid_recommend(user_skills, user_level, user_id,
                     tfidf, tfidf_matrix, df, svd,
                     top_n=10, alpha=0.6):
    query = user_skills + " " + user_level
    content_results = get_content_recommendations(query, tfidf, tfidf_matrix, df, top_n=200)

    candidate_ids = content_results["course_id"].tolist()
    collab_preds = get_collab_predictions(user_id, svd, candidate_ids)

    merged = content_results.merge(collab_preds, on="course_id", how="left")
    merged["predicted_rating"] = merged["predicted_rating"].fillna(merged["predicted_rating"].median())

    cs_min, cs_max = merged["content_score"].min(), merged["content_score"].max()
    pr_min, pr_max = merged["predicted_rating"].min(), merged["predicted_rating"].max()

    merged["norm_content"] = (merged["content_score"] - cs_min) / (cs_max - cs_min) if cs_max > cs_min else 0.5
    merged["norm_collab"] = (merged["predicted_rating"] - pr_min) / (pr_max - pr_min) if pr_max > pr_min else 0.5

    merged["hybrid_score"] = alpha * merged["norm_collab"] + (1 - alpha) * merged["norm_content"]

    if user_level and user_level != "All Levels":
        level_boost = merged["level"] == user_level
        merged.loc[level_boost, "hybrid_score"] += 0.1

    merged = merged.sort_values("hybrid_score", ascending=False).head(top_n)
    return merged


def evaluate_model(predictions, k=10, threshold=3.5):
    print("evaluating recommendations...")
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = []
    recalls = []
    f1s = []
    aps = []
    ndcgs = []

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])

        p = n_rel_and_rec_k / n_rec_k if n_rec_k > 0 else 0
        r = n_rel_and_rec_k / n_rel if n_rel > 0 else 0

        precisions.append(p)
        recalls.append(r)
        f1s.append(2 * (p * r) / (p + r) if (p + r) > 0 else 0)

        ap_sum = 0
        running_hits = 0
        for i, (est, true_r) in enumerate(user_ratings[:k]):
            if true_r >= threshold and est >= threshold:
                running_hits += 1
                ap_sum += running_hits / (i + 1)
        aps.append(ap_sum / min(n_rel, k) if n_rel > 0 else 0)

        hits = [1 if (true_r >= threshold and est >= threshold) else 0 for (est, true_r) in user_ratings[:k]]
        dcg = sum(hits[i] / np.log2(i + 2) for i in range(len(hits)))
        ideal_hits = sorted(hits, reverse=True)
        idcg = sum(ideal_hits[i] / np.log2(i + 2) for i in range(len(ideal_hits)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)

    print("\n--- evaluation metrics ---")
    print(f"  Precision@10: {np.mean(precisions):.4f}")
    print(f"  Recall@10:    {np.mean(recalls):.4f}")
    print(f"  F1-Score@10:  {np.mean(f1s):.4f}")
    print(f"  MAP@10:       {np.mean(aps):.4f}")
    print(f"  NDCG@10:      {np.mean(ndcgs):.4f}")


def generate_learning_path(goal, tfidf, tfidf_matrix, df):
    candidates = get_content_recommendations(goal, tfidf, tfidf_matrix, df, top_n=200)
    
    path = []
    levels = ["Beginner", "Intermediate", "Advanced"]
    
    for i, level in enumerate(levels):
        filtered = candidates[candidates["level"] == level]
        if not filtered.empty:
            best = filtered.sort_values(by=["rating", "content_score"], ascending=False).iloc[0]
            path.append({
                "step": len(path) + 1,
                "level": level,
                "course_id": best["course_id"],
                "title": best["title"],
                "platform": best["platform"],
                "rating": best["rating"]
            })
            
    # if we missed intermediate or advanced, fill using high-scoring 'All Levels' courses
    if len(path) < 3:
        all_levels = candidates[candidates["level"] == "All Levels"]
        if not all_levels.empty:
            for _, row in all_levels.sort_values(by=["rating", "content_score"], ascending=False).iterrows():
                if len(path) >= 3:
                    break
                # don't add duplicate titles
                if row["title"] not in [p["title"] for p in path]:
                    path.append({
                        "step": len(path) + 1,
                        "level": "General/Specialization",
                        "course_id": row["course_id"],
                        "title": row["title"],
                        "platform": row["platform"],
                        "rating": row["rating"]
                    })
                    
    return path


def save_models(tfidf, tfidf_matrix, svd, df, ratings_df):
    print("\nsaving artifacts...")
    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(MODEL_DIR, "tfidf_matrix.pkl"), "wb") as f:
        pickle.dump(tfidf_matrix, f)
    with open(os.path.join(MODEL_DIR, "svd_model.pkl"), "wb") as f:
        pickle.dump(svd, f)
    with open(os.path.join(MODEL_DIR, "course_data.pkl"), "wb") as f:
        pickle.dump(df, f)
    ratings_df.to_csv(os.path.join(DATASET_DIR, "synthetic_ratings.csv"), index=False)


def load_models():
    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "tfidf_matrix.pkl"), "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "svd_model.pkl"), "rb") as f:
        svd = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "course_data.pkl"), "rb") as f:
        df = pickle.load(f)
    return tfidf, tfidf_matrix, svd, df



def main():
    df = load_unified_data()
    tfidf, tfidf_matrix, df = train_content_model(df)

    ratings_df = generate_synthetic_ratings(df)
    
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[["user_id", "course_id", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.20, random_state=42)

    svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    svd.fit(trainset)
    
    predictions = svd.test(testset)
    evaluate_model(predictions)

    full_trainset = data.build_full_trainset()
    final_svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    final_svd.fit(full_trainset)

    save_models(tfidf, tfidf_matrix, final_svd, df, ratings_df)

    print("\n--- sample recommendations for a user ---")
    recs = hybrid_recommend(
        user_skills="python data science machine learning",
        user_level="Beginner",
        user_id=1,
        tfidf=tfidf,
        tfidf_matrix=tfidf_matrix,
        df=df,
        svd=final_svd,
        top_n=10,
        alpha=0.6
    )
    for _, row in recs.iterrows():
        print(f"  [{row['platform']:10}] {row['title'][:60]:60} | rating: {row['rating']:.1f} | score: {row['hybrid_score']:.3f}")

    print("\n--- generated learning path for 'Machine Learning Engineer' ---")
    path = generate_learning_path("Machine Learning Engineer", tfidf, tfidf_matrix, df)
    for step in path:
        print(f"  Step {step['step']} ({step['level']}): {step['title']} [{step['platform']}] (Rating: {step['rating']})")


if __name__ == "__main__":
    main()

