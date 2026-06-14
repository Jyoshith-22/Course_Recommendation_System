import pandas as pd
import numpy as np
import re
import os
import warnings

warnings.filterwarnings("ignore")

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DATASET")
OUTPUT_PATH = os.path.join(DATASET_DIR, "unified_courses.csv")


# --- duration parsing for each platform (they all store it differently smh) ---

def parse_coursera_duration(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()

    if "less than" in s and "hour" in s:
        return 1.5

    # some coursera entries are in spanish lol
    m = re.search(r"(\d+)\s*meses", s)
    if m:
        return int(m.group(1)) * 120

    m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*month", s)
    if m:
        return ((int(m.group(1)) + int(m.group(2))) / 2) * 120

    m = re.search(r"(\d+)\s*month", s)
    if m:
        return int(m.group(1)) * 120

    m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*week", s)
    if m:
        return ((int(m.group(1)) + int(m.group(2))) / 2) * 20

    m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*year", s)
    if m:
        return ((int(m.group(1)) + int(m.group(2))) / 2) * 1440

    return np.nan


def parse_skillshare_duration(val):
    # format is like "1h 19m" or just "55m"
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    hours, mins = 0, 0
    h = re.search(r"(\d+)h", s)
    m = re.search(r"(\d+)m", s)
    if h: hours = int(h.group(1))
    if m: mins = int(m.group(1))
    return round(hours + mins / 60, 2)


def parse_udemy_duration(val):
    # "22 total hours" -> 22.0
    if pd.isna(val):
        return np.nan
    m = re.search(r"([\d.]+)\s*total\s*hour", str(val).lower())
    return float(m.group(1)) if m else np.nan


# --- parsing review counts and student counts ---

def parse_review_count(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower().replace(",", "")

    m = re.search(r"([\d.]+)\s*k", s)
    if m:
        return int(float(m.group(1)) * 1000)

    m = re.search(r"([\d.]+)\s*m", s)
    if m:
        return int(float(m.group(1)) * 1_000_000)

    try:
        return int(float(s))
    except ValueError:
        return np.nan


def parse_students(val):
    # skillshare format: "133,422 students"
    if pd.isna(val):
        return np.nan
    s = str(val).replace(",", "").replace("students", "").strip()
    try:
        return int(s)
    except ValueError:
        return np.nan


# --- coursera stores skills in a weird curly-brace format ---

def clean_skills_str(val):
    if pd.isna(val):
        return ""
    s = str(val).replace("{", "").replace("}", "").replace('"', "")
    skills = [x.strip() for x in s.split(",") if x.strip()]
    return ", ".join(skills)


# --- normalize levels across platforms ---

def normalize_level(val):
    if pd.isna(val):
        return "All Levels"
    s = str(val).strip().lower()
    if s in ("beginner", "introductory"):
        return "Beginner"
    elif s == "intermediate":
        return "Intermediate"
    elif s in ("advanced", "expert"):
        return "Advanced"
    return "All Levels"


# --- try to guess category from the course title ---
# not perfect but works for most cases

CATEGORY_MAP = {
    "Data Science": ["data science", "data analysis", "data analytics", "big data", "data engineering"],
    "Machine Learning": ["machine learning", "deep learning", "neural network", "nlp",
                         "computer vision", "artificial intelligence", "ai "],
    "Web Development": ["web development", "html", "css", "javascript", "react", "angular",
                        "node.js", "django", "flask", "frontend", "backend", "full stack", "web design"],
    "Mobile Development": ["android", "ios", "flutter", "react native", "swift", "kotlin", "mobile app"],
    "Cloud & DevOps": ["aws", "azure", "google cloud", "docker", "kubernetes", "devops",
                       "cloud computing", "terraform"],
    "Cybersecurity": ["cybersecurity", "ethical hacking", "penetration testing", "network security"],
    "Programming": ["python", "java ", "c++", "c#", "golang", "rust", "programming", "coding",
                    "software development", "algorithm"],
    "Database": ["sql", "mysql", "postgresql", "mongodb", "database", "nosql"],
    "Business": ["business", "management", "marketing", "finance", "accounting",
                 "project management", "entrepreneurship", "excel", "power bi", "tableau"],
    "Design": ["graphic design", "ui/ux", "ux design", "photoshop", "illustrator",
               "figma", "adobe", "animation", "3d modeling", "blender"],
    "Photography & Video": ["photography", "video editing", "filmmaking", "premiere",
                            "after effects", "cinematography"],
    "Music & Audio": ["music", "guitar", "piano", "audio", "sound design", "music production"],
    "Writing": ["writing", "copywriting", "content writing", "creative writing", "blogging"],
    "Personal Development": ["productivity", "communication skills", "public speaking",
                             "meditation", "time management"],
}

def guess_category(title):
    if pd.isna(title):
        return "Other"
    t = str(title).lower()
    for cat, keywords in CATEGORY_MAP.items():
        for kw in keywords:
            if kw in t:
                return cat
    return "Other"


# ============================================================
# cleaning each platform's data
# ============================================================

def process_coursera(df):
    print(f"  coursera: {len(df)} rows")
    out = pd.DataFrame()
    out["title"] = df["course"].str.strip()
    out["platform"] = "Coursera"
    out["description"] = ""
    out["instructor"] = df["partner"].str.strip()
    out["skills"] = df["skills"].apply(clean_skills_str)
    out["level"] = df["level"].apply(normalize_level)
    out["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    out["review_count"] = df["reviewcount"].apply(parse_review_count)
    out["duration_hours"] = df["duration"].apply(parse_coursera_duration)
    out["url"] = ""

    # try to figure out category from skills, fall back to title
    out["category"] = ""
    for i, row in out.iterrows():
        cat = guess_category(row["skills"]) if row["skills"] else "Other"
        if cat == "Other":
            cat = guess_category(row["title"])
        out.at[i, "category"] = cat

    return out


def process_edx(df):
    df = df.dropna(subset=["title"]).copy()
    print(f"  edx: {len(df)} rows")
    out = pd.DataFrame()
    out["title"] = df["title"].str.strip()
    out["platform"] = "edX"
    out["description"] = ""
    out["instructor"] = df["institution"].fillna("Unknown").str.strip()
    out["skills"] = df["associatedskills"].fillna("").str.strip()
    out["category"] = df["subject"].fillna("Other").str.strip()
    out["level"] = df["level"].apply(normalize_level)
    out["rating"] = np.nan
    out["review_count"] = np.nan
    out["duration_hours"] = np.nan
    out["url"] = df["link"].fillna("")
    return out


def process_skillshare(df):
    print(f"  skillshare: {len(df)} rows")
    out = pd.DataFrame()
    out["title"] = df["title"].str.strip()
    out["platform"] = "Skillshare"
    out["description"] = ""
    out["instructor"] = df["instructor"].str.strip()
    out["skills"] = ""
    out["category"] = df["title"].apply(guess_category)
    out["level"] = "All Levels"
    out["rating"] = np.nan
    out["review_count"] = df["students"].apply(parse_students)
    out["duration_hours"] = df["duration"].apply(parse_skillshare_duration)
    out["url"] = df["link"].fillna("")
    return out


def process_udemy(df):
    print(f"  udemy: {len(df)} rows")
    out = pd.DataFrame()
    out["title"] = df["title"].str.strip()
    out["platform"] = "Udemy"
    out["description"] = df["description"].fillna("").str.strip()
    out["instructor"] = df["instructor"].str.strip()
    out["skills"] = ""
    out["category"] = df["title"].apply(guess_category)
    out["level"] = df["level"].apply(normalize_level)
    out["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    out["review_count"] = df["reviewcount"].apply(parse_review_count)
    out["duration_hours"] = df["duration"].apply(parse_udemy_duration)
    out["url"] = ""
    return out


# ============================================================
# put everything together
# ============================================================

def run_pipeline():
    print("loading raw datasets...")
    coursera = pd.read_csv(os.path.join(DATASET_DIR, "Coursera.csv"))
    edx = pd.read_csv(os.path.join(DATASET_DIR, "edx.csv"))
    skillshare = pd.read_csv(os.path.join(DATASET_DIR, "skillshare.csv"))
    udemy = pd.read_csv(os.path.join(DATASET_DIR, "Udemy.csv"))

    print("cleaning each platform...")
    c = process_coursera(coursera)
    e = process_edx(edx)
    s = process_skillshare(skillshare)
    u = process_udemy(udemy)

    # merge everything into one dataframe
    print("merging datasets...")
    df = pd.concat([c, e, s, u], ignore_index=True)
    df.insert(0, "course_id", range(1, len(df) + 1))
    print(f"  total before cleanup: {len(df)}")

    # remove duplicate titles on the same platform
    before = len(df)
    df = df.drop_duplicates(subset=["title", "platform"], keep="first")
    print(f"  removed {before - len(df)} duplicates")

    # drop courses with super short titles (probably junk)
    df = df[df["title"].str.len() >= 5].copy()

    # fill missing ratings with platform median, then global median
    for plat in df["platform"].unique():
        mask = (df["platform"] == plat) & (df["rating"].isna())
        med = df.loc[df["platform"] == plat, "rating"].median()
        if not pd.isna(med):
            df.loc[mask, "rating"] = med
    df["rating"] = df["rating"].fillna(df["rating"].median())
    df["rating"] = df["rating"].clip(1.0, 5.0)

    # fill missing durations similarly
    for plat in df["platform"].unique():
        mask = (df["platform"] == plat) & (df["duration_hours"].isna())
        med = df.loc[df["platform"] == plat, "duration_hours"].median()
        if not pd.isna(med):
            df.loc[mask, "duration_hours"] = med
    df["duration_hours"] = df["duration_hours"].fillna(df["duration_hours"].median())
    df["duration_hours"] = df["duration_hours"].round(1)

    df["review_count"] = df["review_count"].fillna(0).astype(int)

    # reset ids after cleanup
    df = df.reset_index(drop=True)
    df["course_id"] = range(1, len(df) + 1)

    # quick summary
    print(f"\nfinal dataset: {len(df)} courses")
    print(f"  platforms: {df['platform'].value_counts().to_dict()}")
    print(f"  levels: {df['level'].value_counts().to_dict()}")
    print(f"  avg rating: {df['rating'].mean():.2f}")
    print(f"  median duration: {df['duration_hours'].median():.1f} hrs")

    # save it
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\nsaved to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    df = run_pipeline()
