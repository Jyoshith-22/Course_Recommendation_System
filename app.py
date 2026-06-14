import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recommendation_engine import load_models, hybrid_recommend, generate_learning_path

st.set_page_config(
    page_title="Course Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── cached model loader ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def get_models():
    return load_models()


# ── helpers ─────────────────────────────────────────────────────
def star_str(rating):
    filled = int(round(rating))
    return "★" * filled + "☆" * (5 - filled)


# ── sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 Course Recommender")
    st.caption("Find courses that match your skills and goals.")
    st.divider()

    st.subheader("How it works")
    st.write(
        "TF-IDF + cosine similarity finds relevant courses from 41K+ options. "
        "SVD collaborative filtering predicts how much you'd like each one. "
        "Both scores are blended into a single hybrid rank."
    )
    st.divider()

    st.subheader("Model Performance")
    st.table({
        "Metric": ["Precision@10", "Recall@10", "F1@10", "MAP@10", "NDCG@10"],
        "Score":  [0.854, 0.988, 0.907, 0.933, 0.960]
    })
    st.divider()

    st.subheader("Dataset")
    st.write(
        "- Udemy: 25,799 courses\n"
        "- Coursera: 1,106 courses\n"
        "- Skillshare: 14,223 courses\n"
        "- edX: 783 courses\n"
        "- **Total: 41,911 courses**"
    )


# ── header ──────────────────────────────────────────────────────
st.title("Course Recommendation System")
st.write("Enter your skills or career goal below to get personalised course recommendations.")
st.divider()


# ── load models ─────────────────────────────────────────────────
try:
    tfidf, tfidf_matrix, svd, df = get_models()
except FileNotFoundError:
    st.error("Models not found. Run `python recommendation_engine.py` first.")
    st.stop()


# ── tabs ────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Course Recommendations", "Learning Path Generator"])


# ────────────────────────────────────────────────────────────────
# TAB 1 — Recommendations
# ────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Find Courses")

    col_q, col_lvl = st.columns([4, 1])
    with col_q:
        query = st.text_input(
            "What do you want to learn?",
            placeholder="e.g. python data science, web development react, ui ux design"
        )
    with col_lvl:
        level = st.selectbox("Experience Level", ["All Levels", "Beginner", "Intermediate", "Advanced"])

    col_p, col_n, col_a = st.columns([1, 1, 2])
    with col_p:
        plat = st.selectbox("Platform", ["All", "Udemy", "Coursera", "Skillshare", "edX"])
    with col_n:
        top_n = st.selectbox("Number of Results", [5, 10, 15, 20], index=1)
    with col_a:
        alpha = st.slider(
            "Collaborative  ←  →  Content-based",
            0.0, 1.0, 0.6, 0.05,
            help="Higher = more personalised (collaborative). Lower = more keyword-matching (content)."
        )

    go = st.button("Search Courses", key="btn_rec")

    if go:
        if not query.strip():
            st.warning("Please enter at least one skill or topic.")
        else:
            with st.spinner("Searching..."):
                recs = hybrid_recommend(
                    user_skills=query, user_level=level,
                    user_id=1, tfidf=tfidf, tfidf_matrix=tfidf_matrix,
                    df=df, svd=svd, top_n=top_n * 4, alpha=alpha
                )
                if plat != "All":
                    recs = recs[recs["platform"] == plat]
                recs = recs.head(top_n).reset_index(drop=True)

            st.divider()
            if recs.empty:
                st.info("No courses matched those filters. Try a broader query or change the platform filter.")
            else:
                st.write(f"Showing **{len(recs)}** results for **\"{query}\"**")
                st.write("")

                for i, (_, row) in enumerate(recs.iterrows()):
                    category = row.get("category", "")
                    rating = float(row.get("rating", 0))
                    score = float(row.get("hybrid_score", row.get("content_score", 0)))
                    platform = row.get("platform", "")
                    lv = row.get("level", "")

                    with st.container():
                        c1, c2 = st.columns([10, 1])
                        with c1:
                            st.markdown(f"**{i+1}. {row['title']}**")
                            meta = f"Platform: {platform} &nbsp;|&nbsp; Level: {lv}"
                            if category and category != "Other":
                                meta += f" &nbsp;|&nbsp; Category: {category}"
                            meta += f" &nbsp;|&nbsp; Match: {score:.2f}"
                            st.caption(meta)
                        with c2:
                            st.write(f"⭐ {rating:.1f}")
                            st.caption(star_str(rating))
                    st.divider()

                csv = recs[["title", "platform", "category", "level", "rating", "hybrid_score"]].to_csv(index=False)
                st.download_button("Download Results as CSV", csv, "recommendations.csv", "text/csv")


# ────────────────────────────────────────────────────────────────
# TAB 2 — Learning Path
# ────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Learning Path Generator")
    st.write("Enter a career goal and get a step-by-step course roadmap.")

    goal = st.text_input(
        "Career Goal",
        placeholder="e.g. Machine Learning Engineer, Full-Stack Web Developer, Data Analyst"
    )
    go2 = st.button("Generate Roadmap", key="btn_path")

    if go2:
        if not goal.strip():
            st.warning("Please enter a career goal.")
        else:
            with st.spinner(f"Building roadmap for '{goal}'..."):
                path = generate_learning_path(goal, tfidf, tfidf_matrix, df)

            st.divider()
            if not path:
                st.info("Couldn't generate a path for that goal. Try something like 'Python Developer' or 'Data Scientist'.")
            else:
                st.write(f"**Your roadmap to become a {goal}:**")
                st.caption("Work through these courses in order, from basics to job-ready.")
                st.write("")

                for step in path:
                    platform = step["platform"]
                    rating = float(step["rating"])
                    with st.container():
                        st.markdown(f"**Step {step['step']} — {step['level']}**")
                        st.write(step["title"])
                        st.caption(f"Platform: {platform} &nbsp;|&nbsp; Rating: {rating:.1f} {star_str(rating)}")
                    st.divider()

                path_df = pd.DataFrame(path)[["step", "level", "title", "platform", "rating"]]
                st.download_button(
                    "Download Roadmap as CSV",
                    path_df.to_csv(index=False),
                    f"roadmap_{goal.replace(' ', '_')}.csv",
                    "text/csv"
                )
