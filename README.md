# 🎓 Hybrid Course Recommender System

An end-to-end Course Recommendation Engine that integrates **41K+ course listings** from major learning platforms (Udemy, Coursera, edX, and Skillshare) and utilizes **Hybrid Recommendation Filtering** (combining Content-Based Filtering and Collaborative Filtering) to deliver personalized course suggestions and structured learning paths.

---

## 🚀 Key Features

*   **Multi-Platform Integration:** Aggregates and standardizes data from **Udemy** (25,799 courses), **Skillshare** (14,223 courses), **Coursera** (1,106 courses), and **edX** (783 courses).
*   **Hybrid Recommendation Engine:**
    *   **Content-Based:** Extracts feature text blobs and maps similarity using **TF-IDF Vectorization** (5,000 features, 1-2 n-grams) and **Cosine Similarity**.
    *   **Collaborative Filtering:** Personalizes results using **Singular Value Decomposition (SVD)** matrix factorization trained on user rating histories.
    *   **Weighted Blending:** Integrates both scoring methods via a dynamic parameter ($\alpha$) to adjust between personalized and keyword-driven matching.
*   **Learning Path Generator:** Automatically generates structured, step-by-step learning roadmaps (Beginner $\rightarrow$ Intermediate $\rightarrow$ Advanced) for targeted career goals.
*   **Interactive Web Application:** Built with **Streamlit** to search courses, filter by platform, tweak recommendations, and download results as CSV.
*   **Command Line Tool:** Quick terminal querying using `query_recommender.py`.

---

## 🛠️ Tech Stack & Libraries

*   **Core Logic:** Python 3.13
*   **Machine Learning:** Scikit-Learn (TF-IDF, Cosine Similarity), Surprise (SVD Matrix Factorization)
*   **Web App:** Streamlit
*   **Data Analysis:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn

---

## 📁 Repository Structure

```
├── DATASET/                  # Raw CSV files (Coursera, Udemy, edX, Skillshare)
│   └── unified_courses.csv   # Automatically generated cleaned & merged dataset
├── models/                   # Serialized ML models (SVD, TF-IDF matrices)
├── app.py                    # Streamlit web application
├── data_preprocessing.py     # Data pipeline to ingest, clean, and merge datasets
├── recommendation_engine.py  # Algorithms, SVD training, and hybrid formula
├── query_recommender.py      # Command-line utility to query recommendations
├── Course_Recommender_Project.ipynb # Jupyter notebook with EDA and step-by-step analysis
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation
```

---

## ⚡ Setup & Installation

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd <repository-directory>
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Run:
```bash
pip install -r requirements.txt
```

### 3. Build & Train the Models
Before launching the application, process the datasets and train the recommendation models by running the pipeline script:
```bash
python recommendation_engine.py
```
This script will:
*   Incorporate datasets from `DATASET/`.
*   Clean duration formats and impute missing ratings.
*   Generate the unified course catalogue (`unified_courses.csv`).
*   Train SVD collaborative filters and generate the TF-IDF vocabulary.
*   Save the model state files to the `models/` folder.

---

## 🖥️ Running the Applications

### Interactive Web Dashboard (Streamlit)
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```
Open the provided local URL (typically `http://localhost:8501`) in your browser to interact with the system.

### Command-Line Interface (CLI)
Search for recommendations directly from your terminal:
```bash
python query_recommender.py "data science beginner"
```

---

## 📈 Model Performance & Evaluation

The SVD Collaborative Filtering model was evaluated on a 20% validation split. Metrics computed at $k=10$:

| Evaluation Metric | Score | Explanation |
| :--- | :---: | :--- |
| **Precision@10** | **0.854** | Out of top-10 recommended courses, 85.4% are relevant to the user's taste. |
| **Recall@10** | **0.988** | 98.8% of user's liked courses were successfully retrieved. |
| **F1@10** | **0.907** | Harmonic mean of Precision and Recall. |
| **MAP@10** | **0.933** | Mean Average Precision of recommendations ranking. |
| **NDCG@10** | **0.960** | Normalized Discounted Cumulative Gain showing ideal ranking order. |
