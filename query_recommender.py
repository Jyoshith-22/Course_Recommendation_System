import sys
from recommendation_engine import load_models, hybrid_recommend, generate_learning_path


def main():
    print("=" * 60)
    print("  Course Recommendation System — Interactive Query")
    print("=" * 60)

    try:
        tfidf, tfidf_matrix, svd, df = load_models()
    except FileNotFoundError:
        print("Error: Pre-trained models not found in the 'models/' directory.")
        print("Please run 'python recommendation_engine.py' first to train and save models.")
        sys.exit(1)

    print("\nModels loaded successfully!")

    while True:
        print("\n" + "-" * 50)
        print("Select an option:")
        print("  1. Get Personalized Recommendations (Hybrid)")
        print("  2. Generate a Learning Path (Roadmap)")
        print("  3. Exit")
        choice = input("Enter choice (1-3): ").strip()

        if choice == "3":
            print("Goodbye!")
            break

        elif choice == "1":
            skills = input("Enter your skills/interests (e.g., python web development): ").strip()
            level = input("Enter experience level (Beginner, Intermediate, Advanced, All Levels): ").strip()
            if not level:
                level = "All Levels"

            print(f"\nSearching courses for skills: '{skills}' | level: '{level}'...")
            
            # Using user_id=1 as a default for SVD prediction
            recs = hybrid_recommend(skills, level, 1, tfidf, tfidf_matrix, df, svd, top_n=10)
            
            print("\nTop 10 Recommendations:")
            for idx, row in recs.reset_index().iterrows():
                print(f"  {idx+1}. [{row['platform']:10}] {row['title']} (Rating: {row['rating']:.1f})")

        elif choice == "2":
            goal = input("Enter career goal/role (e.g., Machine Learning Engineer): ").strip()
            
            print(f"\nGenerating learning path for: '{goal}'...")
            path = generate_learning_path(goal, tfidf, tfidf_matrix, df)
            
            print("\nRecommended Roadmap:")
            for step in path:
                print(f"  Step {step['step']} ({step['level']}):")
                print(f"    Title   : {step['title']}")
                print(f"    Platform: {step['platform']}")
                print(f"    Rating  : {step['rating']}")


if __name__ == "__main__":
    main()
