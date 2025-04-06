# Movie Recommendation System – ML Model Comparison Benchmark

A system for evaluating collaborative and content-based filtering approaches for personalized movie recommendations.

## 🚀 Features
- Benchmarks recommendation algorithms for prediction accuracy.
- Compares filtering techniques, training time, and scalability.
- Real-time feedback using interactive UI.

## 📊 Benchmark Goals
- Evaluate performance on large user-item datasets.
- Automate model training and comparison.
- Ensure reproducibility with clean ML pipelines.

## 🛠️ Tech Stack
- Python, Pandas, scikit-learn
- Streamlit (for interactive UI)
- Surprise library for recommender models

## 📁 Structure
- `/models`: Filtering algorithms
- `/data`: MovieLens dataset
- `/scripts`: Training and benchmarking utilities
- `/ui`: Streamlit dashboard

## 📌 Reproducibility
Use `pip install -r requirements.txt` to install dependencies.  
Run `streamlit run ui/app.py` to launch UI and benchmark.

## 📈 Sample Output
- Accuracy: 92%
- Model training time reduced by 25%
