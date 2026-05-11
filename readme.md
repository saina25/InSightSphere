# InsightSphere

AI-Powered Customer Sentiment & Business Intelligence Dashboard

---

## Live Demo

🔗 [Launch Live Dashboard](YOUR_STREAMLIT_LINK_HERE)

---

## Overview

InsightSphere is an AI-powered analytics platform that transforms raw customer reviews into meaningful business insights.

The platform combines Natural Language Processing (NLP), Sentiment Analysis, and Customer Segmentation to help businesses understand customer behavior, identify recurring complaints, monitor product performance, and generate actionable recommendations.

Unlike traditional ML dashboards focused only on technical metrics, InsightSphere presents insights in a business-friendly format suitable for product managers, marketing teams, and decision-makers.

---

## Key Features

### Business Intelligence Dashboard
- Executive summary metrics
- Customer mood analysis
- Product performance insights
- Customer behavior groups
- AI-generated recommendations

### NLP & Sentiment Analysis
- Text preprocessing
- TF-IDF vectorization
- Logistic Regression sentiment classification
- Positive & negative review analysis

### Customer Segmentation
- K-Means clustering
- Customer persona generation
- PCA-based visualization

### Interactive Visualizations
- Sentiment distribution
- Ratings analysis
- Word clouds
- Cluster analysis
- Product comparisons

### Flexible Dataset Support
Supports:
- Flipkart Reviews
- Amazon Reviews
- Food Review Datasets
- App Review Datasets
- Any CSV-based customer feedback dataset

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core Programming |
| Streamlit | Web Dashboard |
| Scikit-learn | ML Models |
| NLP | Text Processing |
| Pandas | Data Handling |
| Matplotlib | Visualization |
| Seaborn | Statistical Charts |
| WordCloud | Review Trend Analysis |

---

## Project Architecture

```text
Dataset → NLP Processing → Sentiment Analysis →
Customer Segmentation → Business Insights Dashboard
```

---

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPO_LINK
```

Move into the project folder:

```bash
cd UNSUP-NLP-PROJ
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run main.py
```

## Future Improvements

- Real-time sentiment tracking
- Deep learning-based NLP models
- Auto dataset column detection
- Multilingual sentiment analysis
- PDF report generation
- Live database integration
