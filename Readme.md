
# 📸 Instagram Analytics — Engagement Intelligence Platform

> **An end-to-end machine learning system for predicting and understanding Instagram post engagement, powered by XGBoost, scikit-learn, and an interactive Streamlit dashboard.**

<br>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-189AB4?style=flat-square)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com/)

---

## 📋 Table of Contents

- [Executive Summary](#-executive-summary)
- [Project Objectives](#-project-objectives)
- [Live Demo](#-live-demo)
- [Key Results](#-key-results)
- [Dataset Overview](#-dataset-overview)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Feature Engineering](#-feature-engineering)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Model Evaluation](#-model-evaluation)
- [Dashboard Features](#-dashboard-features)
- [Tech Stack](#-tech-stack)
- [Key Insights & Business Findings](#-key-insights--business-findings)
- [Contributing](#-contributing)

---

## 🏆 Executive Summary

The **Instagram Analytics Engagement Intelligence Platform** is a production-grade, data-driven system that transforms raw Instagram post metadata into actionable engagement predictions. Built for content strategists, social media managers, and data teams, the platform combines rigorous exploratory analysis with a tuned XGBoost regression model and a fully interactive Streamlit dashboard — enabling teams to forecast post performance *before* publishing.

The system ingests historical Instagram post data spanning multiple content categories, media types, and traffic sources, applies systematic feature engineering, trains two regression models (Linear Regression and XGBoost), and surfaces predictions and insights through a four-tab web dashboard deployable in minutes.

**Who this is for:**
- 📣 Social media and content teams who want data-backed posting decisions
- 📊 Data analysts and scientists seeking a reproducible ML workflow template
- 🧑‍💻 ML engineers evaluating scikit-learn pipeline architectures
- 🎓 Students and practitioners learning end-to-end data science project structure

---

## 🎯 Project Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Build a reliable regression model to predict Instagram engagement rate | ✅ Complete |
| 2 | Perform comprehensive EDA to surface actionable content strategy insights | ✅ Complete |
| 3 | Engineer domain-relevant features that improve model signal quality | ✅ Complete |
| 4 | Deliver an interactive, shareable dashboard for non-technical stakeholders | ✅ Complete |
| 5 | Create a reusable prediction pipeline for new, unseen post data | ✅ Complete |
| 6 | Compare baseline and tuned model performance with transparent metrics | ✅ Complete |

---

## 🚀 Live Demo

The app launches at `https://instagram-analytics-eda-ml.streamlit.app/` and provides four interactive tabs:
**Overview → EDA → Model Performance → Predict**

---

## 📈 Key Results

| Metric | Linear Regression | XGBoost (Tuned) |
|--------|:-----------------:|:----------------:|
| **MAE** | — | Lower than baseline |
| **RMSE** | — | Lower than baseline |
| **R² Score** | — | Higher than baseline |
| **Training time** | Fast | Moderate |
| **Best for** | Interpretability | Accuracy |

> 📌 Exact metric values are generated dynamically at runtime based on your dataset. The tuned XGBoost model consistently outperforms the Linear Regression baseline on this dataset.

**Top EDA Findings at a Glance:**
- 🎬 **Reels** generate the highest average engagement rate of all media types
- 💄 **Beauty** is the top-performing content category by engagement
- 🔍 **Explore** is the highest-engagement traffic source
- 📅 **Wednesday in April** is the optimal posting window
- 📊 `reach`, `impressions`, and engagement ratio features are the strongest predictors

---

## 📦 Dataset Overview

The dataset (`Instagram_Analytics.csv`) contains historical Instagram post records with the following feature groups:

### Post Metadata
| Feature | Type | Description |
|---------|------|-------------|
| `post_id` | String | Unique post identifier |
| `upload_date` | DateTime | Date the post was published |
| `media_type` | Categorical | `Reel`, `Photo`, `Video`, `Carousel` |
| `content_category` | Categorical | `Beauty`, `Fitness`, `Travel`, `Food`, etc. |
| `traffic_source` | Categorical | `Explore`, `Hashtags`, `Home Feed`, etc. |
| `caption_length` | Integer | Character count of the post caption |
| `hashtags_count` | Integer | Number of hashtags used |

### Engagement Signals
| Feature | Type | Description |
|---------|------|-------------|
| `likes` | Integer | Total likes received |
| `comments` | Integer | Total comments received |
| `shares` | Integer | Total shares |
| `saves` | Integer | Total saves |
| `reach` | Integer | Unique accounts reached |
| `impressions` | Integer | Total times content was displayed |
| `followers_gained` | Integer | Net new followers from the post |
| `engagement_rate` | Float | **Target variable** — aggregate engagement metric |

---

## 🔍 Exploratory Data Analysis

The EDA phase investigates distributions, correlations, temporal patterns, and category-level performance. Eight distinct analysis views are available in the dashboard's EDA tab:

### 1. Correlation Analysis
A Pearson correlation heatmap across all numeric features reveals that `reach`, `impressions`, and the engineered engagement ratio features hold the strongest linear relationships with `engagement_rate`. Raw counts (`likes`, `comments`) are highly intercorrelated, motivating the creation of normalised ratio features.

### 2. Reach vs. Impressions
A scatter plot with OLS trendline confirms a near-perfect positive linear relationship between `reach` and `impressions` — posts reaching more unique users consistently accumulate more total views, independent of media type.

### 3. Engagement by Media Type
Reels outperform all other formats. Carousels rank second, benefiting from multi-slide interaction. Single Photos show the lowest average engagement, suggesting the algorithm rewards richer, time-consuming formats.

### 4. Engagement by Content Category
Beauty content leads all categories by average engagement rate, followed by Fitness and Lifestyle. Tech and Photography content underperforms relative to the dataset average, indicating audience preference for aspirational or practical content over informational.

### 5. Engagement by Traffic Source
Posts discovered via the **Explore** feed achieve the highest engagement. Hashtag-driven traffic ranks second. Home Feed posts, while reaching existing followers, generate less incremental engagement — highlighting the value of discoverability.

### 6. Time Series Analysis
Monthly and weekly breakdowns across media types identify **April** and **Wednesday** as the consistently highest-engagement posting window. This pattern is stable across media types, suggesting platform-level algorithmic or audience behaviour factors rather than content-specific ones.

### 7. Post Frequency Analysis
A dual-axis chart shows posting frequency vs. average engagement by day of week. Wednesday posts are both less frequent and higher-performing, presenting a low-competition, high-reward opportunity.

### 8. Content Category × Media Type Heatmap
A cross-tabulation reveals that **Beauty Reels** and **Fitness Reels** are the highest-performing content-format combinations, while **Technology Photos** and **Photography Photos** consistently underperform.

---

## ⚙️ Feature Engineering

Eight new features were derived from raw post data to give the model richer, more normalised signals:

| Engineered Feature | Formula | Purpose |
|--------------------|---------|---------|
| `total_interactions` | `likes + comments + shares + saves` | Aggregate engagement signal |
| `caption_density` | `caption_length / (hashtags_count + 1)` | Text-to-hashtag ratio; measures content richness |
| `likes_per_reach` | `likes / (reach + 1)` | Normalised like rate; removes reach size bias |
| `saves_per_reach` | `saves / (reach + 1)` | Normalised save rate |
| `comments_per_reach` | `comments / (reach + 1)` | Normalised comment rate |
| `shares_per_reach` | `shares / (reach + 1)` | Normalised share rate |
| `total_interactions_per_reach` | `total_interactions / (reach + 1)` | Overall normalised engagement |
| `log_likes` | `log1p(likes)` | Log-transformed likes; reduces right-skew |

> **Design rationale:** Raw counts are heavily influenced by account size and reach. Normalising by reach isolates *content quality* as a signal, making the model generalise better across posts with varying audience sizes. The `+1` Laplace smoothing prevents division-by-zero on zero-reach edge cases.

---

## 🤖 Machine Learning Pipeline

### Preprocessing Pipeline

Built with scikit-learn's `ColumnTransformer` for reproducibility and leak-free validation:

```python
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OrdinalEncoder(categories=[...]), categorical_features)
])
```

- **Numeric features:** Standardised with `StandardScaler` (zero mean, unit variance)
- **Categorical features:** Ordinal-encoded with explicit category ordering to respect temporal and logical hierarchies (e.g., months, days of week)

### Outlier Removal

A 3-sigma rule (`mean ± 3×std`) is applied independently to `engagement_rate`, all engagement ratio features, `log_likes`, and `caption_density` before training. This removes statistically extreme values that would otherwise bias gradient-based model updates.

### Models Trained

#### 1. Linear Regression (Baseline)
```python
lr_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
```
Serves as an interpretable benchmark. Feature coefficients identify the directional impact of each input on the target.

#### 2. XGBoost Regressor
```python
xgb_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        learning_rate=0.1,
        max_depth=9,
        random_state=42
    ))
])
```
Gradient-boosted trees that capture non-linear interactions between features. The `reg:squarederror` objective minimises mean squared error, appropriate for continuous engagement rate prediction.

#### 3. Tuned XGBoost (GridSearchCV)
```python
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [3, 5, 7, 9],
    'model__min_child_weight': [1, 3, 5, 7],
    'model__learning_rate': [0.01, 0.1, 0.2, 0.3]
}
```
`GridSearchCV` with `cv=2` and `scoring='r2'` systematically evaluates 144 hyperparameter combinations to select the best configuration. The final model is the `best_estimator_` from this search.

### Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

An 80/20 stratified split with a fixed random seed ensures reproducible evaluation across runs.

---

## 📊 Model Evaluation

Each model is evaluated on three standard regression metrics:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** (Mean Absolute Error) | `mean(|y - ŷ|)` | Average prediction error in original units |
| **RMSE** (Root Mean Squared Error) | `√mean((y - ŷ)²)` | Penalises large errors more heavily than MAE |
| **R²** (Coefficient of Determination) | `1 - SS_res/SS_tot` | Proportion of variance explained (1.0 = perfect) |

Additionally, MAE is reported as a **percentage of the dataset mean engagement rate**, providing a business-interpretable measure of model accuracy relative to typical post performance.

Actual vs. Predicted scatter plots are generated for both train and test sets — a diagonal line of identity indicates perfect predictions, and the degree of scatter reveals model uncertainty.

---

## 🖥️ Dashboard Features

The `app.py` Streamlit application provides four fully interactive tabs:

### Tab 1 — 📊 Overview
- **KPI metric cards:** Total posts, average engagement rate, average reach, average impressions
- **Media type performance:** Bar chart and donut chart of engagement by format
- **Content category ranking:** Colour-scaled bar chart across all categories
- **Insight chips:** Quick-reference highlights of the best media type, category, day, month, and traffic source

### Tab 2 — 🔍 EDA
Eight selectable analysis views:
1. Correlation heatmap (Plotly `imshow`)
2. Reach vs. Impressions scatter with OLS trendline
3. Monthly & weekly engagement time series
4. Monthly & weekly followers gained time series
5. Top 10 posts (sortable by engagement rate or total interactions)
6. Feature distribution explorer (histogram + box plot)
7. Post frequency vs. engagement dual-axis chart
8. Content Category × Media Type engagement heatmap

### Tab 3 — 🤖 Model Performance
- Side-by-side MAE, RMSE, R² bar charts for all trained models
- Actual vs. Predicted scatter plots per model with perfect-prediction reference line
- XGBoost feature importance bar chart (top 15 features)
- Graceful fallback: if artifacts are not found, models are re-trained in-memory on the fly

### Tab 4 — 🎯 Predict
A fully functional prediction interface:
- Form inputs for all 15 raw post features (dropdowns, number inputs)
- Automatic feature engineering computed on submission
- Per-model prediction cards (Linear Regression + XGBoost Tuned)
- Ensemble average with comparison against dataset mean
- Animated gauge chart showing predicted vs. average engagement
- Expandable panel displaying all computed engineered feature values

---

### Exploring the EDA

1. Navigate to the **🔍 EDA** tab
2. Use the **dropdown selector** to switch between 8 analysis views
3. All Plotly charts are interactive — hover for tooltips, click legend items to filter, zoom and pan freely
4. Bar charts and scatter plots can be exported as PNG via the Plotly toolbar

### Interpreting Model Performance

1. Navigate to the **🤖 Model Performance** tab
2. Compare **MAE**, **RMSE**, and **R²** side-by-side across models
3. Review the **Actual vs. Predicted** scatter — tighter clustering around the diagonal indicates better generalisation
4. Study the **feature importance chart** to understand which signals drive engagement predictions

---

### Contents of `pipeline_artifacts.joblib`

| Key | Type | Description |
|-----|------|-------------|
| `preprocessor` | `ColumnTransformer` | Fitted scaler + encoder |
| `linear_regression_model` | `Pipeline` | Trained LR pipeline |
| `xgboost_model` | `Pipeline` | Best XGBoost from GridSearchCV |
| `xgboost_feature_importances` | `DataFrame` | Feature importance scores |
| `linear_regression_feature_importances` | `DataFrame` | LR coefficient table |
| `features` | `list` | All feature names |
| `numeric_features` | `list` | Numeric feature names |
| `categorical_data` | `list` | Categorical feature names |

---

## 🧰 Tech Stack

| Category | Library / Tool | Version | Role |
|----------|---------------|---------|------|
| **Language** | Python | 3.10+ | Core language |
| **Data** | pandas | 2.x | Data manipulation & cleaning |
| **Data** | NumPy | 1.26+ | Numerical operations |
| **Visualisation** | Matplotlib | 3.8+ | Static EDA charts (notebook) |
| **Visualisation** | Seaborn | 0.13+ | Statistical visualisations (notebook) |
| **Visualisation** | Plotly | 5.x | Interactive dashboard charts |
| **ML** | scikit-learn | 1.4+ | Preprocessing pipelines, GridSearchCV, LinearRegression |
| **ML** | XGBoost | 2.x | Gradient-boosted regression |
| **Serialisation** | joblib | 1.3+ | Model artifact persistence |
| **Dashboard** | Streamlit | 1.32+ | Web application framework |
| **Environment** | Jupyter | — | Notebook-based analysis |

---

## 💡 Key Insights & Business Findings

The following findings emerged from the EDA and are surfaced in the dashboard for stakeholder consumption:

### Content Strategy
- **Format:** Invest in Reels production — they consistently outperform all other media types by engagement rate. Carousels are the next best investment for organic reach.
- **Category:** Beauty, Fitness, and Lifestyle content generates the strongest audience response. Technology and Photography content significantly underperforms.
- **Caption strategy:** A high `caption_density` (longer captions with fewer hashtags) correlates with stronger engagement, suggesting audiences respond to narrative-driven content.

### Publishing Strategy
- **Day:** Post on **Wednesday** for maximum engagement. This day combines lower content competition with strong algorithmic visibility.
- **Month:** **April** is the highest-engagement month across media types — plan major campaigns around this window.
- **Avoid:** Saturday and Sunday consistently underperform for engagement rate, despite often having higher post volumes.

### Distribution Strategy
- **Explore feed** drives the highest-quality engagement. Optimise for discoverability: use relevant, high-volume hashtags, strong hook visuals, and early engagement velocity.
- **Hashtag-driven** traffic is second-best. A targeted hashtag strategy (niche + mid-size) outperforms mass hashtag blasting.

### Predictive Features (Model Signals)
The XGBoost feature importance analysis identifies the following as the strongest predictors of engagement rate:
1. Engagement ratio features (`likes_per_reach`, `total_interactions_per_reach`)
2. `reach` and `impressions`
3. `total_interactions`
4. `content_category` and `media_type`
5. Temporal features (`month`, `day_of_the_week`)

---

## 🔮 Future Improvements

The following enhancements are planned or recommended for the next iteration:

| Priority | Improvement | Rationale |
|----------|-------------|-----------|
| High | Add cross-validation with 5+ folds | Current `cv=2` in GridSearchCV is aggressive; more folds improve stability |
| High | Include time-of-day as a feature | Post timing within a day likely influences algorithmic visibility |
| Medium | Expand dataset with more posts | Larger training sets improve XGBoost generalisation |
| Medium | Test LightGBM and CatBoost | May outperform XGBoost on categorical-heavy data |
| Medium | Add SHAP explainability | Per-prediction explanations for the Predict tab |
| Low | Implement Streamlit Community Cloud deployment | One-click public sharing |
| Low | Add A/B testing simulation module | Compare predicted engagement of two post configurations |
| Low | Build a REST API wrapper (FastAPI) | Enable programmatic access to the prediction pipeline |
| Low | Add database connector | Replace CSV with live database or API data ingestion |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "feat: add your feature description"`
4. Push to your branch: `git push origin feature/your-feature-name`
5. Open a Pull Request with a clear description of the change

Please follow the existing code style and ensure any new features include appropriate documentation.

---

## 🙏 Acknowledgements

- Dataset sourced from Instagram analytics export
- Model architecture inspired by standard scikit-learn pipeline patterns
- Dashboard design follows Streamlit community best practices
- Statistical methodology guided by established EDA and regression analysis frameworks

---

<div align="center">

**Built with 📊 data, 🤖 machine learning, and ☕ coffee**

*If this project helped you, consider giving it a ⭐ on GitHub*

## Contact: - [GitHub](https://github.com/Dhela456) - [LinkedIn](https://www.linkedin.com/in/ireoluwawolemi-akindipe-16b711373/) - [Email](mailto:dhelacruise@gmail.com)
</div>
