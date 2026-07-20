# Neurofive ML Track

## Titanic Dataset - End-to-End Data Science & Machine Learning Project

This project was completed as part of the Neurofive Solutions ML Track. It covers the complete workflow from raw data exploration to building, evaluating, and systematically optimizing a predictive machine learning model.

### Tasks Completed
- Data inspection and exploratory data analysis (EDA)
- Missing value handling (Median for Age, Mode for Embarked, and dropping Cabin)
- Outlier detection using boxplots (Focused on Fare distribution)
- Data visualization using Matplotlib and Seaborn (Histogram, Boxplot, Bar Chart, Heatmap)
- Feature engineering and categorical encoding using `pd.get_dummies`
- Trained baseline classification models to predict passenger survival
- Evaluated performance beyond basic accuracy using **Precision, Recall, F1-Score**, and **Classification Report**
- Applied **GridSearchCV** for hyperparameter tuning (`max_depth`, `min_samples_split`, `n_estimators`)

### Dataset
- Titanic - Machine Learning from Disaster (Kaggle)

### Tools Used
- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn (RandomForestClassifier, GridSearchCV, Metrics)
- Google Colab

---

### Key Insights & Model Evaluation

* **EDA Discovery:** The analysis showed that **Sex** has the strongest impact on survival, followed by **Pclass**. Female passengers had a significantly higher survival rate than male passengers due to evacuation protocols.
* **Why Accuracy Alone is Misleading:** Relying solely on accuracy can be deceptive on imbalanced datasets. A naive model can achieve high overall accuracy while failing completely on the minority class. Therefore, we prioritized **Precision, Recall, and F1-score** to ensure reliable minority-class prediction.

---

### Model Optimization & GridSearchCV Results

We performed systematic hyperparameter tuning using `GridSearchCV` with 5-fold cross-validation targeting the F1-score.

**Best Parameters Found:**
* `max_depth`: 5
* `min_samples_split`: 5
* `n_estimators`: 100

#### Before vs. After Performance Comparison

| Metric / Feature | Baseline Model | Tuned Model | Difference / Impact |
| :--- | :--- | :--- | :--- |
| **Parameters** | Default Settings | `max_depth=5, min_samples_split=5, n_estimators=100` | Regularized / Overfitting Control |
| **Accuracy** | 0.82 | 0.81 | -0.01 |
| **Precision (Class 1)** | 0.80 | **0.82** | **+0.02 (Improved)** |
| **Recall (Class 1)** | 0.74 | 0.69 | -0.05 (Trade-off) |
| **F1-Score (Class 1)** | 0.77 | 0.75 | -0.02 |

**Key Takeaway:** By setting depth constraints (`max_depth=5`), the tuned model reduced overfitting and improved **Precision** for positive predictions (Class 1) from **80% to 82%**, making survival predictions more dependable..
