# Neurofive ML Track

## Titanic Dataset - End-to-End Data Science & Machine Learning Project

This project was completed as part of the Neurofive Solutions ML Track. It covers the complete workflow from raw data exploration to building a predictive machine learning model.

### Tasks Completed
- Data inspection and exploratory data analysis (EDA)
- Missing value handling (Median for Age, Mode for Embarked, and dropping Cabin)
- Outlier detection using boxplots (Focused on Fare distribution)
- Data visualization using Matplotlib and Seaborn (Histogram, Boxplot, Bar Chart, Heatmap)
- Feature engineering and categorical encoding using `pd.get_dummies`
- Trained a **Logistic Regression** classification model to predict passenger survival
- Evaluated model performance using an **Accuracy Score** and **Confusion Matrix**

### Dataset
- Titanic - Machine Learning from Disaster (Kaggle)

### Tools Used
- Python
- Pandas
- NumPy
- Matplotlib & Seaborn
- Scikit-learn (for Machine Learning)
- Google Colab

### Key Insights & Model Results
* **EDA Discovery:** The analysis showed that **Sex** has the strongest impact on survival, followed by **Pclass**. Female passengers had a significantly higher survival rate than male passengers due to evacuation protocols.
* **Model Performance:** The Logistic Regression baseline model achieved a final accuracy of **81.01%** on the test set, successfully predicting 145 out of 179 passenger outcomes.
* **Evaluation Insight:** Through the confusion matrix, I noticed the model is slightly better at predicting non-survival (85.7% accuracy) than survival (74.3%), indicating a slight class imbalance that provides a great opportunity for future optimization.
