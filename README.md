# Mobile Price Prediction Model

Welcome to the Mobile Price Prediction Model project! In this machine learning endeavor, the goal was to predict mobile prices based on various features. The project involved preprocessing the `train.csv` dataset, including data cleaning using pandas, followed by exploratory data analysis (EDA) using seaborn and matplotlib for visual insights. The data was then scaled using StandardScaler to prepare it for modeling.

## Project Overview

### Dataset

- **Mobile Price Dataset:**
  - Two datasets: `train.csv` for model training and `test.csv` for predictions.
  - Acquired from Kaggle.

### Data Preprocessing and EDA

- **Data Cleaning:**
  - The `train.csv` dataset underwent preprocessing to handle missing values and ensure data integrity.

- **Exploratory Data Analysis (EDA):**
  - Seaborn and Matplotlib were employed to visually analyze and explore the dataset, gaining insights into feature distributions and relationships.

### Model Training

- **Four Manual Models:**
  - SVM, KNN, Decision Tree Classifier, and Random Forest Classifier were manually trained on the `train.csv` dataset.

- **Stratified KFold Cross Validation:**
  - The models were retrained using Stratified KFold cross-validation, and the average score for each model was computed.

### Hyperparameter Tuning

- **GridSearchCV:**
  - SVM emerged as the top-performing model with 96% accuracy after hyperparameter tuning using GridSearchCV.

### Model Finalization and Prediction

- **Finalized Model:**
  - The SVM model was selected as the final model for predicting mobile prices.

- **Test Dataset Prediction:**
  - The finalized SVM model was applied to the `test.csv` dataset to predict mobile prices based on the provided features.

## How to Use

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/mobile-price-prediction.git
   cd mobile-price-prediction
   ```

2. **Run the Jupyter Notebook:**
   - Execute the Jupyter Notebook (`mobile_price_prediction.ipynb`) in your preferred environment.

3. **Follow Notebook Instructions:**
   - Run the notebook cells sequentially to load the data, preprocess it, train models, and predict mobile prices.

Feel free to explore, modify, and utilize this project for mobile price prediction tasks. If you have any questions or suggestions, please create an issue in the repository.

Predict mobile prices with confidence using the Mobile Price Prediction Model! 📱💡🚀
