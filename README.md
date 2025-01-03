# Breast-Cancer-Prediction-Analysis

The primary goal is to analyze breast cell nuclei to aid in breast cancer diagnosis. The aim was to build an accurate machine learning model to classify breast cell nuclei as 'malignant' or 'benign' based on real-valued features through identifing the most effective classification algorithm. It also aimed to uncover hidden patterns within cell nucleus characteristics to reveal possible variations in cell nucleus characteristics relevant for diagnosis and treatment.

# Dataset
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Data Set. 

Relevant Information:

- Source: Kaggle - /kaggle/input/breast-cancer/breast-cancer-wisconsin-data_data.csv

- Description: Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

- Number of Instances: 569

- Number of Attributes: 32 (ID, diagnosis, 30 real-valued input features)

- Diagnosis: M (malignant), B (benign)

# Requirements

To run this project, you will need to install the following Python libraries:

- numpy: For numerical operations and handling arrays.

- pandas: For data manipulation and analysis.

- os: For interacting with the operating system.

- seaborn: For statistical data visualization.

- matplotlib: For creating plots and visualizations.

- scikit-learn: For machine learning algorithms, data splitting, and metrics.

- statsmodels: For statistical modeling.

You can install the required libraries using pip:

  pip install numpy pandas seaborn matplotlib scikit-learn statsmodels


# How to Run the Code

1. Clone or download this repository.

2. Ensure the dataset file breastcancerdata.csv is in the same directory as the code.

3. Run the Jupyter Notebook or Python script for analysis:

- jupyter notebook breast_cancer_analysis.ipynb
or
- python breast_cancer_analysis.py

# Results 
This analysis identified key cell nucleus features including size, shape, texture, compactness, and concavity, as strongly associated with tumor diagnosis, with higher values indicating malignancy. Principal Component Analysis (PCA) successfully reduced the dataset’s dimensionality, retaining 100% of the variance across five principal components. Four classification models were developed and evaluated: Logistic Regression, Decision Trees, K-Nearest Neighbors (KNN), and Random Forest. All models achieved a high accuracy of 96% across metrics (accuracy, precision, recall, and F1 score), however, Logistic Regression and Random Forest showed slightly better balance in reducing false negatives, which is crucial for early detection and treatment.

