# Customer Churn Prediction in the Telecommunications Industry using Decision Trees

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%2CSklearn%2CMatplotlib%2CSeaborn-brightgreen.svg)

## Overview

This project focuses on developing a predictive model to identify customers at high risk of churn within the telecommunications industry. By leveraging historical customer data and various service and demographic attributes, the goal is to build an accurate and reliable churn prediction system. The project utilizes the Decision Tree Classifier algorithm from scikit-learn to achieve this objective.

## Dataset

The project utilizes the publicly available "Telco Customer Churn" dataset, which contains information about telecom customers, including their demographics, services used, account details, and whether they churned or not.

## Key Features

* **Data Exploration and Preprocessing:** Comprehensive analysis of the dataset, including handling missing values and visualizing key features.
* **Feature Engineering:** Transformation of categorical variables into numerical representations using Label Encoding to make them suitable for the machine learning model.
* **Robust Model with Hyperparameter Tuning:** Implementation of the Decision Tree Classifier, optimized using **GridSearchCV** to find the best combination of hyperparameters for improved performance.
* **Performance Evaluation:** Thorough evaluation of the model's predictive accuracy using metrics such as accuracy, precision, recall, and the classification report.
* **Data Visualization:** Creation of insightful visualizations to understand feature distributions and their relationship with customer churn.

## Technologies Used

* **Python:** Programming language used for the entire project.
* **Pandas:** For data manipulation and analysis.
* **Scikit-learn (sklearn):** For data splitting, preprocessing (Label Encoding, StandardScaler), model implementation (Decision Tree Classifier), and evaluation metrics.
* **Matplotlib and Seaborn:** For creating insightful data visualizations.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    https://github.com/Abhinav-Marlingaplar/Customer-Churn-Decision-Tree.git
    cd Churn-Prediction-Decision-Tree
    ```

2.  **Install required libraries manually:**

    This project relies on the following Python libraries. Please ensure you have them installed in your environment. You can install them using pip:

    ```bash
    pip install pandas scikit-learn matplotlib seaborn jupyter
    ```

3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

4.  **Open the notebook:**
    * Jupyter Notebook will open in your default web browser. You will see a list of files and folders.
    * Navigate to the directory containing the notebook and open the file named `Customer_Churn.ipynb`.

## Usage

The Jupyter Notebook `Customer_Churn.ipynb` contains the complete end-to-end workflow of the project, including data loading, preprocessing, model training, and evaluation. Simply run the cells in the notebook sequentially to reproduce the results.

## Results

The tuned Decision Tree Classifier model achieved an overall accuracy of approximately **0.78** on the test dataset.This indicates that while the model demonstrates good performance in predicting customers who will not churn (class 0) and the ones that will churn(class 1).


## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Author

Abhinav Marlingaplar
