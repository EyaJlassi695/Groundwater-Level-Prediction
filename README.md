# Groundwater-Level-Prediction

The primary objective of this project is to build an AI model capable of predicting the watertable/groundwater levels of French piezometric stations, with a particular emphasis on the summer months. Beyond developing the AI model, the project extends to exploring realistic market applications and assessing its impact in a real-world context.

## Deliverables 📦

### 1. **AI Model and Predictions**  
Develop a robust machine learning model to predict the `piezo_groundwater_level_category`.


## Dataset 📊

The dataset provided for this challenge includes over 3 million rows with 136 columns, divided into training and testing datasets:

- **Training Set (`X_train_Hi5.csv`)**  
  - Contains approximately 2,800,000 rows.  
  - Covers data from 2020 to 2023, excluding the summer months (June, July, August, September) of 2022 and 2023.  
  -The training set is hosted externally due to size constraints. You can download it from [Google Drive](https://drive.google.com/file/d/1J_QyJi7JSlMB2uFVOpk8UB8fsgouApUH/view?usp=drive_link)
  - Contains approximately 600,000 rows.  
  - Includes data for the summer months (June, July, August, September) of 2022 and 2023.

- **Test Submission Format (`y_test_submission_example_Hi5.csv`)**  
  - Follow this example for submitting results to the leaderboard.  
  - Use the `row_index` variable to match predictions with rows.

### Target Variable 📌  
The target variable for prediction is `piezo_groundwater_level_category`.



## Methodology ⚙️

1. **Data Analysis and Preprocessing**  
   - Handle missing data, normalize features, and explore variable importance.
   
2. **Feature Engineering**  
   - Extract meaningful features from weather, hydrology, water withdrawal, and economic data.

3. **Modeling**  
   - Train and compare state-of-the-art models, such as LightGBM and XGBoost, to determine the best performance.

4. **Evaluation**  
   - Use appropriate metrics to evaluate model performance and validate predictions on the test dataset.

## Models and Libraries Used 🤖📚

### Models
This project leverages cutting-edge machine learning models and frameworks to predict groundwater levels with high accuracy:

1. **LightGBM**  
   - A gradient boosting framework based on decision trees.  
   - Known for its speed and efficiency, especially with large datasets.  
   - Supports categorical features natively, reducing preprocessing efforts.  
   - Key Features:
     - Histogram-based learning algorithm for fast training.  
     - Leaf-wise tree growth for higher accuracy.  

2. **XGBoost**  
   - An optimized gradient boosting library designed for performance and flexibility.  
   - Frequently used in data science competitions for its robust performance.  
   - Key Features:
     - Regularization techniques to prevent overfitting.  
     - Efficient handling of missing values.  

3. **AutoGluon.Tabular**  
   - A state-of-the-art AutoML framework for tabular data.  
   - Automates model selection, hyperparameter tuning, and ensemble creation.  
   - Configured to include **XGBoost** and **LightGBM** among its base models for optimal performance.  
   - Key Features:
     - Seamlessly integrates multiple models and ensembles them for robust predictions.  
     - Automatically tunes hyperparameters of models like **XGBoost** and **LightGBM** for best results.

### Optimization Framework

4. **Optuna**  
   - A powerful, lightweight hyperparameter optimization framework.  
   - Enables efficient exploration of parameter spaces to find the best model configurations.  
   - Works in conjunction with models to fine-tune critical hyperparameters like learning rates, number of estimators, and tree depths.  
   - Key Features:
     - Implements advanced algorithms like Tree-structured Parzen Estimators (TPE).  
     - Supports distributed optimization for large-scale experiments.  

### Libraries
The following Python libraries were used to preprocess data, train models, and optimize their performance:

- **pandas**: For data manipulation and preprocessing.  
- **NumPy**: For numerical operations and array handling.  
- **scikit-learn**: For machine learning utilities, including train-test splitting and metric evaluations.  
- **LightGBM**: For building and training the LightGBM model.  
- **XGBoost**: For building and training the XGBoost model.  
- **AutoGluon.Tabular**: For automated machine learning with minimal configuration, including the integration of **XGBoost** and **LightGBM**.  
- **Optuna**: For hyperparameter optimization and performance tuning.  
- **matplotlib**: For creating visualizations to understand data distributions and model performance.  
- **seaborn**: For advanced statistical data visualization.

## Key Features of the Project 🌟

- **Scalability**: The model is designed to handle large datasets with millions of rows efficiently.  
- **Interpretability**: Emphasis on understanding the factors influencing water shortage.  
- **Real-World Impact**: Solutions are developed with practical applications and sustainability goals in mind.


## Instructions 🚀

1. Clone the repository and install necessary dependencies:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   pip install -r requirements.txt
   ```

2. Place the datasets in the appropriate folder structure:
   ```
   data/
   ├── X_train_Hi5.csv
   ├── X_test_Hi5.csv
   └── y_test_submission_example_Hi5.csv
   ```

3. Run the notebook to preprocess data, train the model, and generate predictions.

4. Submit predictions following the format in `y_test_submission_example_Hi5.csv`.

---

## Tools & Technologies 🛠️

- **Programming Language**: Python  
- **Libraries**: LightGBM, XGBoost, pandas, NumPy, scikit-learn, matplotlib  
- **Visualization**: Matplotlib, seaborn  

---

## Results 📈

- Detailed evaluation results will be available in the notebook outputs and the scientific document.  


## License 📜

This project is licensed under the MIT License.


## Acknowledgments 🙏

Thanks to the organizers and contributors for providing the datasets and setting up this meaningful challenge. Special thanks to the hackathon community for their support and feedback.

