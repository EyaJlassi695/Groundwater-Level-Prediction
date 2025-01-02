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

- **Testing Set (`X_test_Hi5.csv`)**  
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


## Key Features of the Project 🌟

- **Scalability**: The model is designed to handle large datasets with millions of rows efficiently.  
- **Interpretability**: Emphasis on understanding the factors influencing water shortage.  
- **Real-World Impact**: Solutions are developed with practical applications and sustainability goals in mind.

---

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

---

## Acknowledgments 🙏

Thanks to the organizers and contributors for providing the datasets and setting up this meaningful challenge. Special thanks to the hackathon community for their support and feedback.

