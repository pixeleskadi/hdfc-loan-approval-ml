HDFC Loan Approval Prediction System
Overview
------
This project implements an end-to-end machine learning pipeline for predicting loan approval decisions. It simulates a real-world banking workflow including data preprocessing, model training, evaluation, and inference.
The system is designed with separation between training and prediction modules, along with saved preprocessing artifacts for consistent deployment.

------
Problem Statement
------
Loan approval decisions depend on multiple applicant features such as income, credit history, employment status, and financial background.
This project builds a predictive model to determine whether a loan application should be approved or rejected based on historical data.


------
Tech Stack
------
Python
Pandas
NumPy
Scikit-learn
Pickle (model persistence)


------
Features Implemented
------
Data cleaning & preprocessing
Missing value handling (Imputer)
Feature scaling (StandardScaler)
Model training
Model evaluation
Model persistence (.pkl files)
Separate prediction pipeline
Logging & dataset update scripts

Model Accuracy:
97.66 %

Confusion Matrix:
[[314   9]
 [ 11 521]]

 Classification Report:
Class	Precision	Recall	F1-Score	Support
0	       0.97	   0.97	    0.97	    323
1      	 0.98	   0.98	    0.98	    532
Total Samples: 855

------
How to Run
------
 Install Dependencies
pip install -r requirements.txt

 Train Model
python hdfc_train.py

 Run Prediction
python hdfc_predict.py

------
 Key Concepts Demonstrated
------
End-to-end ML workflow
Data preprocessing pipeline
Model persistence using Pickle
Separation of training and inference logic
Structured project organization
------
 Future Improvements
------
Add cross-validation
Hyperparameter tuning
Feature importance analysis
Convert into API using Flask/FastAPI
Deploy model on cloud

------
 Author
------
Aditya Bhardwaj
Engineering Student | Python & Automation Enthusiast
