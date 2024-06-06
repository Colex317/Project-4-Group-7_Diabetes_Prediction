<img src="images/Header Pics/Cover Pic 1.png" width="817" height="456">


# Project 4 Group 7: Diabetes Prediction

## Purpose/Goal of the project 

### Obtain, clean, and transform the dataset:

1. Obtain the dataset [Diabetes Prediction by Mohammed Mustafa](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) from Kaggle and download it to the local machine

2. Create a database and table schema in PostgreSQL. Then, verify the data was imported correctly.

<img width="1054" alt="image" src="https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/assets/148498483/6d36bfb2-93ea-49c1-84cb-e7ee22737375">

3. Retrieve the dataset using SQLAlchemy in Python

4. Clean and transform the dataset:
   
   a. Using Label and Ordinal Encoding:   
   - convert gender and smoking_history to numeric boolean values.
   - convert BMI, HbA1c, and blood glucose levels according to ranges.

   b. Using One-Hot Encoding and StandardScalar:
   - One-Hot Encoding to convert the categorical columns (gender and smoking_history)
   - Standardize the numerical columns using StandardScaler (age, BMI, HbA1c, and blood glucose level)



. Predictive modeling in Scikit-learn

. Binary classification on Diabetes dataset using multiple predictive modeling techniques

### Modeling Techniques Utilized

1. Logistic Regression
2. Decision Tree
3. Random Forest 
4. Support Vector Machine
5. Deep Learning



## Logistic Regression
### Model Description
<br>



## Decision Tree
### Model Description
<br>
## Random Forest
### Model Description
<br>



## Support Vector Machine
### Model Description
<br>



## Deep Learning Neural Network
### Model Description
<br>




# Result - First dataset (Ordinal and Label Encoding using Pandas)

| Models                  |   Accuracy   |
|-------------------------|--------------|
| Logistic Regression     |      0.93    |
| Decision Tree           |      0.92    |
| Random Forest           |      0.92    |
| Support Vector Machine  |      0.92    |
| Deep Learning           |      0.93    |
| Deep Learning (Tuned)   |      0.93    |

# Result - Second dataset (OneHotEncoder and StandardScaler)

| Models                  |   Accuracy   |
|-------------------------|--------------|
| Logistic Regression     |      0.96    |
| Decision Tree           |      0.95    |
| Random Forest           |      0.96    |
| Support Vector Machine  |      0.97    |
| Deep Learning           |      0.97    |
| Deep Learning (Tuned)   |      0.97    |


## References

Centers for Disease Control and Prevention. (2024). Body Mass Index (BMI). Retrieved from https://www.cdc.gov/healthyweight/assessing/bmi/index.html

Centers for Disease Control and Prevention. (n.d.). Testing for Diabetes. Retrieved from https://www.cdc.gov/diabetes/diabetes-testing/?CDC_AAref_Val=https://www.cdc.gov/diabetes/basics/getting-tested.html

Geekforgeeks. (n.d.). Connecting PostgreSQL with SQLAlchemy in Python. Retrieved from https://www.geeksforgeeks.org/connecting-postgresql-with-sqlalchemy-in-python/

Yale Medicine. (n.d.). What do A1C test results mean? Retrieved from https://www.yalemedicine.org/conditions/hemoglobin-a1c-test-hba1c-test-blood-sugar
