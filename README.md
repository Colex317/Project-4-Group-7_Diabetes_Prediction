![image](https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/assets/148498483/ec423f32-9ff0-446e-b8a2-5d3175e2890d)



# Project 4 Group 7: Diabetes Prediction using Machine Learning

### Contributors:
Marsha Cole, Kim Reiser, Michael Leonard, and Gregory Schreiter

## Background
Diabetes is a chronic condition characterized by the body's inability to produce sufficient insulin or effectively use the insulin it produces. Insulin is crucial for regulating blood glucose levels. Globally, the number of people living with diabetes surged from 108 million in 1980 to 422 million in 2014, with prevalence increasing more rapidly in low- and middle-income countries compared to high-income countries. Diabetes is a leading cause of severe health complications such as blindness, kidney failure, heart attacks, stroke, and lower limb amputations (WHO, 2023). Since there is no permanent cure for diabetes, early detection is vital. Machine learning (ML) algorithms can significantly aid in predicting diabetes, facilitating timely diagnosis and intervention. Therefore, our project focuses on developing and utilizing various ML models to predict the likelihood of diabetes.

<br>

## Purpose/Goal of the project 

**Purpose**

This project aims to develop a predictive model for diabetes diagnosis using a dataset containing patient health metrics. It aims to leverage machine learning techniques to accurately predict the presence of diabetes based on various health indicators, including gender, age, hypertension, heart disease, smoking history, BMI, HbA1c levels, and blood glucose levels.

**Project Goal**

The goal of the project is to create, optimize, and evaluate a machine-learning model that can achieve at least 75% classification accuracy or an R-squared value of 0.80 or higher. The project will involve the following key steps below.

<br>

## Dataset Used 

The Diabetes Prediction dataset by Mohammed Mustafa was retrieved from Kaggle: (https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset). The dataset has 100,000 rows across 9 columns:

- Gender - refers to the biological sex of the individual, which can impact their susceptibility to diabetes.
  
- Age - is an important factor, as diabetes is more commonly diagnosed in older adults. The age range in this dataset is 0 to 80.
  
- Hypertension - is a medical condition in which the blood pressure in the arteries is persistently elevated. It has values of 0 or 1, where 0 indicates no hypertension and 1 means they have hypertension.
  
- Heart disease is a medical condition associated with an increased risk of developing diabetes. It has values of 0 or 1, where 0 indicates no heart disease and 1 means they have heart disease.
  
- Smoking history -  is considered a risk factor for diabetes and can exacerbate the complications associated with it. The dataset has five categories: not current, former, No Info, current, never, and ever.
  
- BMI (Body Mass Index) - is a measure of body fat based on weight and height. Higher BMI values are linked to a higher risk of diabetes. The range of BMI in the dataset is from 10.16 to 71.55.
  
- HbA1c (Hemoglobin A1c) level - is a measure of a person's average blood sugar level over the past 2-3 months. Higher levels indicate a greater risk of developing diabetes.
  
- Blood glucose level - is a measure of a person's average blood sugar level over the past 2-3 months. Higher levels indicate a greater risk of developing diabetes.

- Diabetes - (the target) is a chronic medical disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. It has values of 0 or 1, where 0 indicates no diabetes and 1 means they have diabetes.

<br>

## Steps:

### Data Retrieval and Preprocessing:

1. Obtain the dataset [Diabetes Prediction by Mohammed Mustafa](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) from Kaggle and download it to the local machine.

<img width="1003" alt="image" src="https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/assets/148498483/ca690187-da68-4e67-b747-387a0cb36d75">



----------------------------------------------------------------------------------------------------------------------------------------------------



2. Create a database and table schema ([diabetes_prediction_schema](https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/blob/main/diabetes_prediction_schema.sql)) in PostgreSQL. Then, verify the data was imported correctly.

<img width="1054" alt="image" src="https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/assets/148498483/6d36bfb2-93ea-49c1-84cb-e7ee22737375">

<br>


3. Retrieve the dataset using SQLAlchemy in Python.

4. Clean, normalize, and standardize the data to ensure it is suitable for modeling:
   
   a. Using Label and Ordinal Encoding:   
   - convert gender and smoking_history to numeric format.
   - convert BMI, HbA1c, and blood glucose levels according to ranges.

   b. Using One-Hot Encoding and StandardScalar:
   - One-Hot Encoding to convert the categorical columns (gender and smoking_history).
   - Standardize the numerical columns using StandardScaler (age, BMI, HbA1c, and blood glucose level).

### Model Initialization, Training, and Evaluation:

5. Implement a Python script to initialize, train, and evaluate the machine learning model. For this project, we used multiple models.

6. Select appropriate algorithms and techniques to achieve high predictive accuracy.

### Model Optimization:

7. Perform iterative optimization to enhance the model's performance.

<br>

## Modeling Techniques Utilized

Different models were used to estimate the likelihood of an individual developing diabetes. By analyzing the features (gender, age, hypertension, heart disease, smoking history, BMI, HbA1c levels, and blood glucose levels), the models aid in predicting whether an individual is at risk for diabetes.

1. Logistic Regression
2. Decision Tree
3. Random Forest 
4. Support Vector Machine
5. Deep Learning

<br>


## Logistic Regression

Logistic regression estimates/models the probability of an event occurring based on a given dataset. This dataset contains independent variables, or predictors, and their corresponding dependent variables, or responses. Logistic regression is a widely used statistical method for binary classification problems, making it an ideal choice for diabetes prediction. 

1. [Logistic Regression Model (using the label and ordinal encoding dataset)](https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/blob/main/diabetes_prediction_logistic_regression_label_encoding.ipynb)
2. [Logistic Regression Model (using the One-Hot Encoding and StandardScaler dataset)](https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/blob/main/diabetes_prediction_logistic_regression_one_hot_encoding.ipynb)

<img width="1075" alt="image" src="https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/assets/148498483/75ec8377-1a66-4697-be92-de56b8ecb10a">

<br>


## Decision Tree

Decision tree analysis offers a graphic representation of various alternative solutions that are available to solve a given problem in order to determine the most effective courses of action. They are comprised of nodes (a test on an attribute) and branches (represent potential alternative outcomes). Decision trees are a powerful method for classification problems and offer several advantages that make them suitable for diabetes prediction. They are  simple to interpret, flexible, robust, and explain how different health metrics contribute to the prediction of diabetes.

1. [Decision Tree Model (using the label and ordinal encoding dataset)](https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/blob/main/Diabetes_Prediction_Decisiontree_label_encoding.ipynb)
2. [Decision Tree Model (using the One-Hot Encoding and StandardScaler dataset)](https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/blob/main/Diabetes_Prediction_Decisiontree_one_hot_encoding.ipynb)

<img width="1085" alt="image" src="https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/assets/148498483/cd7cff06-3506-4913-8363-50128828d979">


<br>

## Random Forest

Random forest produces multiple decision trees, randomly choosing features to make decisions when splitting nodes to create each tree. It then takes these randomized observations from each tree and averages them out to build a final model. Building multiple decision trees and merging them together enhances accuracy and produces more stable predictions, making this model ideal for diabetes prediction.

1. [Random Forest Model (using the label and ordinal encoding dataset)](https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/blob/main/diabetes_prediction_random_forest_label_encoding.ipynb)
2. [Random Forest Model (using the One-Hot Encoding and StandardScaler dataset)](https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/blob/main/diabetes_prediction_random_forest_hot_encoding.ipynb)

<img width="1083" alt="image" src="https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/assets/148498483/8c0397dc-6f2a-44dc-82fd-a840fc3bc285">
<br>

## Support Vector Machine
A support vector machine (SVM) is a supervised machine learning algorithm that classifies data by finding an optimal line or hyperplane that maximizes the distance between each class in an N-dimensional space. SVM is a good choice when predicting diabetes as it can handle high-dimensional data efficiently and support different kernel functions, enabling flexibility. The accuracy scores below are from the polynomial kernel output.

1. [Support Vector Machine (using the label and ordinal encoding dataset)](https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/blob/main/Diabetes_prediction_svm_nn_label_encoding.ipynb) (Upper section of this file)
2. [Support Vector Machine (using the One-Hot Encoding and StandardScaler dataset)](https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/blob/main/Diabetes_prediction_svm_nn_one_hot_encoding.ipynb) (Upper section of this file)

<img width="1086" alt="image" src="https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/assets/148498483/effb1fab-15a3-47c6-8445-33563b792b9d">

<br>

## Deep Learning Neural Network

Deep learning neural networks, or artificial neural networks, attempt to mimic the human brain through a combination of data inputs, weights, and biases. These elements work together to accurately recognize, classify, and describe objects within the data. Deep learning neural networks are highly adaptable and can be fine-tuned for specific tasks, such as predicting the onset of diabetes. For this project, following best practice guidelines, the number of nodes in the first hidden layer was chosen to be two times the number of input features within the dataset.

1. [Deep Learning Neural Network (using the label and ordinal encoding dataset)](https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/blob/main/Diabetes_prediction_svm_nn_label_encoding.ipynb) (Middle section of this file)
2. [Deep Learning Neural Network (using the One-Hot EncodindScaler dataset)](https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/blob/main/Diabetes_prediction_svm_nn_one_hot_encoding.ipynb) (Middle section of this file)

![image](https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/assets/148498483/c37626fb-2c9c-46ec-9ce7-5a69844a5c16)


<br>


## Result
Each model used in this project with the diabetes prediction datasets demonstrated meaningful predictive power more significant than 75% classification accuracy:

### First dataset (Label and Ordinal Encoding using Pandas)

| Models                  |   Accuracy   |
|-------------------------|--------------|
| Logistic Regression     |      0.93    |
| Decision Tree           |      0.92    |
| Random Forest           |      0.92    |
| Support Vector Machine  |      0.92    |
| Deep Learning           |      0.92    |
| Deep Learning (Tuned)   |      0.93    |

### Second dataset (OneHotEncoder and StandardScaler)

| Models                  |   Accuracy   |
|-------------------------|--------------|
| Logistic Regression     |      0.96    |
| Decision Tree           |      0.95    |
| Random Forest           |      0.96    |
| Support Vector Machine  |      0.97    |
| Deep Learning           |      0.97    |
| Deep Learning (Tuned)   |      0.97    |

## Additional Models
We also utilized three other machine-learning models in this project to predict diabetes using two different datasets processed with different encoding and scaling techniques. Both datasets and all the models surpassed the performance benchmark, with the second dataset showing slightly higher accuracy for the majority of the models. This demonstrates that the choice of data preprocessing techniques, such as OneHotEncoding and StandardScaler, can have a substantial impact on model performance.

1. Yggdrasil Decision Tree  
2. Gaussian Naive Bayes 
3. XGBoost 

### First dataset (Label and Ordinal Encoding using Pandas)

| Models                  |   Accuracy   |
|-------------------------|--------------|
| Yggdrasil Decision Tree |      0.93    |
| Gaussian Naive Bayes    |      0.67    |
| XGBoost                 |      0.92    |

### Second dataset (OneHotEncoder and StandardScaler)

| Models                  |   Accuracy   |
|-------------------------|--------------|
| Yggdrasil Decision Tree |      0.97    |
| Gaussian Naive Bayes    |      0.64    |
| XGBoost                 |      0.97    |

   1. [Additional Models (using the label and ordinal encoding dataset)](https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/blob/main/Diabetes_prediction_svm_nn_label_encoding.ipynb) (Lower section of this file)
   2. [Additional Models Network (using the One-Hot EncodindScaler dataset)](https://github.com/Colex317/Project-4-Group-7_Diabetes_Prediction/blob/main/Diabetes_prediction_svm_nn_one_hot_encoding.ipynb) (Lower section of this file)

## Conclusion
In this project, we developed predictive models for diabetes diagnosis using a dataset containing patient health metrics. We utilized various models to assess their effectiveness in predicting diabetes using two different datasets processed with different encoding and scaling techniques. The overall performance of the models was greater than 75% predictive accuracy, providing a comprehensive view of the model's effectiveness in predicting diabetes. With 11.6% of the US population affected by diabetes (diagnosed and undiagnosed), the models in this project offer potentially valuable insights in predicting diabetes and facilitating timely intervention and management.

The use of StandardScaler allowed us to gain a 4-5% increase in the accuracy scores compared to the non-scaled data. StandardScaler impacted the importance of age, reducing it from the most important to the third most important feature in the Random Forest model, while increasing the importance of hba1c and blood glucose level predictors. In the scaled dataset, we see accuracies of 95-97%. The simpler models (logistic regression / decision tree) are more interpretable and easy to implement, whereas the deep learning model is more complex and requires advanced tuning techniques to achieve a performance of 97%. The model recommendation for this project is either SVM or Deep Learning which both achieved 97% accuracy. One advantage between the two is that Support Vector Machines are more simple to implement than Deep Learning, so that would be the first choice.


## [Link to the PowerPoint presentation](https://docs.google.com/presentation/d/16Kw_yZAl5BYWi3dwS2XfINaWJo_B7_lPB-taVGtd5q0/edit?usp=sharing)

## References

Centers for Disease Control and Prevention. (2024). Body Mass Index (BMI). Retrieved from https://www.cdc.gov/healthyweight/assessing/bmi/index.html

Centers for Disease Control and Prevention. (n.d.). Testing for Diabetes. Retrieved from https://www.cdc.gov/diabetes/diabetes-testing/?CDC_AAref_Val=https://www.cdc.gov/diabetes/basics/getting-tested.html

Geekforgeeks. (n.d.). Connecting PostgreSQL with SQLAlchemy in Python. Retrieved from https://www.geeksforgeeks.org/connecting-postgresql-with-sqlalchemy-in-python/

Heavy.AI. (2024). Decision Tree Analysis. Retrieved from https://www.heavy.ai/technical-glossary/decision-tree-analysis#:~:text=A%20decision%20tree%20is%20a,the%20best%20courses%20of%20action

Ibm.com. (2023, December 27). What are support vector machines (SVMs)? Retrieved from https://www.ibm.com/topics/support-vector-machine

Ibm.com. (n.d.). What is deep learning? Retrieved from https://www.ibm.com/topics/deep-learning#:~:text=Deep%20learning%20neural%20networks%2C%20or,describe%20objects%20within%20the%20data

Niklas Donges. (2024, March 8). Random Forest: A Complete Guide for Machine Learning. Retrieved from https://builtin.com/data-science/random-forest-algorithm

Samantha Lomuscio. (2022, September 22). Logistic Regression Four Ways with Python. Retrieved from https://library.virginia.edu/data/articles/logistic-regression-four-ways-with-python#:~:text=Logistic%20regression%20is%20a%20predictive,corresponding%20dependent%20variables%2C%20or%20responses

World Health Organization. (2023). Diabetes. Retrieved from https://www.who.int/news-room/fact-sheets/detail/diabetes#:~:text=Diabetes%20is%20a%20chronic%20disease,hormone%20that%20regulates%20blood%20glucose.

Yale Medicine. (n.d.). What do A1C test results mean? Retrieved from https://www.yalemedicine.org/conditions/hemoglobin-a1c-test-hba1c-test-blood-sugar

