# Credit Card Customer Churn Prediction: Exploratory Data Analysis and LightGBM Modeling

<img src="images/churn.jpg">

## 1. Description

This project involves an in-depth **exploratory data analysis (EDA)** and the development of a predictive machine learning model to estimate the probability of a customer churning from a bank's credit card service.

The core objective is a **supervised classification problem**, where the target variable is **customer attrition** (1 if the customer churned/attrited, 0 otherwise). We utilize the powerful **LightGBM** algorithm to build a high-performance predictive model.

The analysis and modeling were initially conducted and documented within **Jupyter Notebooks**, encompassing:
* **Comprehensive EDA:** To understand the dataset structure, identify key features, analyze customer demographics, and uncover patterns related to attrition.
* **Feature Engineering and Preprocessing:** Preparing the data for machine learning, including handling missing values, encoding categorical variables, and scaling numerical features.
* **Model Training and Optimization:** Training a LightGBM classifier and using techniques like **Optuna** for hyperparameter tuning to maximize predictive performance.
* **Model Interpretation:** Using **SHAP** (SHapley Additive exPlanations) to interpret the model's predictions and understand the key drivers of customer churn.

---

## 2. Technologies and Tools

The following technologies and tools were used for the data analysis and model development:

| Category | Tools/Libraries | Purpose |
| :--- | :--- | :--- |
| **Programming** | **Python** | Core language for development. |
| **Data Analysis** | **Pandas, NumPy** | Data manipulation, cleaning, and numerical operations. |
| **Visualization** | **Matplotlib, Seaborn** | Creating statistical graphics for EDA. |
| **Machine Learning** | **Scikit-Learn** | Data preprocessing, model selection, and evaluation. |
| **Core Algorithm** | **LightGBM** | The primary gradient boosting framework used for classification. |
| **Model Optimization** | **Optuna** | Hyperparameter optimization for the LightGBM model. |
| **Model Interpretation** | **SHAP** | Explaining model outputs and feature importance. |
| **Environment** | **Jupyter Notebook** | Environment used for conducting and documenting EDA and modeling. |
| **Version Control** | **Git, GitHub** | Managing and tracking project changes. |
| **Development** | **Anaconda (Terminal), Visual Studio Code** | Project development and environment management. |

# 3. Business problem and project objective

**3.1 What is the business problem?**
- A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them how likely is a customer to churn so they can proactively go to the customers to provide them better services and turn their decisions in the opposite direction.

**3.2 What is the context?**
- When a bank acquires a customer for its credit card service, three essential Key Performance Indicators (KPIs) to consider include:
    1. Customer Acquisition Cost (CAC): This measures the expenses associated with acquiring each credit card customer, encompassing marketing, sales, and related costs. Lower CAC reflects efficient customer acquisition.
    2. Customer Lifetime Value (CLV): CLV estimates the total revenue the bank can expect to generate from a credit card customer over their relationship. A higher CLV indicates that the customer's value surpasses the acquisition cost, ensuring long-term profitability.
    3. **Churn Rate:** Churn rate is typically expressed as a percentage and represents the number of credit card customers who have left during a specific period divided by the total number of customers at the beginning of that period.
- These KPIs help the bank assess the effectiveness of its strategies in acquiring credit card customers and gauge the potential long-term financial benefit of these acquisitions.
- In order to maximize profitability, the bank aims to minimize CAC and Churn while maximizing CLV.

**3.3 Which are the project objectives?**
1. Identify the factors associated with customer churn.
2. Construct a model capable of accurately predicting the probability of a customer to churn.
3. Offer action plans for the bank to reduce credit card customer churn.

**3.4 Which are the project benefits?**
1. Cost Savings.
2. Improved Customer Retention.
3. Enhanced Customer Experience.
4. Targeted Marketing.
5. Revenue Protection.
- And as a result, the mentioned business problem will be resolved.

**3.5 Conclusion**
- When deploying the model so that the bank can make predictions, the primary objective is to generate probability scores for each customer. This is typically more valuable for businesses when compared to making binary predictions (1/0), as it enables better decision-making and more effective customer retention strategies.
- For instance, predicting the probability of churn provides more actionable insights. Instead of simply determining whether a customer will churn or not, you gain an understanding of how likely it is to happen. This information enables the bank to allocate its efforts and resources more effectively. For example, it can concentrate its retention efforts on customers with a high probability of churning.

# 4. Main business insights
1. The bank presents a high churn rate. Approximately 16% of the customers churned. This motivates our project.

<img src="images/target.png">

2. Customers who churn tend to have lower credit limits, revolving balances, transaction amounts, transaction counts, changes in transaction counts, and average utilization ratios. This makes total sense because when a client is about to discontinue the service, he will make fewer transactions, resulting in lower values of these attributes.

<img src="images/trans_ct_churn.png">

<img src="images/trans_amt_churn.png">

3. The bank must be alert to the distribution of the average utilization ratio. Being right-skewed means that most customers have a low utilization of the service. Particularly, about 25% of them present zero utilization ratio, and this is where most of the churners are located. It is a good indicator for churn event. Therefore, the company must plan strategies to increase utilization and, consequently, revenue.

<img src="images/utilization_ratio_churn.png">

5. 75% of the customers contacted the bank at least 2 times in the last 12 months. The bank must pay attention to this and discover the reasons why customers are calling. Are they dissatisfied with the credit card service? If so, what specifically is causing their 
dissatisfaction? Also, when a client makes more than 5 contacts in a year, he will probably skip the service.

<img src="images/contacts_ct.png">

6. 90% of the customers were inactive from 1 to 3 months in the last 12 months. The bank must plan strategies not only to increase the utilization ratio, but also to avoid customer inactivity.

<img src="images/months_inactive.png">

7. The churn rate consistently increases monotonically as the number of contacts with the bank rises. Especially, all customers who made six contacts churned.

<img src="images/churn_contacts.png">

8. Customers with doctorate educational level present the highest churn rate among the other levels.

<img src="images/churn_education.png">

# 5. Modelling
0. I created two preprocessors. One for testing linear models, in which I applied one-hot encoding to categorical variables (linear models can leverage this encoding technique to preserve linearity assumption) and standard scaling to numerical ones (linear models are sensitive to scale because they used distance-based calculations or optimization algorithms such as gradient descent). For the tree-based models preprocessor, I applied ordinal encoding to ordinal categoric features in order to preserve this characteristic and target encoding to the reamining one. This is because one-hot encoding can be harmful for tree models, due to sparse representation and increased dimensionality. Finally, these algorithms don't require scaling, thus, numerical variables were included without any transformation. An important observation is that I performed feature engineering before preprocessing, creating a lot of relevant attributes with respect to churn discrimination. Some include average transaction amount, proportion of months inactive with respect to customer's tenure, and total spending. All these preprocessing steps were divided into transformer classes such that I could integrate everything into a sklearn's pipeline, facilitating the deployment in production environment.
1. Then, I chose a set of linear a tree-based models to comparison through stratified k-fold cross validation, analyzing the ROC-AUC score. Accuracy is not a good metric because the target is imbalanced. The objective was to select the best model for the next steps.
2. As LightGBM had the highest average validation score, I chose it for feature selection, hyperparameter tuning and final model evaluation. Although it is overfitted, it also has an outstanding performance on validation data. However, this is not due to data leakage or modeling problems, but the quality of this dataset. The set of independent variables we have clearly separates churners and non-churners.

<img src="images/treemodels_cv.png">

3. I applied feature selection with the Recursive Feature Elimination (RFE) technique, which recursively select a subset of features with highest feature importances until the desired number of features is reached. As a result, 25 of the 40 variables were selected, including a lot of variables created in feature engineering step, ilustrating the importance of this task.
4. I tuned LightGBM model with Bayesian Search because it uses probabilistic models to intelligently explore the hyperparameter space, balancing exploration and exploitation. An important point here was to define a class_weight hyperparameter, such that the estimator was able to better learn the patterns in minority target class (churn customers).
5. The final LigthGBM performance was excellent. A 0.89 recall indicates that the model correctly identifies 89% of the churners. In practical terms, looking at the confusion matrix, it has accurately predicted 290 out of 325 churners. Furthermore, a 0.90 precision indicates that, out of all customers predicted as churn, 90% of them are actually churners. In practical terms, considering the confusion matrix, out of 324 customers predicted as churn, 297 of them are indeed churners. Finally, the probability scores follow an order, with churners being assigned to higher probabilities, demonstrating that the model outcomes are reliable.

<img src="images/confusion_matrix.png">

| Model    | Accuracy | Precision | Recall   | F1-Score | ROC-AUC  | KS       | Gini     | PR-AUC   | Brier    |
|----------|----------|-----------|----------|----------|----------|----------|----------|----------|----------|
| LightGBM | 0.965943 | 0.895062  | 0.892308 | 0.893683 | 0.991279 | 0.898897 | 0.982559 | 0.964932 | 0.025852 |

6. Although one characteristic of ensemble models like LightGBM is the lack of interpretability, it was possible to interpret and confirm that the estimator results make sense and reinforce the insights found in the EDA (Exploratory Data Analysis) step by examining shap values. It is noticeable that lower values of transaction counts in the last 12 months have a positive impact in the log-odds of churning, and, consequently, in the probability of churning, while higher values have a negative impact. Surprisingly, higher values of average transaction amount tend to reflect a positive impact in the log-odds of churning, and, consequently, in the probability of churning, while lower values tend to have a negative impact.

<img src="images/shap_beeswarm.png">

# 6. Financial results
- I have estimated a baseline financial outcome to assess whether the project is worthwhile. Although I did not have access to specific data to obtain the exact financial gain, a common revenue source for credit card companies is charging fees on the outstanding balance, known as the total revolving balance, which cardholders carry from month to month. Thus, I assumed the bank's fee is 18% (a common value) and performed the estimation based on it. The calculation was derived from the difference between the true positive gain (a 10% fee charged on outstanding balances as a retention strategy), the cost of retaining false positives (an 8% discount given on outstanding balances), and the cost of false negatives churning (the entire 18% fee on outstanding balances that would have been received). Consequently, it was possible to achieve an estimated gain of $171,477, excellent! 
- Finally, the results are excellent and the project objective was achieved. Now, the bank can access customer churn probabilities, facilitating informed decisions. This empowers strategic retention efforts and enhances decision-making. Thus, the business problem is solved.

# 7. Dataset link
The dataset was collected from kaggle.

Link: https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers?sort=votes

# 9. Contact me
Linkedin: http://www.linkedin.com/in/rathidaksh

Gmail: dakshrathi9125@gmail.com
