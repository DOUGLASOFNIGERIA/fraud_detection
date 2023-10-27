# Fraud Detection
Dataset Source from kaggle: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023
## What's The Problem?
Currency exchange is one of the most common operation in the banks. This operation allows economy to grow rapidly; thats term is simple. 

However, the problem is more complicated than that term, some cashiers fail to register the exchange operation and instead keep the difference for themselves. It is a small operational error but it poses a significant risk to reputational banks, since small schemes happens it will be grow to the big schemes, because small schemes can evolve into more significant frauds in the future. 

Furthermore, these incident are also hard to detect since internal audit teams need to process manually review all documents

## How to solve it?
One of the best and most common ways to solve this problem is through statistical analysis to explore the data and utilize machine learning for fraud detection. Models commonly used include is: 
- Logistic Regression
- GaussianNB
- SGDClassicifier
- Catboost
- Neural Network

### Exploration Data Analysis
![outlier_graphs](fraud.jpeg)
As you can see from that image, each fraud have higher value than normal outlier that is represented by p-value
