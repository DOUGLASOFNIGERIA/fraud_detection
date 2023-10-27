# Fraud Detection
Dataset Source from kaggle: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023
## What's The Problem?
Currency exchange is one of the most common operation in the banks. This operation allows economy to grow rapidly; that's term is simple. 

However, the problem is more complicated than that term, some cashiers fail to register the exchange operation and instead keep the difference for themselves. It is a small operational error but it poses a significant risk to reputational banks, since small schemes happens it will be grow to the big schemes, because small schemes can evolve into more significant frauds in the future. 

Furthermore, these incident are also hard to detect since internal audit teams need to process manually review all documents.

## How to solve it?
One of the best and most common ways to solve this problem is through statistical analysis to explore the data and utilize machine learning for fraud detection. Models commonly used include is: 

- Logistic Regression
- GaussianNB
- SGDClassicifier
- Catboost
- Neural Network

### Exploration Data Analysis
#### Outlier Event
![outlier_graphs](fraud.jpeg)

As evident from the image, each instance of fraud exhibits a higher value than typical outliers, as indicated by the p-value.

#### Q-Q Plot
![quantile_plot](quantil.png)

From the graph, we utilize three parameters for evaluate the data: the Shapiro-Wilk test represented by Wα Anderson-Darling Normality test denoted as Aα and Skewness measured by skew. These three parameters provide insight into the distribution of the graph.

- Shapiro-Wilk test assesses the degree of normality of the data, examining its proximity to a normal distribution.
- Anderson-Darling Normality test measures how far the data deviate from a normal distribution.
- Skewness offers valuable insights into the type of distribution being observed.
