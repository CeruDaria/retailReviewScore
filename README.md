# Review Score Prediction
Predicting the review score of products for an e-retailer. The project is done in Databricks.

## Context
BLU is a French e-commerce player offering B2C and B2B customers a broad range of products across more than 100 product categories ranging from kitchen utensils to computer games. They have been active for 5 years and achieved a nice growth of 12% in 2020, 9% in 2021, and 7% in 2022. To get more insights how the company is performing compared to its competitors, the marketing team ordered a report from a consumer insights agency comparing BLU to its two main competitors Amazon.fr and Cdiscount.fr. This revealed that BLU is able to acquire a larger share of new customers compared to the other two (7.8% versus 2.8% for Amazon and 3.4% for Cdiscount), but is underperforming in repeat business from existing customers (6.2% vs. 19.6% for Amazon and 14.4% for Cdiscount). As a next step, BLU’s marketing team wants to have a deeper understanding of its customer base and use predictive modeling to inform its business decisions.

## Data Summary
### Raw Data
Raw data is organized into 2 groups: training and test data. The goal of the project is to build a machine learning model that predicts on the test data and returns the highest accuracy.

Data tables include: orders, order_items, order_payments, order_reviews and products. The **order_reviews** table only exists for the training data as this table contains the target variable to be predicted using the model.

### Data preprocessing & Feature engineering
Standard data cleaning tasks were preformed.

Features were engineered and grouped into 2 categories: customer-journey-related and product-related. Examples of customer-journey variables: delivery day in days, delivery time, delivery status. Examples of product variables: number of items in one order, total amount paid, product categories, description length, etc.

The dataset already contain a product category variable but there were more than 70 categories. To reduce the number of categories, a new variable was created by grouping category names together based on cosine similarity between phrases and words of the original category names using K-means clustering.

## Modeling
2 models 
