# Marketplace-price-prediction
At a large scale, determining product pricing becomes increasingly challenging due to the vast number of products sold online. Different product categories, such as clothing and electronics, exhibit unique pricing trends. Clothing prices are heavily influenced by seasonal trends and brand names, while electronics prices fluctuate based on product specifications.

Mercari, Japan's largest community-powered shopping app, is well aware of this complexity. They aim to provide pricing suggestions to sellers on their marketplace. However, this task is difficult as Mercari's sellers have the freedom to list almost anything or any combination of items, making it a complex and diverse marketplace.

Based on the product listing dataset provided by Mercari, I worked on two different projects.
- **Project #1**: Build a predictive tool to estimate the product listing price using regression modeling
- **Project #2**: Perform Topic modeling to identify 10 frequently appeared topics on the description section 
  
GOAL OF the PROJECTS: 
- Use Regression models to predict reasonable Selling Prices based on features of categories on Mercari, an online marketplace.
- Identify clusters of frequently available topics in the text data of product description section by rigorous data cleaning and preprocessing.

## Project 1: Predictive modeling

## Project 2: Topic modeling and clustering
In this project, I performed Topic Modeling to identify 10 frequently appearing topics on the 'item_description' column of the Mercari dataset. I applied text preprocessing involving 

Before starting the MLP process, I visualized the number of the words in 'item_description' dataset:

![dist_word_count] (https://github.com/amytakeuchi/Mmarketplace-Price-Prediction/blob/images/dist_word_count.png)
