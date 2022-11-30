# Sentiment Based Product Recommendation

## Problem Statement

The e-commerce business is quite popular today. Here, you do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm and Snapdeal.

Suppose you are working as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.

With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.

As a senior ML Engineer, you are asked to build a model that will improve the recommendations given to the users given their past reviews and ratings. 

## Solution

* [github link](https://github.com/Vinay26k/sentiment-based-recommendation-system)
* [Heroku App](https://sntmnt-based-product-recommend.herokuapp.com/)

### Built with following requirements

```sh
Flask==2.0.2
nltk==3.6.7
numpy==1.22.2
pandas==1.4.3
gunicorn==20.1.0
scikit-learn==1.0.2
xgboost==1.5.0
matplotlib==3.5.1
```

### Steps Covered

- Data sourcing and sentiment analysis
- Building a recommendation system
- Improving the recommendations using the sentiment analysis model
- Deploying the end-to-end project with a user interface


### Directories and Files
- `data` : contains data and attributes description
- `models` : contains all the necessary pickle files generated through out the process
- `app.py, model.py, templates/` : flask app related files
- `Procfile` : heroku deployment specific file
- `SentimentBasedRecommendationSystem.ipynb`: end-to-end notebook
- `notebooks` : some intermediate notebooks generated throughout the process  

### Solution Approach

* EDA
* Data Cleaning
* Text processing
* Feature Extraction
* Training a text classification model
  * handle class imbalance
  * perform hyper parameter tuning
  * train on different models
    * Logistic
    * Random Forest
    * XGBoost
    * Naive Bayes
*  Building recommendation system
   *  User-based
   *  Item-based
*  Improving recommendation using sentiment classification model
*  Flask App with basic UI development
*  End-to-end deployment to heroku as flask application