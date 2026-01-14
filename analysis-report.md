
## Introduction

Streaming platforms rely on their algorithms to compete for listeners' attention. Understanding skip patterns can reduce listener mismatches generated from the algorithm. This has direct implications to the streaming economy, as it informs playlist creation, user retention and the recommendation of content.

Spotify's algorithm uses a hybrid mix of collaborative filtering and content-based filtering to curate user's listening experience. They'll present content that users with similar tastes consume, and examine a user’s own granular preferences based on audio features. A feature that is not frequently explored is the semantic content within the songs themselves.

For this study, I will develop and evaluate thee prediction models - Elastic Net Logistic Regression, XGBoost, and LSTM - to determine the effect that semantic content has on skip prediction accuracy. I will examine the features that best predict skip patterns, as well as how the order of tracks affects this. Below are the results.
 
## Methodology

The datasets are taken from the Kaggle Instacart Market Basket Analysis competition. The challenge asked users to predict which previously purchased product will be purchased next by a user. For this analysis, I will be using the same datasets to perform customer segmentation.

## Variables

The datasets contain customer order information. There are two challenges at first glance, there are multiple datasets that contain variables of interest and will need to be joined. In addition, There are more than 150 columns, which can cause issues like overfitting. We will need to perform multiple joins and dimension reduction through the PCA technique. Below are the datasets used and their content.

- **Aisles:** Has aisle names and their corresponding unique id.
- **All Order Products:** Has order id and product id for each purchase along with add to cart order of product.
- **Departments:** Has department name and their corresponding department id.
- **Orders:** Has information on order including the user who placed the order, day of the week, hour order was placed, and days since last purchase.
- **Products:** Has information on products including the name, corres
- **User Features:** Has information on each user's purchasing history on items and the days they've made purchased on.

## Limitations
