
## Introduction

Streaming platforms rely on their algorithms to compete for listeners' attention. Understanding skip patterns can reduce listener mismatches generated from the algorithm. This has direct implications to the streaming economy, as it informs playlist creation, user retention and the recommendation of content.

Spotify's algorithm uses a hybrid mix of collaborative filtering and content-based filtering to curate user's listening experience. They'll present content that users with similar tastes consume, and examine a user’s own granular preferences based on audio features. A feature that is not frequently explored is the semantic content within the songs themselves.

For this study, I will develop and evaluate thee prediction models - Elastic Net Logistic Regression, XGBoost, and LSTM - to determine the effect that semantic content has on skip prediction accuracy. I will examine the features that best predict skip patterns, as well as how the order of tracks affects this. Below are the results.
 
## Methodology

**Data Acquisition**

Using the "Download your data" feature in Spotify, I requested and aquired my listening data history, spanning from 2013 to 2025. 

To increase the complexity of the dataset, I gathered the audio features (for ex. energy, key, tempo) of the songs in my history from kaggle datasets. In addition, I web scraped the lyrics to each song from the AZlyrics website.

**Model Selection**

The dataset I'm using high-dimensional (+100 columns when inlcuding TDIF toakens), and contains varied categorical and numerical data types. Both Elastic Net Logistic Regression and XGBoost were selected because they can both address this. LSTM will be used to look at temporal 

- **Elastic Net:** A linear model that handles feature selection through it's L1 and L2 regularization. Any irrelevant features will have their coefficient reduced or remove altogether. A key advantage is the interpretability of the coefficients, which reveals the direction of the relationship between the features and the likelihood of a skip event.  

- **XGBoost:** features an ensemble of decision trees that optimizes through gradient boosting. It is effective finding non-linear relationships between different data types. It also has built-in handling of class imbalance through scale_pos_weight.

- **LSTM:** Long Short Term Memory Networks (LSTM) is a recurrent neural network model that will look at sequences of songs within a session and find patterns leading to a skip. It excels at modeling temporal events.

**Lyrics vs. No Lyrics Model Variants**

Each model (Elastic Net, XGBoost, and LSTM) was trained twice with distinct feature set to compare the performance changes with and without the lyrical features.

- **With Lyrics:** Audio features + top 100 lyric tokens + temporal/genre features
- **Without lyrics:** Audio features + temporarl/genre features only

Audio features include information on genre, wether track is explicit, tempo, energy, key, mode, and time signature. 

## Data Overview

[Enter Picture]

The listening history dataset contains over 174,400+ rows and 30 columns of data. Our target variable is “skipped”, which indicates if a track was skipped. A “True” value indicates a skip event, while “False” indicates a listening event.

## Data Exploration

[Enter Picture]

There is a significant class imbalance, which can pose problems. If this isn’t addressed, the model will be better at predicting listens than skips. I’ll need to use a custom weight to penalize the model for misclassifying a skip. This will cause the model to work harder to learn patterns that separate skips from listens.

[Enter Picture[

Both the number of songs played and the number of skips significantly increase after morning time. This suggests possible listening fatigue or changes in music preferences throughout the day.

I also noticed that I’m more satisfied with my listening choices in the morning (30.4%), while I’m most selective at night (38.3%)

[Enter Picture]

There is a large representation of Urbano Latino, Trap Latino, and Reggaeton on both listens and skips. They are played more often than skipped.

For genres like Pop, R&B, Rap, and Hip Hop, their skip rates are higher than their listen rate, showing greater selectivity.

[Enter Picture]

Popularity has a weak positive correlation (+0.17) with skipping. Which means that this listening profile is one that leans towards deeper cuts or less popular tracks. It is the strongest correlation. Overall however, all correlations are weak.

Looking at correlations among the features, Energy and Loudness have a strong positive correlation (+0.70) - since they measure similar aspects. As Energy goes up, so does Loudness. Valence and Danceability also have a positive relationship (+0.32). Energy and Acousticness have a moderate negative relationship (-0.45).

## Preprocessing 

- **Missing Information:** Given the volume of multi-lingual tracks (English, Spanish, Portuguese) and instrumental tracks, it was difficult to find the audio features, genres, and lyrics to all the songs in the dataset. Due to sparsity of information available, songs that were missing lyrics, or any audio feature were excluded from the training set. This filter process would result in a 58% percent reduction. 

- **Clean Text:**
- **Scraping:**
- **NLP:**
 - ggfgf
 - gfgfgf
