## Skip Prediction Analysis

This project analyzes whether song lyrics have predictive power in determining if a listener will skip or complete a track. Using my listening data from Spotify, machine learning models were built and compared with two distinct features: One that uses audio characteristics alone (tempo, valence, energy, etc. and another containing and audio and lyric data.

For the full report [click here](https://github.com/curiostegui/Instacart-customer-segmentation-analysis/blob/main/analysis-report.md) 

To access the Tableau dashboard [click here](https://public.tableau.com/views/Instacart_Dashboard/ExecutiveSummary?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link) 

## Objective

This project analyzes whether song lyrics have predictive power in determining if a listener will skip or complete a track. Using my listening data from Spotify, machine learning models were built and compared with two distinct features: One that uses audio characteristics alone (tempo, valence, energy, etc. and another containing and audio and lyric data.

## Approach

- Combined a personal listening history export of +150,000 rows, multiple Kaggle datasets with audio metadata (tempo, valence, energy, etc.), and lyrics scraped from AZLyrics using a custom web scraper.
- Manually built artist-to-genre dictionaries to fill in missing genre values using domain knowledge, handling 200+ rows of missing data.
- Feature engineered different variables, such as the binary classification label (tracks skipped vs. completed), and extracted the hour of day, day of week, and month from the timestamp data.
- Process lyrics using Natural Language Processing (NLP). Applied TF-IDF with stop word removal to convert raw lyrics into numeric features for modeling.
- Used StandardScaler to normalize all numerical audio features before modeling.
- Trained and compared six machine learning models. Built Elastic Net, XGBoost, and LSTM models with two feature sets: audio features alone vs audio features + lyrics.


## Key Insights

- Uncovered key behavior patterns driving skip behavior. This includes the role of valence (emotional tone) and genre throughout a listening session, as well as temporal trends on when skips are more likely to occur.
- Observed that lyrics in most models had a negative impact, with only the Elastic Net model showing improvements in accuracy (+2.67%). 
- Developed general streaming model recommendations for various audiences: streaming platforms, record labels, users, and data teams.


 ## Impact
 Analyzed my personal skip behavior and delivered general insights on how to keep users engaged on streaming platforms, and how data teams can better capture listeners’ interests through the algorithm.
 
 #### Keywords: Analyzed my personal skip behavior and delivered general insights on how to keep users engaged on streaming platforms, and how data teams can better capture listeners’ interests through the algorithm.

 
