# Similarities

This repository explores and recommends the most similar games based on the provided data.

To run the code one must first create an environment such as "conda create --name ds python=3.12 numpy pandas matplotlib seaborn scikit-learn scipy jupyterlab". Running the scripts requires folder "raw_game_data" with ".slot" files to be contained in the working directory. Running "python parse_slot_data.py" will parse the raw data and construct two csv files.

## Data Preprocessing and Modeling

The "numeric_similarity_model.py" uses only numeric values:
- Removes text-based features (name, url, provider, provider_url, review_text, dates...) and categorical features
- Predicts most similar slots for each slot

To run the similarity model:
1. Ensure you have the processed data from parse_slot_data.py
2. Run "python numeric_similarity_model.py" to train and evaluate the model
3. The model will output evaluation metrics and save the trained model for future use


## Ideas for future work
 - using similar games for estimating missing features
 - using categorical features (binary encoding)
 - year of game creation
 - days since last update
 - tf-idf of review_text
 - data split (k-fold, provider, last year, genre)
 - feature encoding (RTP 90%, 95%)
 - encoding LAYOUT, FEATURES, TECHNOLOGY
 - graphs to model related_slot, most popular and connected slots (games that connect clusters)
 