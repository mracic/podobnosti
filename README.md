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

## API Service

A FastAPI service that provides game similarity recommendations through a REST API.

### Prerequisites

Before running the API service, ensure you have:
1. Run the model training script (`python numeric_similarity_model.py`) to generate the trained model file (`api_service/model.pkl`)

### Setup and Running

1. Build the Docker image from the project root directory:
```bash
docker build -t game-similarity-api -f api_service/Dockerfile .
```

2. Run the container:
```bash
docker run -p 8000:8000 game-similarity-api
```

The API will be available at http://localhost:8000

### API Endpoint

#### GET /predlagaj/{game_id}

Get similar games for a given game ID.

Parameters:
- `game_id`: ID of the game to find similar games for
- `predlogi`: Number of similar games to return (default: 8)

Example:
```bash
curl "http://localhost:8000/predlagaj/123?predlogi=5"
```

Response:
```json
{
    "game_id": 123,
    "similar_games": [
        {"slot_id": 456, "distance": 0.123},
        {"slot_id": 789, "distance": 0.234},
        ...
    ]
}
```

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
 