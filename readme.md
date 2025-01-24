# Event detection in Tweets Related to Football Matches

This project aims to analyze tweets related to football matches to detect specific events such as goals, red cards, penalties, etc. The project uses natural language processing (NLP) techniques and machine learning models to classify the tweets.

## Project Structure

- `code/` : Contains Python notebooks and scripts for analysis and modeling.

  - `Best_Model_XGBoost.ipynb` : Notebook containing the final XGBoost model.
  - `MLP_Occurences_Matrix.ipynb` : Notebook for creating occurrence matrices and training an MLP model.
  - `Dataset.py` : Script for loading and cleaning data.
  - `Experimentations.ipynb` : Notebook for experimenting with different models and techniques.

- `data/` : Contains CSV files of training and evaluation tweets.

  - `train_tweets/` : Training tweets.
  - `eval_tweets/` : Evaluation tweets.

- `Report.pdf` : Report of the project. (not up to date)

## Prerequisites

- Python 3.12.5
- Python libraries: numpy, pandas, scikit-learn, nltk, gensim, xgboost, keras, matplotlib

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/TristanDonze/Data-Challenge-Football-Event-Classifier
   cd Data-Challenge-Football-Event-Classifier
   ```
2. Install the required libraries:
   ```sh
    pip install -r requirements.txt
   ```
3. Run the notebook of your choice in the `code/` directory.
