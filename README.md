# 🎵 Spotify Genre Prediction

This project provides an interactive application for predicting the genre of songs using machine learning models and Spotify data. Built with **Streamlit**, the app enables users to select songs, predict their genres, and explore detailed information, including links to Spotify tracks.

## 📜 Features

- **Interactive Song Selection**: Choose songs from a predefined dataset.
- **Machine Learning Prediction**: Utilizes ensemble classifiers (Random Forest, XGBoost, CatBoost, and LightGBM) for accurate genre prediction.
- **Spotify Integration**: Fetch Spotify URLs and preview songs.
- **Visual Insights**: Displays comprehensive details about the track, including predicted and original genres, key signature, and tempo.
- **Genre Mapping**: Includes primary and sub-genres for in-depth analysis.

## 🛠️ Components

### Scripts Overview

1. **`Genre_Prediction_Model_Generator_Script.py`**:
   - Handles data loading, preprocessing, and model training.
   - Implements ensemble models with optimized hyperparameters.
   - Saves trained models, scalers, and encoders.

   **Important:** This script must be executed first to generate the model file required by the Streamlit app.

2. **`genre_prediction.py`**:
   - Defines genre prediction logic.
   - Encodes features, scales data, and integrates Spotify data.
   - Provides utility functions for feature transformation.

3. **`main.py`**:
   - Hosts the Streamlit app.
   - Enables song selection, background music, and result display.

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/spotify-genre-prediction.git
   cd spotify-genre-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Spotify API credentials:
   - Create an app on the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/).
   - Add your `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` as environment variables.

4. Launch the app:
   ```bash
   streamlit run main.py
   ```

## 🗂️ Data Sources

- **Dataset**: A cleaned and preprocessed version of Spotify's public data containing features like danceability, energy, and tempo.
- **Genre Mapping**: Maps primary genres to broader groups for analysis.

## 🪪 Model Pipeline

1. **Data Preprocessing**:
   - Feature selection.
   - Handling missing values and categorical encoding.
2. **Training**:
   - Ensemble models: VotingClassifier with Random Forest, XGBoost, CatBoost, and LightGBM.
   - Evaluation metrics: Classification Report, Cohen's Kappa, and F1-Score.
3. **Prediction**:
   - Encodes input features using trained scalers and encoders.
   - Predicts the genre and integrates Spotify metadata.

## 🔥 Demo

Check out a quick video or GIF showcasing the app's functionality (add link or image).

## 💡 Future Work

- Incorporate real-time Spotify API search for song metadata.
- Enhance UI with additional visualizations.
- Expand genre mappings for global datasets.

