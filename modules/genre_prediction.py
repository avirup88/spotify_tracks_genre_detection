import streamlit as st
from joblib import load
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import traceback as tb
from datetime import datetime as dt
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os





#Function to encode categorical features
def fn_encode_categorical(data_df, feature_list):
        """Encode categorical features in the specified feature columns."""
        
        # Select categorical columns only from the explicitly defined features
        categorical_cols = [col for col in feature_list if data_df[col].dtype in ['object', 'category']]
        
        # One-hot encode categorical columns
        
        if categorical_cols:
            
            new_data_df = pd.get_dummies(data_df, columns=categorical_cols, drop_first=False)
            
            # Update self.features to reflect the new one-hot encoded columns
            encoded_cols = [col for col in data_df.columns if col.startswith(tuple(categorical_cols))]
            
            feature_list_new = [col for col in feature_list if col not in categorical_cols] + encoded_cols

        else:
            new_data_df = data_df
            feature_list_new = feature_list
            
        return new_data_df, feature_list_new

# Function to convert song length
def fn_convert_time(ms):
    total_seconds = ms // 1000
    hours = total_seconds // 3600
    remaining_seconds = total_seconds % 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60
    
    if hours > 0:
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    else:
        return f"{minutes:02}:{seconds:02}"

# Cache the model loading step
@st.cache_resource
def load_model():
    """Load the genre prediction model."""
    return load('./model_files/genre_prediction_model.joblib')


def predict_genre(track_name):
    # Display progress messages
    st.write("🚀 **Loading the pre-trained model...**")
    # Load the pre-trained model
    model = load_model()

    st.write("🔄 **Loading the pre-generated encoders...**")
    # Load the pre-generated scaler
    scaler = load('./model_files/scaler.pkl')
    label_encoder = load('./model_files/label_encoder.pkl')

    st.write("📂 **Reading the sample dataset and filtering the track...**")
    # Read the sample dataset
    sample_data_df = pd.read_csv('./datasets/sample_music_dataset.csv')
    sample_data_df = sample_data_df[sample_data_df['track_name'] == track_name]

    st.write("🔗 **Joining album and artist data...**")
    # Read the album and artist data
    albums_df = pd.read_excel('./datasets/spotify_datasets.xlsx', sheet_name='dim_albums')
    artists_df = pd.read_excel('./datasets/spotify_datasets.xlsx', sheet_name='dim_artists')
    mapping_df = pd.read_excel('./datasets/spotify_datasets.xlsx', sheet_name='artist_track_mapping')

    # Map the albums to the tracks
    sample_data_df = pd.merge(sample_data_df, albums_df, on='album_id', how='inner')
    mapping_df = mapping_df[mapping_df['track_id'].isin(sample_data_df.track_id.to_list())]
    mapping_df = pd.merge(mapping_df, artists_df, on='artist_id', how='inner')
    mapping_df = mapping_df.groupby('track_id').agg({'artist_name': lambda x: ', '.join(x)}).reset_index()
    sample_data_df = pd.merge(sample_data_df, mapping_df, on='track_id', how='inner')

    st.write("🧹 **Pre-processing the sample dataset...**")
    # Select and encode features
    feature_list = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness',
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'explicit']
    sample_data_df, new_feature_list = fn_encode_categorical(sample_data_df, feature_list)

    st.write("📏 **Scaling numeric features...**")
    numeric_features = sample_data_df[new_feature_list].select_dtypes(include=['number'])
    scaled_numeric_df = scaler.transform(numeric_features)
    data_df_scaled = sample_data_df.copy()
    data_df_scaled[numeric_features.columns] = scaled_numeric_df

    st.write("🧠 **Predicting the genre of the track...**")
    # Predict on Test Data
    y_pred_encoded = model.predict(data_df_scaled[new_feature_list])
    y_pred_decoded = label_encoder.inverse_transform(y_pred_encoded)
    sample_data_df['predicted_genre'] = y_pred_decoded

    st.write("📂 **Applying genre mapping and preparing output...**")
    # Read genre mapping file
    genre_mapping_df = pd.read_csv('./datasets/genre_groups.csv')
    sample_data_df = pd.merge(sample_data_df, genre_mapping_df, how='inner', on='primary_genre')
    sample_data_df = sample_data_df.rename(columns={'genre_group': 'original_genre', 'primary_genre': 'sub_genre'})

    # Song Length Minutes
    sample_data_df['song_length'] = sample_data_df['duration_ms'].apply(fn_convert_time)

    # Get relevant columns
    sample_data_df = sample_data_df[['track_id', 'track_name', 'artist_name', 'album_name', 'key_signature',
                                     'time_signature', 'song_length', 'tempo', 'explicit', 'danceability',
                                     'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
                                     'liveness', 'valence', 'original_genre', 'predicted_genre', 'sub_genre']]

    return sample_data_df


#Function to authenticate the spotify client
def spotify_authentication(SPOTIFY_CLIENT_ID,SPOTIFY_CLIENT_SECRET):

        if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
            raise ValueError("Spotify credentials not found. Please set them as environment variables.")

        # Spotify authentication
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        ))

        return sp

# Function to fetch Spotify URL with caching
@st.cache_data(show_spinner=True)
def get_spotify_url(track_name, artist_name):

    # Initialize Spotipy client
    SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
    SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
    sp = spotify_authentication(SPOTIFY_CLIENT_ID,SPOTIFY_CLIENT_SECRET)


    """Fetches the Spotify URL for a song based on track name and artist name."""
    query = f"track:{track_name} artist:{artist_name}"
    results = sp.search(q=query, type='track', limit=1)
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        spotify_url = track['external_urls']['spotify']
        preview_url = track.get('preview_url', None)
        return spotify_url,preview_url
    else:
        return "No results found on Spotify.", None
