import streamlit as st
import pandas as pd
from modules import genre_prediction as gp
import random
import time  # Import the time module

#######################################################################
### Streamlit Application
#######################################################################

# Title of the app
st.title("🎧 Song Genre Prediction App 🎶")

# Read the sample dataset
sample_data_df = pd.read_csv('./datasets/sample_music_dataset.csv')

# List the song names
song_list = ["Choose Your Tune"] + sample_data_df.track_name.to_list()

# Dropdown menu for song selection
song_name = st.selectbox(label="🎼 Tunes List", options=song_list, index=0)

# Pick a background track at random
bg_track_num = str(random.randint(1, 20))

# Buttons layout above the output
col1, col2 = st.columns([4, 1])

# Submit button
with col1:
    if st.button("🎤 Submit"):
        # Start the timer
        start_time = time.time()

        # Add the background music component
        st.markdown(
            f"""
            <audio id="background-music" autoplay loop>
                <source src="https://www.soundhelix.com/examples/mp3/SoundHelix-Song-{bg_track_num}.mp3" type="audio/mpeg">
            </audio>
            """,
            unsafe_allow_html=True
        )

        if song_name != "Select a song":
            # Call the prediction function
            out_df = gp.predict_genre(track_name=song_name)

            # Fetch the Spotify URL
            spotify_url, preview_url = gp.get_spotify_url(
                track_name=out_df['track_name'].iloc[0],
                artist_name=out_df['artist_name'].iloc[0]
            )

            # Determine highlight colors
            if out_df['predicted_genre'].iloc[0] == out_df['original_genre'].iloc[0]:
                predicted_genre_color = "#e3f7d7"  # Green for match
                original_genre_color = "#e3f7d7"  # Green for match
            else:
                predicted_genre_color = "#f7d7d7"  # Red for mismatch
                original_genre_color = "#e3f7d7"  # Green for original

            # Display the details of the selected song in a professional layout
            st.subheader(f"Details for the song:")
            st.markdown(
                f"""
                <div style="background-color:#ffffff; color:#000000; padding:20px; border-radius:10px; max-width: 600px; margin: auto; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">
                    <p style="font-size:18px; line-height:2;"><b>🎶 Track Name:</b> {out_df['track_name'].iloc[0]}</p>
                    <p style="font-size:18px; line-height:2;"><b>🎤 Artist Name:</b> {out_df['artist_name'].iloc[0]}</p>
                    <p style="font-size:18px; line-height:2;"><b>🌍 Album Name:</b> {out_df['album_name'].iloc[0]}</p>
                    <p style="font-size:18px; line-height:2;"><b>⏱️ Song Length:</b> {out_df['song_length'].iloc[0]}</p>
                    <p style="font-size:18px; line-height:2;"><b>🎵 Key Signature:</b> {out_df['key_signature'].iloc[0]}</p>
                    <p style="font-size:18px; line-height:2; background-color:{predicted_genre_color}; border-radius:5px; padding:5px;"><b>🎹 Predicted Genre:</b> {out_df['predicted_genre'].iloc[0]}</p>
                    <p style="font-size:18px; line-height:2; background-color:{original_genre_color}; border-radius:5px; padding:5px;"><b>🔁 Original Genre:</b> {out_df['original_genre'].iloc[0]}</p>
                    <p style="font-size:18px; line-height:2;"><b>🎼 Sub Genre:</b> {out_df['sub_genre'].iloc[0]}</p>
                    <p style="font-size:18px; line-height:2;"><b>🎷 Spotify URL:</b> <a href="{spotify_url}" target="_blank">{spotify_url}</a></p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # End the timer and calculate execution time
            end_time = time.time()
            execution_time = end_time - start_time

            st.markdown("<br>", unsafe_allow_html=True)

            # Display the execution time
            st.markdown(f"⏳ **Execution Time:** {execution_time:.2f} seconds")

        else:
            st.warning("🚨 **Please select a valid song from the dropdown.**")

# Reset button
with col2:
    if st.button("🔄 Reset"):
        st.snow()
