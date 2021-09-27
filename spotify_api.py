"""
Justin Clark
CSYS 300
Final Project
spotify_api.py

Create dataframes based on user chosen Spotfy Playlists and
save to .csv file for later retrieval
"""

####################
### IMPORT ##
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
import os
import pandas as pd
import lyricsgenius
import re
import nltk
from nltk.sentiment import SentimentAnalyzer
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
################################

def analyze_playlist(creator, playlist_id):
    """
    Create dataframe of song information for given spotify playlist
    Arguments:
        creator: string of creator id
        playlist_id: string of playlist id
    Return
        playlist_df: dataframe of songs/variables of interest for given playlist
    
    """
    
    # Create empty dataframe
    playlist_features_list = ["artist","album","track_name",  "track_id","popularity","danceability","energy","key","loudness","mode", "speechiness","instrumentalness","liveness","valence","tempo", "duration_ms","time_signature"]
    
    playlist_df = pd.DataFrame(columns = playlist_features_list)
    
    # Loop through every track in the playlist, extract features and append the features to the playlist df
    
    playlist = sp.user_playlist_tracks(creator, playlist_id)["items"]
    for track in playlist:
        # Create empty dict
        playlist_features = {}
        # Get metadata
        playlist_features["artist"] = track["track"]["album"]["artists"][0]["name"]
        playlist_features["album"] = track["track"]["album"]["name"]
        playlist_features["track_name"] = track["track"]["name"]
        playlist_features["track_id"] = track["track"]["id"]
        playlist_features["popularity"] = track["track"]["popularity"]
        # Get audio features
        audio_features = sp.audio_features(playlist_features["track_id"])[0]
        for feature in playlist_features_list[5:]:
            playlist_features[feature] = audio_features[feature]
        
        # Concat the dfs
        track_df = pd.DataFrame(playlist_features, index = [0])
        playlist_df = pd.concat([playlist_df, track_df], ignore_index = True)
        
    return playlist_df

def createYearlyDataFrames(dictionary):
    """
    Create pandas dataframe for each year and save file to
    csv for later use
    Arguments:
        dictionary:
            key = String of Year
            value[0] = Creator ID
            value[1] = Playlist ID
    Return:
        none
    """
    for key in dictionary:
        year = key
        print("Year: {}".format(year))
        creator = dictionary[key][0]
        playlist_id = dictionary[key][1]
        df = analyze_playlist(creator,playlist_id)
        df.to_csv("rap_df_{}.csv".format(year),index = False)
        

################################
### SPOTIPY ####################

os.environ['SPOTIPY_CLIENT_ID'] = '1e19cf87b9524b8bacd09a98c4eef4c4'
os.environ['SPOTIPY_CLIENT_SECRET'] = 'facdf5ffa42e40ae9e335bebaeea0609'
os.environ['SPOTIPY_REDIRECT_URI'] = 'http://localhost:8888/callback/'
auth_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(auth_manager=auth_manager)
###############################
genius_client_id = '9g5ycExMAL9fWr_r0Y9Lyf7aoZDIYei8FN4ENPf1LZe6RjFX4sBT3xbULYB9qa7x'
genius_client_secret = 'oIQt0GoOWNP9wcVM2yn2tqTbhYAcPp4YOdjNu4VSkB1coCoM0FG7_Nh6az9f0JTYGMdzeEo47OCbhu3y14L4hQ'
genius_client_access_token = 'vequbCKgDuQVe3UtDd-RD7dzuG2sMxZaNz_J8pELZk01SXTK2KvxDZK68E2NtJ9h'
genius = lyricsgenius.Genius(genius_client_access_token)
sentim_analyzer = SentimentAnalyzer()

#df_2019 = analyze_playlist('AT MusicPedia','1YEm3mSbOeDnoftDjSkcYz')
dictionary_of_playlists = {
        "2019":['gentr187','6uM6KjEV4WEc4QyBDa0rIY'],
        "2018":["David Rex",'7AKfZ66t8zH0JnZ8VyQyDP'],
        "2017":["boardboymusicworldwide",'7LJTmZfNGAg8lsbiFVsNSx'],
        "2016":["dflanzer","28UZeYxXikNMM8M0mZqvfR"],
        "2015":["dflanzer","6hELk7zwzw2U9YQFbJ8NYL"],
        "2014":["dflanzer","5jmWFprGlGbMBFM70iuv1Q"],
        "2013":["dflanzer","5e5no0Qlmtv8NtmdYbqEah"],
        "2012":["dflanzer","66FkqVJ6bd396zxk7Vd5LD"],
        "2011":["dflanzer","10OjYXANvPfEGCRTGS1y3E"],
        "2010":["dflanzer","4lDf4pIdtUSspWqxWU3SDh"],
        "2009":["dflanzer","5S2OVw5DQzM0G7HKViisaW"],
        "2008":["dflanzer","4JNGQ3ZLXsfynMaUm8NvQw"],
        "2007":["dflanzer","1lfrSvG67UYWrfM0vR9hep"],
        "2006":["dflanzer","6EHJP6BEs0LfdwrLePgUor"],
        "2005":["dflanzer","78cZbPV04LVKaR6xKhR8Wc"],
        "2004":["dflanzer","12BE4vKU8wjDDaUduQqLOX"],
        "2003":["dflanzer","4cCgvfPlpQ93DfIT4Rg2TJ"],
        "2002":["dflanzer","0C8hVoysyqYbBlJEcuwVHC"],
        "2001":["dflanzer","189rjUuEEImCuMHCy9yfTz"],
        "2000":["dflanzer","425OyboixVIWXHrc3brLLO"],
        "1999":["dflanzer","4YQ7d1FnO8ZGA2XpVRMELC"],
        "1998":["dflanzer","5iZxDKkDZjSaDTiBMLboNJ"],
        "1997":["dflanzer","5t6INQzZjdA4KH2olKOOGO"],
        "1996":["dflanzer","1kvtznOaz6daVUICAiGs3Y"],
        "1995":["dflanzer","7c7qriUiW6yS5bmP7NXamq"],
        "1994":["dflanzer","1YyEwsjFrqE0zKigHb8v8L"],
        "1993":["dflanzer","3EsRGqzUcJdKUWWOzIVWsH"]
        }

createYearlyDataFrames(dictionary_of_playlists)
