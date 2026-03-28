


import os  
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
from requests import Response
import random
import glob
import unicodedata
import re
import unicodedata
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import nltk # type: ignore
from nltk.corpus import stopwords # type: ignore
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import matplotlib.pyplot as plt


#### Project Objective
# Analyze the predictive power of song lyrics in predicting whether a listener will skip or listen to a track
# Will use two distinct feature sets: One with both lyrics and audio features (tempo, valence, key, etc.)
# and another with just audio features.

#### Data Acquistion
# The goal in this section is to aggregate information from three different sources:
# 1. My Spotify listening history file (timestamp, tracks, artist name, platform) called full_df
# 2. Three large Kaggle datasets that contain audio metadata features (danceability, valence, acousticness, etc.)
# 3. Web scraped lyrics from the AZLyrics Website 
 
### Download my spotify listening history
full_df = pd.read_csv(f"C:\\Users\\urios\\full_merge_spotify.csv", low_memory=False) # update location

### Get audio metadata features from kaggle datasets

# Define main features
target_features = [
    "track_id", "acousticness", "album_release_date", "danceability", 
    "energy", "explicit", "genres", "instrumentalness", "key", 
    "liveness", "loudness", "mode", "popularity", "speechiness", 
    "tempo", "time_signature", "valence"
]
# Load Files
kaggle_folder_path = "C:\\Users\\urios\\spotifykaggle\\*.csv"
kaggle_files = glob.glob(kaggle_folder_path)

dfs = [] # This creates an empty Python LIST
for file in kaggle_files:
    temp_df = pd.read_csv(file, low_memory=False)
    
    # Filter columns immediately to save memory
    cols_to_keep = temp_df.columns.intersection(target_features)
    
    # Add the DataFrame to list
    dfs.append(temp_df[cols_to_keep])

# Concatenate
# This turns the LIST of DataFrames into ONE single DataFrame
master_library = pd.concat(dfs, ignore_index=True)

# Clean and merge
master_library.drop_duplicates(subset=['track_id'], keep='first', inplace=True)

# Merge with listening history (assuming full_df is already loaded)
df_merged = full_df.merge(master_library, on='track_id', how='left')

### Scrape lyrics from AZ Lyrics website

# WARNING: This is a large dataset. Set this to True ONLY if you want to scrape
# Default is False to prevent long execution.
RUN_SCRAPER = False 

# Saves to folder where the script is running
save_file = "savepoint_tracks_v1.3.csv" 

INVALID_CHARACTERS = "`',-/.!"

# Special Artist Rewrites for AZLyrics (Global definition)
SPECIAL_ARTIST_MAP = {
    "beyonce": "beyonceknowles",
    "beyoncé": "beyonceknowles",
    "big pun": "bigpunisher",
    "bigpun": "bigpunisher",
    "a$ap rocky": "asaprocky",
    "asap rocky": "asaprocky",
    "a$ap ferg": "asapferg",
    "asap ferg": "asapferg",
    "the weeknd": "weeknd",
    "weeknd": "weeknd",
    "the police": "police",
    "the who": "who",
    "the notorious big": "notoriousbig",
    "the notorious b.i.g.": "notoriousbig",
    "the roots": "roots",
    "the game": "game"
}

class AZLyrics():
    """
    Scrape song's lyrics from: https://www.azlyrics.com
    """
    def __init__(self, artist: str, song: str):
        self._artist = artist
        self._song = song
        
    def remove_accents(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        # Normalize atext
        normalized = unicodedata.normalize("NFKD", text)
        
        # Filter out combining marks (accents)
        return "".join([c for c in normalized if not unicodedata.combining(c)])

    def _parse_artist(self) -> str:
        # 1. Remove accents
        text = self.remove_accents(self._artist)
        # 2. Lowercase
        text = text.lower().strip()
        # 3. Special-case rewrites BEFORE removing spaces/punctuation
        if text in SPECIAL_ARTIST_MAP:
            return SPECIAL_ARTIST_MAP[text]
        # 4. Remove "the " prefix if applicable
        if text.startswith("the "):
            text = text[4:]
        # 5. Normalize $ -> s (A$AP -> asap)
        text = text.replace("$", "s")
        # 6. Remove spaces
        text = text.replace(" ", "")
        # 7. Remove punctuation
        for c in INVALID_CHARACTERS:
            text = text.replace(c, "")
        return text

    def _parse_song(self) -> str:
        text = self.remove_accents(self._song)
        # 1. FIX: Preserve text inside parentheses
        text = re.sub(r"\(([^)]*)\)", r" \1 ", text)
        # 2. Remove bracket contents completely []
        text = re.sub(r"\[[^]]*\]", "", text)
        # 3. Remove feat/ft/featuring sections
        text = re.sub(r"\b(feat\.?|ft\.?|featuring)\b.*", "", text, flags=re.IGNORECASE)
        # 4. Remove hyphens and everything after them
        text = re.sub(r"-.*$", "", text)
        # 5. Lowercase
        text = text.lower()
        # 6. Remove punctuation and symbols
        for c in INVALID_CHARACTERS:
            text = text.replace(c, "")
        # Remove ellipses, slashes, hashes, and misc
        text = re.sub(r"[.#/]", "", text)
        text = text.replace("...", "").replace("..", "")
        # 7. Remove all spaces (after fixing parentheses)
        text = re.sub(r"\s+", "", text)
        return text.strip()
    
    def scrape(self) -> str | None: 
        try:
            response = requests.get(self.url(), timeout=10) 
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {self._artist} - {self._song}: {e}")
            return None
        
        lyrics = None
        if response.ok:
            lyrics = self._scrape_lyrics(response)
        return lyrics

    def url(self) -> str:
        url_syntax = "https://www.azlyrics.com/lyrics/{}/{}.html"
        return url_syntax.format(self._parse_artist(), self._parse_song())

    def _scrape_lyrics(self, r: Response) -> str | None:
        try:
            dom = BeautifulSoup(r.text, "html.parser")
            body = dom.body
            divs = body.find_all("div", {"class": "col-xs-12 col-lg-8 text-center"})[0]
            
            # Heuristics: find the child element with the most <br> tags
            target = {0: 0} 
            for i, d in enumerate(divs.children): 
                try:
                    if hasattr(d, 'find_all'):
                         n_br = len(d.find_all("br"))
                    else:
                         n_br = 0
                    if n_br > list(target.values())[0]:
                        target = {i: n_br}
                except Exception:
                    pass

            target_index = list(target.keys())[0]
            
            lyrics_div = divs.find_all('div', class_=None, recursive=False)
            if len(lyrics_div) >= 2:
                lyrics_text = lyrics_div[1].get_text(separator="\n").strip() 
                return lyrics_text
            
            lyrics = list(divs.children)[target_index].text.strip()
            return lyrics if lyrics else None

        except Exception as e:
            print(f"Error scraping lyrics content: {e}")
            return None


# --- EXECUTION LOGIC (Protected by Safety Switch) ---

if RUN_SCRAPER:
    print("--- ⚠️ STARTING LIVE SCRAPE ---")
    # controls the max number of tracks to be scraped
    max_tracks = 100
    
    # 1. Define tracks from the main dataframe (Dynamic)
    tracks = df_merged[[
        'master_metadata_album_artist_name', 
        'master_metadata_track_name', 
        'spotify_track_uri'
    ]].dropna().drop_duplicates(subset=['spotify_track_uri']).reset_index(drop=True)

    tracks_to_process = tracks
    tracks_subset = tracks

    # HOW TO IMPLEMENT CHUNKING (Uncomment below to run in batches):
    # Example: If you only want to run the first 100 songs:
    # tracks_subset = tracks.iloc[0:100]

    # Example: If you already finished the first 1000 and want to do the next 500:
    # tracks_subset = tracks.iloc[1000:1500]

    # Example: If the scraper crashed at index 2050, restart from there:
    # tracks_subset = tracks.iloc[2050:]

    print(f"Starting scraping for the first {len(tracks_subset)} tracks with missing lyrics...")

    for i, (index, row) in enumerate(tracks_subset.iterrows(), 1):
        artist = row['master_metadata_album_artist_name']
        song = row['master_metadata_track_name']
        
        print(f"({i}/{len(tracks_subset)}) Scraping: {artist} - {song}")

        # --- Instantiate and Scrape ---
        az = AZLyrics(artist=artist, song=song)
        lyrics = az.scrape()

        # --- Update the Original DataFrame ---
        if lyrics:
            tracks.loc[index, 'lyrics'] = lyrics
            print(f"   -> Successfully scraped lyrics.")
        else:
            print(f"   -> Failed to find or scrape lyrics.")

        # --- Delays ---
        delay = random.uniform(2, 5)
        print(f"   -> Pausing for {delay:.2f} seconds...")
        time.sleep(delay)

        if i % 20 == 0:
            print("*** Waiting 30 seconds after 20 songs... ***\n")
            time.sleep(30)

        # --- Checkpoints ---
        if i % 5 == 0:
            print(f"--- CHECKPOINT: Saving progress to '{save_file}' ---")
            tracks.to_csv(save_file, index=False)
            print("--- Save complete. ---")

        if i == max_tracks:
            break

    print(f"\n--- SCRAPING FINISHED: Saving final progress to '{save_file}' ---")
    tracks.to_csv(save_file, index=False)
    print("--- Final Save complete. ---")

else:
    print("--- SKIPPING LIVE SCRAPING (RUN_SCRAPER = False) ---")
    print("Code for scraping class AZLyrics is loaded but disabled.")


### IMPORTANT: Look at Scraping Project Lyrics datasets for lyrics that were scrapped
### Note that there are 3 datasets
### Here - I'm joining them into one dataset

# folder were I kept the scraped lyrics datasets
folder_path = "C:\\Users\\urios\\Scraping Project Lyrics All\\*.csv" # must update path
files = glob.glob(folder_path)

lyric_dfs = [pd.read_csv(f) for f in files] # upload all files into list
lyrics_all = pd.concat(lyric_dfs, ignore_index=True) # merge them together in one dataframe

# found out that the merged dataframes had duplicates in it
lyrics_all.duplicated(subset=["spotify_track_uri"]).sum()

# dropped all nulls in artist + track name
lyrics_all_clean = lyrics_all.dropna(subset=["spotify_track_uri"])

# dropped all duplicates in my dataframe
lyrics_unique = lyrics_all_clean.drop_duplicates(
    subset=["spotify_track_uri"],
    keep="first"
)

### In this section I merge the lyrics into my main Spotify listening history dataset

# Performed join to add lyrics to dataframe to create df_merged2

df_merged2 = df_merged.merge(
    lyrics_unique[["spotify_track_uri", "lyrics"]],
    on="spotify_track_uri",
    how="left",
)

#### Data Cleaning & Imputation
### In this section, I handle missing data, particularly for the genres column:
### 1. For tracks that do have the genre assignment, they are turned into python lists
### 2. For tracks that have empty ([]) or null genres, I created large dictionaries
### to manually imputate missing information
### I used domain knowledge and occassionally, used the Genre assignments from the Genius website
### For rows that have genre info, turn it into a list

df = df_merged2  

# Keep only rows with non-null genres
df_with_genres = df[df["genres"].notna()].copy()

def parse_genre_list(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    try:
        return [g.strip() for g in ast.literal_eval(x)]
    except Exception:
        # fallback: treat whole thing as a single genre string
        return [str(x).strip()]

df_with_genres["genre_list"] = df_with_genres["genres"].apply(parse_genre_list)

### Find rows that have empty brackets for genres

empty_genre_rows = df_with_genres[
    df_with_genres["genres"].astype(str).str.strip() == "[]"
]

artists_with_empty = (
    empty_genre_rows["master_metadata_album_artist_name"]
    .unique()
)

### Fill in genre info for artists that have the brackets []

manual_artist_genres = {
    "Rich Gang": ['hip hop', 'rap'],
    "Alice DJ": ['eurodance'],
    "Daddy Yankee": ['latin hip hop', 'reggaeton', 'trap latino', 'urbano latino'],
    "MellowHype": ['rap'],
    "John Lloyd Young": ['pop'],
    "Nore": ['reggaeton', 'urbano latino'],
    "Notch": ['reggaeton', 'urbano latino'],
    "Sha Na Na": ['musicals', 'pop', 'soundtrack'],
    "La Factoria": ['latin music', 'reggaeton'],
    "Yolanda Be Cool & DCUP": ['australian dance', 'australian house', 'bass house'],
    "Angel Y Khriz": ['latin music', 'reggaeton'],
    "Sergio Mendes & Brasil '66": ['bossa nova', 'pop', 'samba'],
    "La Secta": ['puerto rican rock'],
    "Rockwell": ['dance', 'electronic', 'freestyle'],
    "Zacari": ['rap', 'soundtrack'],
    "Eddie Dee": ['latin music', 'latin pop', 'latin urban', 'reggaeton'],
    "Raphael": ['spanish pop'],
    "THE SCOTTS": ['hip hop', 'rap', 'trap'],
    "Scott Storch": ['latin music', 'latin pop', 'latin urban', 'reggaeton'],
    "Andrea Bocelli": ['en espanol', 'pop'],
    "Buika": ['en espanol', 'pop'],
    "Billie Joe Armstrong": ['rock'],
    "Luis Figueroa": ['latin music', 'salsa'],
    "Chris Lebron": ['latin music', 'latin pop', 'latin urban'],
    "FLO": ['girl group', 'r&b'],
    "Noreaga": ['hip hop', 'new york rap', 'rap'],
    "Ed Sheeran": ['pop', 'singer-songwriter pop', 'uk pop'],
    "Coco Jones": ['contemporary r&b', 'neo soul', 'r&b'],
    "Samara Joy": ['jazz', 'pop', 'vocal jazz'],
    "Teddy Swims": ['adult contemporary', 'alternative r&b', 'pop', 'r&b'],
    "Nas, 50 Cent, & Nature": ['east coast rap', 'gangsta rap', 'new york rap', 'rap'],
    "Tyla": ['amapiano', 'afrobeats', 'pop', 'r&b'],
    "310babii": ['hip hop', 'rap', 'trap'],
    "Teezo Touchdown": ['hip hop', 'rap', 'trap'],
    "Colby O'Donis": ['pop'],
    "Â¥$": ['experimental rap', 'hip hop', 'rage', 'rap', 'trap'],
    "4batz": ['alternative r&b', 'r&b'],
    "Roy Jones Jr.": ['hip hop', 'rap'],
    "DJ Memo": ['latin music', 'latin pop', 'latin urban', 'reggaeton']
}

### fill by matching artist name and where genre column is brackets

for artist, genre_list in manual_artist_genres.items():
    mask = (
        (df["master_metadata_album_artist_name"] == artist) &
        (df["genres"].astype(str).str.strip() == "[]")
    )
    df.loc[mask, "genres"] = str(genre_list)


### Individual cases

mask_r_kelly = (
    (df["master_metadata_album_artist_name"] == "R. Kelly") &
    (df["master_metadata_track_name"].str.lower() != "burn it up") &
    (df["genres"].astype(str).str.strip() == "[]")
)

df.loc[mask_r_kelly, "genres"] = str(['r&b'])


mask_r_kelly_burn = (
    (df["master_metadata_album_artist_name"] == "R. Kelly") &
    (df["master_metadata_track_name"].str.lower() == "burn it up") &
    (df["genres"].astype(str).str.strip() == "[]")
)

df.loc[mask_r_kelly_burn, "genres"] = str(['reggaeton', 'urbano latino'])


### Now look at genre column to find nulls outside of brackets

df_null_genres = df[df["genres"].isna()]

## In addition drop those that have nulls in these columns
columns_to_check = [
    "tempo",
    "energy",
    "key",
    "popularity",
    "mode",
    "time_signature",
    "speechiness",
    "danceability",
    "valence",
    "acousticness",
    "liveness",
    "instrumentalness",
    "loudness",
    "lyrics"
]

df_null_genres_clean = df_null_genres.dropna(subset=columns_to_check)

### Get the unique spotify tracks

unique_null_tracks = df_null_genres_clean.drop_duplicates(
    subset=["spotify_track_uri"]
)


### Fill in rows where genre has rows with nulls in it

artist_genres2 = {
    "2Pac": ['g funk', 'gangster rap', 'hip hop', 'rap', 'west coast rap'],
    "50 Cent": ['east coast hip hop', 'gangster rap', 'hardcore rap', 'hip hop', 'rap'],
    "A$AP Rocky": ['alternative', 'east coast hip hop', 'hip hop', 'rap', 'trap'],
    "Ã‘engo Flow": ['reggaeton', 'trap latino', 'urbano latino'],
    "Abel Pintos": ['argentina', 'en espanol', 'latin music', 'pop'],
    "Accept": ['metal', 'rock', 'traditional heavy metal'],
    "AKA": ['nigeria', 'rap'],
    "Al Jarreau": ['bossa nova', 'jazz', 'jazz fusion', 'r&b', 'smooth jazz'],
    "Alejandro Sanz": ['en espanol', 'espana', 'rock'],
    "Alex Sensation": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "Alexis y Fido": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "Alkilados": ['en espanol', 'pop'],
    "Almighty":	['latin music', 'latin pop', 'latin urban'],
    "Amenazzy": ['latin music', 'latin urban', 'latin trap', 'rap', 'trap'],
    "Ana Gabriel": ['en espanol', 'pop'],
    "Andy Rivera":	['latin music', 'latin urban', 'pop', 'reggaeton'],
    "Anthony Hamilton":	['r&b', 'soul'],
    "Aqua":	['eurodance'],
    "Aretha Franklin": ['jazz', 'r&b', 'soul', 'soul jazz'],
    "Aurora": ['alternative', 'electropop', 'pop', 'singer-songwriter'],
    "Aventura":	['bachata', 'latin music', 'latin pop', 'latin urban'],
    "Aya Nakamura":	['fench pop', 'french r&b'],
    "Backstreet Boys":	['boy band', 'pop'],
    "Bad Gyal":	['spanish urban', 'spanish pop', 'reggaeton'],
    "Baha Men":	['pop'],
    "Bazzi": ['pop', 'r&b'],
    "BeÃ©le": ['latin music', 'latin urban', 'latin pop', 'pop'],
    "Bee Gees":	['pop'],
    "Beenie Man": ['dancehall'],
    "Beret": ['en espanol', 'pop'],
    "BeyoncÃ©":	['pop', 'r&b'],
    "Big Pun":	['bronx hip hop', 'east coast hip hop', 'hardcore hip hop'],
    "Big Sean":	['detroit hip hop', 'hip hop', 'r&b', 'rap', 'southern hip hop', 'trap'],
    "Billie Holliday": ['jazz', 'r&b', 'soul', 'soul jazz'],
    "Black Box": ['dance', 'house', 'pop'],
    "Black Coffee": ['afro house', 'house'],
    "Black Eyed Peas": ['afrobeats', 'hip hop', 'latin music', 'latin urban', 'latin pop', 'pop'],
    "Bob Seger": ['adult contemporary', 'ballad', 'rock', 'soft rock'],
    "Booba": ['french pop', 'rap'],
    "Boyz II Men": ['r&b', 'soul'],
    "Boza": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "Brandy": ['contemporary r&b', 'dance pop', 'hip pop', 'r&b', 'urban contemporary'],
    "Brian McKnight": ['r&b', 'soul'],
    "Bruce Springsteen": ['heartland rock', 'pop', 'rock'],
    "Brytiago": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "Build To Spill": ['alternative rock', 'indie rock', 'rock'],
    "CÃ©line Dion": ['adult contemporary', 'pop'],
    "Caitlyn Smith": ['country'],
    "Cali Y El Dandee": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "Calle 13": ['pop'],
    "Camilo Sesto": ['en espanol', 'pop'],
    "Canserbero": ['conscious rap', 'en espanol', 'latin music', 'rap'],
    "Carl Thomas": ['r&b'],
    "Carly Simon": ['pop'],
    "Carole King": ['ballad', 'rock', 'singer-songwriter'],
    "Casper Magico": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "Cazzu": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "CeCe Peniston": ['dance', 'r&b'],
    "Chamillionaire": ['rap'],
    "Chayanne": ['en espanol', 'pop'],
    "Cherish": ['crunk', 'girl group', 'rap', 'r&b'],
    "Chimbala": ['dembow', 'latin music', 'latin pop', 'latin urban'],
    "Chris Brown": ['rap', 'r&b'],
    "Christian Nodal": ['banda', 'latin music', 'regional mexicano'],
    "Chucky73": ['latin music', 'latin urban', 'latin rap', 'rap', 'trap'],
    "Ciara": ['r&b'],
    "CJ": ['drill', 'latin drill', 'latin music', 'latin urban', 'latin rap', 'rap'],
    "CNCO": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "Commodores": ['adult standards', 'disco', 'funk', 'mellow gold', 'motown', 'quiet storm', 'soft rock', 'soul'],
    "Common": ['conscious rap', 'hip hop', 'rap'],
    "Cosculluela": ['latin music', 'latin urban', 'latin rap', 'rap'],
    "Curtis Mayfield": ['funk', 'funk rock', 'r&b', 'soul', 'soul rock'],
    "Dadju": ['french pop', 'french rap', 'pop', 'rap'],
    "Dalex": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "D'Angelo": ['indie soul', 'neo soul', 'soul'],
    "Daniel Powter": ['pop'],
    "Danna Paola": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "Darell": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "David Bisbal": ['en espanol', 'pop'],
    "David Guetta": ['big room', 'dance pop', 'edm', 'pop', 'pop dance'],
    "Dean Martin": ['adult standards', 'easy listening', 'lounge', 'vocal jazz'],
    "Demis Roussos": ['adult contemporary', 'baroque pop'],
    "Dennis Brown": ['Reggae'],
    "Diana Ross": ['r&b', 'soul'],
    "DJ BoBo": ['eurodance'],
    "DMX": ['east coast hip hop', 'gangster rap', 'hardcore hip hop', 'hip hop', 'rap'],
    "Donna Summer": ['electropop', 'pop', 'r&b'],
    "Donny Hathaway": ['pop', 'r&b'],
    "Duki": ['latin music', 'latin urban', 'latin trap', 'trap'],
    "Duran Duran": ['new wave', 'synth-pop'],
    "Dvicio": ['en espanol', 'pop'],
    "dvsn": ['canadian contemporary r&b', 'r&b'],
    "Eagle-Eye Cherry": ['pop rock'],
    "Eagles": ['album rock', 'classic rock', 'heartland rock', 'mellow gold', 'rock', 'soft rock', 'yacht rock'],
    "ECKO": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "El Alfa": ['dembow', 'rap dominicano', 'trap latino', 'urbano latino'],
    "El Fantasma": ['latin music', 'regional mexicano'],
    "Eladio Carrion": ['latin music', 'latin urban', 'latin trap', 'trap'],
    "Ella Fitzgerald": ['adult standards', 'jazz', 'jazz blues', 'soul', 'swing', 'vocal jazz'],
    "Elton John": ['glam rock', 'mellow gold', 'piano rock', 'rock'],
    "Eminem": ['detroit hip hop', 'hip hop', 'rap'],
    "Erykah Badu": ['afrofuturism', 'alternative r&b', 'neo soul', 'r&b'],
    "Fishbone": ['alternative rock', 'funk metal', 'funk rock', 'groove metal', 'punk rock', 'rock'],
    "Fleetwood Mac": ['album rock', 'classic rock', 'rock', 'soft rock', 'yacht rock'],
    "Fonseca": ['en espanol', 'latin pop', 'pop'],
    "Four Tops": ['classic soul', 'disco', 'motown', 'quiet storm', 'soul'],
    "Franco De Vita": ['latin music', 'latin pop'],
    "Frank Sinatra": ['adult standards', 'easy listening', 'lounge'],
    "Future": ['atl hip hop', 'hip hop', 'rap', 'southern hip hop', 'trap'],
    "George Harrison": ['album rock', 'classic rock', 'folk rock', 'mellow gold', 'rock', 'singer-songwriter', 'soft rock'],
    "Gerardo Ortiz": ['regional mexicano'],
    "Ghostface Killah": ['east coast hip hop', 'gangster rap', 'hardcore hip hop', 'hip hop', 'rap'],
    "Gianluca Grignani": ['en espanol', 'pop'],
    "Gigolo Y La Exce": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "Gipsy Kings": ['flamenco', 'rumba', 'world'],
    "Gorilla Zoe": ['atl hip hop', 'crunk', 'dirty south rap', 'futuristic swag', 'southern hip hop', 'trap'],
    "Green Day": ['modern rock', 'permanent wave', 'punk', 'rock'],
    "Guaynaa": ['cumbia', 'latin music', 'latin pop', 'latin urban'],
    "Gustavo Cerati": ['alternative', 'alternative rock', 'latin rock', 'pop'],
    "Gyptian": ['Reggae'],
    "Ice Mc": ['dancehall', 'eurodance', 'rap', 'pop'],
    "ICE-T": ['rap'],
    "Ivy Queen": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "J Balvin": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "J. Holiday": ['contemporary r&b', 'r&b', 'urban contemporary'],
    "Janet Jackson": ['r&b', 'urban contemporary'],
    "Jazmine Sullivan": ['neo soul', 'r&b', 'urban contemporary'],
    "John Legend": ['neo soul', 'pop', 'pop soul', 'urban contemporary'],
    "Johnny Gill": ['new jack swing', 'r&b'],
    "Jory Boy": ['latin music', 'latin pop', 'latin urban', 'pop', 'reggaeton'],
    "JosÃ© JosÃ©": ['balada', 'latin music', 'pop'],
    "Juan Gabriel": ['balada', 'latin music', 'pop'],
    "Juan MagÃ¡n": ['dance', 'latin music', 'latin pop'],
    "Juanes": ['latin music', 'latin pop'],
    "Juanka": ['latin music', 'latin pop', 'latin urban', 'pop', 'reggaeton'],
    "Juhn": ['latin music', 'latin pop', 'latin urban', 'pop', 'reggaeton'],
    "Julieta Venegas": ['latin pop'],
    "Julio Iglesias": ['latin pop'],
    "Justin Bieber": ['contemporary r&b', 'r&b'],
    "Justin Quiles": ['latin music', 'latin pop', 'latin urban', 'pop', 'reggaeton'],
    "Karen Clark Sheard": ['pop'],
    "KAROL G": ['latin music', 'latin pop', 'latin urban', 'pop', 'reggaeton'],
    "Kelly Rowland": ['contemporary r&b', 'r&b'],
    "Kenia OS": ['latin music', 'pop', 'reggaeton'],
    "KEVIN ROLDAN": ['latin music', 'latin pop', 'latin urban', 'pop', 'reggaeton'],
    "KEVVO": ['latin music', 'latin pop', 'latin urban', 'pop', 'reggaeton'],
    "King Promise": ['afrobeat', 'afropop'],
    "Kwesta": ['kwaito'],
    "La Bouche": ['eurodance'],
    "Lady Gaga": ['country', 'country rock', 'rock'],
    "Lalo Ebratt": ['latin music', 'latin pop', 'latin urban', 'pop', 'reggaeton'],
    "Laura Pausini": ['pop'],
    "Lauv": ['pop'],
    "Led Zeppelin": ['album rock', 'classic rock', 'hard rock', 'rock'],
    "Lenny Kravitz": ['permanent wave', 'rock'],
    "Lil Tjay": ['new york rap', 'rap'],
    "Lil Wayne": ['hip hop', 'new orleans rap', 'pop rap', 'rap', 'trap'],
    "LIT killah": ['latin music', 'latin urban', 'latin rap', 'rap', 'trap'],
    "Living Colour": ['funk metal', 'funk rock', 'hard rock', 'rock'],
    "Loose Ends": ['contemporary r&b', 'r&b'],
    "Los Bukis": ['latin music', 'pop'],
    "Los Enanitos Verdes": ['latin music', 'latin pop', 'latin rock', 'pop'],
    "Lucenzo": ['latin music', 'latin pop', 'latin urban', 'reggaeton'],
    "Lunay": ['latin music', 'latin pop', 'latin urban', 'reggaeton'],
    "Luther Vandross": ['quiet storm', 'soul'],
    "Mac DeMarco": ['indie pop', 'indie rock', 'rock'],
    "Macy Gray": ['neo soul', 'r&b', 'soul', 'soul pop'],
    "Maikel Delacalle": ['spanish pop', 'spanish urban', 'pop', 'reggaeton'],
    "Maluma": ['latin music', 'latin pop', 'latin urban', 'reggaeton'],
    "Manuel Turizo": ['latin music', 'latin pop', 'latin urban', 'reggaeton'],
    "Marc Anthony": ['latin music', 'latin pop', 'pop', 'salsa'],
    "Marc Cohn": ['rock'],
    "Marco Antonio SolÃ­s": ['balada', 'latin music', 'pop'],
    "Marky Mark And The Funky Bunch": ['rap'],
    "Marsha Ambrosius": ['neo soul', 'r&b'],
    "Marvin Gaye": ['classic soul', 'motown', 'neo soul', 'northern soul', 'quiet storm', 'soul'],
    "Massive Attack": ['electronic', 'trip-hop'],
    "Master P": ['rap'],
    "Mavado": ['dancehall', 'rap'],
    "Maxwell": ['contemporary r&b', 'neo soul', 'quiet storm', 'r&b', 'urban contemporary'],
    "MC Fioti": ['funk', 'latin music'],
    "MC Kevinho": ['baile funk', 'pop'],
    "MC Magic": ['rap'],
    "Mecano": ['en espanol', 'pop'],
    "Melendi": ['latin music', 'latin pop'],
    "Memphis Bleek": ['east coast hip hop'],
    "Michael Bolton": ['soft rock'],
    "Michel TelÃ³": ['sertanejo'],
    "Miky Woodz": ['latin music', 'latin urban', 'latin trap', 'trap'],
    "Mint Condition": ['r&b', 'soul'],
    "Mobb Deep": ['east coast hip hop', 'hardcore hip hop', 'hip hop', 'queens hip hop'],
    "Molotov": ['alternative rock', 'latin music', 'latin rock', 'rock'],
    "Monica": ['contemporary r&b', 'r&b'],
    "Mora": ['latin music', 'latin pop', 'reggaeton', 'pop'],
    "Mos Def": ['conscious hip hop', 'east coast hip hop', 'hip hop'],
    "Mr. Vegas": ['dancehall'],
    "MS MR": ['rock'],
    "Myriam Hernandez": ['balada', 'latin music', 'pop'],
    "N.E.R.D": ['alternative rap', 'rap'],
    "N.W.A.": ['hardcore rap', 'west coast rap', 'rap'],
    "Nas": ['conscious hip hop', 'east coast hip hop', 'gangster rap', 'hardcore hip hop', 'hip hop', 'queens hip hop', 'rap'],
    "NATTI NATASHA": ['latin music', 'latin pop', 'latin urban', 'reggaeton'],
    "Naughty By Nature": ['east coast rap', 'rap'],
    "Nelly": ['hip hop', 'pop rap', 'rap', 'st louis rap', 'urban contemporary'],
    "Nicky Jam": ['latin music', 'latin pop', 'latin urban', 'reggaeton'],
    "Nil Moliner": ['pop'],
    "Nina Simone": ['soul', 'pop'],
    "Nio Garcia": ['latin music', 'latin pop', 'latin urban', 'reggaeton'],
    "No Mercy": ['pop'],
    "Noriel": ['latin music', 'latin urban', 'latin trap', 'trap'],
    "Otis Redding": ['r&b', 'soul'],
    "Outkast": ['atl hip hop', 'dirty south rap', 'hip hop', 'old school atlanta hip hop', 'rap', 'southern hip hop'],
    "Ozuna": ['latin music', 'latin pop', 'latin urban', 'reggaeton'],
    "Paloma Mami": ['latin music', 'latin pop', 'latin urban', 'reggaeton'],
    "Patti LaBelle": ['r&b', 'soul' 'synth-pop'],
    "Paul Anka": ['pop', 'singer-songwriter'],
    "Paulina Rubio": ['latin pop', 'pop'],
    "Peabo Bryson": ['pop', 'r&b', 'soul'],
    "Pepe Aguilar": ['pop'],
    "Pereza": ['rock', 'spanish rock'],
    "Peter Cetera": ['pop'],
    "Pharrell Williams": ['pop'],
    "Piso 21": ['latin music', 'latin pop', 'latin urban', 'reggaeton'],
    "Pitizion": ['latin music', 'latin pop', 'latin urban', 'reggaeton'],
    "PNL": ['french rap', 'rap'],
    "Pop Smoke": ['brooklyn drill', 'rap'],
    "Popcaan": ['dancehall'],
    "Prince": ['funk', 'funk rock', 'minneapolis sound', 'rock', 'synth funk'],
    "Prince Royce": ['bachata', 'latin hip hop', 'latin pop', 'urbano latino'],
    "Quincy Jones": ['r&b'],
    "Raekwon": ['east coast hip hop', 'gangster rap', 'hardcore hip hop', 'hip hop', 'rap'],
    "Rauw Alejandro": ['latin music', 'latin pop', 'latin urban', 'reggaeton'],
    "RBD": ['latin music', 'latin pop', 'pop'],
    "Real McCoy": ['eurodance'],
    "Reik": ['latin music', 'latin pop', 'pop'],
    "Ricardo Arjona": ['latin music', 'latin pop' 'singer-songwriter'],
    "Ricardo Montaner": ['latin music', 'latin pop' 'singer-songwriter'],
    "Richie Spice": ['reggae'],
    "Rick James": ['disco', 'funk', 'motown', 'p funk', 'quiet storm', 'soul', 'synth funk'],
    "Rick Ross": ['dirty south rap', 'gangster rap', 'hip hop', 'rap', 'southern hip hop', 'trap'],
    "Rihanna": ['barbadian pop', 'pop', 'urban contemporary'],
    "Roberto Carlos": ['brazil', 'pop'],
    "Ruff Ryders": ['east coast hip hop', 'gangster rap', 'hardcore hip hop', 'hip hop', 'rap'],
    "Rufus": ['funk', 'r&b', 'soul'],
    "Sam Cooke": ['r&b', 'soul', 'soul pop'],
    "Sammy Davis Jr.": ['big band', 'jazz', 'pop'],
    "Santana": ['blues rock', 'jazz fusion', 'latin rock', 'psychedelic rock', 'pop rock'],
    "Santigold": ['alternative dance', 'art pop', 'indietronica', 'neo-synthpop'],
    "Sash!": ['electronic', 'eurotrance'],
    "Sean Paul": ['dancehall', 'pop'],
    "Selena": ['cumbia', 'latin music', 'latin pop', 'pop'],
    "Selena Gomez": ['pop', 'post-teen pop'],
    "Slim Thug": ['dirty south rap'],
    "Snoop Dogg": ['g funk', 'gangster rap', 'hip hop', 'pop rap', 'rap', 'west coast rap'],
    "Sophie B. Hawkins": ['lilith', 'new wave pop'],
    "Sorriso Maroto": ['pagode', 'pop'],
    "Stevie Wonder": ['adult contemporary', 'jazz fusion', 'progressive pop', 'progressive soul', 'r&b', 'soul', 'soul pop'],
    "Super Yei": ['latin music', 'latin urban', 'latin pop', 'pop', 'reggaeton'],
    "T.O.K": ['dancehall'],
    "Tamar Braxton": ['contemporary r&b', 'r&b'],
    "Tamia": ['contemporary r&b', 'r&b'],
    "Tata Young": ['pop'],
    "Teddy Pendergrass": ['classic soul', 'funk', 'motown', 'philly soul', 'quiet storm', 'soul', 'urban contemporary'],
    "Tems": ['afro r&b', 'afrobeats', 'alte', 'nigerian pop'],
    "Tevin Campbell": ['contemporary r&b', 'new jack swing', 'r&b', 'urban contemporary'],
    "The Alan Parsons Project": ['album rock', 'art rock', 'classic rock', 'mellow gold', 'progressive rock', 'soft rock', 'symphonic rock'],
    "The Barden Bellas": ['pop'],
    "The Beatles": ['british invasion', 'classic rock', 'merseybeat', 'psychedelic rock', 'rock'],
    "The Game": ['gangsta rap', 'hardcore rap', 'hip hop', 'rap'],
    "The Guess Who": ['blues rock', 'hard rock', 'psychedelic rock', 'rock'],
    "The Manhattans": ['r&b', 'soul'],
    "The Marcels": ['doo-wop', 'pop'],
    "The Marvelettes": ['motown', 'r&b', 'soul pop'],
    "The Miracles": ['disco', 'motown', 'pop', 'r&b', 'soul'],
    "The Monkees": ['bubblegum pop', 'piano rock', 'pop', 'power pop'],
    "The Offspring": ['rock'],
    "The Partridge Family": ['bubblegum pop', 'pop', 'soundtrack', 'tv'],
    "The Righteous Brothers": ['adult standards', 'brill building pop', 'folk rock', 'mellow gold', 'motown', 'rock-and-roll', 'rockabilly'],
    "The Rolling Stones": ['album rock', 'british invasion', 'classic rock', 'rock'],
    "The Staple Singers": ['christian', 'christian r&b', 'gospel', 'r&b'],
    "The Supremes": ['classic girl group', 'classic soul', 'motown', 'soul'],
    "The Temptations": ['classic soul', 'memphis soul', 'motown', 'soul'],
    "Three 6 Mafia": ['crunk', 'dirty south rap', 'gangster rap', 'memphis hip hop', 'southern hip hop', 'tennessee hip hop', 'trap'],
    "Tinashe": ['r&b'],
    "TINI": ['balada', 'latin music', 'pop'],
    "Toni Braxton": ['contemporary r&b', 'r&b', 'urban contemporary'],
    "Tony Dize": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "Tony! Toni! TonÃ©!": ['new jack swing', 'r&b', 'soul pop'],
    "Tracy Chapman": ['folk', 'lilith', 'singer-songwriter', 'pop'],
    "Trey Songz": ['contemporary r&b', 'r&b'],
    "Trueno": ['latin music', 'latin urban', 'latin trap', 'rap', 'tango', 'trap'],
    "USHER": ['contemporary r&b', 'dance pop', 'pop', 'r&b', 'urban contemporary'],
    "Vetusta Morla": ['pop', 'spanish pop'],
    "Vicente FernÃ¡ndez": ['latin music', 'ranchera'],
    "Whigfield": ['eurodance'],
    "Whitney Houston": ['pop'],
    "Wisin": ['latin music', 'latin pop', 'latin urban', 'reggaeton'],
    "Wizkid": ['afrobeats', 'pop'],
    "Wu-Tang Clan": ['east coast hip hop', 'gangster rap', 'hardcore hip hop', 'hip hop', 'rap'],
    "Yandel": ['latin music', 'latin pop', 'latin urban', 'reggaeton'],
    "Yelawolf": ['rap'],
    "YSY A": ['latin music', 'latin urban', 'latin rap', 'rap', 'trap'],
    "Yuri": ['en espanol', 'pop'],
    "Zion": ['latin music', 'latin pop', 'latin urban', 'pop', 'reggaeton']
}


### These pairings are generalized
### Some artists are more predictable in the genres category
### For these artists, assign them the genres listed for every one of their songs

for artist, genre_list in artist_genres2.items():
    mask = (
        (df["master_metadata_album_artist_name"] == artist) &
        df["spotify_track_uri"].notna() &      # must have a URI
        df["genres"].isna()                    # only where genre is NULL
    )
    df.loc[mask, "genres"] = str(genre_list)

### These are more complex assignments for artists that have more diverse genres 
### track specific assignments

track_specific_genres = {
    "spotify:track:1sk1GwmPDkEECI2h5hNmpI": ['en espanol', 'latin music', 'mexico', 'regional mexicano', 'pop'],
    "spotify:track:4PzGASKw1yO2e8fvDNjpzp": ['bachata', 'en espanol', 'pop'],

    "spotify:track:3cNKhoBUhyuXhQTdQjOiT8": ['latin music', 'latin urban', 'pop', 'reggaeton'],  # Anitta – Loco

    "spotify:track:630ryPVdooTZr78p63sCFB": ['latin music', 'latin urban', 'latin rap', 'rap', 'trap'],
    "spotify:track:4vAsjqRZHlqGvY8Rqog4il": ['latin music', 'latin urban', 'latin rap', 'rap', 'trap'],

    "spotify:track:7MjaU5iFujwT1gOyvJnqNp": ['latin music', 'latin urban', 'latin rap', 'rap', 'trap'],

    "spotify:track:2cpteAYHcd4cjSxAeCkA52": ['latin music', 'latin urban', 'pop', 'reggaeton'],

    "spotify:track:00GbPd84bEyYS477RSymJW": ['latin music', 'latin urban', 'pop', 'reggaeton'],

    "spotify:track:5yJCrSdwrLFzboxg0z9xMd": ['latin music', 'latin pop', 'latin urban', 'pop', 'reggaeton'],

    "spotify:track:1Me0238KTGrw7POxazgl77": ['adult contemporary', 'ballad', 'pop', 'r&b'],

    "spotify:track:5Ufz5dUvW8aZ5AdVYRGmsx": ['balada', 'latin music', 'latin pop', 'pop'],

    "spotify:track:3h99fBGkTn3J2NsrYU8rcO": ['latin music', 'latin urban', 'rap', 'reggaeton'],
    "spotify:track:4yORNsoYe4XnK99EXhKhWB": ['latin music', 'latin urban', 'mambo', 'reggaeton'],
    "spotify:track:6vd3N43OyH9VymDR2rx4rH": ['latin music', 'latin urban', 'latin rap', 'rap'],
    "spotify:track:0TOfUc4AO3nk46xtVS1K4t": ['latin music', 'latin urban', 'latin rap', 'rap'],
    "spotify:track:6leYvwqxAPnzgfb86CdNfl": ['latin music', 'latin urban', 'latin rap', 'rap'],  # also De La Ghetto

    "spotify:track:0fQ9kbOBSpxEU13CqKeeGb": ['latin music', 'latin urban', 'rap', 'reggaeton'],

    "spotify:track:1Sr5Z0pqqEWbNAp2Sos1Dt": ['ballad', 'duet', 'pop'],

    "spotify:track:3eL5jJwutVux7277wM5qzV": ['Mambo', 'Merengue'],
    "spotify:track:752g22kc1OSv5Qb8ywwHYE": ['Mambo', 'Merengue'],
    "spotify:track:6pNjmMB8FTMEcqGSgs7Mji": ['latin music', 'latin urban', 'rap', 'reggaeton'],
    "spotify:track:7zQ42TLChYE29oJiyoWUIK": ['bachata', 'latin music', 'latin pop', 'latin urban'],
    "spotify:track:1En6OzqLa2XWiY1We4jHms": ['dancehall', 'latin music', 'latin pop', 'latin urban'],
    "spotify:track:2K1mqI6DBj5qv83oif1Zj5": ['dembow', 'latin music', 'reggaeton', 'trap latino', 'urbano latino'],
    "spotify:track:1clTca5X3tpBjNgfpsd3wu": ['latin music', 'latin reggae', 'latin urban', 'pop', 'reggae', 'reggaeton'],

    "spotify:track:6zQhJcyuZGX7ADNMZF1VHL": ['dancehall', 'latin music', 'latin urban', 'pop', 'reggaeton'],

    "spotify:track:2OKKeDxai9VJ5uW8zHLP3O": ['latin music', 'latin pop', 'pop'],

    "spotify:track:5ohqhIfOYly0ijyCos7Q6x": ['forro', 'pop'],

    "spotify:track:664N81BanO9zNURGi2iliG": ['latin music', 'latin pop', 'latin urban', 'pop', 'reggaeton'],

    "spotify:track:1if4Ro7Rr0ceUfjS8IOvEy": ['ballad', 'latin pop', 'pop'],
    "spotify:track:4C61EbxaKE4fX2qKSLsSFN": ['pop'],
    "spotify:track:2gcnNWEv9x0g6FhEeWtYuV": ['latin pop', 'salsa'],

    "spotify:track:4n8cY8SZgldxSbq9uJ05Tq": ['balada', 'latin music', 'pop'],
    "spotify:track:237wGvq8S48RC4uCUHUzo6": ['latin music', 'latin pop', 'rap'],
    "spotify:track:2megK9fKcCpg3Swni5irwX": ['balada', 'latin music', 'pop'],

    "spotify:track:1Ou4q1gX3QAvKhAFd3xsXT": ['gospel', 'r&b', 'soul'],
    "spotify:track:5mIqtDBiw3rqMxsJc4UVM6": ['pop', 'r&b'],

    "spotify:track:4F77lELTYwFvP3ESu8f2s4": ['latin music', 'pop', 'rock'],
    "spotify:track:56gsLSSdKzRZM0FHsISTNO": ['latin music', 'pop', 'rock'],

    "spotify:track:4nWZKEZc09aCT68d7cIoGT": ['motown', 'r&b', 'soul pop'],
    "spotify:track:08QvVJT8y6b0i2nH9mUeMj": ['r&b', 'soul pop'],

    # Mike Bahía – note: two different URIs for “Esta Noche”
    "spotify:track:7vFmnmcsp41uzBUiw1zxfw": ['bachata', 'latin music', 'latin pop', 'pop'],
    "spotify:track:4kitLAHbmBBXYHt9g22md3": ['dancehall', 'latin music', 'latin pop', 'latin urban'],
    "spotify:track:4vhkI8x68EaGQR3Kn7Pwhc": ['bachata', 'latin music', 'latin pop', 'pop'],
    "spotify:track:11ufNpNUBDwTGlU8sdWZaP": ['latin music', 'latin pop', 'reggaeton', 'pop'],

    "spotify:track:7z9chavxReUpyOZoAkUnUb": ['latin music', 'latin urban', 'latin trap', 'trap'],

    "spotify:track:4w5COSx5k82xKpkdJJFBjK": ['bachata', 'latin music', 'latin pop'],
    "spotify:track:6bASrkbE4U2NqxNSgl2Gez": ['latin music', 'latin pop', 'latin urban'],
    "spotify:track:6oX3bKrl5871KrltZqrcut": ['bachata', 'latin music', 'latin pop'],
    "spotify:track:1ZsJdeKS6c7zCXGPi4AXKw": ['latin music', 'r&b'],
    "spotify:track:1X1DsLPm00DUq0sSAreASF": ['bachata', 'latin music', 'latin pop'],

    "spotify:track:6WSAPEkvEfcEUgSTAo1D3S": ['latin music', 'latin pop', 'pop'],

    "spotify:track:7AyXdBQAgRLJBK72gqq0Hz": ['latin music', 'latin pop', 'latin urban', 'latin trap', 'rap', 'trap'],
    "spotify:track:12cUU5c82Lwo5lvsxdUXdQ": ['latin music', 'latin pop', 'latin urban', 'latin trap', 'rap', 'trap'],
    "spotify:track:3s49gzIl93cvAbtimjs2c2": ['latin music', 'latin pop', 'pop'],
    "spotify:track:3DJHqEPwg9F8DUOvc3iRAi": ['latin music', 'latin pop', 'latin urban', 'r&b'],
}

### Merge 
for uri, genre_list in track_specific_genres.items():
    mask = (
        (df["spotify_track_uri"] == uri) &
        df["genres"].isna()
    )
    df.loc[mask, "genres"] = str(genre_list)


### More assignments

artist_genres3 = {
    "Alejandro Fernandez": ['en espanol', 'pop'],
    "Anitta": ['brazil', 'pop', 'reggae', 'r&b'],
    "Anuel AA": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "ArcÃ¡ngel": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "Bad Bunny": ['latin music', 'latin urban', 'latin rap', 'rap', 'trap'],
    "Bryant Myers": ['latin music', 'latin urban', 'latin rap', 'rap', 'trap'],
    "Carlos Vives": ['latin music', 'latin pop', 'pop'],
    "Chaka Khan": ['soul', 'soul pop', 'r&b'],
    "Cristian Castro": ['en espanol', 'pop'],
    "Daddy Yankee": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "De La Ghetto": ['latin music', 'latin urban', 'reggaeton'],
    "Don Omar": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "Enrique Iglesias": ['pop'],
    "Farruko": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "Feid": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "Felipe PelÃ¡ez": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "Gusttavo Lima": ['pop', 'sertanejo'],
    "Jay Wheeler": ['latin music', 'latin pop', 'latin urban', 'pop'],
    "Kany GarcÃ­a": ['latin music', 'latin pop', 'pop'],
    "Marc Anthony": ['latin music', 'latin pop', 'pop', 'salsa'],
    "Myke Towers": ['latin music', 'latin pop', 'latin urban', 'reggaeton'],
    "Romeo Santos": ['bachata', 'latin music', 'latin pop'],
    "Sebastian Yatra": ['latin music', 'latin urban', 'pop', 'reggaeton'],
    "Sech": ['latin music', 'latin urban', 'pop', 'reggaeton'],
}


for artist, genre_list in artist_genres3.items():
    mask = (
        (df["master_metadata_album_artist_name"] == artist) &
        df["genres"].isna()
    )
    df.loc[mask, "genres"] = str(genre_list)


### Create new clean dataset that only contains information of interest

core_cols = [
    "master_metadata_track_name",
    "master_metadata_album_artist_name",
    "spotify_track_uri",
    "genres",
    "tempo",
    "energy",
    "key",
    "popularity",
    "mode",
    "time_signature",
    "speechiness",
    "danceability",
    "valence",
    "acousticness",
    "liveness",
    "instrumentalness",
    "loudness",
    "lyrics",
]

df = df.dropna(subset=core_cols).copy()

### fill in missing values  

fill_values = {
    "reason_start": "unknown",
    "reason_end": "unknown",
    "explicit": "unknown",             
}

### drop any remaining nulls

df = df.fillna(value=fill_values)


### drop album_release_date as it was difficult to find this data

df = df.drop(columns=["album_release_date"])

# Create a copy
df_clean = df.copy()

# Convert target to integer
df_clean['skipped'] = df_clean['skipped'].astype(int)


#### Feature Engineering
### The following transformations were performed:
### Part A.
### Extracted Hour, Day of Week, and Month from trimestamped
### Created time_of_day, is_weekend
### One-hot encoded time categories

### Part B. 
### Performed Standard Scaler to all numerical continuous variables
### Like tempo, energy, popularity, valence, acousticness, etc.

### Part C.
### Identified top 20 most frequent genres. Everything else labeled as "other"
### One hot encoded these top genres

### Part D.  
### Tokenized song lyrics
### Create custom stop words to remove text noise (slang, curse words, adlibs, vocalizations)


### Extracted features from timestamp data

# Convert timestamp to datetime object
df_clean['ts'] = pd.to_datetime(df_clean['ts'])

# 1. Extract Basic Components
df_clean['hour'] = df_clean['ts'].dt.hour
df_clean['day_of_week'] = df_clean['ts'].dt.dayofweek  # Monday=0, Sunday=6
df_clean['month'] = df_clean['ts'].dt.month

# 2. Create "Is Weekend" (Saturday or Sunday)
df_clean['is_weekend'] = df_clean['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# 3. Create "Time of Day" Bins (Reduces 24 hours down to 4 categories)
def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 22:
        return 'Evening'
    else:
        return 'Night'

df_clean['time_of_day'] = df_clean['hour'].apply(get_time_of_day)

# One-Hot Encode categorical time features
time_cols_to_encode = ['time_of_day'] 

X_time_dummies = pd.get_dummies(df_clean[time_cols_to_encode], drop_first=True)

### Reduce genres to top 20 and one hot encode

# Parse string list to actual list
df_clean['genres_list'] = df_clean['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# Identify Top 20 Genres
all_genres = [genre for sublist in df_clean['genres_list'] for genre in sublist]
top_20_genres = pd.Series(all_genres).value_counts().head(20).index.tolist()

# Create One-Hot columns for Top 20 only
for genre in top_20_genres:
    col_name = f"genre_{genre.replace(' ', '_')}"
    df_clean[col_name] = df_clean['genres_list'].apply(lambda x: 1 if genre in x else 0)

# Isolate these new genre columns
genre_cols = [c for c in df_clean.columns if c.startswith('genre_')]
X_genres = df_clean[genre_cols]

#### Data Exploration

# Set style for professional-looking charts
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# ============================================================
# Chart 1: Target Distribution (Skip vs. Listen)
# ============================================================

# Calculate counts and percentages
skip_counts = df_clean['skipped'].value_counts()
skip_percentages = df_clean['skipped'].value_counts(normalize=True) * 100

# Ensure consistent order: Listen (0) then Skip (1)
labels = ['Listen (0)', 'Skip (1)']
counts = [int(skip_counts.get(0, 0)), int(skip_counts.get(1, 0))]
pcts   = [float(skip_percentages.get(0, 0)), float(skip_percentages.get(1, 0))]

# Create single figure
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#2ecc71', '#e74c3c']  # Green for Listen, Red for Skip
bars = ax.bar(labels, pcts, color=colors, edgecolor='black', linewidth=1.5)

# Add percentage labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add sample size labels under x-axis ticks (no overlap)
# Place at y=0 in axes-fraction coords, then offset downward in points
for i, count in enumerate(counts):
    ax.annotate(
        f'n={count:,}',
        xy=(i, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -28), textcoords='offset points',
        ha='center', va='top',
        fontsize=10, style='italic', color='gray',
        clip_on=False
    )

ax.set_ylabel('Percentage of Tracks (%)', fontsize=12, fontweight='bold')
ax.set_title('Listen vs. Skip Rate', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, max(pcts) * 1.2)  # add headroom for % labels

# Give extra room at the bottom for the n=... labels
fig.subplots_adjust(bottom=0.22)
ax.tick_params(axis='x', pad=10)

plt.show()

# Print summary statistics
print("=" * 60)
print("TARGET DISTRIBUTION SUMMARY")
print("=" * 60)
print(f"Total Tracks: {len(df_clean):,}")
print(f"Listened (0): {counts[0]:,} ({pcts[0]:.2f}%)")
print(f"Skipped (1):  {counts[1]:,} ({pcts[1]:.2f}%)")
print(f"Skip Rate: {pcts[1]:.2f}%")
print(f"Class Imbalance Ratio: {counts[0] / counts[1]:.2f}:1 (Listen:Skip)")
print("=" * 60 + "\n")

# ============================================================
# Chart 2: Temporal Patterns (Skip Rate by Time of Day)
# ============================================================

# Calculate skip rates by time of day
# Ensure time_of_day is in the correct order
time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
time_skip_rates = df_clean.groupby('time_of_day')['skipped'].agg(['mean', 'count'])
time_skip_rates['percentage'] = time_skip_rates['mean'] * 100

# Reindex to ensure correct order
time_skip_rates = time_skip_rates.reindex(time_order)

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

# Create bar chart with gradient colors (darker = higher skip rate)
colors_gradient = plt.cm.Reds(np.linspace(0.4, 0.8, len(time_skip_rates)))
bars = ax.bar(time_skip_rates.index, 
               time_skip_rates['percentage'], 
               color=colors_gradient,
               edgecolor='black', 
               linewidth=1.5)

# Add percentage labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add sample size labels below x-axis
for i, (idx, row) in enumerate(time_skip_rates.iterrows()):
    ax.text(i, -2, f'n={int(row["count"]):,}',
            ha='center', va='top', fontsize=10, style='italic', color='gray')

# Formatting
ax.set_xlabel('Time of Day', fontsize=12, fontweight='bold')
ax.set_ylabel('Skip Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Skip Rate by Time of Day', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, time_skip_rates['percentage'].max() * 1.2)  # Add 20% headroom

# Add a horizontal line for overall skip rate
overall_skip_rate = df_clean['skipped'].mean() * 100
ax.axhline(y=overall_skip_rate, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Overall Skip Rate ({overall_skip_rate:.1f}%)')
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()

# Print detailed statistics
print("=" * 60)
print("TEMPORAL PATTERNS SUMMARY")
print("=" * 60)
print(f"Overall Skip Rate: {overall_skip_rate:.2f}%\n")
print("Skip Rate by Time of Day:")
print("-" * 60)
for time_period, row in time_skip_rates.iterrows():
    diff = row['percentage'] - overall_skip_rate
    direction = "↑" if diff > 0 else "↓"
    print(f"{time_period:12} | Skip Rate: {row['percentage']:5.2f}% | n={int(row['count']):6,} | {direction} {abs(diff):4.2f}% vs. avg")
print("=" * 60 + "\n")


# ============================================================
# Chart 3: Top 10 Listened Vs Skipped by Genre
# ============================================================

print("=" * 70)
print("GENRE ANALYSIS: TOP 10 LISTENED VS. TOP 10 SKIPPED")
print("=" * 70)

# Explode genre lists (since each track can have multiple genres)
genre_data = []
for idx, row in df_clean.iterrows():
    genres = row['genres_list']
    skipped = row['skipped']
    for genre in genres:
        genre_data.append({'genre': genre, 'skipped': skipped})

df_genres = pd.DataFrame(genre_data)

# Separate listened and skipped
df_listened_genres = df_genres[df_genres['skipped'] == 0]
df_skipped_genres = df_genres[df_genres['skipped'] == 1]

# Get top 10 for each
top_listened = df_listened_genres['genre'].value_counts().head(10)
top_skipped = df_skipped_genres['genre'].value_counts().head(10)

# Calculate percentages
total_listened_genres = len(df_listened_genres)
total_skipped_genres = len(df_skipped_genres)

top_listened_pct = (top_listened / total_listened_genres * 100).round(1)
top_skipped_pct = (top_skipped / total_skipped_genres * 100).round(1)

# Create side-by-side plot
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# --- LEFT: Top 10 Listened Genres ---
ax1 = axes[0]
bars1 = ax1.barh(range(len(top_listened)), top_listened_pct.values, color='#2ecc71', edgecolor='black', linewidth=1.2)

# Add percentage and count labels
for i, (pct, count) in enumerate(zip(top_listened_pct.values, top_listened.values)):
    ax1.text(pct + 0.5, i, f'{pct:.1f}% (n={count:,})', 
             va='center', fontsize=10, fontweight='bold')

ax1.set_yticks(range(len(top_listened)))
ax1.set_yticklabels(top_listened.index)
ax1.invert_yaxis()  # Highest at top
ax1.set_xlabel('Percentage of All Listened Genre Tags (%)', fontsize=11, fontweight='bold')
ax1.set_title('Top 10 Genres in Listened Songs', fontsize=13, fontweight='bold', pad=15, color='#27ae60')
ax1.set_xlim(0, max(top_listened_pct.values) * 1.25)

# --- RIGHT: Top 10 Skipped Genres ---
ax2 = axes[1]
bars2 = ax2.barh(range(len(top_skipped)), top_skipped_pct.values, color='#e74c3c', edgecolor='black', linewidth=1.2)

# Add percentage and count labels
for i, (pct, count) in enumerate(zip(top_skipped_pct.values, top_skipped.values)):
    ax2.text(pct + 0.5, i, f'{pct:.1f}% (n={count:,})', 
             va='center', fontsize=10, fontweight='bold')

ax2.set_yticks(range(len(top_skipped)))
ax2.set_yticklabels(top_skipped.index)
ax2.invert_yaxis()
ax2.set_xlabel('Percentage of All Skipped Genre Tags (%)', fontsize=11, fontweight='bold')
ax2.set_title('Top 10 Genres in Skipped Songs', fontsize=13, fontweight='bold', pad=15, color='#c0392b')
ax2.set_xlim(0, max(top_skipped_pct.values) * 1.25)

plt.suptitle('Listened vs. Skipped Genres', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# Print statistics
print("\nTOP 10 LISTENED GENRES:")
print("-" * 70)
for i, (genre, count, pct) in enumerate(zip(top_listened.index, top_listened.values, top_listened_pct.values), 1):
    print(f"{i:2}. {genre:25} | Count: {count:6,} | {pct:5.1f}%")

print("\nTOP 10 SKIPPED GENRES:")
print("-" * 70)
for i, (genre, count, pct) in enumerate(zip(top_skipped.index, top_skipped.values, top_skipped_pct.values), 1):
    print(f"{i:2}. {genre:25} | Count: {count:6,} | {pct:5.1f}%")

print(f"\nTotal Genre Tags (Listened): {total_listened_genres:,}")
print(f"Total Genre Tags (Skipped):  {total_skipped_genres:,}")
print("=" * 70 + "\n")


# ============================================================
# Chart 4: Correlation Matrix (Audio Features + Skipped)
# ============================================================

corr_cols = [
    "tempo", "energy", "popularity", "danceability", "valence",
    "acousticness", "liveness", "instrumentalness", "loudness", "speechiness",
    "key", "mode", "time_signature",
    "skipped"
]

corr_df = df_clean[corr_cols].copy()  # df_clean / skipped already exist in your script :contentReference[oaicite:0]{index=0}

# ensure numeric
for c in corr_cols:
    corr_df[c] = pd.to_numeric(corr_df[c], errors="coerce")

corr_df = corr_df.dropna()
corr_matrix = corr_df.corr(method="pearson")


mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(14, 12))
sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap="coolwarm",              # darker / more contrast than coolwarm
    vmin=-0.6, vmax=0.6,      # <- tighten range to make colors pop (adjust as needed)
    center=0,
    linewidths=0.5,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 9},
    cbar_kws={"shrink": 0.8}
)
plt.title("Correlation Matrix: Audio Features + Skipped", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


### Apply standard scaler to numerical features

# Feature Engineering: Audio (Scaling)
# ---------------------------------------
audio_cols = ['tempo', 'energy', 'popularity', 'danceability', 'valence', 
              'acousticness', 'liveness', 'instrumentalness', 'loudness', 'speechiness']

scaler = StandardScaler()
X_audio_scaled = scaler.fit_transform(df_clean[audio_cols])
X_audio_df = pd.DataFrame(X_audio_scaled, columns=audio_cols, index=df_clean.index)


#### NLP section

### Setup & Helper Functions

def remove_accents(input_str):
    """
    Normalizes a string to remove accents (e.g., 'música' -> 'musica').
    Crucial for matching standard stop words to your cleaned text.
    """
    if not isinstance(input_str, str): return input_str
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def fix_mojibake(text):
    """
    Fixes encoding artifacts common in Latin lyrics (e.g., 'Ã¡' -> 'á').
    """
    if not isinstance(text, str): return text
    try:
        return text.encode('latin-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text

# Apply cleaning steps to the raw dataframe
# 1. Fix encoding errors
df_clean['lyrics_clean'] = df_clean['lyrics'].apply(fix_mojibake)
# 2. Fill NaNs
df_clean['lyrics_clean'] = df_clean['lyrics_clean'].fillna('')

# ---------------------------------------------------------
### Build The Stop Word List

# A. Standard Lists (Processed to match your text)
english_stops = stopwords.words('english')
spanish_stops_raw = stopwords.words('spanish')
# Strip accents from the standard Spanish list so 'también' catches 'tambien'
spanish_stops_clean = [remove_accents(w) for w in spanish_stops_raw]

# B. Vocal Fillers (Reggaeton & Pop Ad-libs)
vocal_fillers = [
    'pa', 'yeah', 'oh', 'ah', 'eh', 'uh', 'ooh', 'ey', 'ay', 'mm', 'ayy', 
    'huh', 'na', 'da', 'ta', 'sa', 'dey', 'woah', 'hey', 
    'yeh', 'ra', 'vo', 'nah', 'brr' 
]

# C. Slang, Dialect & Contractions
slang = [
    'toy', 'gon', 'gonna', 'bout', 'tryna', 'ma', 'mami', 
    'vamo', 'ere', 'em', 'us','je'
]

# D. Generic Pop Verbs & Adverbs (High frequency, low meaning)
generic_fillers = [
    # Spanish Common Verbs/Grammar
    'hacer', 'mejor', 'ven', 'ando', 'pongo', 'puse', 'darte', 'quieren', 
    'mismo', 'mas', 'aqui', 'asi', 'tan', 'cada', 'nadie', 'conmigo', 
    'contigo', 'voy', 'dime', 'ahora', 'quiere', 'hace', 'ver', 'hoy', 
    'gusta', 'sabe', 'sabes', 'quieres', 'puedo', 'ser', 'dice', 'veo', 
    'vas', 'siento', 'paso', 'creo', 'aunque', 'vida', 'vez', 'tiempo', 
    'cama', 'dale', 'solo', 'siempre', 'bien', 'quiero', 'va', 
    'nunca', 'amor', 'noche', 'dia', 'sola', 'ves', 'sigo', 'dejo', 'pase', 'di', 'llegaron'
    
    # English Common Verbs/Grammar
    'nobody', 'would', 'every', 'new', 'made', 'show', 'try', 'around',
    'like', 'know', 'got', 'get', 'go', 'want', 'make', 'say', 'come',
    'tell', 'look', 'think', 'baby', 'bebe', 'wanna', 'gotta', 'let', 
    'back', 'one', 'cause', 'see', 'never', 'need', 'take', 'right', 
    'way', 'still', 'even', 'better', 'girl', 'man', 'time', 'life', 
    'feel', 'good', 'bad', 'real', 'really', 'said', 'keep', 'put', 
    'give', 'could', 'call', 'hit', 'turn', 'everything', 'much', 
    'boy', 'day', 'long', 'stop', 'thing', 'things', 'came', 'many', 'might', 'little', 'lil'
]

# E. Explicit Content
profanity = [
    'bitch', 'bitches', 'fuck', 'fucks',
    'shit', 'bullshit', 'ass', 'damn', 'hell'
]

# F. Numbers / Miscellaneous
numbers = ['21', 'dos', 'two', 'par', 'cuatro'] # 'par' means pair/couple but often used as filler

# COMBINE ALL LISTS
custom_stop_words = (
    english_stops + 
    spanish_stops_clean + 
    vocal_fillers + 
    slang + 
    generic_fillers + 
    profanity + 
    numbers
)

### Generate Features (TF-IDF) 

tfidf = TfidfVectorizer(
    stop_words=custom_stop_words, 
    max_features=100,      # Keeping top 100 most predictive words
    min_df=2,              # Ignore unique typos
    max_df=0.7,            # Ignore words that are EVERYWHERE
    strip_accents='unicode'
)

# Fit and transform
X_lyrics_sparse = tfidf.fit_transform(df_clean['lyrics_clean'])

# Convert to dataFrame
X_lyrics_df = pd.DataFrame(
    X_lyrics_sparse.toarray(), 
    columns=[f"lyric_{word}" for word in tfidf.get_feature_names_out()],
    index=df_clean.index
)

### Review Results

print("Final Feature Set (Top 50):")
print(X_lyrics_df.sum().sort_values(ascending=False).head(50))

#### Modeling
### Created split-test datasets (80% Train/ 20% Test) and created 6 distinct models: 
### Model 1 Elastic Net Logistic Regression (lyrics+audio)
### Model 2 XGBoost (lyrics+audio)
### Model 3 Elastic Net Logistic Regression (audio only)
### Model 4 XGBoost (audio only) 
### Model 5 LSTM RNN (lyrics+audio)
### Model 6 LSTM RNN (audio only)

# Combine all Feature Sets
# We use set_index for time_dummies just to be safe, though they should match
X_combined = pd.concat([
    X_audio_df,
    X_lyrics_df,
    X_time_dummies.set_index(df_clean.index),
    df_clean[['is_weekend']],
    X_genres
], axis=1)

# Define Target
y = df_clean['skipped'].astype(int)

# Split Data (80% Train, 20% Test)
# Stratify=y ensures we have the same ratio of skips in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)


# ======================================================
# Model 1: Elastic Net Logistic Regression (lyrics+audio)
# ======================================================

print("\n--- Starting Model 1: Elastic Net Tuning ---")

from sklearn.linear_model import LogisticRegressionCV

# Setup the Model
# solver='saga' is required to support Elastic Net penalties
model_cv = LogisticRegressionCV(
    cv=5,
    penalty='elasticnet', 
    solver='saga', 
    l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
    Cs=10,
    class_weight='balanced',
    scoring= 'roc_auc', 
    random_state=42,
    max_iter=5000,      
    n_jobs=-1
)

# Get the Best Model
model_cv.fit(X_train, y_train)

best_model_en = model_cv

### Evaluation & Interpretation

# Predict on Test Set
y_pred = model_cv.predict(X_test)

# Metrics
print("\nElastic Performance (Lyrics & Audio):")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance (Coefficients)
# Create a dataframe of features and their coefficients
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': best_model_en.coef_[0]
})

# Calculate absolute value to sort by "magnitude of impact"
coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Abs_Coef', ascending=False)

# Display Top 10 Predictors
print("\nTop 10 Most Influential Features:")
print(coef_df[['Feature', 'Coefficient']].head(15))

top_features = coef_df.head(15)

# Create a color mapping based on the coefficient sign
# Positive Coef (Skip) -> Red
# Negative Coef (Listen) -> Blue
colors = ['red' if x > 0 else 'blue' for x in top_features['Coefficient']]

plt.figure(figsize=(10, 6))
sns.barplot(
    x='Coefficient', 
    y='Feature', 
    data=top_features, 
    palette=colors # Manually applying the logic
)

plt.title('Feature Importance (Corrected Colors)')
plt.xlabel('Negative (Blue) = You KEEP this | Positive (Red) = You SKIP this')
plt.axvline(0, color='black', linestyle='--')
plt.show()

# ==========================================
# Model 2: XGBoost Model (lyrics+audio)
# ==========================================

print("--- Starting Model 2: XGBoost Classifier ---")

# Calculate Class Imbalance Ratio for 'scale_pos_weight'
# Formula: (Count of Listens) / (Count of Skips)
# This forces the model to pay attention to the minority class (Skips)
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
print(f"Calculated Imbalance Ratio: {ratio:.2f}")

# Setup the Model
# objective='binary:logistic': Standard for Yes/No problems
# eval_metric='logloss': Removes warning messages
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=ratio,  
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

# Define Hyperparameter Grid
# We tune the tree structure to find the sweet spot between "too simple" and "overfitting"
param_grid_xgb = {
    'n_estimators': [100, 200, 300, 400],      # How many trees to build
    'learning_rate': [0.01, 0.05, 0.1, 0.2],     # How fast the model learns
    'max_depth': [3, 4, 5, 6, 7, 8],          # How deep each tree can grow
    'min_child_weight': [1, 3, 5],          # Minimum samples per leaf node
    'subsample': [0.6, 0.7, 0.8, 1.0],          # % of data to sample for each tree
    'colsample_bytree': [0.4, 0.6, 0.8, 1.0]     
}

# Run Grid Search
print("Tuning XGBoost... (This may take a minute)")
random_search_xgb = RandomizedSearchCV(
    xgb_model, 
    param_distributions=param_grid_xgb, 
    n_iter=50,            # 50 random combinations
    cv=3,                 # 3-fold CV is faster for complex models
    scoring='roc_auc', 
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search_xgb.fit(X_train, y_train)

# Get Best Model
best_model_xgb = random_search_xgb.best_estimator_
print(f"Best Parameters: {random_search_xgb.best_params_}")

### Evaluation

# Predict
y_pred_xgb = best_model_xgb.predict(X_test)

# Metrics
print("\n--- XGBoost Performance (Lyrics & Audio) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

#### Visualization (Feature Importance)

### XGBoost importance is "Magnitude Only" (0 to 1). 
### Unlike Logistic Regression, it doesn't show Positive/Negative direction 
### in this specific chart, just "Importance" (how much it was used to split trees).

importances = best_model_xgb.feature_importances_
feature_names = X_train.columns

# Create DataFrame
xgb_feat_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(15)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=xgb_feat_df, palette='viridis')
plt.title('Top 15 Most Important Features (XGBoost w/ Lyrics)')
plt.xlabel('Importance Score (0-1)')
plt.show()



# ====================================================
# Model 3: Elastic Net Logistic Regression (audio only)
# ====================================================

# We take everything EXCEPT X_lyrics_df
X_no_lyrics = pd.concat([
    X_audio_df, 
    X_time_dummies.set_index(df_clean.index),
    df_clean[['is_weekend']],
    X_genres
], axis=1)

y = df_clean['skipped'].astype(int)

# Split (Must use same random_state=42 to match previous models)
X_train_nl, X_test_nl, y_train_nl, y_test_nl = train_test_split(
    X_no_lyrics, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Feature Set Created: {X_train_nl.shape[1]} columns (Audio + Time + Genre)")


print("\nTraining Elastic Net (No Lyrics)...")

model_en_nl = LogisticRegression(
    penalty='elasticnet', solver='saga', max_iter=5000,
    class_weight='balanced', random_state=42
)

param_grid_en = {'l1_ratio': [0.1, 0.5, 0.9], 'C': [0.1, 1, 10]}

grid_en_nl = GridSearchCV(model_en_nl, param_grid_en, cv=5, scoring='accuracy', n_jobs=-1)
grid_en_nl.fit(X_train_nl, y_train_nl)

y_pred_en_nl = grid_en_nl.predict(X_test_nl)

# Print summary table
print("\n--- Elastic Net (No Lyrics) Performance ---")
print(f"Best Params: {grid_en_nl.best_params_}")
print(f"Accuracy: {accuracy_score(y_test_nl, y_pred_en_nl):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_nl, y_pred_en_nl))

### feature importance

# Extract the Best Model and Coefficients
best_en_model = grid_en_nl.best_estimator_
coefs = best_en_model.coef_[0]
feature_names = X_train_nl.columns

# Create a DataFrame
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefs
})

# Calculate Absolute Value for Sorting (Impact Magnitude)
coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()

# Sort and Take Top 15
top_features = coef_df.sort_values(by='Abs_Coef', ascending=False).head(15)

# Create Color Logic: Red for Skip (Positive), Blue for Listen (Negative)
colors = ['red' if x > 0 else 'blue' for x in top_features['Coefficient']]

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(
    x='Coefficient', 
    y='Feature', 
    data=top_features, 
    palette=colors
)

plt.title('Top 15 Drivers of Taste (Elastic Net - No Lyrics)\nBlue = Likely to Listen | Red = Likely to Skip', fontsize=16)
plt.xlabel('Coefficient Magnitude', fontsize=12)
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.show()

# 7. Print the list for your report
print("\nTop 15 Most Influential Features (Linear):")
print(top_features[['Feature', 'Coefficient']])


# ==========================================
# Model 4: XGBoost Model (audio only)
# ==========================================

# Calculate Ratio again
ratio = float(np.sum(y_train_nl == 0)) / np.sum(y_train_nl == 1)

xgb_model_nl = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=ratio,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Using same params as before for fair comparison
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1]
}

grid_xgb_nl = GridSearchCV(xgb_model_nl, param_grid_xgb, cv=3, scoring='accuracy', n_jobs=-1)
grid_xgb_nl.fit(X_train_nl, y_train_nl)

y_pred_xgb_nl = grid_xgb_nl.predict(X_test_nl)

### Results & Comparison

# Print summary table
print("\n--- XGBoost (No Lyrics) Performance ---")
print(f"Best Params: {grid_xgb_nl.best_params_}")
print(f"Accuracy: {accuracy_score(y_test_nl, y_pred_xgb_nl):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_nl, y_pred_xgb_nl))

### Feature Importance

best_xgb_model = grid_xgb_nl.best_estimator_
importances = best_xgb_model.feature_importances_
feature_names = X_train_nl.columns

# Create a DataFrame
xgb_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort by Importance (Highest to Lowest)
top_features_xgb = xgb_imp_df.sort_values(by='Importance', ascending=False).head(15)

### Plot
plt.figure(figsize=(12, 8))
sns.barplot(
    x='Importance', 
    y='Feature', 
    data=top_features_xgb, 
    palette='viridis' # Green/Purple palette distinct from the Red/Blue linear one
)

plt.title('Top 15 Most Important Features (XGBoost - No Lyrics)', fontsize=16)
plt.xlabel('Importance Score (0 - 1)', fontsize=12)
plt.show()

# Print the list for report
print("\nTop 15 Most Influential Features (Non-Linear):")
print(top_features_xgb[['Feature', 'Importance']])

# ------------------------------------------------
#### LSTM RNN Models 

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Global Random Seed set to: {seed}")

set_seeds(42)

### LSTM Preparation (New DataFrame)

# Create separate dataframe strictly for LSTM 
df_lstm = df_clean.copy()

# Scale audio features
df_lstm[audio_cols] = scaler.fit_transform(df_clean[audio_cols])

# Sorting data by time so LSTM model can learn from the order of events

# Sort by Time (Required for Sequence Models)
df_lstm = df_lstm.sort_values('ts')

# Calculate Time Gaps
# This looks at where listening sessions start and end
df_lstm['prev_ts'] = df_lstm['ts'].shift(1)
df_lstm['diff_minutes'] = (df_lstm['ts'] - df_lstm['prev_ts']).dt.total_seconds() / 60

# Generate Session IDs
# Logic: If gap > 20 mins, it's a new session.
SESSION_THRESHOLD = 20  # minutes

# This creates a unique ID for every listening session
df_lstm['is_new_session'] = (df_lstm['diff_minutes'] > SESSION_THRESHOLD) | (df_lstm['diff_minutes'].isna())
df_lstm['session_id'] = df_lstm['is_new_session'].astype(int).cumsum()

# Filter Short Sessions
### We remove sessions with < 3 songs because they are too short to learn a sequence.
session_counts = df_lstm['session_id'].value_counts()
long_sessions = session_counts[session_counts >= 3].index

df_lstm = df_lstm[df_lstm['session_id'].isin(long_sessions)].copy()

# Create primary_genre column for visualization
genre_cols = [c for c in df_lstm.columns if c.startswith('genre_')]

def get_primary_genre(row):
    for col in genre_cols:
        if row[col] == 1:
            return col.replace('genre_', '').replace('_', ' ').title()
    return 'Other'

df_lstm['primary_genre'] = df_lstm.apply(get_primary_genre, axis=1)
print(f"Primary genres found: {df_lstm['primary_genre'].nunique()} unique genres")

### Summary of LSTM dataset

print(f"Base Dataset (df_clean): {len(df_clean)} rows")
print(f"LSTM Dataset (df_lstm): {len(df_lstm)} rows (Short sessions removed)")
print(f"Unique Sessions available for LSTM: {df_lstm['session_id'].nunique()}")

### Create function to visualize high-confidence predictions with genre sequences

def visualize_with_genres(model, X_test, y_test, genre_sequences, n_examples, model_name, feature_cols_source):
    probs = model.predict(X_test, verbose=0)
    high_conf_indices = np.where((probs > 0.85) & (y_test.reshape(-1, 1) == 1))[0]
    
    # Sort by confidence (highest first)
    high_conf_probs = probs[high_conf_indices].flatten()
    sorted_order = np.argsort(high_conf_probs)[::-1]
    high_conf_indices = high_conf_indices[sorted_order]
    
    numeric_features = ['popularity', 'loudness', 'speechiness', 'valence']
    feature_indices = [feature_cols_source.index(f) for f in numeric_features if f in feature_cols_source]
    
    print(f"Found {len(high_conf_indices)} high-confidence skip predictions")
    
    for i in range(min(n_examples, len(high_conf_indices))):
        idx = high_conf_indices[i]
        sequence_numeric = X_test[idx][:, feature_indices]
        confidence = probs[idx][0]
        genre_seq = genre_sequences[idx]
        
        create_individual_heatmap(
            sequence_numeric,
            numeric_features,
            confidence,
            idx,
            model_name,
            genre_sequence=genre_seq
        )

def create_sequences(df, session_col, feature_cols, target_col, n_steps=2):
    """
    Transforms a 2D DataFrame into 3D arrays for LSTM.
    
    Returns:
    X (numpy array): Shape (Samples, n_steps, n_features)
    y (numpy array): Shape (Samples,)
    """
    X, y = [], []
    
    # Group by session to ensure we don't mix data from different sessions
    grouped = df.groupby(session_col)
    
    for _, group in grouped:
        data = group[feature_cols].values
        target = group[target_col].values
        
        # If the session is too short for the window, skip it
        if len(data) < n_steps + 1:
            continue
            
        # Sliding Window Logic
        for i in range(len(data) - n_steps):
            # Gather past 'n_steps' songs as input
            X.append(data[i:(i + n_steps)])
            # The target is the skip status of the NEXT song
            y.append(target[i + n_steps])
            
    return np.array(X), np.array(y)

print("Fixing data types for Neural Network...")

# Fix 'explicit' column
df_lstm['explicit'] = df_lstm['explicit'].astype(str).replace(
    {'True': 1, 'False': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
)
df_lstm['explicit'] = df_lstm['explicit'].fillna(0).astype(int)

# Fix Boolean columns
bool_cols = df_lstm.select_dtypes(include=['bool']).columns
for col in bool_cols:
    df_lstm[col] = df_lstm[col].astype(int)



####  Setup features for next model 

# Create the TWO Dataframes
# Branch A: The Lyrics Dataframe
# We use 'inner' join to ensure we match the rows correctly with the NLP features from before
df_lstm_lyrics = df_lstm.join(X_lyrics_df, how='inner')

# Branch B: The Audio-Only Dataframe
df_lstm_audio = df_lstm.copy()

print(f"Dataset 1 (Lyrics): {df_lstm_lyrics.shape}")
print(f"Dataset 2 (Audio): {df_lstm_audio.shape}")

# ==========================================
# LSTM RNN Model 5 (lyrics + audio features)
# ==========================================
print("\n--- Starting Model 5: LSTM (Audio + Lyrics) ---")

# 1. Define Features
# Select numeric columns (this includes audio, metadata, AND 'lyric_...' columns)
all_cols = df_lstm_lyrics.select_dtypes(include=['number']).columns.tolist()

# Drop technical columns
### dropped skipped since it's our target variable
### remove other variables since they won't provide helpful info to the model
cols_to_drop = ['skipped', 'session_id', 'is_new_session', 'ms_played', 'diff_minutes', 'Unnamed: 0']
feature_cols_5 = [c for c in all_cols if c not in cols_to_drop]

# 2. Create Sequences (Window = 5 for Fatigue Detection)
X_seq_5, y_seq_5 = create_sequences(
        df_lstm_lyrics, 
    'session_id', 
    feature_cols_5, 
    'skipped', 
    n_steps=5  # Using 5 previous songs to predict the next skip
)

def get_genre_sequences(df, session_col, n_steps=5):
    """Extract genre sequences matching the X sequences"""
    genre_sequences = []
    grouped = df.groupby(session_col)
    
    for _, group in grouped:
        genres = group['primary_genre'].values
        if len(genres) < n_steps + 1:
            continue
        for i in range(len(genres) - n_steps):
            genre_sequences.append(list(genres[i:(i + n_steps)]))
    
    return genre_sequences

# Get genre sequences for Model 5
genre_sequences_5 = get_genre_sequences(df_lstm_lyrics, 'session_id', n_steps=5)

# 3. Time-Based Split (80/20) 
### We train on the past (first 80%) and test on the future (last 20%)
train_size_5 = int(len(X_seq_5) * 0.8)
genre_test_5 = genre_sequences_5[train_size_5:]
X_train_5, X_test_5 = X_seq_5[:train_size_5], X_seq_5[train_size_5:]
y_train_5, y_test_5 = y_seq_5[:train_size_5], y_seq_5[train_size_5:]

# 4. Build & Train
model_5 = Sequential()

model_5.add(Input(shape=(X_train_5.shape[1], X_train_5.shape[2]))) # Input Layer: Matches the shape of our sequences
model_5.add(LSTM(128, return_sequences=False)) # Larger units for complex text data
model_5.add(Dropout(0.4)) # Higher dropout to prevent memorizing lyrics
model_5.add(Dense(1, activation='sigmoid')) # squeeze to 0-1 output

model_5.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_5 = model_5.fit(
    X_train_5, y_train_5, 
    epochs=20, batch_size=32, 
    validation_data=(X_test_5, y_test_5), 
    verbose=1
)

# ==========================================
# MODEL 6: LSTM (Audio Only)
# ==========================================

print("\n--- Starting Model 6: LSTM (Audio Only) ---")

# Define Features
# Select numeric columns from the Audio dataframe (which HAS NO LYRICS)
all_cols_audio = df_lstm_audio.select_dtypes(include=['number']).columns.tolist()

# Drop technical columns
feature_cols_6 = [c for c in all_cols_audio if c not in cols_to_drop]

# Double check: Ensure no 'lyric_' columns snuck in
feature_cols_6 = [c for c in feature_cols_6 if not c.startswith('lyric_')]

print(f"Model 6 Feature Count: {len(feature_cols_6)}")

# Create Sequences (Window = 5)
X_seq_6, y_seq_6 = create_sequences(
    df_lstm_audio, 
    'session_id', 
    feature_cols_6, 
    'skipped', 
    n_steps=5 
)

# For Model 6 (Audio Only)
genre_sequences_6 = get_genre_sequences(df_lstm_audio, 'session_id', n_steps=5)

# Time-Based Split (80/20)
train_size_6 = int(len(X_seq_6) * 0.8)
genre_test_6 = genre_sequences_6[train_size_6:]
X_train_6, X_test_6 = X_seq_6[:train_size_6], X_seq_6[train_size_6:]
y_train_6, y_test_6 = y_seq_6[:train_size_6], y_seq_6[train_size_6:]

# Build & Train
model_6 = Sequential()

# We use 64 units because audio data is less complex compared to dataset with lyrics
model_6.add(Input(shape=(X_train_6.shape[1], X_train_6.shape[2])))
model_6.add(LSTM(64, return_sequences=False)) # Smaller units (less noise)
model_6.add(Dropout(0.3)) 
model_6.add(Dense(1, activation='sigmoid'))

model_6.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_6 = model_6.fit(
    X_train_6, y_train_6, 
    epochs=20, batch_size=32, 
    validation_data=(X_test_6, y_test_6), 
    verbose=1
)

# Get predictions
def decode_patterns_robust(model, X_test, y_test, feature_cols, model_name="LSTM"):
    print(f"\n--- DECODING PREDICTIVE PATTERNS ({model_name}) ---") 

    probs = model.predict(X_test, verbose=0)

    # 2. Find High-Confidence "SKIP" predictions (Prob > 0.85)
    # We look for cases where the model was SURE you'd skip, and you DID skip.
    high_conf_indices = np.where((probs > 0.85) & (y_test.reshape(-1, 1) == 1))[0]


    print(f"Found {len(high_conf_indices)} high-confidence correct skip predictions.\n")
    
    if len(high_conf_indices) > 0:
        # Look at the first 3 examples
        for idx in high_conf_indices[:3]:
            print(f"Example #{idx}")
            print(f"Model Confidence: {probs[idx][0]:.2%}")
            
            # Get the sequence history
            sequence = X_test[idx] # Shape (n_steps, Features)
            
            # --- CRITICAL FIX: Look at the LAST 2 songs in the sequence ---
            # These are the ones representing the immediate transition
            steps_to_show = [-2, -1] # Second to last, and Last song
            labels = ["Song A (Previous)", "Song B (Current/Transition)"]
            
            for step_idx, label in zip(steps_to_show, labels):
                print(f"  {label}:")
                
                # Define which features you specifically want to inspect
                # Make sure these match the columns actually in your dataframe
                key_feats = [
                    'primary_genre', 'popularity', 'loudness', 
                    'speechiness', 'valence'
                ]
                
                for f in key_feats:
                    if f in feature_cols:
                        feat_idx = feature_cols.index(f)
                        val = sequence[step_idx][feat_idx]
                        print(f"    - {f}: {val:.2f}")
                        
            print("  -> PREDICTION: SKIP (Correct)\n")
    else:
        print("No predictions met the 85% confidence threshold. Try lowering it to 0.70.")

# --- Run it ---
# Pass the correct feature columns for each model
decode_patterns_robust(model_5, X_test_5, y_test_5, feature_cols_5, "Model 5 (Lyrics)")
decode_patterns_robust(model_6, X_test_6, y_test_6, feature_cols_6, "Model 6 (Audio)")

### Visualizing Listening Patterns 


def create_individual_heatmap(sequence_data, feature_names, model_confidence, 
                               example_num, model_name, genre_sequence=None):
    """
    Creates a single heatmap table for one sequence, with genre as text row.
    
    Parameters:
    -----------
    sequence_data : numpy array
        Shape (n_steps, n_features) - numeric features only
    feature_names : list
        Names of the numeric features (excluding genre)
    model_confidence : float
        Model's confidence score (0-1)
    example_num : int
        The example number from the output
    model_name : str
        Name of the model
    genre_sequence : list, optional
        List of genre names for each song in the sequence
    """
    
    n_steps = sequence_data.shape[0]
    n_features = len(feature_names)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 2.5 + n_features * 0.6))
    ax.axis('tight')
    ax.axis('off')
    
    # Add title
    title = f"Example #{example_num} - {model_name}\nModel Confidence: {model_confidence:.2%}"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Prepare column headers
    col_labels = ['Feature'] + [f'Song {i+1}' for i in range(n_steps-1)] + ['Song 5 (SKIP) ⏭']
    
    # Prepare table data
    table_data = []
    cell_colors = []
    
    # Add genre row first (if provided)
    if genre_sequence is not None:
        genre_row = ['Genre'] + genre_sequence
        table_data.append(genre_row)
        # Genre row gets neutral gray coloring
        genre_colors = ['white'] + ['#F0F0F0'] * n_steps
        cell_colors.append(genre_colors)
    
    # Add numeric feature rows
    for feat_idx, feat_name in enumerate(feature_names):
        row = [feat_name.replace('_', ' ').title()]
        row_colors = ['white']  # Feature name column is white
        
        for step in range(n_steps):
            value = sequence_data[step, feat_idx]
            row.append(f'{value:.2f}')
            
            # Normalize value for color
            row_values = sequence_data[:, feat_idx]
            val_min, val_max = row_values.min(), row_values.max()
            
            if val_max - val_min > 0:
                normalized = (value - val_min) / (val_max - val_min)
            else:
                normalized = 0.5
            
            # Color intensity: darker blue = higher value
            intensity = normalized
            color = plt.cm.Blues(0.3 + intensity * 0.6)
            row_colors.append(color)
        
        table_data.append(row)
        cell_colors.append(row_colors)
    
    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center',
                     cellColours=cell_colors)
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(len(col_labels)):
        cell = table[(0, i)]
        cell.set_facecolor('#E8E8E8')
        cell.set_text_props(weight='bold', fontsize=11)
        cell.set_height(0.15)
    
    # Style feature name column (first column)
    for i in range(1, len(table_data) + 1):
        cell = table[(i, 0)]
        cell.set_text_props(weight='bold', ha='left', fontsize=11)
        cell.set_width(0.15)
    
    # Add note at bottom
    note_text = "Shows feature intensities across the listening sequence. Darker = higher values."
    fig.text(0.5, 0.02, note_text, ha='center', fontsize=9, 
             style='italic', color='gray')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08)
    plt.show()


### Visualize first 3 examples


# Model 5 (Audio + Lyrics) - First 3 high-confidence examples
print("=" * 60)
print("MODEL 5: AUDIO + LYRICS")
print("=" * 60)

visualize_with_genres(
    model_5, X_test_5, y_test_5, genre_test_5, 
    n_examples=3, 
    model_name="Model 5 (Audio + Lyrics)",
    feature_cols_source=feature_cols_5
)
# Model 6 (Audio Only) - First 3 high-confidence examples
print("=" * 60)
print("MODEL 6: AUDIO ONLY")
print("=" * 60)

visualize_with_genres(
    model_6, X_test_6, y_test_6, genre_test_6, 
    n_examples=3, 
    model_name="Model 6 (Audio Only)",
    feature_cols_source=feature_cols_6
)


# =========================================================
# Comparison between all models
# =========================================================


# 1. Compile Data from All 6 Models
# These values are taken directly from your classification reports.

data = {
    'Model': [
        # --- Models with Lyrics ---
        'Elastic Net\n(Lyrics)', 
        'XGBoost\n(Lyrics)', 
        'LSTM\n(Lyrics)', 

        # --- Models with Audio Only ---
        'Elastic Net\n(Audio)', 
        'XGBoost\n(Audio)', 
        'LSTM\n(Audio)'
    ],
    
    'Type': [
        'Linear (Snapshot)', 'Non-Linear (Snapshot)', 'Deep Learning (Sequence)',
        'Linear (Snapshot)', 'Non-Linear (Snapshot)', 'Deep Learning (Sequence)'
    ],

    # 1. Accuracy: Overall correctness
    'Accuracy': [
        0.6108, 0.6706, 0.40,  # Lyrics Models (EN, XGB, LSTM)
        0.5843, 0.6730, 0.46   # Audio Models (EN, XGB, LSTM)
    ],
    
    # 2. Recall (Class 1 - Skips): "How many of the actual skips did we catch?"
    # High Recall = Good for cleaning up a playlist (catches all bad songs)
    'Recall': [
        0.63, 0.72, 0.32, 
        0.64, 0.70, 0.40
    ], 
    
    # 3. Precision (Class 1 - Skips): "When we predicted a skip, were we right?"
    # High Precision = Trustworthy. The LSTM dominates here (0.75).
    'Precision': [
        0.48, 0.54, 0.69, 
        0.45, 0.54, 0.75
    ], 
    
    # 4. F1-Score (Class 1): The Harmonic Mean (Balance)
    # This is the "Fair" metric for imbalanced data.
    'F1-Score': [
        0.54, 0.62, 0.44, 
        0.53, 0.61, 0.52
    ] 
}

df_scores = pd.DataFrame(data)

# 2. Melt for Plotting (Convert to Long Format)
# This allows us to plot multiple metrics on the same chart using 'hue'
df_melted = df_scores.melt(
    id_vars=['Model', 'Type'], 
    value_vars=['Accuracy', 'Recall', 'Precision', 'F1-Score'],
    var_name='Metric', 
    value_name='Score'
)

# 3. Generate the Comparison Visualization
plt.figure(figsize=(16, 8))
sns.set_style("whitegrid")

# We use a grouped bar chart
ax = sns.barplot(
    x='Metric', 
    y='Score', 
    hue='Model', 
    data=df_melted, 
    palette='Paired' # Paired palette helps group Lyrics/Audio pairs visually
)

# 4. Formatting the Chart
plt.title('Performance Across XGBoost, ElasticNet & Sequence Models (LSTM)', fontsize=18, pad=20)
plt.xlabel('Evaluation Metric', fontsize=12)
plt.ylabel('Score (0.0 - 1.0)', fontsize=12)
plt.ylim(0, 0.90) # Set limit slightly higher than max score
plt.legend(title='Model Architecture', bbox_to_anchor=(1.01, 1), loc='upper left')

# 5. Add Value Labels on Bars
for p in ax.patches:
    if p.get_height() > 0: 
        # Only label if the bar is visible
        ax.annotate(
            format(p.get_height(), '.2f'), 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha = 'center', va = 'center', 
            xytext = (0, 9), 
            textcoords = 'offset points',
            fontsize=9, fontweight='bold'
        )

plt.tight_layout()
plt.show()

# 6. Print the Raw Table for Report
print("\n--- Final Project Summary Table ---")
# Reorder columns to put Type first for readability
cols = ['Model', 'Type', 'Accuracy', 'Recall', 'Precision', 'F1-Score']
print(df_scores[cols])

