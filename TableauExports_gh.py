

#### Objective
# Purpose of this file is to create and export the dataframes needed to create visualizations for the Tableau dashboard.
# The exports will be used to some of the following visualizations:
# 1. KPI cards (total skips, total listens, skip rate, etc.)
# 2. Top artists, songs, and genres tables (wide format with skip/listen counts)
# 3. word cloud data (top words driving skips vs listens, with importance scores)
# 4. Skip rate heatmap (day of week × hour)
# 5. Correlation Matrix
# 6. Choose Your Vibe (song recommendation engine based on audio features)


# Import

import os
import re
import ast
import numpy as np
import pandas as pd
from itertools import product
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt


# Upload data

CSV_PATH = "C:\\Users\\urios\\df_clean_spotify.csv"   # ← update path as needed

df_clean = pd.read_csv(CSV_PATH, low_memory=False)

# Ensure timestamp is parsed as datetime
df_clean['ts'] = pd.to_datetime(df_clean['ts'])

# Ensure skipped is integer
df_clean['skipped'] = df_clean['skipped'].astype(int)

print(f"df_clean loaded: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")


# =============================================================================
# Feature Engineering
# =============================================================================

# Temporal features 

df_clean['hour']        = df_clean['ts'].dt.hour
df_clean['day_of_week'] = df_clean['ts'].dt.dayofweek   # Monday = 0
df_clean['month']       = df_clean['ts'].dt.month
df_clean['is_weekend']  = df_clean['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

def get_time_of_day(hour):
    if   5  <= hour < 12: return 'Morning'
    elif 12 <= hour < 17: return 'Afternoon'
    elif 17 <= hour < 22: return 'Evening'
    else:                 return 'Night'

df_clean['time_of_day'] = df_clean['hour'].apply(get_time_of_day)

# Genre list column 

def parse_genre_list(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    try:
        return [g.strip() for g in ast.literal_eval(x)]
    except Exception:
        return [str(x).strip()]

df_clean['genres_list'] = df_clean['genres'].apply(parse_genre_list)

# Genre one-hot columns (top 20) 

all_genres   = [g for sublist in df_clean['genres_list'] for g in sublist]
top_20_genres = pd.Series(all_genres).value_counts().head(20).index.tolist()

for genre in top_20_genres:
    col = f"genre_{genre.replace(' ', '_')}"
    df_clean[col] = df_clean['genres_list'].apply(lambda x: 1 if genre in x else 0)

genre_cols = [c for c in df_clean.columns if c.startswith('genre_')]

print("Shared feature engineering complete.")
print(f"  Temporal columns : hour, day_of_week, month, is_weekend, time_of_day")
print(f"  Genre OHE columns: {len(genre_cols)} columns")


# =============================================================================
# Skip Heatmap Export
# =============================================================================


day_labels = {0: 'Mon', 1: 'Tues', 2: 'Wed', 3: 'Thurs',
              4: 'Fri', 5: 'Sat',  6: 'Sun'}

df_heatmap = (
    df_clean
    .groupby(['day_of_week', 'hour'], as_index=False)
    .agg(
        Total_Skips=('skipped', 'sum'),
        Total_Plays=('skipped', 'count')
    )
)

df_heatmap['Skip_Rate']  = (df_heatmap['Total_Skips'] / df_heatmap['Total_Plays']).round(4)
df_heatmap['Day_Label']  = df_heatmap['day_of_week'].map(day_labels)

export_path_heatmap = 'skips_heatmap_for_tableau_v2.xlsx'
df_heatmap.to_excel(export_path_heatmap, index=False)

print(f"\n[Heatmap] Exported {len(df_heatmap)} rows → '{export_path_heatmap}'")
print(df_heatmap.head(10))

# =============================================================================
# Decision Tree
# =============================================================================

# This was originally intended to create a Sankey flow visual for Tableau.
# I ultimately, decided not to include it because of difficulty  in interpretation.
# It was still useful to me in identifying which features to include in the "Pick Your Vibe" recommendation engine.

df_sankey = df_clean.copy()

# Define genre priority order for dominant-genre assignment
genre_priority_sankey = [
    'genre_reggaeton',
    'genre_urbano_latino',
    'genre_trap_latino',
    'genre_rap',
    'genre_hip_hop',
    'genre_latin_pop',
    'genre_pop',
    'genre_dance_pop',
    'genre_pop_rap',
    'genre_reggaeton_colombiano',
    'genre_trap',
    'genre_puerto_rican_pop',
    'genre_trap_boricua',
    'genre_latin_hip_hop',
    'genre_trap_triste',
    'genre_urban_contemporary',
    'genre_colombian_pop',
    'genre_reggaeton_flow',
    'genre_afrobeats'
]

def assign_dominant_genre(row):
    for genre in genre_priority_sankey:
        if genre in row.index and row[genre] == 1:
            return genre.replace('genre_', '').replace('_', ' ').title()
    return 'Other'

df_sankey['dominant_genre'] = df_sankey.apply(assign_dominant_genre, axis=1)

print("\n=== Dominant Genre Distribution ===")
print(df_sankey['dominant_genre'].value_counts())
print(f"Rows with no genre assigned (Other): {(df_sankey['dominant_genre'] == 'Other').sum()}")

# Genre consolidation mapping 

genre_consolidation = {
    'Rap':               'Rap/Hip Hop',
    'Hip Hop':           'Rap/Hip Hop',
    'Trap':              'Rap/Hip Hop',
    'Urban Contemporary':'Rap/Hip Hop',
    'Trap Latino':       'Urbano Latino',
    'Latin Hip Hop':     'Urbano Latino',
    'Trap Triste':       'Urbano Latino',
    'Colombian Pop':     'Latin Pop',
    'Pop Rap':           'Pop',
    'Dance Pop':         'Pop'
}

df_sankey['dominant_genre'] = df_sankey['dominant_genre'].replace(genre_consolidation)

print("\n=== Consolidated Genre Distribution ===")
print(df_sankey['dominant_genre'].value_counts())
print(f"Total unique genres: {df_sankey['dominant_genre'].nunique()}")

# Decision Tree V2 (without popularity)

sankey_features_v2 = [
    'dominant_genre', 'energy', 'danceability', 'valence', 'tempo',
    'acousticness', 'speechiness', 'loudness', 'hour', 'is_weekend', 'time_of_day'
]

df_tree_v2 = df_sankey[sankey_features_v2 + ['skipped']].dropna().copy()

le_genre = LabelEncoder()
le_time  = LabelEncoder()

df_tree_v2['dominant_genre_encoded'] = le_genre.fit_transform(df_tree_v2['dominant_genre'])
df_tree_v2['time_of_day_encoded']    = le_time.fit_transform(df_tree_v2['time_of_day'])

feature_cols_v2 = [
    'dominant_genre_encoded', 'energy', 'danceability', 'valence', 'tempo',
    'acousticness', 'speechiness', 'loudness', 'hour', 'is_weekend', 'time_of_day_encoded'
]

X_tree_v2 = df_tree_v2[feature_cols_v2]
y_tree_v2 = df_tree_v2['skipped']

dt_v2 = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
dt_v2.fit(X_tree_v2, y_tree_v2)

# Feature importance report
importance_df_v2 = pd.DataFrame({
    'Feature':    feature_cols_v2,
    'Importance': dt_v2.feature_importances_
}).sort_values('Importance', ascending=False)

importance_df_v2['Feature'] = importance_df_v2['Feature'].replace({
    'dominant_genre_encoded': 'dominant_genre',
    'time_of_day_encoded':    'time_of_day'
})

print("\n=== Feature Importances V2 (No Popularity) ===")
print(importance_df_v2.to_string(index=False))

feature_names_v2 = [
    'dominant_genre', 'energy', 'danceability', 'valence', 'tempo',
    'acousticness', 'speechiness', 'loudness', 'hour', 'is_weekend', 'time_of_day'
]

print("\n=== Decision Tree V2 Split Rules ===")
print(export_text(dt_v2, feature_names=feature_names_v2))

# Save decision tree diagram
plt.figure(figsize=(24, 10))
plot_tree(
    dt_v2,
    feature_names=feature_names_v2,
    class_names=['Listen', 'Skip'],
    filled=True,
    rounded=True,
    fontsize=10,
    impurity=True,
    proportion=False
)
plt.title('Decision Tree V2: Predicting Skip vs Listen (No Popularity)', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('decision_tree_v2_sankey_v3.png', dpi=150, bbox_inches='tight')
plt.show()
print("Decision tree V2 saved to 'decision_tree_v2_sankey_v3.png'")

# =============================================================================
# Song Mood/Pick Your Vibe Export
# =============================================================================


# Genre priority list (same order used throughout capstone)
genre_priority_vibe = [
    'genre_reggaeton', 'genre_urbano_latino', 'genre_trap_latino',
    'genre_rap', 'genre_hip_hop', 'genre_latin_pop', 'genre_pop',
    'genre_dance_pop', 'genre_pop_rap', 'genre_reggaeton_colombiano',
    'genre_trap', 'genre_puerto_rican_pop', 'genre_trap_boricua',
    'genre_latin_hip_hop', 'genre_trap_triste', 'genre_urban_contemporary',
    'genre_colombian_pop', 'genre_reggaeton_flow', 'genre_afrobeats'
]

genre_display_vibe = {
    g: g.replace('genre_', '').replace('_', ' ').title()
    for g in genre_priority_vibe
}

def assign_genre_vibe(row):
    for g in genre_priority_vibe:
        if g in row.index and row[g] == 1:
            return genre_display_vibe[g]
    return 'Other'

# Play count filter — exclude bottom 25% by play frequency
play_counts = (
    df_clean
    .groupby(['master_metadata_track_name', 'master_metadata_album_artist_name'])
    .size()
    .reset_index(name='play_count')
)
threshold = play_counts['play_count'].quantile(0.25)
eligible  = play_counts[play_counts['play_count'] > threshold]

print(f"\n[Vibe] Play count threshold (25th pct): {threshold}")
print(f"[Vibe] Tracks before filter: {len(play_counts)} → after filter: {len(eligible)}")

# Build core track table
features_vibe = ['danceability', 'energy', 'valence', 'acousticness']
keep_cols_vibe = (
    ['master_metadata_track_name', 'master_metadata_album_artist_name']
    + features_vibe
    + genre_priority_vibe
)

# Only keep columns that actually exist in df_clean (some genre cols may be absent)
keep_cols_vibe = [c for c in keep_cols_vibe if c in df_clean.columns]

df_vibe = (
    df_clean[keep_cols_vibe]
    .drop_duplicates(subset=['master_metadata_track_name', 'master_metadata_album_artist_name'])
    .dropna(subset=features_vibe)
)

df_vibe = df_vibe.merge(
    eligible,
    on=['master_metadata_track_name', 'master_metadata_album_artist_name'],
    how='inner'
)

df_vibe.rename(columns={
    'master_metadata_track_name':          'track_name',
    'master_metadata_album_artist_name':   'artist'
}, inplace=True)

df_vibe['genre'] = df_vibe.apply(assign_genre_vibe, axis=1)

print(f"[Vibe] Eligible unique tracks: {len(df_vibe)}")

# Bin each feature: Low / Mid / High
bin_edges  = [0, 1/3, 2/3, 1.0]
bin_labels = ['Low', 'Mid', 'High']

for f in features_vibe:
    df_vibe[f + '_bin'] = pd.cut(
        df_vibe[f], bins=bin_edges, labels=bin_labels, include_lowest=True
    )

# Vibe label lookup
vibe_labels = {
    ('High', 'High', 'High', 'Low'): '🔥 Hype Mode!',
    ('High', 'High', 'High', 'Mid'): '🔥 Hype Mode!',
    ('High', 'High', 'High', 'High'): '🎉 Festival Ready',
    ('High', 'High', 'Mid',  'Low'): '💪 Power Hour',
    ('High', 'High', 'Mid',  'Mid'): '💪 Power Hour',
    ('High', 'High', 'Low',  'Low'): '😤 Rage Fuel',
    ('High', 'High', 'Low',  'Mid'): '😤 Rage Fuel',
    ('Mid',  'High', 'High', 'Low'): '🌟 Feel Good Flow',
    ('Mid',  'High', 'Mid',  'Low'): '🏃 Runner\'s High',
    ('Low',  'High', 'Low',  'Low'): '🌩️ Dark Energy',
    ('Low',  'Low',  'High', 'High'): '🌤️ Sunday Morning',
    ('Low',  'Low',  'High', 'Mid'): '☀️ Good Vibes Only',
    ('Mid',  'Low',  'High', 'High'): '🎸 Acoustic Feels',
    ('Low',  'Low',  'Mid',  'High'): '🌙 Late Night Drive',
    ('Low',  'Low',  'Low',  'High'): '🌧️ Rainy Day Mood',
    ('Mid',  'Low',  'Mid',  'High'): '🍂 Cozy & Calm',
    ('Low',  'Low',  'Low',  'Mid'): '😶‍🌫️ Introspective',
    ('High', 'Mid',  'High', 'Low'): '💃 Dance Party',
    ('High', 'Mid',  'High', 'Mid'): '💃 Dance Party',
    ('High', 'Mid',  'Mid',  'Low'): '🎶 Groove Session',
    ('Mid',  'Mid',  'High', 'Low'): '😊 Happy Place',
    ('Mid',  'Mid',  'Mid',  'Low'): '🎵 Everyday Bop',
    ('Mid',  'Mid',  'Mid',  'Mid'): '🎵 Everyday Bop',
    ('Mid',  'Mid',  'Low',  'Low'): '🌀 Moody Vibes',
    ('Low',  'Mid',  'High', 'Mid'): '🎹 Melodic Feels',
    ('Low',  'Mid',  'Low',  'Mid'): '🖤 Brooding',
}
default_vibe = '🎵 Everyday Bop'

bin_center   = {'Low': 1/6, 'Mid': 0.5, 'High': 5/6}
track_vals   = df_vibe[features_vibe].values
bin_combos   = list(product(bin_labels, repeat=4))

records = []
for combo in bin_combos:
    d_bin, e_bin, v_bin, a_bin = combo
    centroid = np.array([
        bin_center[d_bin], bin_center[e_bin],
        bin_center[v_bin], bin_center[a_bin]
    ])
    dists   = np.sqrt(np.sum((track_vals - centroid) ** 2, axis=1))
    top_idx = np.argsort(dists)[:5]
    vibe    = vibe_labels.get(combo, default_vibe)

    for rank, idx in enumerate(top_idx, start=1):
        row = df_vibe.iloc[idx]
        records.append({
            'danceability_bin':  d_bin,
            'energy_bin':        e_bin,
            'valence_bin':       v_bin,
            'acousticness_bin':  a_bin,
            'vibe_label':        vibe,
            'rank':              rank,
            'track_name':        row['track_name'],
            'artist':            row['artist'],
            'genre':             row['genre'],
            'play_count':        int(row['play_count']),
            'danceability':      round(row['danceability'], 3),
            'energy':            round(row['energy'], 3),
            'valence':           round(row['valence'], 3),
            'acousticness':      round(row['acousticness'], 3),
            'distance':          round(dists[idx], 4),
        })

result_vibe = pd.DataFrame(records)

output_path_vibe = 'pick_your_vibe_tableau_v3.xlsx'
result_vibe.to_excel(output_path_vibe, index=False)
print(f"\n[Vibe] Exported {len(result_vibe)} rows → '{output_path_vibe}'")
print(result_vibe.head(10).to_string())


# =============================================================================
# Top Artists / Songs / Genres Export 
# =============================================================================

df_vis = df_clean.copy()

def parse_genres_vis(x):
    try:
        if isinstance(x, list):
            return x
        return ast.literal_eval(x)
    except Exception:
        return []

df_vis['genres_list'] = df_vis['genres'].apply(parse_genres_vis)
df_vis['skipped']     = df_vis['skipped'].astype(bool)

def create_wide_stats(df, group_cols, top_n=10):
    """
    Returns a wide-format DataFrame with Skipped_Count and Completed_Count
    for the top_n most-skipped and top_n most-completed items.
    """
    skips       = df[df['skipped'] == True].groupby(group_cols).size().rename('Skipped_Count')
    completions = df[df['skipped'] == False].groupby(group_cols).size().rename('Completed_Count')
    combined    = pd.concat([skips, completions], axis=1).fillna(0).astype(int)

    top_skipped_keys   = skips.sort_values(ascending=False).head(top_n).index
    top_completed_keys = completions.sort_values(ascending=False).head(top_n).index
    relevant_keys      = top_skipped_keys.union(top_completed_keys)

    final_df = combined.loc[relevant_keys].reset_index()
    final_df = final_df.sort_values(
        by=['Completed_Count', 'Skipped_Count'], ascending=False
    )
    return final_df

# Artists
df_artists_wide = create_wide_stats(df_vis, 'master_metadata_album_artist_name')
df_artists_wide.rename(
    columns={'master_metadata_album_artist_name': 'Artist'}, inplace=True
)

# Songs
df_songs_wide = create_wide_stats(
    df_vis, ['master_metadata_track_name', 'master_metadata_album_artist_name']
)
df_songs_wide.rename(columns={
    'master_metadata_track_name':        'Track Name',
    'master_metadata_album_artist_name': 'Artist Name'
}, inplace=True)

# Genres (explode so each genre tag gets its own row)
df_exploded_vis  = df_vis.explode('genres_list')
df_genres_wide   = create_wide_stats(df_exploded_vis, 'genres_list')
df_genres_wide.rename(columns={'genres_list': 'Genre'}, inplace=True)

output_filename_wide = 'spotify_data_wide_format_v2.xlsx'
with pd.ExcelWriter(output_filename_wide, engine='openpyxl') as writer:
    df_artists_wide.to_excel(writer, sheet_name='Artists', index=False)
    df_songs_wide.to_excel(  writer, sheet_name='Songs',   index=False)
    df_genres_wide.to_excel( writer, sheet_name='Genres',  index=False)

print(f"\n[Wide Format] Exported → '{output_filename_wide}'")
print(f"  Artists : {len(df_artists_wide)} rows")
print(f"  Songs   : {len(df_songs_wide)} rows")
print(f"  Genres  : {len(df_genres_wide)} rows")


# =============================================================================
# Correlation Matrix
# =============================================================================

corr_features = [
    'tempo', 'energy', 'popularity', 'danceability', 'valence',
    'acousticness', 'liveness', 'instrumentalness', 'loudness',
    'speechiness', 'key', 'mode', 'time_signature', 'skipped'
]

df_corr     = df_clean[corr_features].copy()
corr_matrix = df_corr.corr()

corr_long = corr_matrix.reset_index().melt(
    id_vars='index',
    var_name='Variable2',
    value_name='Correlation'
)
corr_long.rename(columns={'index': 'Variable1'}, inplace=True)
corr_long['Correlation'] = corr_long['Correlation'].round(2)

corr_long.to_excel('correlation_matrix_tableau_v2.xlsx', index=False)
print(f"\n[Correlation Matrix] Exported {len(corr_long)} rows "
      f"→ 'correlation_matrix_tableau_v2.xlsx'")
print(corr_long.head(20))


# =============================================================================
# Word Cloud Lyrics Importance
# =============================================================================

df_wc = df_clean.copy()


lyric_stopwords_wc = [
    'ain', 'don', 'll', 've', 're', 'm', 's', 'd', 'em',
    'gon', 'na', 'ta', 'wan', 'gonna', 'wanna', 'gotta',
    'bout', 'cause', 'cuz', 'tryna', 'finna', 'dey',
    'ah', 'oh', 'ooh', 'eh', 'ey', 'ay', 'ayy', 'la', 'da', 'pa', 'ra',
    'ya', 'ye', 'yeah', 'yo', 'whoa', 'ha', 'hmm', 'uh', 'huh', 'baby', 'babe',
    'hey', 'hi', 'hello', 'who', 'what', 'where', 'when', 'why', 'yah', 'vo'
]

curse_words_wc = [
    'bitch', 'bitches', 'fuck', 'fucks',
    'shit', 'bullshit', 'ass', 'damn', 'hell'
]

spanish_stopwords_wc = [
    'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'lo', 'al', 'del',
    'de', 'a', 'en', 'con', 'por', 'para', 'sin', 'sobre', 'entre', 'hacia', 'hasta',
    'yo', 'tu', 'ella', 'nosotros', 'usted', 'mi', 'ti', 'si', 'te', 'se',
    'me', 'le', 'les', 'nos', 'os', 'su', 'sus', 'mis', 'tus', 'mio', 'tuyo',
    'conmigo', 'contigo', 'eso', 'ese', 'esa', 'esos', 'esas', 'esto', 'esta', 'este',
    'nada', 'nadie', 'nunca', 'siempre', 'algo', 'alguien', 'todo', 'todos', 'otro', 'otra',
    'y', 'o', 'e', 'ni', 'que', 'pero', 'mas', 'porque', 'cuando', 'donde',
    'como', 'no', 'ya', 'muy', 'tan', 'aqui', 'ahora', 'bien', 'mal',
    'asi', 'entonces', 'pues', 'hoy', 'ayer',
    'es', 'son', 'soy', 'eres', 'somos', 'fue', 'era', 'ser',
    'esta', 'estas', 'estoy', 'estan', 'estamos', 'estado', 'estar',
    'hay', 'habia', 'he', 'has', 'han', 'tener', 'tengo', 'tiene',
    'voy', 'vas', 'va', 'vamos', 'van', 'ir',
    'hace', 'hacer', 'hago', 'dime', 'di', 'dice', 'decir',
    'quiero', 'quiere', 'gusta', 'dale', 'ven', 'ver', 'sabe', 'sabes',
    'estaa', 'quaa', 'ma', 'maa', 'maas'
]

encoding_artifacts_wc = [
    'mã', 'tãº', 'sã', 'estã', 'tã', 'quã', 'âº', 'â³n', 'asã', 'aquã',
    'bebã', 'yeh', 'uah', 'woo', 'â³',
    '_x000d_', 'x000d', '000d', 'x000d_'
]

generic_pop_words_wc = [
    'like', 'know', 'got', 'just', 'want', 'let', 'make', 'say',
    'come', 'tell', 'think', 'thing', 'things', 'day', 'did',
    'right', 'look', 'good', 'really', 'need', 'way', 'girl',
    'man', 'feel', 'won', 'going', 'said', 'told', 'live'
]

missing_spanish_grammar_wc = [
    'aunque', 'desde', 'entonces', 'mientras', 'contra',
    'puedo', 'puedes', 'puede', 'podemos',
    'quieres', 'quiere', 'quisiera',
    'tienes', 'tiene', 'tienen',
    'siento', 'sientes', 'siente',
    've', 'vez', 'veces',
    'tan', 'tal', 'sola', 'solo', 'solamente',
]

dialect_variations_wc = [
    'ere', 'vamo', 'tas', 'pa',
    'fuckin', 'lil', 'nah', 'mm', 'hmm', 'woah',
    'til', 'em', 'cha', 'brr', 'skrt',
    'quien', 'je', 'pas'
]

common_suffixes_wc = [
    'feeling', 'feelings', 'loving', 'loved', 'loves',
    'living', 'lived', 'lives', 'thinking', 'thought',
    'wanted', 'wants', 'knowing', 'knew', 'knows'
]

full_stopword_list_wc = (
    list(ENGLISH_STOP_WORDS) +
    lyric_stopwords_wc + curse_words_wc + spanish_stopwords_wc +
    encoding_artifacts_wc + generic_pop_words_wc +
    dialect_variations_wc + common_suffixes_wc + missing_spanish_grammar_wc
)

# Lyrics cleaning 

def fix_mojibake_wc(text):
    """Fixes encoding artifacts and removes carriage return escape sequences."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'_x000d_|\\n|\\r', ' ', text, flags=re.IGNORECASE)
    try:
        text = text.encode('latin-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return text

df_wc['lyrics_clean'] = df_wc['lyrics'].apply(fix_mojibake_wc)

# TF-IDF pass tuned for word cloud 

tfidf_wc = TfidfVectorizer(
    stop_words=full_stopword_list_wc,
    max_features=100,
    min_df=2,
    max_df=0.7,
    strip_accents='unicode'
)

X_lyrics_wc = tfidf_wc.fit_transform(df_wc['lyrics_clean'])
X_lyrics_wc_df = pd.DataFrame(
    X_lyrics_wc.toarray(),
    columns=[f"lyric_{w}" for w in tfidf_wc.get_feature_names_out()],
    index=df_wc.index
)

# Lyrics-only Elastic Net

print("\n--- Training Lyrics-Only Elastic Net for Word Cloud Export ---")

y_wc = df_wc['skipped'].astype(int)

X_train_wc, X_test_wc, y_train_wc, y_test_wc = train_test_split(
    X_lyrics_wc_df, y_wc, test_size=0.2, random_state=42, stratify=y_wc
)

model_wc = LogisticRegression(
    penalty='elasticnet', solver='saga', max_iter=5000,
    class_weight='balanced', random_state=42
)

param_grid_wc = {
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    'C':        [0.01, 0.1, 1, 10]
}

grid_wc = GridSearchCV(
    model_wc, param_grid_wc, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
grid_wc.fit(X_train_wc, y_train_wc)

print(f"Best Parameters: {grid_wc.best_params_}")

# Extract coefficients and export 

best_wc_model = grid_wc.best_estimator_

lyrics_coef_df = pd.DataFrame({
    'word':        X_lyrics_wc_df.columns.str.replace('lyric_', '', regex=False),
    'coefficient': best_wc_model.coef_[0]
})

lyrics_coef_df['importance'] = lyrics_coef_df['coefficient'].abs()
lyrics_coef_df['category']   = lyrics_coef_df['coefficient'].apply(
    lambda x: 'skip' if x > 0 else 'listen'
)

wordcloud_df = (
    lyrics_coef_df[lyrics_coef_df['coefficient'] != 0]
    .copy()
    .sort_values('importance', ascending=False)
)

wordcloud_df[['word', 'importance', 'coefficient', 'category']].to_excel(
    'wordcloud_lyrics_importance_v2.xlsx', index=False
)

print(f"\n[Word Cloud] Exported {len(wordcloud_df)} words "
      f"→ 'wordcloud_lyrics_importance_v2.xlsx'")
print(f"  Skip words  : {(wordcloud_df['category'] == 'skip').sum()}")
print(f"  Listen words: {(wordcloud_df['category'] == 'listen').sum()}")


# =============================================================================
# Hourly Skip Rate + KPI Summary Export
# =============================================================================

df_hourly = df_clean.copy()

hourly_skips = df_hourly.groupby('hour').agg(
    total_plays=('skipped', 'count'),
    total_skips=('skipped', 'sum')
).reset_index()

hourly_skips['skip_rate'] = (
    hourly_skips['total_skips'] / hourly_skips['total_plays'] * 100
).round(2)

def get_time_label(hour):
    if hour == 0:
        return "12:00 AM"
    elif hour < 12:
        return f"{hour}:00 AM"
    elif hour == 12:
        return "12:00 PM"
    else:
        return f"{hour - 12}:00 PM"

def get_context(hour):
    if   6  <= hour <= 9:  return "Morning Commute"
    elif 10 <= hour <= 11: return "Late Morning"
    elif 12 <= hour <= 13: return "Lunch"
    elif 14 <= hour <= 17: return "Afternoon"
    elif 18 <= hour <= 21: return "Evening"
    elif 22 <= hour <= 23: return "Night"
    else:                  return "Late Night"

hourly_skips['time_label'] = hourly_skips['hour'].apply(get_time_label)
hourly_skips['context']    = hourly_skips['hour'].apply(get_context)

max_skip_hour = hourly_skips.loc[hourly_skips['skip_rate'].idxmax()]
min_skip_hour = hourly_skips.loc[hourly_skips['skip_rate'].idxmin()]

print("\n--- Hourly Skip Data ---")
print(hourly_skips.to_string(index=False))
print(f"\nMost Impatient : {max_skip_hour['time_label']} "
      f"({max_skip_hour['context']}) — {max_skip_hour['skip_rate']}% skip rate")
print(f"Most Chill     : {min_skip_hour['time_label']} "
      f"({min_skip_hour['context']}) — {min_skip_hour['skip_rate']}% skip rate")

hourly_skips.to_excel('hourly_skip_rate_v2.xlsx', index=False)
print(f"\n[Hourly Skips] Exported {len(hourly_skips)} rows → 'hourly_skip_rate_v2.xlsx'")

# KPI summary
kpi_df = pd.DataFrame({
    'Metric': ['Total Songs', 'Total Skips', 'Skip Rate'],
    'Value': [
        len(df_clean),
        int(df_clean['skipped'].sum()),
        round(df_clean['skipped'].sum() / len(df_clean), 4)
    ]
})

kpi_df.to_csv('spotify_kpis_v2.csv', index=False)
print(f"[KPIs] Exported → 'spotify_kpis_v2.csv'")
print(kpi_df.to_string(index=False))


# =============================================================================
# Summary of Exports
# =============================================================================

print("\n" + "=" * 60)
print("ALL TABLEAU EXPORTS COMPLETE")
print("=" * 60)
print(f"  skips_heatmap_for_tableau_v2.xlsx    → {len(df_heatmap)} rows")
print(f"  decision_tree_v2_sankey_v3.png       → saved")
print(f"  pick_your_vibe_tableau_v3.xlsx       → {len(result_vibe)} rows")
print(f"  spotify_data_wide_format_v2.xlsx     → Artists / Songs / Genres sheets")
print(f"  correlation_matrix_tableau_v2.xlsx   → {len(corr_long)} rows")
print(f"  wordcloud_lyrics_importance_v2.xlsx  → {len(wordcloud_df)} words")
print(f"  hourly_skip_rate_v2.xlsx             → {len(hourly_skips)} rows")
print(f"  spotify_kpis_v2.csv                  → {len(kpi_df)} metrics")
print("=" * 60)
