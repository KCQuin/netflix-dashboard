import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
from io import BytesIO
import requests

# Load Netflix Data
file_id = "1DE2s_g8DkOxr_CneTu_Me1pW_qJE6ITS"
csv_url = f"https://drive.google.com/uc?id={file_id}"
df = pd.read_csv(csv_url)

# Page Config
st.set_page_config(page_title="Netflix Dashboard", page_icon="📺", layout="wide")

# Sidebar Logo and Filters
image_url = "https://drive.google.com/uc?export=download&id=1lxjEicVIKey9iNfm5vF2kiqOtdZFan-X"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
st.sidebar.image(image, width=250)
st.sidebar.header("Filter Netflix Data")

# Theme Toggle
theme = st.sidebar.radio("Select Theme", ["Dark", "Light"])
bg_color = "#000" if theme == "Dark" else "#fff"
text_color = "#fff" if theme == "Dark" else "#000"
palette = ["#E50914", "#B20710"] if theme == "Dark" else ["#880000", "#FF9999"]

# Inline CSS styling
st.markdown(f"""
    <style>
    html, body, [class*="css"] {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
    }}
    .stSelectbox > div > label, .stSlider > label {{
        color: {text_color};
    }}
    </style>
""", unsafe_allow_html=True)

# Sidebar Filters
type_filter = st.sidebar.multiselect("Select Type", df['type'].dropna().unique(), default=df['type'].dropna().unique())
year_min, year_max = int(df['release_year'].min()), int(df['release_year'].max())
year_filter = st.sidebar.slider("Release Year", min_value=year_min, max_value=year_max, value=(year_min, year_max))
genre_filter = st.sidebar.multiselect("Select Genre", sorted(set(", ".join(df['listed_in'].dropna()).split(', '))))
country_filter = st.sidebar.multiselect("Select Country", df['country'].dropna().unique())

# Filter Logic
df_filtered = df[df['type'].isin(type_filter) & df['release_year'].between(year_filter[0], year_filter[1])]
if genre_filter:
    df_filtered = df_filtered[df_filtered['listed_in'].apply(lambda x: any(g in str(x) for g in genre_filter))]
if country_filter:
    df_filtered = df_filtered[df_filtered['country'].isin(country_filter)]

# Title
st.markdown(f"<h1 style='text-align:center; color:{text_color};'>📺 Netflix Dashboard</h1>", unsafe_allow_html=True)

# Row 1: Type Count & Top Countries
col1, col2 = st.columns(2)
with col1:
    st.subheader("Movies vs TV Shows")
    fig, ax = plt.subplots()
    sns.countplot(x='type', data=df_filtered, palette=palette, ax=ax)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlabel('Type', color=text_color)
    ax.set_ylabel('Count', color=text_color)
    ax.tick_params(colors=text_color)
    st.pyplot(fig)

with col2:
    st.subheader("Top Countries")
    top_countries = df_filtered['country'].value_counts().head(10)
    fig, ax = plt.subplots()
    top_countries.plot(kind='barh', color=palette[1], ax=ax)
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)
    ax.set_xlabel("Number of Titles", color=text_color)
    ax.set_ylabel("Country", color=text_color)
    ax.tick_params(colors=text_color)
    st.pyplot(fig)

# Row 2: Year Distribution & Top Genres
col1, col2 = st.columns(2)
with col1:
    st.subheader("Release Year Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_filtered['release_year'], kde=True, color=palette[0], ax=ax)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlabel("Year", color=text_color)
    ax.set_ylabel("Count", color=text_color)
    ax.tick_params(colors=text_color)
    st.pyplot(fig)

with col2:
    st.subheader("Top Genres")
    movie_genres = df_filtered[df_filtered['type'] == 'Movie']['listed_in'].dropna()
    tv_genres = df_filtered[df_filtered['type'] == 'TV Show']['listed_in'].dropna()
    top_movie_genres = pd.Series(", ".join(movie_genres).split(', ')).value_counts().head(5)
    top_tv_genres = pd.Series(", ".join(tv_genres).split(', ')).value_counts().head(5)

    fig1, ax1 = plt.subplots()
    ax1.pie(top_movie_genres, labels=top_movie_genres.index, autopct='%1.1f%%', startangle=140,
            textprops={'color': text_color}, wedgeprops={'edgecolor': bg_color}, colors=palette)
    fig1.patch.set_facecolor(bg_color)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.pie(top_tv_genres, labels=top_tv_genres.index, autopct='%1.1f%%', startangle=140,
            textprops={'color': text_color}, wedgeprops={'edgecolor': bg_color}, colors=palette)
    fig2.patch.set_facecolor(bg_color)
    st.pyplot(fig2)

# Content Ratings
st.subheader("Content Ratings")
ratings = df_filtered.groupby('rating').size().reset_index(name='counts')
ratings = ratings[ratings['rating'].notna()]
pieChart = px.pie(ratings, values='counts', names='rating',
                  color_discrete_sequence=palette + ['#404040', '#5a5a5a'])
pieChart.update_layout(paper_bgcolor=bg_color, font_color=text_color)
st.plotly_chart(pieChart, use_container_width=True)

# Heatmap for Date Added
st.subheader("Netflix Content Updates (by Month & Year)")
netflix_date = df_filtered[['date_added']].dropna()
netflix_date['date_added'] = pd.to_datetime(netflix_date['date_added'], errors='coerce')
netflix_date = netflix_date.dropna()
netflix_date['year'] = netflix_date['date_added'].dt.year.astype(str)
netflix_date['month'] = netflix_date['date_added'].dt.month_name()

month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

pivot = netflix_date.groupby('year')['month'].value_counts().unstack().fillna(0)
pivot = pivot[month_order].T

fig, ax = plt.subplots(figsize=(12, 6))
c = ax.pcolor(pivot, cmap='Reds', edgecolors='white', linewidths=2)
ax.set_xticks(np.arange(0.5, len(pivot.columns), 1))
ax.set_yticks(np.arange(0.5, len(pivot.index), 1))
ax.set_xticklabels(pivot.columns, rotation=45, color=text_color, fontsize=8)
ax.set_yticklabels(pivot.index, color=text_color, fontsize=8)
fig.colorbar(c)
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)
st.pyplot(fig)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("✨ Created by Kristal Quintana | Inspired by Netflix UI ✨")
