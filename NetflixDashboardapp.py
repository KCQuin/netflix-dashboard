import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import requests
from PIL import Image
from io import BytesIO
import pandas as pd

# Load Data
file_id = "1DE2s_g8DkOxr_CneTu_Me1pW_qJE6ITS"
csv_url = f"https://drive.google.com/uc?id={file_id}"
df = pd.read_csv(csv_url)

# Page Configuration
st.set_page_config(page_title="Netflix Dashboard", page_icon="📺", layout="wide")

# Sidebar Logo
image_url = "https://drive.google.com/uc?export=download&id=1lxjEicVIKey9iNfm5vF2kiqOtdZFan-X"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
st.sidebar.image(image, width=250)
st.sidebar.header("Filter Netflix Data")

# Theme Toggle
theme = st.sidebar.radio("Select Theme", ("Dark", "Light"))

# CSS Styling based on theme
if theme == "Dark":
    st.markdown("""
        <style>
        h1, h2, h3, .stTextInput label, .stSelectbox label, .stSlider label { color: white; }
        .css-1d391kg { background-color: #000 !important; }
        .st-cg { background-color: #111 !important; border-radius: 10px; padding: 20px; }
        .css-ffhzg2 { background-color: #111 !important; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        h1, h2, h3, .stTextInput label, .stSelectbox label, .stSlider label { color: black; }
        .css-1d391kg { background-color: #fff !important; }
        .st-cg { background-color: #f0f0f0 !important; border-radius: 10px; padding: 20px; }
        .css-ffhzg2 { background-color: #f0f0f0 !important; }
        </style>
    """, unsafe_allow_html=True)

# Colors
bg_color = "black" if theme == "Dark" else "white"
text_color = "white" if theme == "Dark" else "black"
palette = ["#E50914", "#B20710"] if theme == "Dark" else ["#880000", "#FF9999"]

# Sidebar Filters
type_filter = st.sidebar.multiselect("Select Type", df['type'].unique(), default=df['type'].unique())
year_min = int(df['release_year'].min())
year_max = int(df['release_year'].max())
year_filter = st.sidebar.slider("Release Year", min_value=year_min, max_value=year_max, value=(year_min, year_max))

genre_filter = st.sidebar.multiselect("Select Genre", df['listed_in'].dropna().unique(), default=None)
country_filter = st.sidebar.multiselect("Select Country", df['country'].dropna().unique(), default=None)

# Filter Data
df_filtered = df[df['type'].isin(type_filter) & df['release_year'].between(year_filter[0], year_filter[1])]
if genre_filter:
    df_filtered = df_filtered[df_filtered['listed_in'].isin(genre_filter)]
if country_filter:
    df_filtered = df_filtered[df_filtered['country'].isin(country_filter)]

# Title
st.markdown(f"<h1 style='text-align:center; color:{text_color};'> Netflix Dashboard</h1>", unsafe_allow_html=True)

# Layout Starts
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Movies vs TV Shows")
        fig, ax = plt.subplots()
        sns.countplot(x='type', data=df_filtered, palette=palette, ax=ax)
        ax.set_facecolor(bg_color)
        fig.patch.set_facecolor(bg_color)
        ax.set_title('Movies vs TV Shows', color=text_color)
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
        ax.set_title("Top 10 Countries", color=text_color)
        ax.set_xlabel("Number of Titles", color=text_color)
        ax.set_ylabel("Country", color=text_color)
        ax.tick_params(colors=text_color)
        st.pyplot(fig)

# Second row layout
col1 = st.columns(1)
    with col1:
        st.subheader("Content Ratings")
        rating_data = df_filtered.groupby('rating').size().reset_index(name='counts')
        pie = px.pie(rating_data, values='counts', names='rating',
                     color_discrete_sequence=px.colors.sequential.Reds)
        pie.update_layout(paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                          font_color=text_color, title_font_color=text_color)
        st.plotly_chart(pie, use_container_width=True)

# Third row layout
col1, col2 = st.columns(2)
with col1:
    st.subheader("Release Year Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_filtered['release_year'], kde=True, color=palette[0], ax=ax)
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)
    ax.set_title("Release Year", color=text_color)
    ax.set_xlabel("Year", color=text_color)
    ax.set_ylabel("Count", color=text_color)
    ax.tick_params(colors=text_color)
    st.pyplot(fig)

with col2:
    st.subheader("Top Genres")
    if 'listed_in' in df_filtered.columns:
        movie_genres = df_filtered[df_filtered['type'] == 'Movie']['listed_in'].value_counts().head(5)
        tv_genres = df_filtered[df_filtered['type'] == 'TV Show']['listed_in'].value_counts().head(5)

        fig1, ax1 = plt.subplots()
        ax1.pie(movie_genres, labels=movie_genres.index, autopct='%1.1f%%', startangle=140,
                textprops={'color': text_color}, wedgeprops={'edgecolor': bg_color})
        ax1.set_title('Top 5 Movie Genres', color=text_color)
        fig1.patch.set_facecolor(bg_color)
        ax1.set_facecolor(bg_color)
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.pie(tv_genres, labels=tv_genres.index, autopct='%1.1f%%', startangle=140,
                textprops={'color': text_color}, wedgeprops={'edgecolor': bg_color})
        ax2.set_title('Top 5 TV Show Genres', color=text_color)
        fig2.patch.set_facecolor(bg_color)
        ax2.set_facecolor(bg_color)
        st.pyplot(fig2)

# Heatmap Section
st.subheader("Content Additions Heatmap")
netflix_date = df_filtered[['date_added']].dropna().copy()
netflix_date['date_added'] = pd.to_datetime(netflix_date['date_added'], errors='coerce')
netflix_date.dropna(inplace=True)
netflix_date['year'] = netflix_date['date_added'].dt.year.astype(str)
netflix_date['month'] = netflix_date['date_added'].dt.month_name()

month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

pivot = netflix_date.groupby('year')['month'].value_counts().unstack().fillna(0)
pivot = pivot[month_order]
pivot = pivot.T

fig, ax = plt.subplots(figsize=(12, 6))
c = ax.pcolor(pivot, cmap='Reds', edgecolors='white', linewidths=2)
ax.set_xticks(np.arange(0.5, len(pivot.columns), 1))
ax.set_yticks(np.arange(0.5, len(pivot.index), 1))
ax.set_xticklabels(pivot.columns, rotation=45, color=text_color, fontsize=8)
ax.set_yticklabels(pivot.index, color=text_color, fontsize=8)
ax.set_title('Monthly Netflix Additions', color=text_color)
fig.colorbar(c)
ax.set_facecolor(bg_color)
fig.patch.set_facecolor(bg_color)
st.pyplot(fig)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("✨ Created by Kristal Quintana | Inspired by Netflix UI ✨")
