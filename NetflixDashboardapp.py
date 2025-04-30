import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import requests
from PIL import Image
from io import BytesIO
import pandas as pd

file_id = "1DE2s_g8DkOxr_CneTu_Me1pW_qJE6ITS"
csv_url = f"https://drive.google.com/uc?id={file_id}"


# Read the data
df = pd.read_csv(csv_url)

# Page Settings
st.set_page_config(page_title="NETFLIX DASHBOARD", page_icon="🇳‌🇪‌🇹‌🇫‌🇱‌🇮‌🇽‌", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: black;
       
    }
    h1 {
        color: #E50914;
        font-family: 'Trebuchet MS', sans-serif;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True
)

# Your Drive file ID
file_id = "1lxjEicVIKey9iNfm5vF2kiqOtdZFan-X"

# Generate the direct download URL
image_url = f"https://drive.google.com/uc?export=download&id={file_id}"

# Download the image
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Display the image in Streamlit
st.sidebar.image(image, width=300) 


# Main Title
st.markdown("<h1>Netflix Dashboard</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Filter Your Netflix Data")
type_filter = st.sidebar.multiselect("Select Type", options=df['type'].unique(), default=df['type'].unique())
year_min = int(df['release_year'].min())
year_max = int(df['release_year'].max())
year_filter = st.sidebar.slider('Select Release Year Range', min_value=year_min, max_value=year_max, value=(year_min, year_max))

# Filter Data
df_filtered = df[(df['type'].isin(type_filter)) & (df['release_year'].between(year_filter[0], year_filter[1]))]

# Section: Release Year Distribution
st.subheader("Release Year Distribution")
fig1, ax1 = plt.subplots(figsize=(10,6))
sns.histplot(df_filtered['release_year'], color="#e50914", kde=True, ax=ax1)
ax1.set_title('Release Year Distribution', fontsize=20, fontweight='bold', color='white')
ax1.set_xlabel('Release Year', fontsize=16, color='white')
ax1.set_ylabel('Density', fontsize=16, color='white')
ax1.set_facecolor('black')
fig1.patch.set_facecolor('black')
ax1.grid(False)
st.pyplot(fig1)

# Section: Top Countries
st.subheader("Top 10 Countries with Most Netflix Titles")
top_countries = df_filtered['country'].value_counts().head(10)
fig2, ax2 = plt.subplots(figsize=(10,6))
top_countries.plot(kind='barh', color="#B20710", ax=ax2)
ax2.set_title('Top 10 Countries', fontsize=20, fontweight='bold', color='white')
ax2.set_xlabel('Number of Titles', fontsize=16, color='white')
ax2.set_ylabel('Country', fontsize=16, color='white')
ax2.invert_yaxis()
ax2.set_facecolor('black')
fig2.patch.set_facecolor('black')
ax2.grid(False)
st.pyplot(fig2)

# Section: Movie vs TV Show Count
st.subheader("Movies vs TV Shows")
fig3, ax3 = plt.subplots(figsize=(8,6))
sns.countplot(x='type', data=df_filtered, palette=["#E50914", "#B20710"], ax=ax3)
ax3.set_title('Movies vs TV Shows', fontsize=20, fontweight='bold', color='white')
ax3.set_xlabel('Type', fontsize=16, color='white')
ax3.set_ylabel('Count', fontsize=16, color='white')
ax3.set_facecolor('black')
fig3.patch.set_facecolor('black')
ax3.grid(False)
st.pyplot(fig3)

# Section: Ratings Pie Chart
st.subheader("Content Ratings")
n = df_filtered.groupby(['rating']).size().reset_index(name='counts')
pieChart = px.pie(n, values='counts', names='rating',
                  title='Distribution of Content Ratings',
                  color_discrete_sequence=["#E50914", "#B20710", '#404040', '#5a5a5a'])
pieChart.update_layout(title_font=dict(size=24, color='white', family='Arial'),
                       paper_bgcolor='black', plot_bgcolor='black', font_color='white')
st.plotly_chart(pieChart)

# Section: Top Genres Pie Charts
st.subheader("Top Genres")

# Netflix color palette
colors = ["#E50914", "#B20710", '#404040', '#5a5a5a', '#737373', '#8c8c8c', '#a6a6a6', '#bfbfbf']

# Function to wrap labels into two lines
def wrap_label(label, max_words=1):
    words = label.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + '\n' + ' '.join(words[max_words:])
    return label

# Check and get top genres (using 5 here as in your Streamlit version)
if 'listed_in' in df_filtered.columns:
    movie_genres = df_filtered[df_filtered['type'] == 'Movie']['listed_in'].value_counts().head(5)
    tv_genres = df_filtered[df_filtered['type'] == 'TV Show']['listed_in'].value_counts().head(5)
else:
    raise KeyError("The 'listed_in' column is missing from the DataFrame.")

# Wrap genre labels
movie_labels_wrapped = [wrap_label(label) for label in movie_genres.index]
tv_labels_wrapped = [wrap_label(label) for label in tv_genres.index]

# Streamlit layout
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    ax1.pie(
        movie_genres,
        labels=movie_labels_wrapped,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        textprops={'color': 'white', 'fontsize': 11},
        labeldistance=1.1,
        pctdistance=0.8,
        wedgeprops={'edgecolor': 'black'})
    ax1.set_title('Top 5 Genres - Movies', color='white', fontsize=16, fontweight='bold')
    fig1.patch.set_facecolor('black')
    ax1.set_facecolor('black')
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.pie( tv_genres, labels=tv_labels_wrapped, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'color': 'white', 'fontsize': 11}, labeldistance=1.1, pctdistance=0.8, wedgeprops={'edgecolor': 'black'})
    ax2.set_title('Top 5 Genres - TV Shows', color='white', fontsize=16, fontweight='bold')
    fig2.patch.set_facecolor('black')
    ax2.set_facecolor('black')
    st.pyplot(fig2)

# Section: Heatmap of Additions
st.subheader("Netflix Content Updates Heatmap")
netflix_date = df_filtered[['date_added']].dropna().copy()
netflix_date['date_added'] = pd.to_datetime(netflix_date['date_added'], errors='coerce')
netflix_date = netflix_date.dropna()
netflix_date['year'] = netflix_date['date_added'].dt.year.astype(str)
netflix_date['month'] = netflix_date['date_added'].dt.month_name()

month_order = ['December', 'November', 'October', 'September', 'August', 'July',
               'June', 'May', 'April', 'March', 'February', 'January']
pivot_table = netflix_date.groupby('year')['month'].value_counts().unstack().fillna(0)
pivot_table = pivot_table[month_order]
pivot_table = pivot_table.T

fig6, ax6 = plt.subplots(figsize=(14,8), dpi=200)
c = ax6.pcolor(pivot_table, cmap='Reds', edgecolors='black', linewidths=2)
ax6.set_xticks(np.arange(0.5, len(pivot_table.columns), 1))
ax6.set_yticks(np.arange(0.5, len(pivot_table.index), 1))
ax6.set_xticklabels(pivot_table.columns, fontsize=8, color='white', rotation=45)
ax6.set_yticklabels(pivot_table.index, fontsize=8, color='white')
ax6.set_title('Netflix Content Updates by Month and Year', fontsize=20, fontweight='bold', color='white')
fig6.colorbar(c)
ax6.set_facecolor('white')
fig6.patch.set_facecolor('black')
st.pyplot(fig6)

# Footer
st.caption('Created by Kristal Quintana')
