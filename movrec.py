import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv('netflix_titles.csv')

netflix_data = df.copy()

tfidf = TfidfVectorizer(stop_words='english')
netflix_data['description'] = netflix_data['description'].fillna('')
tfidf_matrix = tfidf.fit_transform(netflix_data['description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(netflix_data.index, index=netflix_data['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return netflix_data[['title','description']].iloc[movie_indices]

st.header("Netflix Movie Recommendation System")

movie_list = netflix_data['title'].values

selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

def Table(df):
    fig = go.Figure(go.Table(
        columnorder=[1, 2],
        columnwidth=[10, 28],
        header=dict(
            values=['Title', 'Description'],
            line_color='black',
            font=dict(color='black', size=19),
            height=40,
            fill_color='#dd571c',
            align=['left', 'center']
        ),
        cells=dict(
            values=[df.title, df.description],
            fill_color='#ffdac4',
            line_color='grey',
            font=dict(color='black', size=16),
            align='left'
        )
    ))

    fig.update_layout(
        height=600,
        title={
            'text': "Top 10 movie recommendations",
            'font': {'size': 22}
        },
        title_x=0.5
    )

    return st.plotly_chart(fig, use_container_width=True)


if st.button('Show recommendation'):
    recommended_movie_names = get_recommendations(selected_movie)
    Table(recommended_movie_names)