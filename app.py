from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load Dataset
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# User-Movie Matrix
user_movie_matrix = ratings.pivot_table(
    index='movieId',
    columns='userId',
    values='rating'
).fillna(0)

# Similarity Matrix
movie_similarity = cosine_similarity(
    user_movie_matrix
)

movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

# Recommendation Function
def recommend(movie_name, top_n=10):

    matches = movies[
        movies['title'].str.contains(
            movie_name,
            case=False,
            na=False
        )
    ]

    if matches.empty:
        return []

    selected_movie = matches.iloc[0]

    movie_id = selected_movie['movieId']

    if movie_id not in movie_similarity_df.index:
        return []

    similar_movies = movie_similarity_df[
        movie_id
    ].sort_values(
        ascending=False
    )[1:top_n+1]

    recommendations = []

    for mid in similar_movies.index:

        title = movies[
            movies['movieId'] == mid
        ]['title'].values

        if len(title) > 0:

            recommendations.append({
                "title": title[0]
            })

    return recommendations


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def get_recommendation():

    movie_name = request.form['movie']

    recommendations = recommend(
        movie_name
    )

    return render_template(
        'index.html',
        recommendations=recommendations,
        movie=movie_name
    )


if __name__ == '__main__':
    app.run(debug=True)