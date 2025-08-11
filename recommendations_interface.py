import streamlit as st
import pickle
import pandas as pd
import requests as req
from nltk.stem.porter import PorterStemmer
from difflib import get_close_matches
from sklearn.neighbors import NearestNeighbors
import numpy as np

stemmer = PorterStemmer()

def preprocess_input(text):
    if not isinstance(text, str):
        return ''
    return stemmer.stem(text.lower().strip())

def load_data():
    try:
        with open('cos_similarity.pkl', 'rb') as f:
            similarity = pickle.load(f)
        with open('movies_dict.pkl', 'rb') as f:
            movies_dict = pickle.load(f)
        movies_df = pd.DataFrame(movies_dict)
        return similarity, movies_df
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        st.stop()

def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=58f74c7f3d3836bdaf9c5a52e49e0282&language=en-US"
        response = req.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path', '')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        return "https://via.placeholder.com/500x750?text=No+Poster"
    except Exception as e:
        st.error(f"Couldn't fetch poster: {str(e)}")
        return "https://via.placeholder.com/500x750?text=No+Poster"

def find_similar_movies(movie_name, movies_df):
    all_titles = movies_df['title'].tolist()
    matches = get_close_matches(movie_name, all_titles, n=3, cutoff=0.6)
    return matches[0] if matches else None

def matches_preferences(movie_tags, preferences):
    if not isinstance(movie_tags, str):
        return False
    
    stemmed_tags = {preprocess_input(word) for word in movie_tags.split()}
    match_score = 0

    for pref_type, pref_value in preferences.items():
        if pref_value != 'any':
            stemmed_pref = preprocess_input(pref_value)
            if stemmed_pref in stemmed_tags:
                match_score += 1

    if preferences['company'] == 'Family' and 'famili' in stemmed_tags:
        match_score += 1
        
    return match_score >= 1 

def get_recommendations(movie_name, movies_df, similarity, preferences):

    try:
        if movie_name not in movies_df['title'].values:
            similar_movie = find_similar_movies(movie_name, movies_df)
            if similar_movie:
                movie_name = similar_movie
                st.info(f"Showing recommendations similar to '{similar_movie}'")

        movie_idx = movies_df[movies_df['title'] == movie_name].index[0]

        similarity_array = np.array(similarity)

        min_sim = similarity_array.min()
        max_sim = similarity_array.max()
        normalized_sim = (similarity_array - min_sim) / (max_sim - min_sim)

        distance_matrix = 1 - normalized_sim

        n_neighbors = 21  
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='precomputed')

        knn.fit(distance_matrix)

        distances, indices = knn.kneighbors([distance_matrix[movie_idx]])

        sim_scores = 1 - distances[0][1:] 
        neighbor_indices = indices[0][1:]  
        
        recommendations = []
        for idx, score in zip(neighbor_indices, sim_scores):
            movie = movies_df.iloc[idx]
            if matches_preferences(movie.tags, preferences):
                recommendations.append({
                    'title': movie.title,
                    'poster': fetch_poster(movie.movie_id),
                    'score': score,
                    'tags': movie.tags
                })
            if len(recommendations) >= 5:
                break

        if len(recommendations) < 5:
            for idx, score in zip(neighbor_indices, sim_scores):
                movie = movies_df.iloc[idx]
                if movie.title not in [r['title'] for r in recommendations]:
                    recommendations.append({
                        'title': movie.title,
                        'poster': fetch_poster(movie.movie_id),
                        'score': score,
                        'tags': movie.tags
                    })
                if len(recommendations) >= 5:
                    break
        
        return recommendations
    
    except Exception as e:
        st.error(f"Recommendation error: {str(e)}")
        return []

similarity, movies_df = load_data()

st.header("🎬 Smart Movie Recommendation System")
st.markdown("""
Discover personalized movie recommendations based on your mood, preferences, 
and favorite movies. Our AI-powered system suggests films you'll love!
""")

if 'step' not in st.session_state:
    st.session_state.update({
        'step': 0,
        'mood': 'Any',
        'with_whom': 'Alone',
        'genre_pref': 'Any',
        'age_group': 'Any',
        'selected_movie': None,
        'recommendations': []
    })

steps = [
    "Welcome",
    "Your Mood",
    "Viewing Company", 
    "Genre Preference",
    "Age Group",
    "Select Movie",
    "Your Recommendations"
]

def next_step(): st.session_state.step = min(st.session_state.step + 1, len(steps)-1)
def prev_step(): st.session_state.step = max(st.session_state.step - 1, 0)
def skip_to_movie(): st.session_state.step = 5
def reset_app():
    st.session_state.step = 0
    st.session_state.mood = 'Any'
    st.session_state.with_whom = 'Alone'
    st.session_state.genre_pref = 'Any'
    st.session_state.age_group = 'Any'
    st.session_state.selected_movie = None
    st.session_state.recommendations = []

progress = st.session_state.step / (len(steps)-1)
st.progress(progress, text=f"Step {st.session_state.step+1}/{len(steps)}: {steps[st.session_state.step]}")

if st.session_state.step == 0:
    st.header("Welcome to MovieMatch!")
    #st.image("https://via.placeholder.com/800x300?text=Movie+Recommendation+System", use_container_width=True)
    st.markdown("""
    We'll help you find the perfect movie by asking a few simple questions about:
    - Your current mood 😊😢🎭
    - Who you're watching with 👨‍👩‍👧‍👦👫
    - Your favorite genres 🎬🍿
    """)
    st.button("Get Started →", on_click=next_step, type="primary")

elif st.session_state.step == 1:
    st.subheader("How are you feeling today?")
    st.session_state.mood = st.radio(
        "Select your mood:",
        ['Any', 'Happy', 'Sad', 'Adventurous', 'Romantic', 'Tense', 'Nostalgic'],
        horizontal=True
    )
    left, middle, right = st.columns([1,2,1], vertical_alignment="bottom")
    with middle:
      st.button("Next →", on_click=next_step, type="primary")
    with left:
      st.button("← Go Back", on_click=prev_step, type="primary")
    with right:
       st.button("⏭️ Skip To Movie →", on_click=skip_to_movie, type="primary")

elif st.session_state.step == 2:
    st.subheader("Who are you watching with?")
    st.session_state.with_whom = st.radio(
        "Select your viewing company:",
        ['Alone', 'Friends', 'Family', 'Partner', 'Children'],
        horizontal=True
    )
    left, middle, right = st.columns([1,2,1], vertical_alignment="bottom")
    with middle:
      st.button("Next →", on_click=next_step, type="primary")
    with left:
      st.button("← Go Back", on_click=prev_step, type="primary")
    with right:
       st.button("⏭️ Skip To Movie →", on_click=skip_to_movie, type="primary")

elif st.session_state.step == 3:
    st.subheader("What genre are you in the mood for?")
    st.session_state.genre_pref = st.selectbox(
        "Select preferred genre:",
        ['Any', 'Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 
         'Thriller', 'Animation', 'Documentary']
    )
    left, middle, right = st.columns([1,2,1], vertical_alignment="bottom")
    with middle:
      st.button("Next →", on_click=next_step, type="primary")
    with left:
      st.button("← Go Back", on_click=prev_step, type="primary")
    with right:
       st.button("⏭️ Skip To Movie →", on_click=skip_to_movie, type="primary")

elif st.session_state.step == 4:
    st.subheader("For which age group?")
    st.session_state.age_group = st.radio(
        "Select age appropriateness:",
        ['Any', 'Kids', 'Teens', 'Adults'],
        horizontal=True
    )
    left, middle, right = st.columns([1,2,1], vertical_alignment="bottom")
    with middle:
      st.button("Next →", on_click=next_step, type="primary")
    with left:
      st.button("← Go Back", on_click=prev_step, type="primary")
    with right:
       st.button("⏭️ Skip To Movie →", on_click=skip_to_movie, type="primary")

elif st.session_state.step == 5:
    st.subheader("Select a movie you like")
    st.write("Choose a movie to get similar recommendations:")
    
    search_query = st.text_input("Search for a movie:", key="movie_search")
    if search_query:
        matches = movies_df[movies_df['title'].str.contains(search_query, case=False)]['title']
        if not matches.empty:
            st.session_state.selected_movie = st.selectbox(
                "Select from matches:",
                matches
            )
        else:
            st.warning("No matches found. Try a different search term.")
    else:
        st.session_state.selected_movie = st.selectbox(
            "Or select from popular movies:",
            movies_df['title'].sample(50).sort_values()
        )
    
    if st.session_state.selected_movie:
        st.button("Get Recommendations 🎥", on_click=next_step, type="primary")

elif st.session_state.step == 6:
    st.subheader("🍿 Your Personalized Recommendations")
    
    if not st.session_state.selected_movie:
        st.warning("Please select a movie first!")
        st.button("← Go Back", on_click=prev_step)
    else:
        preferences = {
            'mood': st.session_state.mood,
            'company': st.session_state.with_whom,
            'genre': st.session_state.genre_pref,
            'age': st.session_state.age_group
        }
        
        with st.spinner("Finding your perfect movies..."):
            st.session_state.recommendations = get_recommendations(
                st.session_state.selected_movie,
                movies_df,
                similarity,
                preferences
            )
        
        if not st.session_state.recommendations:
            st.error("""
            Couldn't generate recommendations. Please try:
            1. Selecting a different reference movie
            2. Adjusting your preferences
            """)
        else:
            cols = st.columns(5)
            for i, rec in enumerate(st.session_state.recommendations[:5]):
                with cols[i]:
                    st.image(rec['poster'], use_container_width=True)
                    st.markdown(f"**{rec['title']}**")
                    st.caption(f"Match score: {rec['score']:.2f}")
                    st.markdown(f"[Details ↗](https://www.themoviedb.org/movie/{movies_df[movies_df['title'] == rec['title']].movie_id.iloc[0]})", 
                               help="View on TMDB")

            with st.expander("Why these recommendations?"):
                st.write("Your preferences:")
                st.json(preferences)
                st.write("Recommendation details:")
                for rec in st.session_state.recommendations:
                    st.write(f"{rec['title']} (Score: {rec['score']:.2f}) - Tags: {rec['tags']}")
        
        st.button("🔄 Start Over", on_click=reset_app, type="primary")