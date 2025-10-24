import streamlit as st
import pandas as pd
import numpy as np
import ast
import requests
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(
    page_title="BingeRecc >_<",
    page_icon="🎬",
    layout="wide"
)

TMDB_KEY = st.secrets.get("TMDB_API_KEY")
OMDB_KEY = st.secrets.get("OMDB_API_KEY")

COUNTRY_CODES = {
    "United States of America": "US", "United Kingdom": "GB", "India": "IN",
    "Canada": "CA", "France": "FR", "Germany": "DE", "Japan": "JP",
    "China": "CN", "Spain": "ES", "Australia": "AU", "South Korea": "KR",
}

@st.cache_resource
def make_session():
    print("Creating resilient network session...")
    retry = Retry(total=3, status_forcelist=[429, 500, 502, 503, 504], backoff_factor=1)
    adapter = HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

session = make_session()

@st.cache_resource
def load_data():
    print("Loading data and building recommender brain... (This runs only once)")
    try:
        df_base = pd.read_csv('movies_metadata.csv', low_memory=False)
    except FileNotFoundError:
        st.error("Error: 'movies_metadata.csv' not found.")
        st.stop()

    df_base['id'] = pd.to_numeric(df_base['id'], errors='coerce')
    df_base.dropna(subset=['id'], inplace=True)
    df_base['id'] = df_base['id'].astype(int)
    df_base.drop_duplicates(subset=['id'], inplace=True)
    df_base['vote_average'] = pd.to_numeric(df_base['vote_average'], errors='coerce')
    df_base['vote_count'] = pd.to_numeric(df_base['vote_count'], errors='coerce')
    df_base['original_language'] = df_base['original_language'].fillna('N/A')

    def parse_json_list(text, key='name'):
        if pd.isna(text): return []
        try:
            items = ast.literal_eval(text)
            if isinstance(items, list):
                return [i[key] for i in items if isinstance(i, dict) and key in i]
            return []
        except (ValueError, SyntaxError): return []
        except Exception: return []

    def top_3_cast(text):
        if pd.isna(text): return []
        try:
            items = ast.literal_eval(text)
            if isinstance(items, list):
                return [i['name'].replace(" ", "") for i in items[:3] if isinstance(i, dict) and 'name' in i]
            return []
        except (ValueError, SyntaxError): return []
        except Exception: return []

    disc_df = df_base[['id', 'title', 'genres', 'production_countries', 'original_language', 'vote_average', 'vote_count', 'imdb_id']].copy()
    disc_df.dropna(subset=['vote_average', 'vote_count'], inplace=True)
    disc_df['genres_list'] = disc_df['genres'].apply(lambda x: parse_json_list(x, key='name'))
    disc_df['countries_list'] = disc_df['production_countries'].apply(lambda x: parse_json_list(x, key='name'))

    C = disc_df['vote_average'].mean()
    m = disc_df['vote_count'].quantile(0.90)

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        if pd.isna(v) or pd.isna(R) or (v + m) == 0:
            return C
        return (v / (v + m)) * R + (m / (v + m)) * C

    disc_df['weighted_rating'] = disc_df.apply(weighted_rating, axis=1)

    all_countries = set([item for sublist in disc_df['countries_list'] if isinstance(sublist, list) for item in sublist])
    countries = sorted(list(all_countries))
    countries.insert(0, "Any Country")

    all_languages = set(df_base['original_language'].unique())
    languages = sorted([lang for lang in all_languages if isinstance(lang, str) and len(lang) > 1])
    languages.insert(0, "Any Language")

    try:
        credits = pd.read_csv('credits.csv')
    except FileNotFoundError:
        st.error("Error: 'credits.csv' not found.")
        st.stop()
    credits['id'] = pd.to_numeric(credits['id'], errors='coerce')
    credits.dropna(subset=['id'], inplace=True)
    credits['id'] = credits['id'].astype(int)

    try:
        keywords = pd.read_csv('keywords.csv')
    except FileNotFoundError:
        st.error("Error: 'keywords.csv' not found.")
        st.stop()
    keywords['id'] = pd.to_numeric(keywords['id'], errors='coerce')
    keywords.dropna(subset=['id'], inplace=True)
    keywords['id'] = keywords['id'].astype(int)

    content = df_base[['id', 'title', 'imdb_id', 'overview', 'genres', 'production_countries']].copy()
    content = content.merge(credits[['id', 'cast']], on='id', how='left')
    content = content.merge(keywords[['id', 'keywords']], on='id', how='left')

    content['genres_list'] = content['genres'].apply(lambda x: [i.replace(" ", "") for i in parse_json_list(x, key='name')])
    content['cast_list'] = content['cast'].apply(top_3_cast)
    content['keywords_list'] = content['keywords'].apply(lambda x: [i.replace(" ", "") for i in parse_json_list(x, key='name')])
    content['countries_list'] = content['production_countries'].apply(lambda x: parse_json_list(x, key='name'))

    content['overview'] = content['overview'].fillna('').apply(lambda x: x.split())
    content['combined_features'] = (content['genres_list'].apply(lambda x: x if isinstance(x, list) else []) * 3) + (content['cast_list'].apply(lambda x: x if isinstance(x, list) else []) * 3) + (content['keywords_list'].apply(lambda x: x if isinstance(x, list) else []) * 2) + content['overview']
    content['combined_string'] = content['combined_features'].apply(lambda x: " ".join(x).lower() if isinstance(x, list) else "")

    content = content[['id', 'title', 'combined_string', 'imdb_id', 'countries_list']].copy()
    content['countries_list'] = content['countries_list'].apply(lambda x: x if isinstance(x, list) else [])

    tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
    tfidf_features = tfidf.fit_transform(content['combined_string'])
    content = content.reset_index()

    print("...Brain build complete.")
    return tfidf_features, content, disc_df, countries, languages

@st.cache_data
def fetch_genres(api_key):
    print("Fetching genre map from API...")
    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={api_key}&language=en-US"
    genres = {}
    try:
        data = session.get(url, timeout=5).json()
        for g in data.get('genres', []):
            if isinstance(g, dict) and 'name' in g and 'id' in g:
                genres[g['name']] = g['id']
    except Exception as e:
        print(f"Error fetching genre map: {e}")
        genres = {
            'Action': 28, 'Adventure': 12, 'Animation': 16, 'Comedy': 35, 'Crime': 80,
            'Documentary': 99, 'Drama': 18, 'Family': 10751, 'Fantasy': 14, 'History': 36,
            'Horror': 27, 'Music': 10402, 'Mystery': 9648, 'Romance': 10749,
            'Science Fiction': 878, 'TV Movie': 10770, 'Thriller': 53, 'War': 10752, 'Western': 37
        }
    return genres

def movie_info(movie_id):
    details = {
        'poster': "https://via.placeholder.com/500x750.png?text=Poster+Not+Found",
        'title': "Not Found", 'year': "", 'imdb_id': None, 'overview': 'No overview available.', 'cast': [], 'watch_link': '#'
    }
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_KEY}&language=en-US&append_to_response=credits"
        data = session.get(url, timeout=5).json()
        if data.get('poster_path'):
            details['poster'] = "https://image.tmdb.org/t/p/w500" + data['poster_path']
        details['title'] = data.get('title', 'Title not found')
        if data.get('release_date') and isinstance(data['release_date'], str) and '-' in data['release_date']:
            details['year'] = data['release_date'].split('-')[0]
        else:
            details['year'] = 'N/A'
        details['imdb_id'] = data.get('imdb_id')
        details['overview'] = data.get('overview', 'No overview available.')
        credits_data = data.get('credits', {})
        cast_list = credits_data.get('cast', [])
        if isinstance(cast_list, list):
            details['cast'] = [actor['name'] for actor in cast_list[:5] if isinstance(actor, dict) and 'name' in actor]

        url_watch = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers?api_key={TMDB_KEY}"
        data_watch = session.get(url_watch, timeout=5).json()
        results_watch = data_watch.get('results', {})
        link_in = results_watch.get('IN', {}).get('link')
        link_us = results_watch.get('US', {}).get('link')
        if link_in: details['watch_link'] = link_in
        elif link_us: details['watch_link'] = link_us
    except Exception as e:
        print(f"Error in movie_details for ID {movie_id}: {e}")

    if "placeholder" in details['poster'] and details['imdb_id']:
        poster_url_omdb, _ = omdb_info(details['imdb_id'])
        if poster_url_omdb and "placeholder" not in poster_url_omdb:
            details['poster'] = poster_url_omdb
    return details

def tmdb_info(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_KEY}&language=en-US"
    placeholder = "https://via.placeholder.com/500x750.png?text=Poster+Not+Found"
    imdb_id = None
    overview = None
    try:
        data = session.get(url, timeout=5).json()
        if data.get('status_code') and data.get('status_code') != 200 and data.get('success') is False:
            print(f"TMDb API error for ID {movie_id}: {data.get('status_message')}")
            return placeholder, None, None
        imdb_id = data.get('imdb_id')
        poster_path = data.get('poster_path')
        overview = data.get('overview')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path, imdb_id, overview
        else:
            return placeholder, imdb_id, overview
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching TMDb poster for ID {movie_id}: {e}")
        return placeholder, None, None
    except Exception as e:
        print(f"Error fetching TMDb poster for ID {movie_id}: {e}")
        return placeholder, None, None

def omdb_info(imdb_id):
    if not imdb_id or imdb_id == 'N/A':
        return "https://via.placeholder.com/500x750.png?text=Poster+Not+Found", None
    url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_KEY}"
    placeholder = "https://via.placeholder.com/500x750.png?text=Poster+Not+Found"
    overview = None
    try:
        data = session.get(url, timeout=5).json()
        if data.get('Response') == 'False':
            print(f"OMDb API error for ID {imdb_id}: {data.get('Error')}")
            return placeholder, None
        poster_url = data.get('Poster')
        overview = data.get('Plot')
        if poster_url and poster_url != "N/A":
            return poster_url, overview
        else:
            return placeholder, overview
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching OMDb poster for ID {imdb_id}: {e}")
        return placeholder, None
    except Exception as e:
        print(f"Error fetching OMDb poster for ID {imdb_id}: {e}")
        return placeholder, None

def fetch_tmdb_recs(genre_ids, num_recs=15, language=None, region=None):
    api_recs = []
    if not genre_ids: return []
    genre_id_str = ",".join(map(str, genre_ids))
    for page in range(1, 3):
        url = f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_KEY}&with_genres={genre_id_str}&primary_release_date.gte=2018-01-01&sort_by=popularity.desc&page={page}"
        if language and language != "Any Language": url += f"&with_original_language={language}"
        if region: url += f"&region={region}"
        try:
            data = session.get(url, timeout=5).json()
            results = data.get('results', [])
            if not isinstance(results, list): continue
            for movie in results:
                if isinstance(movie, dict):
                    imdb_id = None
                    try:
                        details_url = f"https://api.themoviedb.org/3/movie/{movie.get('id')}?api_key={TMDB_KEY}"
                        details_data = session.get(details_url, timeout=3).json()
                        imdb_id = details_data.get('imdb_id')
                    except Exception: pass
                    api_recs.append({'title': movie.get('title', 'Unknown Title'), 'id': movie.get('id'), 'imdb_id': imdb_id})
        except Exception as e:
            print(f"  >! API rec fetch (page {page}) failed: {e}")
            break
        if len(api_recs) >= num_recs: break
    return api_recs[:num_recs]

def local_recs(idx, actual_title, content, tfidf_features, country_filter="Any Country", num_recs=15):
    brain_recs = []
    seen = {actual_title}
    if not isinstance(idx, (int, np.integer)): return []
    try:
        if idx < 0 or idx >= tfidf_features.shape[0]: return []
        sims_arr = cosine_similarity(tfidf_features[idx], tfidf_features)
        sims = list(enumerate(sims_arr[0]))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
    except Exception as e:
        print(f"Error during similarity calc in local_brain_recs: {e}")
        return []
    for rec_index, score in sims[1:]:
        if len(brain_recs) >= num_recs: break
        if rec_index < 0 or rec_index >= len(content): continue
        try: rec = content.iloc[rec_index]
        except IndexError: continue
        title = rec.get('title', 'Unknown Title')
        valid_country = False
        if country_filter == "Any Country": valid_country = True
        else:
            countries_list = rec.get('countries_list')
            if isinstance(countries_list, list) and country_filter in countries_list: valid_country = True
        if title not in seen and valid_country:
            if 'id' in rec and 'imdb_id' in rec:
                brain_recs.append({'title': title, 'id': rec['id'], 'imdb_id': rec['imdb_id']})
                seen.add(title)
            else:
                print(f"Warning: Skipping brain rec due to missing keys: {rec.get('id', 'N/A')}")
    return brain_recs

def get_recs(search_title, content, tfidf_features, language=None, country="Any Country"):
    potential_recs = []
    actual_title = search_title
    found_id = None
    region = COUNTRY_CODES.get(country) if country and country != "Any Country" else None
    st.session_state.recommendations_full = []
    st.session_state.recommendations_show_count = 10
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_KEY}&query={search_title}"
        if language and language != "Any Language": search_url += f"&language={language}"
        if region: search_url += f"&region={region}"
        data = session.get(search_url, timeout=5).json()
        results = data.get('results', [])
        if not results or not isinstance(results, list):
            st.error(f"No API results for '{search_title}' with selected filters.")
            return [], actual_title, None
        movie = results[0]
        if not isinstance(movie, dict):
            st.error("Invalid movie data received from API.")
            return [], actual_title, None
        actual_title = movie.get('title', search_title)
        api_id = movie.get('id')
        found_id = api_id
        api_genre_ids = movie.get('genre_ids', [])
        if api_id is None:
            st.error("Could not retrieve movie ID from API.")
            return [], actual_title, None
        matches = content[content['id'] == api_id]
        num_api_needed = 30
        if not matches.empty:
            match_row = matches.iloc[0]
            int_pos_idx = match_row.get('index')
            if int_pos_idx is not None and int_pos_idx >= 0:
                brain_list = local_recs(int_pos_idx, actual_title, content, tfidf_features, country_filter=country, num_recs=15)
                potential_recs.extend(brain_list)
                num_api_needed = max(0, 30 - len(potential_recs))
            else:
                print(f"Warning: Found match but invalid index {int_pos_idx} for movie {api_id}")
        else:
            print(f"Movie ID {api_id} ('{actual_title}') not found in local brain dataset.")
        if num_api_needed > 0:
            potential_recs.extend(fetch_tmdb_recs(api_genre_ids, num_recs=num_api_needed, language=language, region=region))
    except requests.exceptions.RequestException as e:
        st.error(f"Network error during API search: {e}")
        return [], actual_title, None
    except Exception as e:
        st.error(f"An unexpected error occurred during recommendation generation: {e}")
        st.exception(e)
        return [], actual_title, None
    final_potential_recs = []
    seen_ids = set()
    for rec in potential_recs:
        rec_id = rec.get('id')
        if rec_id is not None and rec_id not in seen_ids:
            final_potential_recs.append(rec)
            seen_ids.add(rec_id)
    st.session_state.recommendations_full = final_potential_recs
    return st.session_state.recommendations_full[:10], actual_title, found_id

if 'discover_results' not in st.session_state: st.session_state.discover_results = []
if 'discover_show_count' not in st.session_state: st.session_state.discover_show_count = 10
if 'recommendations_full' not in st.session_state: st.session_state.recommendations_full = []
if 'recommendations_show_count' not in st.session_state: st.session_state.recommendations_show_count = 10
if 'last_search' not in st.session_state: st.session_state.last_search = ""
if 'found_title_rec' not in st.session_state: st.session_state.found_title_rec = ""
if 'found_id_rec' not in st.session_state: st.session_state.found_id_rec = None
if 'last_discover_filters' not in st.session_state: st.session_state.last_discover_filters = ""


st.title('BingeRecc >_<')
st.write('Search for a movie to get recommendations, or discover new movies by genre and country.')

try:
    tfidf_features, content, disc_df, countries, languages = load_data()
    genres_map = fetch_genres(TMDB_KEY)
except FileNotFoundError as e:
    st.error(f"Data file not found: {e.filename}. Please place required CSV files in the app folder.")
    st.stop()
except ValueError as e:
    st.error(f"Error unpacking data during loading: {e}. Check load_data_and_brain function's return values.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during data loading.")
    st.exception(e)
    st.stop()


tab1, tab2 = st.tabs(["Recommend by Movie", "Discover Movies"])

with tab1:
    st.header('Search by Movie')
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        selected_language = st.selectbox("Filter by Language (Optional)", options=languages, key='lang_filter')
    with col_filter2:
        selected_country_rec = st.selectbox("Filter by Country (Optional)", options=countries, key='country_filter')
    search_title = st.text_input('Search for a movie...', 'The Dark Knight', key='search_input')
    
    if st.session_state.last_search != search_title:
        st.session_state.recommendations_full = []
        st.session_state.recommendations_show_count = 10
        st.session_state.found_id_rec = None
        st.session_state.last_search = search_title
    
    if st.button('Get Recommendations', key='btn_rec_movie'):
        if search_title:
            with st.spinner(f'Searching for "{search_title}"...'):
                initial_recs, found_title, found_id = get_recs(search_title, content, tfidf_features, selected_language, selected_country_rec)
                st.session_state.found_title_rec = found_title
                st.session_state.found_id_rec = found_id
        else:
            st.warning('Please enter a movie title.')
    
    if st.session_state.get('found_id_rec'):
        st.subheader(f'Showing results for: {st.session_state.found_title_rec}')
        details = movie_info(st.session_state.found_id_rec)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.image(details['poster'], use_container_width=True)
            st.markdown(f"<h3 style='text-align: center;'>{details['title']} ({details['year']})</h3>", unsafe_allow_html=True)
        st.subheader('Synopsis')
        st.write(details['overview'])
        if details['cast']:
            st.subheader('Starring')
            st.write(", ".join(details['cast']))
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if details['watch_link'] != '#':
                st.link_button("Where to Watch", details['watch_link'], use_container_width=True)
            else:
                st.button("Watch Info N/A", disabled=True, use_container_width=True)
        with btn_col2:
            if details['imdb_id']:
                st.link_button("IMDb Page", f"https://www.imdb.com/title/{details['imdb_id']}", use_container_width=True)
            else:
                st.button("IMDb N/A", disabled=True, use_container_width=True)
        st.divider()
    
    if st.session_state.get('recommendations_full'):
        st.subheader('Your Recommendations')
        recs_to_show = st.session_state.recommendations_full[:st.session_state.recommendations_show_count]
        num_cols = 10
        num_rows = (len(recs_to_show) + num_cols - 1) // num_cols
        for row_num in range(num_rows):
            cols = st.columns(num_cols)
            start_index = row_num * num_cols
            end_index = min(start_index + num_cols, len(recs_to_show))
            for i, movie in enumerate(recs_to_show[start_index:end_index]):
                col_index = i
                if col_index < len(cols):
                    col = cols[col_index]
                    if isinstance(movie, dict) and 'id' in movie and 'title' in movie:
                        with col:
                            poster_url, imdb_id, overview = tmdb_info(movie['id'])
                            final_imdb_id = imdb_id if imdb_id else movie.get('imdb_id')
                            if "placeholder" in poster_url and final_imdb_id:
                                poster_url, omdb_overview = omdb_info(final_imdb_id)
                                if not overview: overview = omdb_overview
                            if not overview: overview = "No overview available."
                            st.image(poster_url, use_container_width=True)
                            st.caption(f"**{movie['title']}**")
                            with st.expander("Details"):
                                st.write(overview)
                                if final_imdb_id:
                                    st.link_button("IMDb", f"https://www.imdb.com/title/{final_imdb_id}", use_container_width=True)
                                else:
                                    st.button("IMDb", disabled=True, use_container_width=True, help="IMDb ID not found", key=f"imdb_rec_disabled_{row_num}_{i}_{movie['id']}")
                    else:
                        print(f"Warning: Skipping invalid movie recommendation format: {movie}")
                else:
                    print(f"Warning: Column index {col_index} out of bounds for row {row_num}")
        if len(st.session_state.recommendations_full) > st.session_state.recommendations_show_count:
            if st.button("Load More Recommendations", key='load_more_recs'):
                st.session_state.recommendations_show_count += 10
                st.rerun()

with tab2:
    st.header('Discover Movies by Genre & Country')
    col1, col2 = st.columns(2)
    with col1:
        selected_genres = st.multiselect("Select Genres", options=list(genres_map.keys()), key='genre_discover')
    with col2:
        selected_country_disc = st.selectbox("Select Country", options=countries, key='country_discover')
    sort_option = st.radio("Show results by:", ("Top Rated", "Random"), horizontal=True, key='sort_discover')
    filters_key = f"{selected_genres}-{selected_country_disc}-{sort_option}"
    if st.session_state.last_discover_filters != filters_key:
        st.session_state.discover_results = []
        st.session_state.discover_show_count = 10
        st.session_state.last_discover_filters = filters_key
    if st.button('Discover Movies', key='btn_discover_action'):
        st.session_state.discover_results = []
        st.session_state.discover_show_count = 10
        with st.spinner("Searching for movies..."):
            filtered_df = disc_df.copy()
            if selected_genres:
                def genre_match(genres_list):
                    if not isinstance(genres_list, list): return False
                    return all(g in genres_list for g in selected_genres)
                filtered_df = filtered_df[filtered_df['genres_list'].apply(genre_match)]
            if selected_country_disc != "Any Country":
                def country_match(countries_list):
                    if not isinstance(countries_list, list): return False
                    return selected_country_disc in countries_list
                filtered_df = filtered_df[filtered_df['countries_list'].apply(country_match)]
            if filtered_df.empty:
                st.warning(f"No movies found matching your criteria. Try different filters!")
            else:
                max_results = 50
                if sort_option == "Top Rated":
                    st.session_state.discover_results = filtered_df.sort_values('weighted_rating', ascending=False).head(max_results).to_dict('records')
                else:
                    st.session_state.discover_results = filtered_df.sample(n=min(max_results, len(filtered_df))).to_dict('records')
    if st.session_state.get('discover_results'):
        st.subheader(f"Showing {sort_option} movies matching your filters:")
        results_to_show = st.session_state.discover_results[:st.session_state.discover_show_count]
        num_cols = 10
        num_rows = (len(results_to_show) + num_cols - 1) // num_cols
        for row_num in range(num_rows):
            cols = st.columns(num_cols)
            start_index = row_num * num_cols
            end_index = min(start_index + num_cols, len(results_to_show))
            for i, movie in enumerate(results_to_show[start_index:end_index]):
                col_index = i
                if col_index < len(cols):
                    col = cols[col_index]
                    if isinstance(movie, dict) and 'id' in movie and 'title' in movie:
                        with col:
                            poster_url, _, overview = tmdb_info(movie['id'])
                            imdb_id_discover = movie.get('imdb_id')
                            if "placeholder" in poster_url and imdb_id_discover:
                                poster_url, omdb_overview = omdb_info(imdb_id_discover)
                                if not overview: overview = omdb_overview
                            if not overview: overview = "No overview available."
                            st.image(poster_url, use_container_width=True)
                            st.caption(f"**{movie['title']}**")
                            with st.expander("Details"):
                                st.write(overview)
                                if imdb_id_discover:
                                    st.link_button("IMDb", f"https://www.imdb.com/title/{imdb_id_discover}", use_container_width=True)
                                else:
                                    st.button("IMDb", disabled=True, use_container_width=True, help="IMDb ID not found", key=f"imdb_disc_disabled_{row_num}_{i}_{movie['id']}")
                    else:
                        print(f"Warning: Skipping invalid discover movie format: {movie}")
                else:
                    print(f"Warning: Column index {col_index} out of bounds for row {row_num}")
        if len(st.session_state.discover_results) > st.session_state.discover_show_count and st.session_state.discover_results:
            if st.button("Load More", key='load_more_discover'):
                st.session_state.discover_show_count += 10
                st.rerun()
