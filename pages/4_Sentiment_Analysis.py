import streamlit as st
import pandas as pd
import tweepy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pydeck as pdk
import simplekml
import io

st.set_page_config(page_title="Sentiment Analysis", layout="wide")

st.title("Twitter Sentiment & Geospatial Analysis (API v2 Compatible)")
st.write("Fetch tweets using Twitter API v2, analyze sentiment, geocode user locations, and visualize them on a map.")

# -------------------------------
#       TWITTER AUTH SETUP
# -------------------------------
if "twitter" not in st.secrets:
    st.error("Twitter API Bearer Token missing in secrets. Add it under [twitter].")
    st.stop()

bearer_token = st.secrets["twitter"]["bearer_token"]
client = tweepy.Client(bearer_token=bearer_token)

# -------------------------------
#       VADER SENTIMENT
# -------------------------------
@st.cache_data
def init_vader():
    try:
        import nltk
        nltk.download("vader_lexicon")
    except:
        pass
    return SentimentIntensityAnalyzer()

analyzer = init_vader()

def compute_sentiment(text):
    s = analyzer.polarity_scores(text)
    comp = s["compound"]
    if comp >= 0.05:
        label = "positive"
    elif comp <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return comp, label

# -------------------------------
#     SIDEBAR USER INPUT
# -------------------------------
st.sidebar.header("Search Options")
query = st.sidebar.text_input("Search Keyword", value="climate change")
max_results = st.sidebar.slider("Number of tweets", 10, 100, 50)
geocode_limit = st.sidebar.slider("Max Geocoding Attempts", 10, 200, 50)
enable_kml = st.sidebar.checkbox("Enable KML download", False)

# -------------------------------
#     FETCH TWEETS (TWEEPY)
# -------------------------------
def fetch_tweets(q, limit):
    results = client.search_recent_tweets(
        query=q,
        max_results=limit,
        tweet_fields=["created_at", "text"],
        expansions=["author_id"],
        user_fields=["location", "name", "username"]
    )

    if results.data is None:
        return pd.DataFrame()

    tweets = results.data
    users = {u.id: u for u in results.includes["users"]}

    rows = []
    for t in tweets:
        user = users.get(t.author_id)
        rows.append({
            "date": t.created_at,
            "content": t.text,
            "username": user.username if user else "",
            "display_name": user.name if user else "",
            "user_location": user.location if user and user.location else ""
        })

    return pd.DataFrame(rows)

# -------------------------------
#          GEOCODING
# -------------------------------
def geocode_locations(loc_series, max_attempts=50):
    geolocator = Nominatim(user_agent="streamlit-geocoder")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    lats, lons = [], []
    attempts = 0

    for loc in loc_series:
        if attempts >= max_attempts:
            lats.append(None)
            lons.append(None)
            continue

        if not loc:
            lats.append(None)
            lons.append(None)
            continue

        attempts += 1
        try:
            place = geocode(loc)
            if place:
                lats.append(place.latitude)
                lons.append(place.longitude)
            else:
                lats.append(None)
                lons.append(None)
        except:
            lats.append(None)
            lons.append(None)

    return lats, lons

# -------------------------------
#     RUN SEARCH BUTTON
# -------------------------------
if st.sidebar.button("Run Analysis"):
    with st.spinner("Fetching tweets..."):
        df = fetch_tweets(query, max_results)

    if df.empty:
        st.warning("No tweets found.")
        st.stop()

    st.success(f"Fetched {len(df)} tweets")

    # Sentiment
    df["sentiment_score"], df["sentiment_label"] = zip(*df["content"].apply(compute_sentiment))

    st.subheader("Sentiment Distribution")
    st.write(df["sentiment_label"].value_counts())

    # Geocoding
    with st.spinner("Geocoding user locations (slow due to rate limits)..."):
        df["latitude"], df["longitude"] = geocode_locations(df["user_location"], geocode_limit)

    map_df = df.dropna(subset=["latitude", "longitude"])

    st.subheader("Geospatial Map of Tweets")
    st.write(f"{len(map_df)} tweets geocoded successfully")

    if not map_df.empty:
        map_df["color"] = map_df["sentiment_label"].map({
            "positive": [0, 200, 0],
            "negative": [200, 0, 0],
            "neutral": [200, 200, 50]
        })

        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=map_df["latitude"].mean(),
                longitude=map_df["longitude"].mean(),
                zoom=2,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df,
                    get_position=["longitude", "latitude"],
                    get_color="color",
                    get_radius=30000,
                    pickable=True,
                )
            ],
            tooltip={"text": "{display_name}\n{sentiment_label}\n{content}"}
        ))
    else:
        st.info("No geocoded points available.")

    st.subheader("Raw Data")
    st.dataframe(df)

    # KML DOWNLOAD
    if enable_kml and not map_df.empty:
        kml = simplekml.Kml()
        for _, r in map_df.iterrows():
            p = kml.newpoint(name=r["display_name"], description=r["content"])
            p.coords = [(r["longitude"], r["latitude"])]

        kml_bytes = io.BytesIO()
        kml.save(kml_bytes)
        kml_bytes.seek(0)

        st.download_button(
            "Download KML File",
            data=kml_bytes,
            file_name="tweets.kml",
            mime="application/vnd.google-earth.kml+xml"
        )
