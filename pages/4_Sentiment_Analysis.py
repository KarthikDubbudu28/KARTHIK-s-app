# pages/05_Sentiment_Geospatial.py
import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import snscrape.modules.twitter as sntwitter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pydeck as pdk
import simplekml
import io
import time
from typing import Tuple

st.set_page_config(layout="wide", page_title="Sentiment Geospatial")

# --- Sidebar controls ---
st.sidebar.header("Sentiment & Geospatial - Controls")
query = st.sidebar.text_input("Search keyword or hashtag", value="#climate")
max_tweets = st.sidebar.slider("Max tweets to fetch", min_value=10, max_value=1000, value=200, step=10)
geocode_limit = st.sidebar.number_input("Max geocoding attempts", min_value=10, max_value=500, value=100, step=10)
use_kml = st.sidebar.checkbox("Enable KML export", value=False)

# Option: choose tweet source
st.sidebar.markdown("**Tweet source**")
use_snscrape = st.sidebar.radio("Source", ("snscrape (no keys)", "Twitter API (keys required)")) == "snscrape (no keys)"
st.sidebar.markdown("---")

st.title("Sentiment + Geospatial")
st.write("Search tweets, compute sentiment, geocode user locations, and map results.")

# --- Helpers and cached resources ---
@st.cache_data(show_spinner=False)
def init_vader():
    try:
        from nltk import download
        download("vader_lexicon")
    except Exception:
        pass
    return SentimentIntensityAnalyzer()

analyzer = init_vader()

def analyze_sentiment(text: str) -> Tuple[float, str]:
    if not text:
        return 0.0, "neutral"
    s = analyzer.polarity_scores(str(text))
    c = s["compound"]
    if c >= 0.05:
        label = "positive"
    elif c <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return c, label

@st.cache_data(show_spinner=False)
def scrape_with_snscrape(keyword: str, limit: int) -> pd.DataFrame:
    rows = []
    scraper = sntwitter.TwitterSearchScraper(keyword)
    for i, tweet in enumerate(scraper.get_items()):
        if i >= limit:
            break
        rows.append({
            "date": tweet.date,
            "id": tweet.id,
            "user": getattr(tweet.user, "username", None),
            "display_name": getattr(tweet.user, "displayname", None),
            "content": tweet.content,
            "user_location": getattr(tweet.user, "location", None),
            "tweet_coordinates": getattr(tweet, "coordinates", None)  # sometimes available
        })
    return pd.DataFrame(rows)

# OPTIONAL: placeholder for official twitter api fetch (use st.secrets)
def fetch_with_twitter_api(keyword: str, limit: int) -> pd.DataFrame:
    # This function is an example skeleton if you prefer the official Twitter API.
    # It expects keys in st.secrets["twitter"] and Tweepy configured.
    # If you need this, I can fill it out to match your current app's auth method.
    raise NotImplementedError("Twitter API fetch not implemented in this page. Use snscrape or ask me to implement API-based fetching.")

@st.cache_data(show_spinner=False)
def geocode_locations(loc_series: pd.Series, max_attempts=100):
    geolocator = Nominatim(user_agent="streamlit-sentiment-geospatial")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2, error_wait_seconds=2.0)
    lat, lon, resolved = [], [], []
    attempts = 0
    for loc in loc_series:
        if attempts >= max_attempts:
            lat.append(None); lon.append(None); resolved.append(None); continue
        if not loc or str(loc).strip() == "":
            lat.append(None); lon.append(None); resolved.append(None); continue
        attempts += 1
        try:
            place = geocode(loc)
            if place:
                lat.append(place.latitude)
                lon.append(place.longitude)
                resolved.append(place.address)
            else:
                lat.append(None); lon.append(None); resolved.append(None)
        except Exception:
            lat.append(None); lon.append(None); resolved.append(None)
    return lat, lon, resolved

def create_kml(df: pd.DataFrame) -> bytes:
    kml = simplekml.Kml()
    for _, r in df.dropna(subset=["latitude", "longitude"]).iterrows():
        p = kml.newpoint(name=r.get("display_name") or r.get("user") or "tweet",
                         description=f"{r.get('content')}\nSentiment: {r.get('sentiment_label')} ({r.get('sentiment_score')})")
        p.coords = [(r["longitude"], r["latitude"])]
    bio = io.BytesIO()
    kml.save(bio)
    return bio.getvalue()

# --- Run search button ---
if st.button("Run search"):
    with st.spinner("Fetching tweets..."):
        if use_snscrape:
            df = scrape_with_snscrape(query, max_tweets)
        else:
            # alternative: use Twitter API — requires st.secrets and implementation
            st.warning("Twitter API option selected but not implemented. Falling back to snscrape.")
            df = scrape_with_snscrape(query, max_tweets)

    st.success(f"Fetched {len(df)} tweets")
    if df.empty:
        st.info("No tweets found for this query.")
    else:
        # sentiment
        df["sentiment_score"], df["sentiment_label"] = zip(*df["content"].apply(analyze_sentiment))
        st.subheader("Sentiment counts")
        st.write(df["sentiment_label"].value_counts())

        # geocode user_location (note: many locations are free-text)
        with st.spinner("Geocoding user locations (Nominatim) ..."):
            lats, lons, resolved = geocode_locations(df["user_location"], max_attempts=int(geocode_limit))
            df["latitude"] = lats
            df["longitude"] = lons
            df["resolved_location"] = resolved

        map_df = df.dropna(subset=["latitude", "longitude"])
        st.subheader("Map of geocoded tweets")
        st.write(f"{len(map_df)} tweets geocoded")

        if not map_df.empty:
            def color_from_score(score):
                try:
                    s = float(score)
                except Exception:
                    s = 0.0
                if s >= 0.05:
                    return [34, 200, 34]
                elif s <= -0.05:
                    return [200, 50, 50]
                else:
                    return [200, 200, 50]

            map_df["color"] = map_df["sentiment_score"].apply(lambda s: color_from_score(s))
            map_df["radius"] = map_df["sentiment_score"].apply(lambda s: 50000 + abs(s) * 100000)

            st.pydeck_chart(pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=map_df["latitude"].mean(),
                    longitude=map_df["longitude"].mean(),
                    zoom=2
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=map_df,
                        get_position=["longitude", "latitude"],
                        get_color="color",
                        get_radius="radius",
                        pickable=True
                    ),
                ],
                tooltip={"text": "{display_name}\n{sentiment_label}\n{content}"}
            ))
        else:
            st.info("No geocoded points to display.")

        st.subheader("Sample results")
        st.dataframe(df[["date", "user", "user_location", "resolved_location", "sentiment_label", "sentiment_score", "content"]].head(200))

        if use_kml:
            kml_bytes = create_kml(df)
            st.download_button("Download KML", data=kml_bytes, file_name="tweets.kml", mime="application/vnd.google-earth.kml+xml")

        # Optionally store results in session state for later pages/actions
        st.session_state["last_sentiment_df"] = df
