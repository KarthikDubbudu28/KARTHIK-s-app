# pages/4_Sentiment_Analysis.py
import streamlit as st
import pandas as pd
import requests
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pydeck as pdk
import simplekml
import io

st.set_page_config(page_title="Sentiment Analysis", layout="wide")
st.title("Twitter Sentiment & Geospatial (API v2 — No Tweepy)")

# -----------------------
# Check secrets / bearer
# -----------------------
if "twitter" not in st.secrets or "bearer_token" not in st.secrets["twitter"]:
    st.error("Twitter Bearer Token missing. Add the following to Streamlit Secrets (Settings → Secrets):\n\n[twitter]\nbearer_token = \"YOUR_BEARER_TOKEN_HERE\"")
    st.stop()

BEARER_TOKEN = st.secrets["twitter"]["bearer_token"]

# -----------------------
# Fetch tweets via requests (Twitter API v2)
# -----------------------
def fetch_tweets(query: str, max_results: int) -> pd.DataFrame:
    # Twitter recent search endpoint supports max_results between 10 and 100.
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    params = {
        "query": query,
        "max_results": max(10, min(100, int(max_results))),
        "tweet.fields": "created_at,text",
        "expansions": "author_id",
        "user.fields": "id,name,username,location"
    }
    resp = requests.get(url, headers=headers, params=params, timeout=20)
    if resp.status_code != 200:
        st.error(f"Twitter API error {resp.status_code}: {resp.text}")
        return pd.DataFrame()
    j = resp.json()
    tweets = j.get("data", [])
    users = {u["id"]: u for u in j.get("includes", {}).get("users", [])}
    rows = []
    for t in tweets:
        user = users.get(t.get("author_id"), {})
        rows.append({
            "date": t.get("created_at"),
            "content": t.get("text"),
            "username": user.get("username"),
            "display_name": user.get("name"),
            "user_location": user.get("location")
        })
    return pd.DataFrame(rows)

# -----------------------
# VADER sentiment
# -----------------------
@st.cache_data
def get_vader():
    try:
        import nltk
        nltk.download("vader_lexicon", quiet=True)
    except Exception:
        # If download fails, analyzer may still exist in environment
        pass
    return SentimentIntensityAnalyzer()

analyzer = get_vader()

def analyze_text_sentiment(text: str):
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    scores = analyzer.polarity_scores(text)
    comp = scores["compound"]
    if comp >= 0.05:
        label = "positive"
    elif comp <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return comp, label

# -----------------------
# Geocoding
# -----------------------
def geocode_locations(loc_series, max_attempts=50):
    geolocator = Nominatim(user_agent="streamlit-sentiment-geocoder")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    lat_list, lon_list, resolved = [], [], []
    attempts = 0
    for loc in loc_series:
        if attempts >= max_attempts or not loc or str(loc).strip() == "":
            lat_list.append(None); lon_list.append(None); resolved.append(None)
            continue
        attempts += 1
        try:
            place = geocode(loc)
            if place:
                lat_list.append(place.latitude)
                lon_list.append(place.longitude)
                resolved.append(place.address)
            else:
                lat_list.append(None); lon_list.append(None); resolved.append(None)
        except Exception:
            lat_list.append(None); lon_list.append(None); resolved.append(None)
    return lat_list, lon_list, resolved

# -----------------------
# Sidebar inputs
# -----------------------
st.sidebar.header("Search options")
query = st.sidebar.text_input("Search query (e.g. #climate)", value="climate change")
max_results = st.sidebar.slider("Number of tweets (max 100)", 10, 100, 50)
geo_limit = st.sidebar.number_input("Max geocoding attempts", min_value=10, max_value=500, value=100, step=10)
enable_kml = st.sidebar.checkbox("Enable KML download", value=False)

# -----------------------
# Run analysis
# -----------------------
if st.sidebar.button("Run"):
    with st.spinner("Querying Twitter..."):
        df = fetch_tweets(query, max_results)
    if df.empty:
        st.warning("No tweets returned. Check your query or bearer token.")
        st.stop()

    # sentiment
    df["sentiment_score"], df["sentiment_label"] = zip(*df["content"].apply(analyze_text_sentiment))

    st.subheader("Sentiment counts")
    st.write(df["sentiment_label"].value_counts().to_frame("count"))

    # geocode user locations
    with st.spinner("Geocoding user locations (may be slow due to rate limits)..."):
        lats, lons, resolved = geocode_locations(df["user_location"].fillna(""), max_attempts=int(geo_limit))
        df["latitude"] = lats
        df["longitude"] = lons
        df["resolved_location"] = resolved

    # map
    map_df = df.dropna(subset=["latitude", "longitude"])
    st.subheader(f"Map — {len(map_df)} geocoded tweets")
    if not map_df.empty:
        def color_from_label(lbl):
            return {
                "positive": [0, 200, 0],
                "negative": [200, 0, 0],
                "neutral": [200, 200, 50]
            }.get(lbl, [100, 100, 100])

        map_df["color"] = map_df["sentiment_label"].apply(color_from_label)
        map_df["radius"] = map_df["sentiment_score"].abs().apply(lambda s: 20000 + (abs(s) * 100000))

        midpoint_lat = map_df["latitude"].mean()
        midpoint_lon = map_df["longitude"].mean()

        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=midpoint_lat, longitude=midpoint_lon, zoom=2),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df,
                    get_position=["longitude", "latitude"],
                    get_color="color",
                    get_radius="radius",
                    pickable=True
                )
            ],
            tooltip={"text": "{display_name}\n{sentiment_label}\n{content}"}
        ))
    else:
        st.info("No geocoded points available to map.")

    st.subheader("Tweets (sample)")
    st.dataframe(df[["date", "username", "user_location", "resolved_location", "sentiment_label", "sentiment_score", "content"]])

    # KML export
    if enable_kml and not map_df.empty:
        kml = simplekml.Kml()
        for _, r in map_df.iterrows():
            n = r.get("display_name") or r.get("username") or "tweet"
            p = kml.newpoint(name=n, description=f"{r.get('content')}\nSentiment: {r.get('sentiment_label')}")
            p.coords = [(r["longitude"], r["latitude"])]
        bio = io.BytesIO()
        kml.save(bio)
        bio.seek(0)
        st.download_button("Download KML", data=bio, file_name="tweets.kml", mime="application/vnd.google-earth.kml+xml")
