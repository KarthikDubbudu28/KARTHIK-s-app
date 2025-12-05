# pages/4_Sentiment_Analysis.py

import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import pydeck as pdk
import simplekml
import io

# ----------------------------
# PAGE SETTINGS
# ----------------------------
st.set_page_config(page_title="Sentiment Analysis (Trial Version)", layout="wide")
st.title("📌 Sentiment Analysis & Geospatial Mapping — Trial Version")
st.write("Upload a CSV file containing text data & user locations.")

# ----------------------------
# SIDEBAR — File upload
# ----------------------------
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

geo_limit = st.sidebar.number_input("Max Geocoding Attempts", min_value=10, max_value=500, value=100)
enable_kml = st.sidebar.checkbox("Enable KML Download", value=False)

# ----------------------------
# Sentiment Analyzer
# ----------------------------
@st.cache_data
def load_vader():
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()

analyzer = load_vader()

def analyze_sentiment(text):
    s = analyzer.polarity_scores(str(text))
    c = s["compound"]
    if c >= 0.05:
        label = "positive"
    elif c <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return c, label

# ----------------------------
# Subjectivity / Objectivity
# ----------------------------
def get_subjectivity(text):
    tb = TextBlob(str(text))
    subjectivity = tb.sentiment.subjectivity     # 0 = objective, 1 = subjective
    objectivity = 1 - subjectivity
    return subjectivity, objectivity

# ----------------------------
# Geocoding Function
# ----------------------------
def geocode_locations(location_series, max_attempts=100):
    geolocator = Nominatim(user_agent="streamlit-geocoder")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    lat_list, lon_list, resolved_list = [], [], []
    attempts = 0

    for loc in location_series:
        if attempts >= max_attempts or not loc or str(loc).strip() == "":
            lat_list.append(None)
            lon_list.append(None)
            resolved_list.append(None)
            continue

        attempts += 1
        try:
            place = geocode(loc)
            if place:
                lat_list.append(place.latitude)
                lon_list.append(place.longitude)
                resolved_list.append(place.address)
            else:
                lat_list.append(None)
                lon_list.append(None)
                resolved_list.append(None)
        except:
            lat_list.append(None)
            lon_list.append(None)
            resolved_list.append(None)

    return lat_list, lon_list, resolved_list

# ----------------------------
# PROCESS CSV
# ----------------------------
if uploaded_file is not None:

    # Read CSV
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        st.error("Error reading CSV file. Please upload a valid CSV.")
        st.stop()

    # Required columns
    required_cols = ["content", "user_location"]

    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain the following columns: {required_cols}")
        st.stop()

    st.success("File successfully uploaded!")

    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(df.head())

    # ----------------------------
    # Sentiment Analysis
    # ----------------------------
    st.subheader("🔎 Sentiment Analysis")

    df["sentiment_score"], df["sentiment_label"] = zip(*df["content"].apply(analyze_sentiment))
    df["subjectivity"], df["objectivity"] = zip(*df["content"].apply(get_subjectivity))

    st.write("Sentiment label counts:")
    st.write(df["sentiment_label"].value_counts())

    st.write("Subjectivity / Objectivity example:")
    st.dataframe(df[["content", "subjectivity", "objectivity"]].head())

    # ----------------------------
    # Geocoding
    # ----------------------------
    st.subheader("🌍 Geocoding user locations (may take time)")
    with st.spinner("Geocoding locations..."):
        df["latitude"], df["longitude"], df["resolved_location"] = geocode_locations(
            df["user_location"], geo_limit
        )

    map_df = df.dropna(subset=["latitude", "longitude"])
    st.success(f"Geocoded {len(map_df)} rows successfully")

    # ----------------------------
    # Map Visualization
    # ----------------------------
    st.subheader("🗺 Geospatial Map")

    if not map_df.empty:
        map_df["color"] = map_df["sentiment_label"].map({
            "positive": [0, 200, 0],
            "negative": [200, 0, 0],
            "neutral": [200, 200, 50]
        })

        st.pydeck_chart(
            pdk.Deck(
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
                tooltip={"text": "{content}\nSentiment: {sentiment_label}\nSubjectivity: {subjectivity}"},
            )
        )
    else:
        st.info("No valid geocoded locations found.")

    # ----------------------------
    # Final Table
    # ----------------------------
    st.subheader("📌 Full Processed Data")
    st.dataframe(df)

    # ----------------------------
    # KML Download
    # ----------------------------
    if enable_kml and not map_df.empty:
        kml = simplekml.Kml()
        for _, r in map_df.iterrows():
            p = kml.newpoint(
                name=r.get("username") or "text",
                description=f"{r['content']}\nSentiment: {r['sentiment_label']}"
            )
            p.coords = [(r["longitude"], r["latitude"])]
        bio = io.BytesIO()
        kml.save(bio)
        bio.seek(0)

        st.download_button(
            "Download KML File",
            data=bio,
            file_name="sentiment_map.kml",
            mime="application/vnd.google-earth.kml+xml",
        )

else:
    st.info("Please upload a CSV file to begin analysis.")
