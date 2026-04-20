# Run this with: streamlit run app.py

"""
AI Sentiment Radar — VibeCheck
================================
A Streamlit app that fetches live Google News headlines for any topic
and performs AI-powered sentiment analysis using TextBlob NLP.

Run: streamlit run app.py
"""

import datetime
import xml.etree.ElementTree as ET
from typing import Optional, List, Dict
from html import unescape
import re

import pandas as pd
import requests
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
from textblob import TextBlob

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Sentiment Radar – VibeCheck",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS (Modern Dark Theme)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 18px;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15);
    }

    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 2rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: 0.02em;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
    }

    .news-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 14px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .news-card:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateX(4px);
    }

    .sentiment-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }

    .pill-positive { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }
    .pill-negative { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .pill-neutral  { background: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.4); }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }

    .hero-sub {
        text-align: center;
        color: #64748b;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #c084fc;
        margin-bottom: 0.8rem;
    }

    .vibe-meter {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #f87171, #fbbf24, #4ade80);
        position: relative;
        margin: 10px 0;
    }

    .source-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Google News RSS Logic
# ---------------------------------------------------------------------------
def clean_html(raw_html: str) -> str:
    """Remove HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", unescape(raw_html))


def fetch_news_headlines(search_term: str, max_results: int = 30) -> Optional[pd.DataFrame]:
    """
    Fetch recent news headlines from Google News RSS feed and
    perform sentiment analysis on each headline.
    """
    try:
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(search_term)}&hl=en-IN&gl=IN&ceid=IN:en"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        channel = root.find("channel")
        if channel is None:
            return None

        items = channel.findall("item")[:max_results]
        if not items:
            return None

        processed = []
        for item in items:
            title = clean_html(item.findtext("title", ""))
            source = item.findtext("source", "Unknown")
            link = item.findtext("link", "")
            pub_date = item.findtext("pubDate", "")

            # Sentiment Analysis
            blob = TextBlob(title)
            polarity = blob.sentiment.polarity      # -1 to +1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1

            if polarity > 0.1:
                label = "Positive"
            elif polarity < -0.1:
                label = "Negative"
            else:
                label = "Neutral"

            processed.append({
                "headline": title,
                "source": source,
                "link": link,
                "pub_date": pub_date,
                "polarity": round(polarity, 3),
                "subjectivity": round(subjectivity, 3),
                "sentiment": label,
            })

        return pd.DataFrame(processed)

    except requests.exceptions.ConnectionError:
        st.error("📡 **Connection Error**: Unable to reach Google News.")
        return None
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return None


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def render_sentiment_donut(df: pd.DataFrame) -> plt.Figure:
    """Donut chart of sentiment distribution."""
    counts = df["sentiment"].value_counts().to_dict()
    labels = list(counts.keys())
    sizes = list(counts.values())
    color_map = {"Positive": "#4ade80", "Negative": "#f87171", "Neutral": "#94a3b8"}

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor("none")
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=[color_map.get(l, "#ccc") for l in labels],
        autopct="%1.0f%%",
        startangle=140,
        textprops={"color": "w", "fontsize": 11, "fontweight": "bold"},
        wedgeprops={"width": 0.55, "edgecolor": "none"},
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(12)
    ax.set_title("Sentiment Breakdown", color="w", pad=20, fontsize=14, fontweight="bold")
    return fig


def render_polarity_bar(df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of polarity per headline (top 12)."""
    top = df.head(12).copy()
    top["short"] = top["headline"].str[:50] + "…"
    top = top.iloc[::-1]  # Reverse for bottom-up display

    colors = []
    for p in top["polarity"]:
        if p > 0.1:
            colors.append("#4ade80")
        elif p < -0.1:
            colors.append("#f87171")
        else:
            colors.append("#94a3b8")

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    ax.barh(top["short"], top["polarity"], color=colors, height=0.6, edgecolor="none")
    ax.axvline(x=0, color="#475569", linewidth=0.8, linestyle="--")
    ax.set_xlabel("← Negative          Polarity          Positive →", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_color("#334155")
    ax.set_title("Headline Polarity (Top 12)", color="w", pad=15, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------
def render_news_card(row: pd.Series):
    """Render a single news headline as a styled card."""
    label = row["sentiment"]
    pill_class = "pill-positive" if label == "Positive" else "pill-negative" if label == "Negative" else "pill-neutral"

    polarity_pct = (row["polarity"] + 1) / 2 * 100  # Map -1..1 to 0..100%

    st.markdown(
        f"""
        <div class="news-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px;">
                <div style="flex:1;">
                    <h4 style="margin:0 0 6px 0; color:#e2e8f0; font-size:0.95rem; line-height:1.4;">{row['headline']}</h4>
                    <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-top:8px;">
                        <span class="source-tag">{row['source']}</span>
                        <span style="color:#475569; font-size:0.75rem;">{row['pub_date'][:22] if row['pub_date'] else ''}</span>
                    </div>
                </div>
                <span class="sentiment-pill {pill_class}">{label} ({row['polarity']:+.2f})</span>
            </div>
            <div class="vibe-meter" style="margin-top:12px;">
                <div style="position:absolute; left:{polarity_pct:.0f}%; top:-4px; width:16px; height:16px; background:white; border-radius:50%; box-shadow: 0 0 8px rgba(255,255,255,0.5); transform:translateX(-50%);"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#475569; margin-top:2px;">
                <span>Negative</span>
                <span>Neutral</span>
                <span>Positive</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main():
    # Sidebar
    st.sidebar.markdown("# 🧠 Radar Controls")
    max_articles = st.sidebar.slider("Max Headlines", 10, 50, 25, step=5)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Try These")
    st.sidebar.markdown("""
    - `Narendra Modi`
    - `Bitcoin`
    - `Donald Trump`
    - `AI Technology`
    - `Stock Market India`
    - `Climate Change`
    """)
    st.sidebar.markdown("---")
    st.sidebar.caption("Powered by Google News RSS + TextBlob NLP")

    # Hero
    st.markdown('<div class="hero-title">🧠 AI Sentiment Radar</div>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Real-time sentiment analysis of news headlines — powered by NLP</p>', unsafe_allow_html=True)

    # Search
    search_term = st.text_input(
        "🔍 Search any topic",
        placeholder="e.g. Modi, Bitcoin, AI, Climate Change...",
    )

    if st.button("🛰️ Analyze Sentiment") or (search_term and st.session_state.get("_last") != search_term):
        st.session_state["_last"] = search_term

        if not search_term:
            st.warning("Please enter a search term.")
            return

        with st.spinner(f"Scanning news for **{search_term}**..."):
            df = fetch_news_headlines(search_term, max_results=max_articles)

        if df is None or df.empty:
            st.warning(f"No news found for '{search_term}'. Try a broader term.")
            return

        # ── Metrics Row ──────────────────────────────
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)

        total = len(df)
        pos = len(df[df["sentiment"] == "Positive"])
        neg = len(df[df["sentiment"] == "Negative"])
        neu = len(df[df["sentiment"] == "Neutral"])
        avg_pol = df["polarity"].mean()

        c1.metric("📰 Headlines Analyzed", total)
        c2.metric("😊 Positive", f"{pos}  ({pos/total*100:.0f}%)")
        c3.metric("😠 Negative", f"{neg}  ({neg/total*100:.0f}%)")

        # Overall vibe
        if avg_pol > 0.1:
            vibe = "✅ Positive"
        elif avg_pol < -0.1:
            vibe = "🔴 Negative"
        else:
            vibe = "⚪ Neutral"
        c4.metric("Overall Vibe", vibe)

        st.markdown("---")

        # ── Charts Row ────────────────────────────────
        chart1, chart2 = st.columns([1, 1.5])

        with chart1:
            st.pyplot(render_sentiment_donut(df))

        with chart2:
            st.pyplot(render_polarity_bar(df))

        # ── Top Sources ──────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-title">📡 Top Sources</div>', unsafe_allow_html=True)

        source_stats = df.groupby("source").agg(
            count=("headline", "size"),
            avg_polarity=("polarity", "mean")
        ).sort_values("count", ascending=False).head(8)

        src_cols = st.columns(min(4, len(source_stats)))
        for i, (source, row) in enumerate(source_stats.iterrows()):
            col = src_cols[i % len(src_cols)]
            pol = row["avg_polarity"]
            emoji = "🟢" if pol > 0.1 else "🔴" if pol < -0.1 else "⚪"
            col.metric(f"{emoji} {source}", f"{int(row['count'])} articles", f"Avg: {pol:+.2f}")

        # ── Headlines List ───────────────────────────
        st.markdown("---")
        st.markdown(f'<div class="section-title">📋 {total} Headlines for "{search_term}"</div>', unsafe_allow_html=True)

        # Filter tabs
        tab_all, tab_pos, tab_neg, tab_neu = st.tabs(["🌐 All", "😊 Positive", "😠 Negative", "⚪ Neutral"])

        with tab_all:
            for _, row in df.iterrows():
                render_news_card(row)

        with tab_pos:
            pos_df = df[df["sentiment"] == "Positive"]
            if pos_df.empty:
                st.info("No positive headlines found.")
            for _, row in pos_df.iterrows():
                render_news_card(row)

        with tab_neg:
            neg_df = df[df["sentiment"] == "Negative"]
            if neg_df.empty:
                st.info("No negative headlines found.")
            for _, row in neg_df.iterrows():
                render_news_card(row)

        with tab_neu:
            neu_df = df[df["sentiment"] == "Neutral"]
            if neu_df.empty:
                st.info("No neutral headlines found.")
            for _, row in neu_df.iterrows():
                render_news_card(row)


if __name__ == "__main__":
    main()
