import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Lens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&display=swap');

/* global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* hide streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* main background */
.stApp {
    background-color: #F7F6F2;
}

/* sidebar */
section[data-testid="stSidebar"] {
    background-color: #1C1C1E;
    border-right: none;
}
section[data-testid="stSidebar"] * {
    color: #E8E8E3 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label {
    color: #9E9E9A !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* metric cards */
div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #E8E8E3;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
div[data-testid="metric-container"] label {
    color: #9E9E9A !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #1C1C1E !important;
    font-size: 2rem !important;
    font-weight: 600;
}

/* text input */
.stTextInput input {
    background: white;
    border: 1.5px solid #E8E8E3;
    border-radius: 10px;
    padding: 14px 16px;
    font-size: 0.95rem;
    color: #1C1C1E;
    transition: border-color 0.2s;
}
.stTextInput input:focus {
    border-color: #1C1C1E;
    box-shadow: none;
}

/* button */
.stButton button {
    background: #1C1C1E;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 28px;
    font-size: 0.9rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    transition: all 0.2s;
    width: 100%;
}
.stButton button:hover {
    background: #3A3A3C;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* section cards */
.card {
    background: white;
    border: 1px solid #E8E8E3;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* section labels */
.section-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #9E9E9A;
    font-weight: 500;
    margin-bottom: 4px;
}

/* section title */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #1C1C1E;
    margin-bottom: 20px;
    line-height: 1.2;
}

/* verdict badge */
.verdict-pos {
    display: inline-block;
    background: #ECFDF5;
    color: #065F46;
    border: 1px solid #A7F3D0;
    border-radius: 999px;
    padding: 6px 16px;
    font-size: 0.85rem;
    font-weight: 500;
}
.verdict-neg {
    display: inline-block;
    background: #FEF2F2;
    color: #991B1B;
    border: 1px solid #FECACA;
    border-radius: 999px;
    padding: 6px 16px;
    font-size: 0.85rem;
    font-weight: 500;
}

/* comment cards */
.comment-pos {
    background: #F0FDF4;
    border-left: 3px solid #22C55E;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
    font-size: 0.9rem;
    color: #1C1C1E;
    line-height: 1.5;
}
.comment-neg {
    background: #FFF1F2;
    border-left: 3px solid #F43F5E;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
    font-size: 0.9rem;
    color: #1C1C1E;
    line-height: 1.5;
}

/* divider */
.divider {
    height: 1px;
    background: #E8E8E3;
    margin: 24px 0;
}

/* video info */
.video-info {
    background: #1C1C1E;
    border-radius: 14px;
    padding: 20px 24px;
    color: white;
}
.video-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: white;
    margin-bottom: 8px;
    line-height: 1.3;
}
.video-meta {
    font-size: 0.78rem;
    color: #9E9E9A;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* progress bar custom */
.progress-container {
    background: #F0F0EC;
    border-radius: 999px;
    height: 8px;
    width: 100%;
    margin: 8px 0;
}
.progress-pos {
    background: #22C55E;
    border-radius: 999px;
    height: 8px;
}
.progress-neg {
    background: #F43F5E;
    border-radius: 999px;
    height: 8px;
}

/* hide plotly toolbar */
.js-plotly-plot .plotly .modebar {
    display: none;
}

/* selectbox */
.stSelectbox > div > div {
    background: #2C2C2E;
    border: 1px solid #3A3A3C;
    border-radius: 8px;
    color: white;
}

/* tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #F0F0EC;
    padding: 4px;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-size: 0.85rem;
    font-weight: 500;
    color: #9E9E9A;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #1C1C1E !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}

/* dataframe */
.stDataFrame {
    border: 1px solid #E8E8E3;
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ── helper functions ──────────────────────────────────────────────────────────

def clean_text_for_wordcloud(texts):
    stopwords = {
        "the","a","an","is","it","this","that","was","are","be",
        "have","had","but","for","not","with","he","she","they",
        "we","you","i","to","of","in","and","or","on","at","by",
        "my","your","his","her","their","our","its","will","would",
        "could","should","do","did","does","just","so","if","about",
        "what","how","when","where","who","which","been","being",
        "also","very","too","more","most","some","any","all","one",
        "me","him","us","them","from","up","out","as","into","than",
        "then","there","here","get","got","can","even","really",
        "br","re","ve","ll","amp","http","https","www","com"
    }
    words = []
    for text in texts:
        text = re.sub(r'http\S+', '', str(text))
        text = re.sub(r'[^\w\s]', ' ', text)
        for word in text.lower().split():
            if (word not in stopwords
                    and len(word) > 2
                    and word.isalpha()):
                words.append(word)
    return words


def make_wordcloud(words, colormap, bg="white"):
    if not words:
        return None
    text = " ".join(words)
    wc   = WordCloud(
        width=600, height=300,
        background_color=bg,
        colormap=colormap,
        max_words=60,
        prefer_horizontal=0.85,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(text)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor(bg)
    plt.tight_layout(pad=0)
    return fig


def get_top_words(texts, n=8):
    words = clean_text_for_wordcloud(texts)
    return Counter(words).most_common(n)


def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    elif "shorts/" in url:
        return url.split("shorts/")[1].split("?")[0]
    return None


def plot_pie(pos_pct, neg_pct):
    fig = go.Figure(go.Pie(
        labels=["Positive", "Negative"],
        values=[pos_pct, neg_pct],
        hole=0.65,
        marker=dict(
            colors=["#22C55E", "#F43F5E"],
            line=dict(color="white", width=2)
        ),
        textinfo="percent",
        textfont=dict(size=13, family="DM Sans"),
        showlegend=False,
    ))
    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=f"<b>{max(pos_pct, neg_pct):.0f}%</b><br>"
                 f"<span style='font-size:11px;color:#9E9E9A'>"
                 f"{'positive' if pos_pct >= neg_pct else 'negative'}</span>",
            x=0.5, y=0.5,
            font=dict(size=18, family="DM Sans", color="#1C1C1E"),
            showarrow=False
        )]
    )
    return fig


def plot_top_words(pos_words, neg_words):
    if not pos_words and not neg_words:
        return None

    fig = go.Figure()

    if pos_words:
        words_p, counts_p = zip(*pos_words)
        fig.add_trace(go.Bar(
            y=list(words_p), x=list(counts_p),
            orientation="h",
            name="Positive",
            marker_color="#22C55E",
            marker_line_width=0,
        ))

    if neg_words:
        words_n, counts_n = zip(*neg_words)
        fig.add_trace(go.Bar(
            y=list(words_n), x=list(counts_n),
            orientation="h",
            name="Negative",
            marker_color="#F43F5E",
            marker_line_width=0,
        ))

    fig.update_layout(
        barmode="group",
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=11)
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#F0F0EC",
            zeroline=False,
            title="Frequency",
            titlefont=dict(size=11, color="#9E9E9A")
        ),
        yaxis=dict(
            showgrid=False,
            categoryorder="total ascending"
        ),
        margin=dict(t=10, b=10, l=10, r=10)
    )
    return fig


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 24px'>
        <div style='font-size:0.7rem;text-transform:uppercase;
                    letter-spacing:0.12em;color:#636363;margin-bottom:6px'>
            Tool
        </div>
        <div style='font-family:"DM Serif Display",serif;
                    font-size:1.5rem;color:white;line-height:1.1'>
            Sentiment<br>Lens
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:0.7rem;text-transform:uppercase;
                letter-spacing:0.1em;color:#636363;margin-bottom:12px'>
        Configuration
    </div>
    """, unsafe_allow_html=True)

    max_comments = st.selectbox(
        "Comments to analyze",
        options=[50, 100, 200, 500],
        index=1
    )

    video_type = st.selectbox(
        "Video type",
        options=[
            "General",
            "Music / Song",
            "Comedy",
            "News / Politics",
            "Education",
            "Gaming",
            "Product Review",
            "Vlog",
        ]
    )

    show_table = st.checkbox("Show comments table", value=False)

    st.markdown("""
    <div style='height:1px;background:#2C2C2E;margin:20px 0'></div>
    <div style='font-size:0.7rem;text-transform:uppercase;
                letter-spacing:0.1em;color:#636363;margin-bottom:8px'>
        Model
    </div>
    <div style='font-size:0.8rem;color:#9E9E9A;line-height:1.6'>
        DistilBERT fine-tuned<br>
        on IMDB reviews<br>
        Accuracy: 89.6%
    </div>
    <div style='height:1px;background:#2C2C2E;margin:20px 0'></div>
    <div style='font-size:0.7rem;color:#636363;line-height:1.6'>
        Built by Anubhav Goyal<br>
        MLOps Portfolio Project
    </div>
    """, unsafe_allow_html=True)


# ── main content ──────────────────────────────────────────────────────────────

# header
st.markdown("""
<div style='padding: 32px 0 8px'>
    <div style='font-size:0.72rem;text-transform:uppercase;
                letter-spacing:0.12em;color:#9E9E9A;margin-bottom:8px'>
        YouTube Comment Intelligence
    </div>
    <div style='font-family:"DM Serif Display",serif;
                font-size:2.8rem;color:#1C1C1E;line-height:1.1;
                margin-bottom:12px'>
        What is your audience<br>
        <em>really</em> thinking?
    </div>
    <div style='font-size:0.9rem;color:#9E9E9A;max-width:480px;line-height:1.6'>
        Paste any YouTube URL to analyze comment sentiment
        using a fine-tuned DistilBERT model.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# input row
col_input, col_btn = st.columns([5, 1])
with col_input:
    url = st.text_input(
        "",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed"
    )
with col_btn:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    analyze = st.button("Analyze →", type="primary")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── analysis ──────────────────────────────────────────────────────────────────
if analyze and url:

    video_id = extract_video_id(url)
    if not video_id:
        st.error("Invalid YouTube URL. Please check and try again.")
        st.stop()

    with st.spinner("Fetching comments and running analysis..."):
        try:
            resp = requests.post(
                "http://localhost:8000/analyze/youtube",
                json={"url": url, "max_comments": max_comments},
                timeout=120
            )
            if resp.status_code != 200:
                st.error(f"API Error: {resp.json().get('detail', 'Unknown error')}")
                st.stop()
            data = resp.json()

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach API. Make sure FastAPI is running on port 8000.")
            st.stop()

    comments_df = pd.DataFrame(data["comments"])
    pos_texts   = comments_df[comments_df["sentiment"] == "pos"]["text"].tolist()
    neg_texts   = comments_df[comments_df["sentiment"] == "neg"]["text"].tolist()
    pos_pct     = data["positive_pct"]
    neg_pct     = data["negative_pct"]

    # ── video info bar ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='video-info'>
        <div class='video-meta'>Now analyzing · {video_type}</div>
        <div class='video-title'>
            youtube.com/watch?v={video_id}
        </div>
        <div style='display:flex;gap:20px;margin-top:12px'>
            <div>
                <div style='font-size:0.7rem;color:#636363;
                            text-transform:uppercase;letter-spacing:0.08em'>
                    Comments
                </div>
                <div style='font-size:1.1rem;color:white;font-weight:600'>
                    {data['total']}
                </div>
            </div>
            <div>
                <div style='font-size:0.7rem;color:#636363;
                            text-transform:uppercase;letter-spacing:0.08em'>
                    Inference
                </div>
                <div style='font-size:1.1rem;color:white;font-weight:600'>
                    {data['inference_ms']}ms
                </div>
            </div>
            <div>
                <div style='font-size:0.7rem;color:#636363;
                            text-transform:uppercase;letter-spacing:0.08em'>
                    Verdict
                </div>
                <div style='font-size:1.1rem;font-weight:600;
                            color:{"#4ADE80" if data["overall"]=="positive" else "#F87171"}'>
                    {data['overall'].capitalize()}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── row 1: metrics + pie ──────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 0.3, 1.5])

    with col1:
        st.metric("Total Comments", data["total"])
    with col2:
        st.metric("Positive", f"{pos_pct}%")
    with col3:
        st.metric("Negative", f"{neg_pct}%")
    with col5:
        st.plotly_chart(
            plot_pie(pos_pct, neg_pct),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # sentiment progress bars
    st.markdown(f"""
    <div class='card'>
        <div class='section-label'>Sentiment breakdown</div>
        <div style='margin-top:12px'>
            <div style='display:flex;justify-content:space-between;
                        margin-bottom:4px'>
                <span style='font-size:0.85rem;color:#1C1C1E;font-weight:500'>
                    Positive
                </span>
                <span style='font-size:0.85rem;color:#22C55E;font-weight:600'>
                    {pos_pct}%
                </span>
            </div>
            <div class='progress-container'>
                <div class='progress-pos' style='width:{pos_pct}%'></div>
            </div>
            <div style='display:flex;justify-content:space-between;
                        margin-top:10px;margin-bottom:4px'>
                <span style='font-size:0.85rem;color:#1C1C1E;font-weight:500'>
                    Negative
                </span>
                <span style='font-size:0.85rem;color:#F43F5E;font-weight:600'>
                    {neg_pct}%
                </span>
            </div>
            <div class='progress-container'>
                <div class='progress-neg' style='width:{neg_pct}%'></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── row 2: word clouds ────────────────────────────────────────────────────
    st.markdown("""
    <div class='section-label' style='margin-bottom:8px'>
        Word clouds
    </div>
    """, unsafe_allow_html=True)

    wc_col1, wc_col2 = st.columns(2)

    with wc_col1:
        st.markdown("""
        <div style='font-size:0.8rem;font-weight:500;color:#065F46;
                    margin-bottom:8px'>
            Positive comments
        </div>
        """, unsafe_allow_html=True)
        pos_words_list = clean_text_for_wordcloud(pos_texts)
        fig = make_wordcloud(pos_words_list, "YlGn")
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.markdown("""
            <div style='background:#F0FDF4;border-radius:12px;padding:40px;
                        text-align:center;color:#9E9E9A;font-size:0.85rem'>
                No positive comments found
            </div>
            """, unsafe_allow_html=True)

    with wc_col2:
        st.markdown("""
        <div style='font-size:0.8rem;font-weight:500;color:#991B1B;
                    margin-bottom:8px'>
            Negative comments
        </div>
        """, unsafe_allow_html=True)
        neg_words_list = clean_text_for_wordcloud(neg_texts)
        fig = make_wordcloud(neg_words_list, "OrRd")
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.markdown("""
            <div style='background:#FFF1F2;border-radius:12px;padding:40px;
                        text-align:center;color:#9E9E9A;font-size:0.85rem'>
                No negative comments found
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── row 3: top words chart ────────────────────────────────────────────────
    st.markdown("""
    <div class='section-label' style='margin-bottom:8px'>
        Most frequent words
    </div>
    """, unsafe_allow_html=True)

    pos_words = get_top_words(pos_texts, 8)
    neg_words = get_top_words(neg_texts, 8)
    fig       = plot_top_words(pos_words, neg_words)

    if fig:
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    # ── row 4: notable comments ───────────────────────────────────────────────
    st.markdown("""
    <div class='section-label' style='margin-bottom:12px'>
        Most liked comments
    </div>
    """, unsafe_allow_html=True)

    nc_col1, nc_col2 = st.columns(2)

    with nc_col1:
        st.markdown("""
        <div style='font-size:0.8rem;font-weight:500;color:#065F46;
                    margin-bottom:8px'>
            👍 Top positive
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class='comment-pos'>
            {data['top_positive']}
        </div>
        """, unsafe_allow_html=True)

        # show top 3 positive by likes
        if len(pos_texts) > 1:
            pos_df = comments_df[
                comments_df["sentiment"] == "pos"
            ].nlargest(3, "likes")
            for _, row in pos_df.iterrows():
                st.markdown(f"""
                <div class='comment-pos'>
                    <div>{row['text']}</div>
                    <div style='font-size:0.75rem;color:#9E9E9A;margin-top:6px'>
                        👍 {row['likes']} likes
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with nc_col2:
        st.markdown("""
        <div style='font-size:0.8rem;font-weight:500;color:#991B1B;
                    margin-bottom:8px'>
            👎 Top negative
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class='comment-neg'>
            {data['top_negative']}
        </div>
        """, unsafe_allow_html=True)

        # show top 3 negative by likes
        if len(neg_texts) > 1:
            neg_df = comments_df[
                comments_df["sentiment"] == "neg"
            ].nlargest(3, "likes")
            for _, row in neg_df.iterrows():
                st.markdown(f"""
                <div class='comment-neg'>
                    <div>{row['text']}</div>
                    <div style='font-size:0.75rem;color:#9E9E9A;margin-top:6px'>
                        👍 {row['likes']} likes
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── row 5: comments table (optional) ─────────────────────────────────────
    if show_table:
        st.markdown("""
        <div class='section-label' style='margin-bottom:12px'>
            All comments
        </div>
        """, unsafe_allow_html=True)

        display_df = comments_df[
            ["text", "sentiment", "likes", "date"]
        ].copy()
        display_df["sentiment"] = display_df["sentiment"].map({
            "pos": "✅ Positive",
            "neg": "❌ Negative"
        })
        display_df.columns = ["Comment", "Sentiment", "Likes", "Date"]
        display_df = display_df.sort_values("Likes", ascending=False)

        st.dataframe(
            display_df,
            use_container_width=True,
            height=350,
            hide_index=True
        )

    # ── row 6: download ───────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='section-label' style='margin-bottom:8px'>
        Export
    </div>
    """, unsafe_allow_html=True)

    dl_col1, dl_col2 = st.columns([1, 3])
    with dl_col1:
        csv = comments_df.to_csv(index=False)
        st.download_button(
            label="⬇ Download CSV",
            data=csv,
            file_name=f"sentiment_{video_id}.csv",
            mime="text/csv"
        )

elif analyze and not url:
    st.warning("Please paste a YouTube URL to get started.")

else:
    # empty state
    st.markdown("""
    <div style='text-align:center;padding:60px 20px'>
        <div style='font-size:3rem;margin-bottom:12px'>🔍</div>
        <div style='font-family:"DM Serif Display",serif;
                    font-size:1.3rem;color:#1C1C1E;margin-bottom:8px'>
            Ready to analyze
        </div>
        <div style='font-size:0.88rem;color:#9E9E9A;max-width:360px;
                    margin:0 auto;line-height:1.6'>
            Paste any YouTube URL above and click Analyze
            to see what your audience really thinks.
        </div>
    </div>
    """, unsafe_allow_html=True)