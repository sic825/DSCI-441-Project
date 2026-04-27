"""Central academic stylesheet — injected at the top of every page."""
import streamlit as st

_CSS = """
<style>
/* ── Body: serif for academic feel ──────────────────────────────────────── */
.main, .block-container {
    font-family: Georgia, 'Times New Roman', serif;
    max-width: 900px;
    margin: 0 auto;
}

/* ── Headings: clean sans-serif (hybrid-journal convention) ─────────────── */
h1 {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 2.4em;
    font-weight: 600;
    border-bottom: 2px solid #1F4E79;
    padding-bottom: 0.3em;
    color: #1A1A1A;
}
h2 {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 1.6em;
    margin-top: 1.5em;
    color: #1F4E79;
}
h3 {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 1.25em;
    margin-top: 1.2em;
    color: #2C3E50;
}

/* ── Body text ──────────────────────────────────────────────────────────── */
p {
    line-height: 1.7;
    text-align: justify;
    margin-bottom: 1em;
}
li { line-height: 1.65; margin-bottom: 0.3em; }

/* ── Figure captions ───────────────────────────────────────────────────── */
.figure-caption {
    font-style: italic;
    font-size: 0.88em;
    color: #555;
    text-align: center;
    margin-top: 0.4em;
    margin-bottom: 1.8em;
    line-height: 1.5;
}

/* ── Paper-style tables ─────────────────────────────────────────────────── */
table {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 0.9em;
    border-top: 2px solid #1A1A1A;
    border-bottom: 2px solid #1A1A1A;
    border-left: none;
    border-right: none;
    width: 100%;
    border-collapse: collapse;
}
th {
    border-bottom: 1px solid #1A1A1A;
    font-weight: 600;
    padding: 0.4em 0.8em;
    text-align: left;
}
td { padding: 0.35em 0.8em; }
tr:nth-child(even) { background: #FAFAFA; }

/* ── Abstract / blockquote ─────────────────────────────────────────────── */
blockquote {
    border-left: 3px solid #1F4E79;
    margin: 1em 0 1em 0;
    padding: 0.5em 1em;
    background: #F5F8FC;
    font-style: italic;
    color: #333;
}

/* ── Streamlit overrides ────────────────────────────────────────────────── */
[data-testid="stMetricLabel"] {
    font-family: 'Helvetica Neue', Arial, sans-serif !important;
    font-size: 0.75rem !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: 0.06em; color: #555 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Helvetica Neue', Arial, sans-serif !important;
    font-size: 1.85rem !important; font-weight: 700 !important; color: #1F4E79 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.82rem !important; font-style: italic; }

[data-testid="stExpander"] {
    border: 1px solid #D0D8E4 !important; border-radius: 3px !important;
    background: #FAFCFF !important; margin-bottom: 0.6rem !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Helvetica Neue', Arial, sans-serif !important;
    font-weight: 600 !important; font-size: 0.97rem !important; color: #1F4E79 !important;
}

[data-testid="stDataFrame"] {
    font-family: 'Helvetica Neue', Arial, sans-serif !important; font-size: 0.84rem !important;
}

[data-testid="stTabs"] button {
    font-family: 'Helvetica Neue', Arial, sans-serif !important;
    font-size: 0.93rem !important; font-weight: 600 !important; color: #1F4E79 !important;
}

[data-testid="stAlert"] {
    border-left: 4px solid #1F4E79 !important;
    background: #EEF3F9 !important;
    font-style: italic; font-family: Georgia, serif !important;
}

[data-testid="stSidebar"] {
    background: #F0F4F8 !important; border-right: 1px solid #D0D8E4;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { border-bottom: none !important; }

hr { border: none !important; border-top: 1px solid #D0D8E4 !important; margin: 1.4rem 0 !important; }

code, pre {
    font-family: 'Courier New', monospace !important; font-size: 0.85rem !important;
    background: #F5F5F5 !important; border: 1px solid #E0E0E0 !important;
    border-radius: 3px !important;
}

[data-testid="stSelectbox"] label, [data-testid="stSlider"] label {
    font-family: 'Helvetica Neue', Arial, sans-serif !important;
    font-size: 0.9rem !important; font-weight: 600 !important;
}

.katex { font-size: 1.05rem !important; }
.element-container svg { border: 1px solid #D0D8E4; border-radius: 3px; }
</style>
"""


def inject():
    st.markdown(_CSS, unsafe_allow_html=True)


def figcap(text: str):
    """Render a figure caption in the .figure-caption style."""
    st.markdown(f'<p class="figure-caption">{text}</p>', unsafe_allow_html=True)
