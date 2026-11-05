"""
EchoMind — Mental Health Text Analysis
Streamlit front-end for the BiLSTM text classifier trained in the EchoMind
notebook (3 classes: mental-health / non-suicidal / suicidal).

HOW TO RUN
----------
1) Keep the folder structure as it is:
       app.py
       assets/hero.png
       models/bilstm_model.keras
       models/tokenizer.pkl
       models/label_encoder.pkl
       .streamlit/config.toml
2) pip install -r requirements.txt
3) streamlit run app.py
"""

import base64
import hashlib
import hmac
import os
import pickle
import re
import secrets
import sqlite3
import time
from datetime import datetime
from html import escape as esc
from pathlib import Path
import requests
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "bilstm_model.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
HERO_IMAGE = BASE_DIR / "assets" / "hero.png"
MAX_LEN = 150
MIN_WORDS = 5            # shorter text is not analysed (too little context)
CONF_THRESHOLD = 70.0    # below this the app says "uncertain" instead of a hard label
SUPPORT_PROB = 30.0      # if suicidal probability >= this, still show helpline even when uncertain

PROJECT_AUTHOR = "Support_team"          # shown in the top-right pill
HELPLINE_NUMBER = "1800-599-0019"
HELPLINE_NAME = "KIRAN Mental Health Helpline, India"

# tone -> (main colour, soft background, border)
TONES = {
    "mental-health": ("#12A15C", "#E6F8EE", "#C5EDD6"),
    "non-suicidal": ("#2F6FEB", "#E8F0FF", "#C9DBFF"),
    "suicidal": ("#E5484D", "#FFECEE", "#FAD0D3"),
}
TONES["uncertain"] = ("#B7791F", "#FFF6E0", "#F6E2B0")
DEFAULT_TONE = ("#6D4AFF", "#EFEAFF", "#DCD2FF")
BAR_COLORS = {
    "mental-health": "linear-gradient(90deg,#3FD08C,#2EBE7B)",
    "non-suicidal": "linear-gradient(90deg,#5B9BFF,#3B82F6)",
    "suicidal": "linear-gradient(90deg,#FF7B84,#EF4A57)",
}

NAV = ["Home", "Analyze", "About", "Mental Health Tips", "Contact", "Profile"]
NAV_ICONS = [":material/home:", ":material/find_in_page:", ":material/info:", ":material/spa:", ":material/mail:", ":material/person:"]

st.set_page_config(
    page_title="EchoMind | Mental Health Text Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.session_state.setdefault("dark", False)
st.session_state.setdefault("result", None)
st.session_state.setdefault("page", "Home")
st.session_state.setdefault("advice", None)
st.session_state.setdefault("user", None)

# --------------------------------------------------------------------------
# ICONS (inline SVG, Lucide-style strokes)
# --------------------------------------------------------------------------
ICON_PATHS = {
    "home": '<path d="M3 10.5 12 3l9 7.5V20a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>',
    "analyze": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><circle cx="11.5" cy="14.5" r="2.5"/><path d="m13.5 16.5 2 2"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "leaf": '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>',
    "mail": '<rect width="20" height="16" x="2" y="4" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>',
    "pencil": '<path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/>',
    "chart": '<path d="M5 21V12"/><path d="M12 21V5"/><path d="M19 21v-6"/>',
    "brain": '<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/><path d="M12 18V5"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "pie": '<path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/>',
    "heart": '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>',
    "bulb": '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "phone": '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.13.96.36 1.9.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.91.34 1.85.57 2.81.7A2 2 0 0 1 22 16.92z"/>',
    "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
}


def icon(name: str, size: int = 22, color: str = "currentColor", sw: float = 2, fill: str = "none") -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="{fill}" stroke="{color}" '
        f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        f"{ICON_PATHS[name]}</svg>"
    )


LOGO_SVG = """
<svg width="64" height="64" viewBox="0 0 64 64" fill="none" aria-hidden="true">
  <defs>
    <linearGradient id="lg" x1="6" y1="6" x2="58" y2="58" gradientUnits="userSpaceOnUse">
      <stop stop-color="#4F7CF7"/><stop offset="1" stop-color="#8B4DF0"/>
    </linearGradient>
  </defs>
  <path d="M24 8c-3.6 0-6.4 2.4-7 5.6C12.6 14.4 10 17.8 10 21.6c0 1.6.4 3 1.1 4.2C8.9 27.4 7.5 30 7.5 33c0 3.2 1.6 6 4.1 7.6-.2.7-.3 1.4-.3 2.2 0 4 3.2 7.2 7.2 7.2 1 0 2-.2 2.8-.6C22.7 52.6 25.4 55 28.6 55c1.4 0 2.7-.4 3.7-1.2 1 .8 2.3 1.2 3.7 1.2 3.2 0 5.9-2.4 6.3-5.6.8.4 1.8.6 2.8.6 4 0 7.2-3.2 7.2-7.2 0-.8-.1-1.5-.3-2.2 2.5-1.6 4.1-4.4 4.1-7.6 0-3-1.4-5.6-3.6-7.2.7-1.2 1.1-2.6 1.1-4.2 0-3.8-2.6-7.2-7-8-.6-3.2-3.4-5.6-7-5.6-1.6 0-3.1.5-4.2 1.4C27.1 8.5 25.6 8 24 8Z"
        stroke="url(#lg)" stroke-width="2.6" stroke-linejoin="round"/>
  <path d="M32 45.5c-7-4.6-10.6-8.2-10.6-12.2 0-3 2.3-5.3 5.2-5.3 2.2 0 4.2 1.3 5.4 3.3 1.2-2 3.2-3.3 5.4-3.3 2.9 0 5.2 2.3 5.2 5.3 0 4-3.6 7.6-10.6 12.2Z"
        stroke="url(#lg)" stroke-width="2.6" stroke-linejoin="round"/>
</svg>
"""


def html(s: str) -> str:
    """Strip indentation and blank lines so Markdown never treats HTML as a code block."""
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())


def hero_image_b64() -> str:
    if HERO_IMAGE.exists():
        return base64.b64encode(HERO_IMAGE.read_bytes()).decode()
    return ""


# --------------------------------------------------------------------------
# STYLE
# --------------------------------------------------------------------------
LIGHT_VARS = """
:root {
  --bg1:#E8EEFF; --bg2:#F1ECFF; --bg3:#FBEEF8;
  --side:rgba(250,250,255,.97);
  --card:rgba(255,255,255,.92); --card-solid:#FFFFFF; --line:#E6E9F7;
  --ink:#1B1A4A; --text:#3A3F63; --muted:#7B80A0;
  --brand:#5B5CF0; --brand-soft:#ECE7FF; --nav:#3D3F86;
  --field:#FFFFFF; --field-line:#C4D2FF; --shadow:0 12px 32px rgba(91,92,240,.08);
  --track:#E8EAF4; --lav:#EEEAFF; --img-fade:#000 62%, transparent 100%;
  --img-filter:none; --echo:#161055; --pill-bg:rgba(155,170,255,.28); --pill-fg:#3746B8;
}
"""
DARK_VARS = """
:root {
  --bg1:#12122A; --bg2:#171533; --bg3:#1F1637;
  --side:rgba(24,24,50,.97);
  --card:rgba(32,33,66,.92); --card-solid:#202142; --line:#2F3160;
  --ink:#F2F2FF; --text:#CBCDE8; --muted:#9598BA;
  --brand:#8E8CFF; --brand-soft:#2C2A5C; --nav:#C9CBF0;
  --field:#191A38; --field-line:#3C4076; --shadow:0 12px 32px rgba(0,0,0,.35);
  --track:#2C2E58; --lav:#2A2A57; --img-fade:#000 40%, transparent 88%;
  --img-filter:brightness(.62) saturate(.9); --echo:#F4F3FF; --pill-bg:#2B2E66; --pill-fg:#BAC1FF;
}
"""

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&family=Caveat:wght@500;600&display=swap');

html, body, .stApp, .stApp button, .stApp textarea, .stApp input { font-family:'Inter','Poppins',sans-serif; }
h1,h2,h3,.poppins { font-family:'Poppins',sans-serif; }

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
header[data-testid="stHeader"] { background:transparent; pointer-events:none; }
header[data-testid="stHeader"] * { pointer-events:auto; }

.stApp { background:linear-gradient(150deg,var(--bg1) 0%,var(--bg2) 55%,var(--bg3) 100%); background-attachment:fixed; color:var(--text); }
.block-container { padding:1.2rem 2.2rem 2rem !important; max-width:1280px; }
.stApp p, .stApp label { color:inherit; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background:var(--side); backdrop-filter:blur(14px); border-right:1px solid var(--line); }
section[data-testid="stSidebar"][aria-expanded="true"] { width:300px !important; min-width:300px !important; }
section[data-testid="stSidebar"] > div { padding-top:.6rem; }
.logo { display:flex; align-items:center; gap:12px; padding:6px 4px 26px; }
.logo-name { font:800 30px/1 'Poppins',sans-serif; color:var(--ink); letter-spacing:-.5px; }
.logo-tag { font-size:14px; color:var(--muted); margin-top:4px; }

[class*="st-key-nav_"] button {
  width:100%; justify-content:flex-start; gap:10px; min-height:56px; padding:0 20px; border:0; border-radius:16px;
  background:transparent !important; color:var(--nav) !important; box-shadow:none;
}
[class*="st-key-nav_"] button > div { justify-content:flex-start; width:100%; }
[class*="st-key-nav_"] button p { font:500 18px 'Poppins',sans-serif !important; color:inherit !important; }
[class*="st-key-nav_"] button [data-testid="stIconMaterial"] { font-size:26px; }
.st-key-analyze_btn button [data-testid="stIconMaterial"] { font-size:28px; }
.st-key-theme_toggle button [data-testid="stIconMaterial"] { font-size:26px; }
[class*="st-key-nav_"] button:hover { background:var(--brand-soft) !important; color:var(--brand) !important; }
[class*="st-key-nav_"] button:focus-visible { outline:2px solid var(--brand); }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:.4rem; }
.side-quote {
  margin-top:120px; background:var(--lav); border-radius:22px; padding:30px 20px 22px;
  text-align:center; color:var(--text); font:italic 400 17px/1.5 'Poppins',sans-serif;
}
.side-quote svg { margin-top:10px; }

/* ---------- top bar ---------- */
.st-key-theme_toggle { display:flex; justify-content:flex-end; }
.st-key-theme_toggle button {
  width:52px; height:48px; min-height:48px; padding:0; border-radius:16px;
  background:var(--card); border:1px solid var(--line); box-shadow:var(--shadow); color:var(--brand);
}
.st-key-theme_toggle button:hover { border-color:var(--brand); color:var(--brand); }
.pill {
  display:flex; align-items:center; justify-content:center; gap:12px; height:48px; padding:0 22px;
  border-radius:16px; background:var(--card); border:1px solid var(--line); box-shadow:var(--shadow);
  font:500 15.5px 'Poppins',sans-serif; color:var(--ink); white-space:nowrap;
}
.pill svg { color:var(--brand); }
.st-key-top_logout button, .st-key-prof_logout button {
  height:48px; min-height:48px; border-radius:16px; background:var(--card) !important; border:1px solid var(--line) !important;
  box-shadow:var(--shadow); color:var(--brand) !important;
}
.st-key-top_logout button p, .st-key-prof_logout button p { font:500 15.5px 'Poppins',sans-serif; color:var(--brand) !important; }
.st-key-top_logout button:hover, .st-key-prof_logout button:hover { border-color:var(--brand) !important; }

/* ---------- hero ---------- */
.hero { display:grid; grid-template-columns:1fr 1fr; align-items:center; min-height:250px; margin:-6px 0 8px; }
.hero-left { text-align:center; padding-top:8px; }
.wordmark { font:800 84px/1 'Poppins',sans-serif; letter-spacing:-3px; margin:0; padding:0; }
.w-echo { color:var(--echo); }
.w-mind { background:linear-gradient(90deg,#4F5BF0,#8B5CF6); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
.hero-title { font:600 30px 'Poppins',sans-serif; color:var(--ink); margin-top:8px; }
.hero-sub { font:400 18px 'Poppins',sans-serif; color:var(--muted); margin-top:10px; }
.hero-sub i { font-style:normal; margin:0 10px; }
.hero-pill {
  display:inline-flex; align-items:center; gap:8px; margin-top:20px; padding:10px 26px; border-radius:999px;
  background:var(--pill-bg); color:var(--pill-fg); font-size:15px;
}
.hero-right { position:relative; display:flex; align-items:center; gap:10px; min-height:250px; margin-left:-40px; }
.hero-img {
  width:min(390px,62%); height:auto; filter:var(--img-filter);
  -webkit-mask-image:radial-gradient(ellipse closest-side at 50% 50%, var(--img-fade));
  mask-image:radial-gradient(ellipse closest-side at 50% 50%, var(--img-fade));
}
.script { font:600 34px/1.15 'Caveat',cursive; color:#4B3FA6; transform:rotate(-9deg); margin-left:8px; }
.script span { display:block; white-space:nowrap; }
.script .l2 { padding-left:14px; } .script .l3 { padding-left:26px; } .script .l4 { padding-left:38px; }
body:has(.dark-on) .script { color:#B4B0FF; }

/* ---------- cards ---------- */
.card, .st-key-input_card, .st-key-contact_card {
  background:var(--card); border:1px solid var(--line); border-radius:24px; box-shadow:var(--shadow);
  padding:24px 26px;
}
.st-key-input_card, .st-key-contact_card { gap:.9rem; }
.card-head { display:flex; align-items:center; gap:14px; }
.card-head .ttl { font:600 22px 'Poppins',sans-serif; color:var(--ink); flex:1; }
.chip-ic { width:46px; height:46px; border-radius:14px; display:grid; place-items:center; flex:0 0 46px; }
.lead { font-size:17px; line-height:1.55; color:var(--text); margin:2px 0 0; }

.stTextArea div[data-baseweb="textarea"], .stTextArea div[data-baseweb="base-input"], .stTextArea [data-testid="stTextAreaRootElement"], .stTextArea [data-testid="stTextAreaRootElement"] > div { background:transparent !important; border:0 !important; box-shadow:none !important; }
.stTextArea textarea, .st-key-contact_card input {
  background:var(--field) !important; color:var(--text) !important; font-size:16px !important; line-height:1.6;
  border:1.5px solid var(--field-line) !important; border-radius:16px !important; padding:16px 18px !important;
  box-shadow:0 0 0 4px rgba(99,120,255,.07);
}
.stTextArea textarea:focus, .st-key-contact_card input:focus { border-color:var(--brand) !important; box-shadow:0 0 0 4px rgba(99,102,241,.15); }
.stTextArea textarea::placeholder { color:var(--muted); opacity:1; }
.stTextArea { position:relative; }
.stTextArea [data-testid="InputInstructions"] { position:absolute; right:22px; bottom:14px; color:var(--muted); font-size:14px; opacity:1; }
.stTextArea [data-testid="InputInstructions"] span:not(:last-child) { display:none; }
div[data-testid="stColumn"]:has(.st-key-input_card) > div[data-testid="stVerticalBlock"] { height:100%; }
div[data-testid="stColumn"]:has(.st-key-input_card) [data-testid="stLayoutWrapper"] { flex:1; }
.st-key-input_card { height:100%; }
.st-key-contact_card label p { color:var(--text); font-weight:500; }

.st-key-analyze_btn button, .st-key-contact_card [data-testid="stFormSubmitButton"] button {
  height:60px; border:0 !important; border-radius:16px; color:#fff !important;
  background:linear-gradient(90deg,#8B5CF6 0%,#5B6CF6 100%) !important;
  box-shadow:0 14px 26px rgba(99,102,241,.34); transition:transform .12s, box-shadow .12s;
}
.st-key-analyze_btn button:hover, .st-key-contact_card [data-testid="stFormSubmitButton"] button:hover { transform:translateY(-1px); box-shadow:0 18px 30px rgba(99,102,241,.4); }
.st-key-analyze_btn button, .st-key-analyze_btn button p, .st-key-contact_card [data-testid="stFormSubmitButton"] button p { color:#fff !important; }
.st-key-analyze_btn button p { font:600 20px 'Poppins',sans-serif; }
.st-key-analyze_btn button:disabled { opacity:.55; }
.st-key-analyze_btn button:focus-visible { outline:3px solid rgba(139,92,246,.45); outline-offset:2px; }

/* ---------- result card ---------- */
.done { display:inline-flex; align-items:center; gap:8px; padding:9px 18px; border-radius:12px; font:500 15px 'Poppins',sans-serif;
  color:#12A15C; background:#E6F8EE; border:1px solid #C5EDD6; }
.done.idle { color:var(--muted); background:var(--lav); border-color:var(--line); }
.tiles { display:grid; grid-template-columns:auto 1fr; gap:12px; margin:20px 0 16px; }
.tile { display:flex; align-items:center; gap:12px; padding:18px 18px; border-radius:18px; border:1px solid; min-width:0; }
.tile .ic { width:46px; height:46px; border-radius:50%; display:grid; place-items:center; flex:0 0 46px; }
.tile .k { font:600 16px 'Poppins',sans-serif; }
.tile .v { font:700 27px/1.15 'Poppins',sans-serif; margin-top:2px; white-space:nowrap; }
.tile.conf { background:#EEEAFF; border-color:#DDD5FF; }
.tile.conf .k { color:#26215F; } .tile.conf .v { color:#4F46E5; }
.tile.conf .ic { background:#DED6FF; color:#5B4BE0; }
body:has(.dark-on) .tile.conf { background:#2B2A5E; border-color:#3D3B7C; }
body:has(.dark-on) .tile.conf .k { color:#E7E6FF; } body:has(.dark-on) .tile.conf .v { color:#B7B4FF; }
.probs { border:1px solid var(--line); border-radius:18px; padding:18px 20px 12px; background:var(--card-solid); }
.probs-head { display:flex; align-items:center; gap:12px; font:600 18px 'Poppins',sans-serif; color:var(--ink); margin-bottom:8px; }
.probs-head .chip-ic { width:40px; height:40px; flex-basis:40px; border-radius:50%; background:var(--brand-soft); color:var(--brand); }
.prow { display:grid; grid-template-columns:120px 1fr 64px; align-items:center; gap:14px; padding:9px 0; }
.prow .n { font:500 16px 'Poppins',sans-serif; color:var(--ink); }
.prow .t { height:13px; border-radius:99px; background:var(--track); overflow:hidden; }
.prow .f { height:100%; border-radius:99px; min-width:12px; }
.prow .p { font:600 16px 'Poppins',sans-serif; color:var(--ink); text-align:right; }
.hint { color:var(--muted); font-size:15px; margin:14px 2px 0; }
.support {
  margin-top:14px; padding:14px 16px; border-radius:16px; background:#FFECEE; border:1px solid #FAD0D3; color:#8C1D25; font-size:14.5px; line-height:1.55;
}
.support b { color:#B4232C; }
.note-warn { margin-top:14px; padding:14px 16px; border-radius:16px; background:#FFF6E0; border:1px solid #F6E2B0; color:#7A4E0B; font-size:14.5px; line-height:1.55; }
.note-warn b { color:#8A5A00; }
.rec-card { margin-top:18px; }
.rec-title { font:600 17px 'Poppins',sans-serif; margin:12px 2px 4px; }
.rec-sec { font:600 14px 'Poppins',sans-serif; color:var(--ink); margin:14px 2px 6px; text-transform:uppercase; letter-spacing:.4px; }
.rec-list { margin:0; padding-left:22px; color:var(--text); font-size:15px; line-height:1.6; }
.rec-list li { margin-bottom:4px; }
.rec-seek { margin-top:16px; padding:13px 16px; border-radius:14px; border:1px solid; font-size:14.5px; line-height:1.55; color:var(--text); }

/* ---------- four info cards ---------- */
.grid4 { display:grid; grid-template-columns:repeat(4,1fr); gap:18px; margin-top:22px; }
.grid4 .card { padding:22px 22px 20px; display:flex; flex-direction:column; }
.grid4 .card-head .ttl { font-size:19px; }
.grid4 .chip-ic { width:42px; height:42px; flex-basis:42px; border-radius:50%; }
.steps { list-style:none; margin:18px 0 0; padding:0; display:grid; gap:14px; }
.steps li { display:flex; align-items:center; gap:14px; font-size:15px; color:var(--text); }
.steps .num { width:30px; height:30px; border-radius:50%; display:grid; place-items:center; background:#DCE6FF; color:#2F55D4; font:600 14px 'Poppins',sans-serif; flex:0 0 30px; }
.card p.body { font-size:16px; line-height:1.6; margin:16px 0 0; color:var(--text); }
.mini-quote { margin-top:auto; text-align:center; background:var(--lav); border-radius:14px; padding:16px 14px 12px; font-style:italic; color:var(--text); font-size:15px; }
.mini-quote svg { display:block; margin:8px auto 0; }
.ticks { list-style:none; margin:16px 0 0; padding:0; display:grid; gap:11px; }
.ticks li { display:flex; align-items:center; gap:12px; font-size:15px; color:var(--text); }
.ticks .tick { width:22px; height:22px; border-radius:50%; background:#D9F4E5; color:#12A15C; display:grid; place-items:center; flex:0 0 22px; }
.helpline { margin-top:auto; min-height:64px; display:flex; align-items:center; gap:12px; padding:12px 14px; border-radius:14px; background:#FFE9EE; }
.helpline .ph { width:38px; height:38px; border-radius:12px; background:#FFD3DD; color:#D6336C; display:grid; place-items:center; flex:0 0 38px; }
.helpline b { display:block; color:#B0124A; font:600 14.5px 'Poppins',sans-serif; white-space:nowrap; }
.helpline small { color:#B0124A; font-size:11.5px; }

.foot { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:26px; padding:0 8px; color:var(--muted); font-size:14px; }
.foot .kinder { font:500 30px 'Caveat',cursive; color:#4B3FA6; flex:1; text-align:center; padding-left:120px; }
body:has(.dark-on) .foot .kinder { color:#B4B0FF; }
.foot .made { display:flex; align-items:center; gap:6px; }

/* ---------- inner pages ---------- */
.page-title { font:700 40px 'Poppins',sans-serif; color:var(--ink); margin:6px 0 4px; }
.page-sub { color:var(--muted); font-size:17px; margin-bottom:18px; }
.stack { display:grid; gap:18px; }
.two { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
.tip-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:18px; }
.tip-grid .card { display:flex; gap:14px; align-items:flex-start; font-size:16px; line-height:1.55; }
.tip-grid .ticks-ic { width:38px; height:38px; border-radius:50%; background:#D9F4E5; color:#12A15C; display:grid; place-items:center; flex:0 0 38px; }
.card h3 { margin:0 0 8px; font-size:20px; color:var(--ink); }
.card p { line-height:1.65; }

@media (max-width: 1100px) { .grid4 { grid-template-columns:1fr 1fr; } .tip-grid { grid-template-columns:1fr 1fr; } }
@media (max-width: 900px) {
  .hero { grid-template-columns:1fr; } .hero-right { display:none; }
  .wordmark { font-size:60px; } .hero-title { font-size:24px; }
  .grid4, .two, .tip-grid { grid-template-columns:1fr; } .tiles { grid-template-columns:1fr; }
  .foot { flex-direction:column; } .foot .kinder { padding-left:0; }
  .block-container { padding:1rem 1rem 2rem !important; }
}
@media (max-width: 640px) {
  /* Sidebar becomes a slide-over overlay instead of squeezing the page content */
  section[data-testid="stSidebar"] {
    position:fixed !important; top:0; left:0; height:100vh; z-index:1000000;
    box-shadow:0 0 40px rgba(0,0,0,.35);
  }
  section[data-testid="stSidebar"][aria-expanded="true"] { width:82vw !important; min-width:0 !important; max-width:320px; }
  section[data-testid="stSidebar"][aria-expanded="false"] { width:0 !important; min-width:0 !important; }
}

/* ---------- auth + profile ---------- */
[class*="st-key-form_card"] {
  background:var(--card); border:1px solid var(--line); border-radius:24px; box-shadow:var(--shadow); padding:24px 26px; gap:.9rem;
}
[class*="st-key-form_card"] label p { color:var(--text); font-weight:500; }
[class*="st-key-form_card"] div[data-baseweb="input"] {
  background:var(--field) !important; border:1.5px solid var(--field-line) !important; border-radius:14px !important; overflow:hidden;
}
[class*="st-key-form_card"] div[data-baseweb="input"]:focus-within { border-color:var(--brand) !important; box-shadow:0 0 0 4px rgba(99,102,241,.15); }
[class*="st-key-form_card"] div[data-baseweb="base-input"] { background:transparent !important; border:0 !important; }
[class*="st-key-form_card"] input { background:transparent !important; color:var(--text) !important; font-size:16px !important; padding:12px 14px !important; }
[class*="st-key-form_card"] [data-testid="stFormSubmitButton"] button {
  height:52px; border:0 !important; border-radius:14px; background:linear-gradient(90deg,#8B5CF6 0%,#5B6CF6 100%) !important;
  box-shadow:0 12px 22px rgba(99,102,241,.3);
}
[class*="st-key-form_card"] [data-testid="stFormSubmitButton"] button p { color:#fff !important; font:600 17px 'Poppins',sans-serif; }
[class*="st-key-form_card"] button[role="tab"] p { font:600 16px 'Poppins',sans-serif; }
.auth-brand { text-align:center; margin:14px 0 18px; }
.auth-brand svg { width:60px; height:60px; }
.auth-name { font:800 36px/1.1 'Poppins',sans-serif; color:var(--ink); letter-spacing:-1px; margin-top:6px; }
.auth-tag { color:var(--muted); font-size:15.5px; margin-top:4px; }
.auth-note { color:var(--muted); font-size:13px; line-height:1.5; margin-top:2px; }
.prof-head { display:flex; align-items:center; gap:22px; flex-wrap:wrap; }
.avatar { width:88px; height:88px; border-radius:50%; flex:0 0 88px; display:grid; place-items:center; color:#fff;
  font:700 36px 'Poppins',sans-serif; background:linear-gradient(135deg,#8B5CF6,#5B6CF6); box-shadow:0 12px 24px rgba(99,102,241,.35); }
.prof-name { font:700 28px 'Poppins',sans-serif; color:var(--ink); line-height:1.15; }
.prof-mail { color:var(--text); font-size:16px; margin-top:2px; word-break:break-all; }
.prof-since { color:var(--muted); font-size:14px; margin-top:4px; }
.stats { display:grid; grid-template-columns:repeat(3,1fr); gap:18px; margin:18px 0; }
.stat { background:var(--card); border:1px solid var(--line); border-radius:20px; box-shadow:var(--shadow); padding:18px 20px; }
.stat .k { color:var(--muted); font-size:14px; }
.stat .v { font:700 26px 'Poppins',sans-serif; color:var(--ink); margin-top:4px; }
.lbl { display:inline-block; padding:4px 12px; border-radius:999px; font:500 13.5px 'Poppins',sans-serif; border:1px solid; }
.htable { width:100%; border-collapse:collapse; font-size:15px; color:var(--text); }
.htable th { text-align:left; color:var(--muted); font-weight:500; padding:8px 6px; border-bottom:1px solid var(--line); }
.htable td { padding:11px 6px; border-bottom:1px solid var(--line); }
.empty { color:var(--muted); font-size:15px; padding:8px 2px; }
@media (max-width: 900px) { .stats { grid-template-columns:1fr; } }
@media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
"""


def inject_css() -> None:
    active = NAV.index(st.session_state.page)
    active_css = (
        f'.st-key-nav_{active} button {{ background:linear-gradient(90deg,var(--brand-soft),var(--lav)) !important; '
        f"color:var(--brand) !important; }} .st-key-nav_{active} button p {{ font-weight:600 !important; }}"
    )
    css = (DARK_VARS if st.session_state.dark else LIGHT_VARS) + BASE_CSS + active_css
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    if st.session_state.dark:
        st.markdown('<span class="dark-on"></span>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# TEXT CLEANING — must match the training pipeline exactly
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------
# MODEL LOADING (cached)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading EchoMind model…")
def load_artifacts():
    missing = [p for p in (MODEL_PATH, TOKENIZER_PATH, ENCODER_PATH) if not p.exists()]
    if missing:
        return None, None, None, missing
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder, []


model, tokenizer, label_encoder, missing_files = load_artifacts()


def predict(text: str):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]
    classes = label_encoder.classes_
    order = np.argsort(probs)[::-1]
    top_class = str(classes[order[0]])
    top_conf = float(probs[order[0]] * 100)
    if top_conf < CONF_THRESHOLD:
        top_class = "uncertain"  # model is not sure, do not force a label
    result = [(str(classes[i]), float(probs[i] * 100)) for i in order]
    return top_class, top_conf, result


def generate_advice(user_text: str, prediction_result) -> str:
    """Generate supportive guidance using OpenRouter."""

    try:
        api_key = st.secrets["OPENROUTER_API_KEY"]

        top_class, top_conf, _ = prediction_result

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a supportive wellbeing assistant. "
                            "Give empathetic, non-judgmental general guidance. "
                            "Do not diagnose or claim the prediction is certain. "
                            "Offer three practical coping steps and encourage "
                            "professional help when appropriate. If immediate "
                            "self-harm danger is mentioned, encourage contacting "
                            "local emergency services and a trusted person. "
                            "If the text is about physical health or something unrelated to emotions, "
                            "answer helpfully and briefly without assuming a mental illness. "
                            "Keep the response concise."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Text: {user_text}\n"
                            f"Classifier category: {top_class}\n"
                            f"Classifier confidence: {top_conf:.1f}%\n"
                            "Provide supportive guidance based on this context."
                        ),
                    },
                ],
                "temperature": 0.5,
                "max_tokens": 350,
            },
            timeout=30,
        )

        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        st.caption("AI guidance is unavailable right now (check OPENROUTER_API_KEY in secrets). Built-in recommendations are shown instead.")
        return (
            "Please consider talking with someone you trust or a "
            "qualified mental health professional if you need support."
        )


# class -> (title, what you can do now, daily habits, when to reach out)
RECOMMENDATIONS = {
    "mental-health": (
        "Looking after your mental wellbeing",
        [
            "Pause and take 5 slow breaths (inhale 4s, hold 4s, exhale 6s).",
            "Write down what you are feeling and what triggered it.",
            "Talk to a friend or family member you trust about how you feel.",
        ],
        [
            "Sleep 7-8 hours at a fixed time every day.",
            "Walk or stretch for 20-30 minutes daily.",
            "Limit doom-scrolling and take short screen breaks.",
            "Keep a small daily routine: meals, sunlight, one enjoyable activity.",
        ],
        "If low mood, anxiety or worry lasts more than 2 weeks or affects work, "
        "studies or sleep, book a session with a counsellor, psychologist or psychiatrist.",
    ),
    "non-suicidal": (
        "Keeping the balance going",
        [
            "Keep doing what helps you: hobbies, music, time with people you like.",
            "Share the good and the bad with someone, do not hold it all inside.",
            "Set one small goal for today and celebrate finishing it.",
        ],
        [
            "Stick to a regular sleep and meal schedule.",
            "Exercise, yoga or a walk in daylight a few times a week.",
            "Practice 5-10 minutes of meditation or journaling.",
            "Check in with yourself weekly: how is my mood, sleep and energy?",
        ],
        "If you notice your mood dropping, constant worry or loss of interest in things "
        "you enjoy, do not wait. Talk to a professional early.",
    ),
    "uncertain": (
        "General care tips",
        [
            "If you feel physically unwell (for example vomiting or stomach upset), drink fluids and see a doctor or pharmacist if it continues.",
            "If you want an emotional check, write a few sentences about how you feel and analyze again.",
            "Talk to someone you trust if something is bothering you.",
        ],
        [
            "Rest, drink enough water and eat light, regular meals.",
            "Keep a regular sleep schedule.",
            "Take short breaks and a little daylight or a walk each day.",
        ],
        "If you feel unwell for more than a couple of days, or low or anxious for more than 2 weeks, "
        "please consult a doctor or a mental health professional.",
    ),
    "suicidal": (
        "Please reach out for support right now",
        [
            f"Call {HELPLINE_NAME}: {HELPLINE_NUMBER}, or your local emergency number.",
            "Tell one trusted person (family, friend, teacher) how you are feeling, right now.",
            "Stay with someone, do not be alone, and move away from anything you could use to hurt yourself.",
        ],
        [
            "Take things one hour at a time, you only need to get through today.",
            "Eat something, drink water and rest, the body affects the mind.",
            "Plan a visit to a doctor or mental health professional in the next few days.",
            "Write a short list of people you can call when things feel heavy.",
        ],
        "You deserve support, and these feelings can change with help. This tool is "
        "not a diagnosis, please speak to a professional as soon as you can.",
    ),
}


def recommendation_card(result) -> str:
    """Class-based recommendations shown under the analysis result (works offline)."""
    if not result:
        return ""
    top_class, _, _ = result
    rec = RECOMMENDATIONS.get(top_class)
    if not rec:
        return ""
    title, now, habits, seek = rec
    fg, bg, border = TONES.get(top_class, DEFAULT_TONE)
    now_li = "".join(f"<li>{i}</li>" for i in now)
    habit_li = "".join(f"<li>{i}</li>" for i in habits)
    return html(
        f"""
        <div class="card rec-card">
          <div class="card-head">
            {chip("heart", fg, bg, True)}
            <span class="ttl">Recommendations</span>
          </div>
          <p class="rec-title" style="color:{fg};">{title}</p>
          <div class="rec-sec">What you can do right now</div>
          <ul class="rec-list">{now_li}</ul>
          <div class="rec-sec">Healthy habits that help</div>
          <ul class="rec-list">{habit_li}</ul>
          <div class="rec-seek" style="background:{bg};border-color:{border};">
            <b>When to seek help:</b> {seek}
          </div>
        </div>
        """
    )


# --------------------------------------------------------------------------
# HTML BUILDERS
# --------------------------------------------------------------------------
def chip(name: str, fg: str, bg: str, round_: bool = False) -> str:
    radius = "border-radius:50%;" if round_ else ""
    return f'<span class="chip-ic" style="background:{bg};color:{fg};{radius}">{icon(name, 22, fg)}</span>'


def result_card(result, notice: str = "") -> str:
    head_badge = '<span class="done idle">Waiting for text</span>'
    tiles_pred = ("—", *DEFAULT_TONE)
    conf_text = "—"
    rows = ""
    support = ""

    if result:
        top_class, top_conf, probs = result
        fg, bg, border = TONES.get(top_class, DEFAULT_TONE)
        head_badge = f'<span class="done">Analysis Complete {icon("check", 18, "#12A15C", 2.6)}</span>'
        tiles_pred = (top_class, fg, bg, border)
        conf_text = f"{top_conf:.2f}%"
        for cls, pct in probs:
            rows += (
                f'<div class="prow"><div class="n">{cls.capitalize()}</div>'
                f'<div class="t"><div class="f" style="width:{max(pct, 0.5):.1f}%;background:{BAR_COLORS.get(cls, "#6366F1")}"></div></div>'
                f'<div class="p">{pct:.2f}%</div></div>'
            )
        if top_class == "suicidal":
            support = (
                f'<div class="support">This result suggests language linked to suicide risk. It is '
                f"<b>not a diagnosis</b>. If you or someone you know is struggling, please talk to a "
                f"trusted person or a professional right away. "
                f"<b>{HELPLINE_NAME}: {HELPLINE_NUMBER}</b></div>"
            )
        elif top_class == "uncertain":
            suicidal_p = dict(probs).get("suicidal", 0.0)
            support = (
                '<div class="note-warn">The model is <b>not confident</b> about this text. It is trained only on '
                "emotional / mental-health language (mostly social-media posts), so it can misread other topics "
                "such as physical illness. Try writing a few full sentences about how you feel."
            )
            if suicidal_p >= SUPPORT_PROB:
                support += (
                    f" If you are feeling low or unsafe, you can talk to someone any time: "
                    f"<b>{HELPLINE_NAME}: {HELPLINE_NUMBER}</b>."
                )
            support += "</div>"
    else:
        for cls in ("mental-health", "non-suicidal", "suicidal"):
            rows += (
                f'<div class="prow"><div class="n">{cls.capitalize()}</div>'
                f'<div class="t"></div><div class="p">—</div></div>'
            )

    label, fg, bg, border = tiles_pred
    hint = notice or ("" if result else "Type something on the left and press Analyze Text.")
    hint_html = f'<p class="hint">{hint}</p>' if hint else ""

    return html(
        f"""
        <div class="card">
          <div class="card-head">
            {chip("chart", "#3B5BDB", "#DCE6FF")}
            <span class="ttl">Analysis Result</span>
            {head_badge}
          </div>
          <div class="tiles">
            <div class="tile" style="background:{bg};border-color:{border};">
              <span class="ic" style="background:{border};color:{fg};">{icon("brain", 24, fg)}</span>
              <div><div class="k" style="color:{fg};">Prediction</div>
              <div class="v" style="color:{fg};">{label}</div></div>
            </div>
            <div class="tile conf">
              <span class="ic">{icon("target", 24)}</span>
              <div><div class="k">Confidence</div><div class="v">{conf_text}</div></div>
            </div>
          </div>
          <div class="probs">
            <div class="probs-head">{chip("pie", "#5B4BE0", "#E7E1FF", True)} Class Probabilities</div>
            {rows}
          </div>
          {support}
          {hint_html}
        </div>
        """
    )


def top_bar() -> None:
    logged_in = bool(st.session_state.get("user"))
    widths = [5.0, 0.7, 2.2, 1.5] if logged_in else [6.2, 0.7, 2.3]
    cols = st.columns(widths, vertical_alignment="center", gap="small")
    c_toggle, c_pill = cols[1], cols[2]
    with c_toggle:
        st.button(
            "",
            icon=":material/light_mode:" if not st.session_state.dark else ":material/dark_mode:",
            key="theme_toggle",
            help="Switch to dark mode" if not st.session_state.dark else "Switch to light mode",
            on_click=lambda: st.session_state.update(dark=not st.session_state.dark),
        )
    user = st.session_state.get("user")
    pill_text = f"Hi, {esc(user['name'].split()[0])}" if user else f"Project by {PROJECT_AUTHOR}"
    with c_pill:
        st.markdown(
            f'<div class="pill">{icon("user", 22)}<span>{pill_text}</span></div>',
            unsafe_allow_html=True,
        )
    if logged_in:
        with cols[3]:
            st.button("Logout", icon=":material/logout:", key="top_logout",
                      use_container_width=True, on_click=do_logout)


def hero() -> None:
    img = hero_image_b64()
    img_tag = f'<img class="hero-img" alt="" src="data:image/png;base64,{img}">' if img else ""
    st.markdown(
        html(
            f"""
            <div class="hero">
              <div class="hero-left">
                <div class="wordmark"><span class="w-echo">Echo</span><span class="w-mind">Mind</span></div>
                <div class="hero-title">Mental Health Text Analysis</div>
                <div class="hero-sub">Analyze<i>•</i>Understand<i>•</i>Support</div>
                <div class="hero-pill">Your words matter. Let's make sense of them together {icon("heart", 16, "#3B6CF0", 0, "#3B6CF0")}</div>
              </div>
              <div class="hero-right">
                {img_tag}
                <div class="script"><span class="l1">A</span><span class="l2">Healthier Mind</span><span class="l3">A Brighter</span><span class="l4">Tomorrow {icon("heart", 22, "#4B3FA6", 2)}</span></div>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def analyzer_row() -> None:
    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(key="input_card"):
            st.markdown(
                html(
                    f"""
                    <div class="card-head">{chip("pencil", "#5B4BE0", "#E7E1FF")}
                    <span class="ttl">Enter Your Text</span></div>
                    <p class="lead">Write your thoughts, feelings or any text below and EchoMind will analyze it for you.</p>
                    """
                ),
                unsafe_allow_html=True,
            )
            text = st.text_area(
                "Your text",
                placeholder="Type your text here...\ne.g. I have been feeling very anxious and depressed lately.",
                height=230,
                max_chars=500,
                label_visibility="collapsed",
                key="user_text",
            )
            clicked = st.button(
                "Analyze Text",
                icon=":material/search:",
                key="analyze_btn",
                use_container_width=True,
                disabled=bool(missing_files),
            )

    with right:
        notice = ""

        if clicked:
            if len(text.split()) < MIN_WORDS:
                notice = f"Please write at least a full sentence ({MIN_WORDS}+ words) so EchoMind has enough context."
            else:
                with st.spinner("Analyzing text and preparing guidance…"):
                    st.session_state.result = predict(text)
                    st.session_state.advice = generate_advice(
                        text, st.session_state.result
                    )
                    user = st.session_state.get("user")
                    if user:
                        try:  # only label + confidence are stored, never the text
                            add_history(user["id"], st.session_state.result[0], st.session_state.result[1])
                        except Exception:
                            pass

        st.markdown(
            result_card(st.session_state.result, notice),
            unsafe_allow_html=True,
        )
        st.markdown(
            recommendation_card(st.session_state.result),
            unsafe_allow_html=True,
        )

    if st.session_state.get("advice"):
        st.markdown("### 💙 AI Supportive Guidance")
        st.markdown(st.session_state.advice)
        st.caption("This guidance is for general support, not a medical diagnosis.")


def info_cards() -> None:
    steps = ["Enter your text", "Our AI model analyzes it", "Get instant prediction", "Understand and seek support"]
    tips = ["Take deep breaths", "Talk to someone you trust", "Be kind to yourself", "Seek professional help if needed", "You are not alone"]
    steps_html = "".join(f'<li><span class="num">{i}</span>{s}</li>' for i, s in enumerate(steps, 1))
    ticks_html = "".join(f'<li><span class="tick">{icon("check", 14, "#12A15C", 3)}</span>{t}</li>' for t in tips)
    st.markdown(
        html(
            f"""
            <div class="grid4">
              <div class="card">
                <div class="card-head">{chip("heart", "#E5484D", "#FFE0E4", True)}<span class="ttl">How It Works</span></div>
                <ul class="steps">{steps_html}</ul>
              </div>
              <div class="card">
                <div class="card-head">{chip("bulb", "#E59B00", "#FFF1CC", True)}<span class="ttl">Remember</span></div>
                <p class="body">Your mental health is important. Talking about your feelings is a sign of strength, not weakness.</p>
                <div class="mini-quote">“Small steps lead to big changes.”{icon("heart", 20, "#5B4BE0")}</div>
              </div>
              <div class="card">
                <div class="card-head">{chip("leaf", "#12A15C", "#DDF5E8", True)}<span class="ttl">Self-Care Tips</span></div>
                <ul class="ticks">{ticks_html}</ul>
              </div>
              <div class="card">
                <div class="card-head">{chip("shield", "#4F46E5", "#E3E0FF", True)}<span class="ttl">Disclaimer</span></div>
                <p class="body" style="font-size:14.5px;margin:14px 0 14px;">EchoMind is a research/educational machine learning project and is not a medical diagnosis tool. If you are in crisis, please reach out to a mental health professional or a helpline.</p>
                <div class="helpline"><span class="ph">{icon("phone", 20)}</span>
                  <div><b>Helpline: {HELPLINE_NUMBER}</b><small>({HELPLINE_NAME})</small></div></div>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        html(
            f"""
            <div class="foot">
              <span class="kinder">A kinder tomorrow is possible {icon("heart", 22, "#4B3FA6", 2)}</span>
              <span class="made">Made with {icon("heart", 15, "#F0508A", 0, "#F0508A")} for a healthier world</span>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def missing_warning() -> None:
    if missing_files:
        st.warning(
            "Model files not found. Place `bilstm_model.keras`, `tokenizer.pkl` and "
            "`label_encoder.pkl` inside the `models/` folder next to `app.py`, then rerun."
        )



# --------------------------------------------------------------------------
# ACCOUNTS DATABASE (SQLite, built into Python)
# - Passwords: PBKDF2-HMAC-SHA256, 200k rounds, random salt per user.
# - History stores only the result label + confidence, NEVER the typed text.
# --------------------------------------------------------------------------
DB_PATH = BASE_DIR / "data" / "echomind.db"
PBKDF2_ROUNDS = 200_000
MIN_PASSWORD_LEN = 8
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    """Raised with a user-friendly message."""


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db() -> None:
    with _conn() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                email       TEXT NOT NULL UNIQUE,
                pw_hash     TEXT NOT NULL,
                salt        TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                label       TEXT NOT NULL,
                confidence  REAL NOT NULL,
                created_at  TEXT NOT NULL
            );
            """
        )


# ---------------------------------------------------------------- hashing
def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS).hex()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------- validation
def validate_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not EMAIL_RE.match(email):
        raise AuthError("Please enter a valid email address.")
    return email


def validate_name(name: str) -> str:
    name = " ".join((name or "").split())
    if len(name) < 2:
        raise AuthError("Please enter your name (at least 2 characters).")
    if len(name) > 60:
        raise AuthError("Name is too long (max 60 characters).")
    return name


def validate_password(password: str) -> None:
    if len(password or "") < MIN_PASSWORD_LEN:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LEN} characters.")
    if not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        raise AuthError("Password must contain at least one letter and one number.")


# ------------------------------------------------------------------- users
def register_user(name: str, email: str, password: str, confirm: str) -> dict:
    name = validate_name(name)
    email = validate_email(email)
    validate_password(password)
    if password != confirm:
        raise AuthError("Passwords do not match.")
    salt = secrets.token_bytes(16)
    try:
        with _conn() as con:
            cur = con.execute(
                "INSERT INTO users (name, email, pw_hash, salt, created_at) VALUES (?,?,?,?,?)",
                (name, email, _hash(password, salt), salt.hex(), _now()),
            )
            uid = cur.lastrowid
    except sqlite3.IntegrityError:
        raise AuthError("An account with this email already exists. Please log in.")
    return get_user(uid)


def login_user(email: str, password: str) -> dict:
    email = (email or "").strip().lower()
    with _conn() as con:
        row = con.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    # Same message for unknown email / wrong password (no account enumeration)
    if row is None:
        _hash(password or "", os.urandom(16))  # keep timing similar
        raise AuthError("Incorrect email or password.")
    expected = row["pw_hash"]
    actual = _hash(password or "", bytes.fromhex(row["salt"]))
    if not hmac.compare_digest(expected, actual):
        raise AuthError("Incorrect email or password.")
    return get_user(row["id"])


def get_user(user_id: int) -> dict:
    with _conn() as con:
        row = con.execute(
            "SELECT id, name, email, created_at FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    if row is None:
        raise AuthError("Account not found.")
    return dict(row)


def update_profile(user_id: int, name: str, email: str) -> dict:
    name = validate_name(name)
    email = validate_email(email)
    try:
        with _conn() as con:
            con.execute("UPDATE users SET name = ?, email = ? WHERE id = ?", (name, email, user_id))
    except sqlite3.IntegrityError:
        raise AuthError("That email is already used by another account.")
    return get_user(user_id)


def change_password(user_id: int, current: str, new: str, confirm: str) -> None:
    with _conn() as con:
        row = con.execute("SELECT pw_hash, salt FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        raise AuthError("Account not found.")
    if not hmac.compare_digest(row["pw_hash"], _hash(current or "", bytes.fromhex(row["salt"]))):
        raise AuthError("Current password is incorrect.")
    validate_password(new)
    if new != confirm:
        raise AuthError("New passwords do not match.")
    if new == current:
        raise AuthError("New password must be different from the current one.")
    salt = secrets.token_bytes(16)
    with _conn() as con:
        con.execute(
            "UPDATE users SET pw_hash = ?, salt = ? WHERE id = ?",
            (_hash(new, salt), salt.hex(), user_id),
        )


def delete_user(user_id: int, password: str) -> None:
    with _conn() as con:
        row = con.execute("SELECT pw_hash, salt FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None or not hmac.compare_digest(
        row["pw_hash"], _hash(password or "", bytes.fromhex(row["salt"]))
    ):
        raise AuthError("Password is incorrect.")
    with _conn() as con:
        con.execute("DELETE FROM users WHERE id = ?", (user_id,))  # history cascades


# ----------------------------------------------------------------- history
def add_history(user_id: int, label: str, confidence: float) -> None:
    with _conn() as con:
        con.execute(
            "INSERT INTO history (user_id, label, confidence, created_at) VALUES (?,?,?,?)",
            (user_id, label, float(confidence), _now()),
        )


def get_history(user_id: int, limit: int = 10) -> list:
    with _conn() as con:
        rows = con.execute(
            "SELECT label, confidence, created_at FROM history WHERE user_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_stats(user_id: int) -> dict:
    with _conn() as con:
        total = con.execute("SELECT COUNT(*) c FROM history WHERE user_id = ?", (user_id,)).fetchone()["c"]
        counts = con.execute(
            "SELECT label, COUNT(*) c FROM history WHERE user_id = ? GROUP BY label", (user_id,)
        ).fetchall()
        last = con.execute(
            "SELECT created_at FROM history WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user_id,)
        ).fetchone()
    return {
        "total": total,
        "by_label": {r["label"]: r["c"] for r in counts},
        "last": last["created_at"] if last else None,
    }


def clear_history(user_id: int) -> None:
    with _conn() as con:
        con.execute("DELETE FROM history WHERE user_id = ?", (user_id,))


init_db()


# --------------------------------------------------------------------------
# AUTH + PROFILE
# --------------------------------------------------------------------------
MAX_FAILS, LOCK_SECONDS = 5, 30


def flash(kind: str, msg: str) -> None:
    st.session_state["flash"] = (kind, msg)


def show_flash() -> None:
    f = st.session_state.pop("flash", None)
    if f:
        getattr(st, f[0])(f[1])


def do_logout() -> None:
    for k in ("user", "result", "advice", "user_text", "flash"):
        st.session_state.pop(k, None)
    st.session_state.user = None
    st.session_state.page = "Home"


def fmt_date(value, with_time: bool = False) -> str:
    if not value:
        return "—"
    try:
        d = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        return d.strftime("%d %b %Y, %I:%M %p" if with_time else "%d %b %Y")
    except ValueError:
        return value


def label_chip(label: str) -> str:
    fg, bg, border = TONES.get(label, DEFAULT_TONE)
    return f'<span class="lbl" style="color:{fg};background:{bg};border-color:{border};">{esc(label)}</span>'


def start_session(user: dict) -> None:
    st.session_state.user = user
    st.session_state.page = "Home"
    st.session_state.fails = 0
    st.session_state.lock_until = 0
    st.rerun()


def auth_page() -> None:
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        st.markdown(
            html(
                f"""
                <div class="auth-brand">{LOGO_SVG}
                  <div class="auth-name">EchoMind</div>
                  <div class="auth-tag">Log in or create an account to continue</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        show_flash()
        with st.container(key="form_card_auth"):
            tab_login, tab_reg = st.tabs(["Login", "Create account"])

            with tab_login:
                with st.form("login_form", border=False):
                    email = st.text_input("Email", placeholder="you@example.com")
                    pw = st.text_input("Password", type="password")
                    go = st.form_submit_button("Log in", use_container_width=True)
                if go:
                    now = time.time()
                    lock = st.session_state.get("lock_until", 0)
                    if now < lock:
                        st.error(f"Too many failed attempts. Try again in {int(lock - now) + 1} seconds.")
                    else:
                        try:
                            start_session(login_user(email, pw))
                        except AuthError as e:
                            fails = st.session_state.get("fails", 0) + 1
                            if fails >= MAX_FAILS:
                                st.session_state.lock_until = now + LOCK_SECONDS
                                fails = 0
                            st.session_state.fails = fails
                            st.error(str(e))

            with tab_reg:
                with st.form("register_form", border=False):
                    name = st.text_input("Full name", placeholder="Your name")
                    r_email = st.text_input("Email", placeholder="you@example.com", key="reg_email")
                    r_pw = st.text_input("Password", type="password", key="reg_pw",
                                         help="At least 8 characters, with a letter and a number.")
                    r_pw2 = st.text_input("Confirm password", type="password", key="reg_pw2")
                    go_r = st.form_submit_button("Create account", use_container_width=True)
                st.markdown(
                    '<div class="auth-note">Your password is stored securely (salted and hashed). '
                    "The text you analyze is never saved, only the result label.</div>",
                    unsafe_allow_html=True,
                )
                if go_r:
                    try:
                        start_session(register_user(name, r_email, r_pw, r_pw2))
                    except AuthError as e:
                        st.error(str(e))


def profile_page() -> None:
    user = st.session_state.user
    try:
        user = get_user(user["id"])
        st.session_state.user = user
    except AuthError:
        do_logout()
        st.rerun()

    st.markdown(
        '<div class="page-title">My Profile</div>'
        '<div class="page-sub">Manage your account and see your activity.</div>',
        unsafe_allow_html=True,
    )
    show_flash()

    stats = get_stats(user["id"])
    top = max(stats["by_label"], key=stats["by_label"].get) if stats["by_label"] else None
    initial = esc(user["name"][:1].upper() or "?")
    st.markdown(
        html(
            f"""
            <div class="card prof-head">
              <div class="avatar">{initial}</div>
              <div>
                <div class="prof-name">{esc(user["name"])}</div>
                <div class="prof-mail">{esc(user["email"])}</div>
                <div class="prof-since">Member since {fmt_date(user["created_at"])}</div>
              </div>
            </div>
            <div class="stats">
              <div class="stat"><div class="k">Total analyses</div><div class="v">{stats["total"]}</div></div>
              <div class="stat"><div class="k">Most common result</div><div class="v">{label_chip(top) if top else "—"}</div></div>
              <div class="stat"><div class="k">Last analysis</div><div class="v" style="font-size:18px;">{fmt_date(stats["last"], True)}</div></div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.button("Logout", icon=":material/logout:", key="prof_logout", on_click=do_logout)

    left, right = st.columns(2, gap="medium")

    with left:
        with st.container(key="form_card_edit"):
            st.markdown(
                html(f'<div class="card-head">{chip("user", "#5B4BE0", "#E7E1FF")}<span class="ttl">Edit Profile</span></div>'),
                unsafe_allow_html=True,
            )
            with st.form("edit_profile", border=False):
                new_name = st.text_input("Full name", value=user["name"])
                new_email = st.text_input("Email", value=user["email"])
                if st.form_submit_button("Save changes", use_container_width=True):
                    try:
                        st.session_state.user = update_profile(user["id"], new_name, new_email)
                        flash("success", "Profile updated.")
                        st.rerun()
                    except AuthError as e:
                        st.error(str(e))

        with st.container(key="form_card_pw"):
            st.markdown(
                html(f'<div class="card-head">{chip("shield", "#4F46E5", "#E3E0FF")}<span class="ttl">Change Password</span></div>'),
                unsafe_allow_html=True,
            )
            with st.form("change_pw", border=False, clear_on_submit=True):
                cur = st.text_input("Current password", type="password")
                new = st.text_input("New password", type="password", help="At least 8 characters, with a letter and a number.")
                conf = st.text_input("Confirm new password", type="password")
                if st.form_submit_button("Update password", use_container_width=True):
                    try:
                        change_password(user["id"], cur, new, conf)
                        flash("success", "Password changed successfully.")
                        st.rerun()
                    except AuthError as e:
                        st.error(str(e))

    with right:
        history = get_history(user["id"], 10)
        if history:
            rows = "".join(
                f"<tr><td>{fmt_date(h['created_at'], True)}</td><td>{label_chip(h['label'])}</td>"
                f"<td>{h['confidence']:.1f}%</td></tr>"
                for h in history
            )
            body = f'<table class="htable"><tr><th>Date</th><th>Result</th><th>Confidence</th></tr>{rows}</table>'
        else:
            body = '<div class="empty">No analyses yet. Go to <b>Analyze</b> and your results will appear here.</div>'
        st.markdown(
            html(
                f"""
                <div class="card">
                  <div class="card-head">{chip("chart", "#3B5BDB", "#DCE6FF")}<span class="ttl">Recent Analyses</span></div>
                  <p class="lead" style="font-size:14.5px;color:var(--muted);">Only the result and confidence are saved, never your text.</p>
                  {body}
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        with st.expander("Danger zone"):
            if st.button("Clear my analysis history", key="clear_hist", disabled=not history):
                clear_history(user["id"])
                flash("success", "History cleared.")
                st.rerun()
            st.markdown("**Delete account** — this permanently removes your account and history.")
            with st.form("delete_form", border=False):
                del_pw = st.text_input("Confirm with your password", type="password", key="del_pw")
                sure = st.checkbox("I understand this cannot be undone")
                if st.form_submit_button("Delete my account"):
                    if not sure:
                        st.error("Please tick the box to confirm.")
                    else:
                        try:
                            delete_user(user["id"], del_pw)
                            do_logout()
                            flash("success", "Your account has been deleted.")
                            st.rerun()
                        except AuthError as e:
                            st.error(str(e))

# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
inject_css()

# Guests only see the login / register page (no sidebar)
if not st.session_state.user:
    st.markdown(
        '<style>section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], '
        '[data-testid="collapsedControl"] { display:none !important; }</style>',
        unsafe_allow_html=True,
    )
    top_bar()
    auth_page()
    st.stop()

with st.sidebar:
    st.markdown(
        html(
            f"""
            <div class="logo">{LOGO_SVG}
              <div><div class="logo-name">EchoMind</div><div class="logo-tag">Your Mind Matters</div></div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    for i, (label, mat) in enumerate(zip(NAV, NAV_ICONS)):
        st.button(label, icon=mat, key=f"nav_{i}", use_container_width=True,
                  on_click=lambda l=label: st.session_state.update(page=l))
    st.button("Logout", icon=":material/logout:", key="nav_logout", use_container_width=True, on_click=do_logout)
    st.markdown(
        f'<div class="side-quote">“It’s okay<br>to not be okay.<br>You are not alone.”<br>{icon("heart", 26, "#5B4BE0")}</div>',
        unsafe_allow_html=True,
    )
    


# --------------------------------------------------------------------------
# PAGES
# --------------------------------------------------------------------------
page = st.session_state.page
top_bar()

if page == "Home":
    hero()
    missing_warning()
    analyzer_row()
    info_cards()
    footer()

elif page == "Analyze":
    st.markdown(
        '<div class="page-title">Analyze Text</div>'
        '<div class="page-sub">Paste or write a short piece of text and EchoMind will estimate which category it falls into.</div>',
        unsafe_allow_html=True,
    )
    missing_warning()
    analyzer_row()
    footer()

elif page == "About":
    st.markdown(
        html(
            f"""
            <div class="page-title">About EchoMind</div>
            <div class="page-sub">A deep learning project for understanding mental-health-related language.</div>
            <div class="stack">
              <div class="card">
                <div class="card-head">{chip("info", "#5B4BE0", "#E7E1FF")}<span class="ttl">Project Overview</span></div>
                <p>EchoMind classifies short pieces of text into three categories — <b>mental-health</b>,
                <b>non-suicidal</b> and <b>suicidal</b> — using a Bidirectional LSTM (BiLSTM) model trained on
                combined, cleaned public social-media text datasets.</p>
                <p><b>Pipeline:</b> data cleaning → tokenization and padding → LSTM baseline → BiLSTM model →
                evaluation → deployment.</p>
                <p><b>Stack:</b> Python, TensorFlow/Keras, Scikit-learn, Streamlit.</p>
              </div>
              <div class="card">
                <div class="card-head">{chip("shield", "#4F46E5", "#E3E0FF")}<span class="ttl">Ethics and Limitations</span></div>
                <p>This project is for academic demonstration only. Predictions can be wrong and must never
                replace professional judgment. Accounts store only a name, email and a securely hashed password. For each analysis only the result label and confidence are saved, never the text you type.</p>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    footer()

elif page == "Mental Health Tips":
    tips = [
        "Take deep breaths and pause when you feel overwhelmed.",
        "Talk to someone you trust about how you feel.",
        "Be kind to yourself — progress isn't a straight line.",
        "Seek professional help if things feel heavy.",
        "Keep a simple routine: sleep, food and movement.",
        "You are not alone, and asking for help is a strength.",
    ]
    cards = "".join(f'<div class="card"><span class="ticks-ic">{icon("check", 18, "#12A15C", 3)}</span><div>{t}</div></div>' for t in tips)
    st.markdown(
        html(
            f"""
            <div class="page-title">Mental Health Tips</div>
            <div class="page-sub">Small, gentle habits that can help on a hard day.</div>
            <div class="tip-grid">{cards}</div>
            <div class="card" style="margin-top:18px;">
              <div class="helpline" style="margin-top:0;"><span class="ph">{icon("phone", 20)}</span>
              <div><b>If you need to talk to someone now: {HELPLINE_NUMBER}</b><small>({HELPLINE_NAME})</small></div></div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    footer()

elif page == "Profile":
    profile_page()
    footer()

elif page == "Contact":
    st.markdown(
        '<div class="page-title">Contact</div>'
        '<div class="page-sub">Questions about this academic project? Send a message below.</div>',
        unsafe_allow_html=True,
    )
    with st.container(key="contact_card"):
        with st.form("contact_form", border=False):
            name = st.text_input("Name")
            email = st.text_input("Email")
            message = st.text_area("Message", height=130)
            if st.form_submit_button("Send message", use_container_width=True):
                st.success("Thanks! This is a demo form — connect it to email or a backend to receive messages.")
    footer()