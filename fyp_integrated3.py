# fyp_integrated.py
# ===============================================
# 💬 PhishNet Chat (Streamlit + Google Safe Browsing + ML + Explainability)
# ===============================================

import streamlit as st
st.set_page_config(page_title="PhishNet Chat", page_icon="🛡️", layout="wide")

import pickle, re, socket, csv, time, requests
from urllib.parse import urlparse
import pandas as pd
import tldextract
from typing import Tuple, List, Union

# ====================================================
# 1️⃣ Load models (cached)
# ====================================================
import joblib
import streamlit as st

@st.cache_resource
def load_models():
    try:
        url_model = joblib.load('phishing_url_model.pkl')
    except Exception as e:
        st.error(f"Could not load URL model: {e}")
        raise

    try:
        text_model = joblib.load('smishing_text_model.pkl')
    except Exception as e:
        st.error(f"Could not load text model: {e}")
        raise

    try:
        vectorizer = joblib.load('smishing_vectorizer.pkl')
    except Exception as e:
        st.error(f"Could not load vectorizer: {e}")
        raise

    return url_model, text_model, vectorizer


url_model, text_model, text_vectorizer = load_models()

st.sidebar.markdown("**PhishNet Chat**")
st.sidebar.caption("Models loaded (cached).")

# ====================================================
# 2️⃣ Google Safe Browsing API (replace API_KEY)
# ====================================================
API_KEY = "AIzaSyAzmPbAs2EpeIRejV4GWuOVeX5yHFMlHVA"  # <-- replace with your key or leave empty to skip GSB
GSB_ENDPOINT = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={API_KEY}"

def check_google_safe_browsing(url: str) -> bool:
    """Return True if Google Safe Browsing flags the URL."""
    if not API_KEY or "YOUR_GOOGLE_SAFE_BROWSING_API_KEY" in API_KEY:
        # No key configured: bail out quickly
        return False
    payload = {
        "client": {"clientId": "phishnet-app", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE","SOCIAL_ENGINEERING","PHISHING","UNWANTED_SOFTWARE"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}]
        }
    }
    try:
        r = requests.post(GSB_ENDPOINT, json=payload, timeout=5)
        r.raise_for_status()
        return "matches" in r.json()
    except Exception:
        return False

# ====================================================
# 3️⃣ Helpers, whitelist, utilities
# ====================================================
WHITELIST = {
    "google.com","youtube.com","facebook.com","instagram.com","twitter.com","linkedin.com",
    "github.com","stackoverflow.com","reddit.com","amazon.com","paypal.com","wikipedia.org",
    "bbc.co.uk","cnn.com","nytimes.com","dropbox.com","microsoft.com","office.com",
    "apple.com","spotify.com","netflix.com","tiktok.com","etsy.com","ebay.com",
    "docs.google.com","drive.google.com","utp.edu.my","openai.com","canva.com"
}

def get_base_domain(host: str) -> str:
    try:
        ext = tldextract.extract(host)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
        return host
    except Exception:
        return host

def is_domain_valid(domain: str) -> bool:
    try:
        socket.gethostbyname(domain)
        return True
    except Exception:
        return False

def explain_features(features: List[int], names: List[str]) -> List[str]:
    # return the feature names where feature value is truthy (1)
    return [name for val, name in zip(features, names) if val]

def append_fp_to_csv(url: str, features: List[str], model_outcome: str, file: str = "fp_log.csv"):
    with open(file, "a", newline="", encoding="utf8") as f:
        csv.writer(f).writerow([time.time(), url, model_outcome] + list(features))

# ====================================================
# 4️⃣ URL feature extractor (returns feature vector OR special token)
# ====================================================
PHISHING_FEATURES = [
    'UsingIP','LongURL','ShortURL','Symbol@','Redirecting//','PrefixSuffix-','SubDomains',
    'HTTPS','DomainRegLen','Favicon','NonStdPort','HTTPSDomainURL','RequestURL','AnchorURL',
    'LinksInScriptTags','ServerFormHandler','InfoEmail','AbnormalURL','WebsiteForwarding',
    'StatusBarCust','DisableRightClick','UsingPopupWindow','IframeRedirection','AgeofDomain',
    'DNSRecording','WebsiteTraffic','PageRank','GoogleIndex','LinksPointingToPage','StatsReport'
]

def extract_url_features(url: str) -> Union[str, List[int]]:
    """
    Returns:
      - "WHITELISTED" if exact base domain is whitelisted
      - "FLAGGED_BY_GOOGLE" if Google SafeBrowsing marks it
      - otherwise list[int] of 30 features (same order as PHISHING_FEATURES)
    """
    url = str(url).strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = parsed.netloc.lower().replace("www.", "")
    base = get_base_domain(host)

    # ===== FIXED: whitelist must match base domain exactly (prevents google.com.hacker.ru) =====
    if base in WHITELIST:
        return "WHITELISTED"

    # Google Safe Browsing check
    try:
        if check_google_safe_browsing(url):
            return "FLAGGED_BY_GOOGLE"
    except Exception:
        pass

    # quick domain checks
    domain_ok = is_domain_valid(base)

    using_ip = 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}$', host) else 0
    long_url = 1 if len(host) > 30 else 0
    shorteners = ['bit.ly','tinyurl','goo.gl','ow.ly','t.co']
    short_url = 1 if any(s in host for s in shorteners) else 0
    symbol_at = 1 if '@' in url else 0
    redirecting = 1 if url.count('//') > 1 else 0
    prefix_suffix = 1 if '-' in host else 0
    subdomains = 1 if host.count('.') > 2 else 0
    https = 1 if parsed.scheme == 'https' else 0

    keywords = ['verify','update','bank','free','prize','win','claim','secure','signin','login','account']
    anchor_url = 1 if any(k in url.lower() for k in keywords) else 0

    dns_record = 1 if domain_ok else 0
    website_traffic = 1 if domain_ok else 0
    page_rank = 1 if domain_ok else 0
    google_index = 1 if domain_ok else 0
    links_pointing = 1 if domain_ok else 0
    stats_report = 0

    features = [
        using_ip, long_url, short_url, symbol_at, redirecting, prefix_suffix, subdomains, https,
        0, 1, 0, (0 if https else 1), 0, anchor_url, 0, 0, 0, (1 if (using_ip or anchor_url) else 0),
        0, 0, 0, 0, 0, 0, dns_record, website_traffic, page_rank, google_index, links_pointing, stats_report
    ]
    return features

# ====================================================
# 5️⃣ Decision wrapper for URL (reduce false positives)
# ====================================================
# ---------------------------
# Replace existing decide_url_label
# ---------------------------
def decide_url_label(features: Union[str, List[int]], url: str) -> Tuple[str, List[str]]:
    """
    Decide final label for a URL using:
      - whitelist / Google SB special tokens (handled earlier)
      - ML model probability
      - keyword heuristic bump to reduce false negatives (e.g., "win", "prize")
    Returns (verdict_text, active_feature_names_list)
    """
    if features == "WHITELISTED":
        return "✅ Legitimate URL (trusted domain)", []
    if features == "FLAGGED_BY_GOOGLE":
        return "🚨 Unsafe URL (flagged by Google Safe Browsing)", []

    # get ML probability (fallback to predict if predict_proba not available)
    df = pd.DataFrame([features], columns=PHISHING_FEATURES)
    try:
        ml_proba = float(url_model.predict_proba(df)[0][1])
    except Exception:
        pred = int(url_model.predict(df)[0])
        ml_proba = 1.0 if pred == 1 else 0.0

    active = explain_features(features, PHISHING_FEATURES)

    # ---------- Keyword heuristic ----------
    # these words increase suspicion if present in the full URL (path/query) or host
    url_keywords = ["win", "prize", "free", "claim", "contest", "gift", "urgent", "reward",
                    "verify", "signin", "login", "account", "password", "confirm"]
    text_lower = url.lower()
    keyword_hits = [k for k in url_keywords if k in text_lower]
    n_hits = len(keyword_hits)

    # n_hits bumps the ML probability modestly (25% per keyword, capped)
    # This avoids hard overrides but makes suspicious pages more likely to cross threshold
    if n_hits:
        bump = 0.25 * n_hits
        ml_proba = min(0.99, ml_proba + bump)

    # ---------- Decision thresholds (conservative) ----------
    # If ML is very confident -> phishing
    if ml_proba >= 0.75:
        verdict = f"🚨 Phishing URL Detected (ML {ml_proba:.2f})"
    # If ML is very confident safe -> safe
    elif ml_proba <= 0.40:
        verdict = f"✅ Legitimate URL (ML {ml_proba:.2f})"
    # otherwise uncertain
    else:
        verdict = f"⚠️ Suspicious — Uncertain (ML {ml_proba:.2f})"

    # add keyword list to active features shown to user (for explainability)
    active_keywords = [f"kw:{k}" for k in keyword_hits]
    return verdict, active + active_keywords


# ====================================================
# 6️⃣ Text (smishing) decision wrapper
# ====================================================
# ---------------------------
# Replace existing decide_text_label
# ---------------------------
def decide_text_label(text: str) -> Tuple[str, List[str]]:
    """
    Decide final label for SMS/text messages by combining ML probability
    with a keyword heuristic (detects 'win', 'otp', 'verify', shorteners inside text).
    Returns (verdict_text, triggered_keywords_list)
    """
    tfidf = text_vectorizer.transform([text])
    try:
        proba = text_model.predict_proba(tfidf)[0]  # [prob_ham, prob_smish]
        prob_smish = float(proba[1])
        prob_ham = float(proba[0])
    except Exception:
        pred = int(text_model.predict(tfidf)[0])
        prob_smish = 1.0 if pred == 1 else 0.0
        prob_ham = 1.0 - prob_smish

    # ---------- Text keywords ----------
    text_keywords_high = ["win", "prize", "free", "claim", "click here", "congrat", "won"]
    text_keywords_otp = ["otp", "one-time", "one time", "verification", "pin", "code"]
    text_keywords_action = ["verify", "confirm", "login", "signin", "update account"]

    lowered = text.lower()

    hits_high = [k for k in text_keywords_high if k in lowered]
    hits_otp = [k for k in text_keywords_otp if k in lowered]
    hits_action = [k for k in text_keywords_action if k in lowered]

    # detect shortener links inside text (we will treat as higher suspicion)
    shorteners = ["bit.ly", "tinyurl", "ow.ly", "t.co", "goo.gl"]
    short_hits = [s for s in shorteners if s in lowered]

    # build bonus suspicion score
    # high-risk words (win/prize) add more; OTP/action words moderate
    bonus = 0.0
    if hits_high:
        bonus += 0.30 * len(hits_high)   # big bump per hit
    if short_hits:
        bonus += 0.35 * len(short_hits)  # shorteners are strong signal
    if hits_otp:
        bonus += 0.20 * len(hits_otp)
    if hits_action:
        bonus += 0.15 * len(hits_action)

    bonus = min(0.99, bonus)           # cap

    # combine: new smishing probability = model_prob + bonus * (1 - model_prob)
    smish_score = min(0.99, prob_smish + bonus * (1 - prob_smish))

    # Decision thresholds
    if smish_score >= 0.6:
        verdict = f"🚨 Smishing Message ({smish_score*100:.1f}% confidence)"
    elif smish_score <= 0.35:
        verdict = f"✅ Safe Message ({prob_ham*100:.1f}% confidence)"
    else:
        verdict = f"⚠️ Suspicious — Uncertain ({smish_score*100:.1f}% smish score)"

    triggered = hits_high + hits_otp + hits_action + short_hits
    # make triggered list unique and readable
    triggered = sorted(set(triggered))
    return verdict, triggered


# ====================================================
# Utility: find URLs inside arbitrary text (including shorteners)
# ====================================================
URL_REGEX = re.compile(
    r'(?:(https?:\/\/|www\.)\S+)|'                         # explicit http(s) or www
    r'\b(?:bit\.ly|tinyurl\.com|t\.co|ow\.ly|goo\.gl)\/\S+\b|'  # common shorteners + path
    r'\b(?:[a-z0-9-]+\.)+[a-z]{2,6}(?:\/\S*)?\b',          # bare domains (domain.tld/path)
    flags=re.IGNORECASE
)

def find_urls_in_text(text: str) -> List[str]:
    matches = URL_REGEX.findall(text)
    # regex with groups returns tuples sometimes; simpler to re.findall with a simpler pattern
    # We'll do a safer separate approach:
    tokens = re.findall(r'(https?://\S+|www\.\S+|\b(?:bit\.ly|tinyurl\.com|t\.co|ow\.ly|goo\.gl)/\S+|\b(?:[a-z0-9-]+\.)+[a-z]{2,6}(?:/\S*)?)', text, flags=re.IGNORECASE)
    urls = []
    for u in tokens:
        u = u.strip(".,;:()[]\"'")  # trim punctuation
        # prepend scheme if missing for consistent parsing
        if u.startswith("www.") or u.startswith("bit.ly") or u.startswith("tinyurl"):
            u = "http://" + u
        if not u.lower().startswith(("http://", "https://")) and "." in u:
            u = "http://" + u
        urls.append(u)
    return list(dict.fromkeys(urls))  # unique, preserve order

# ====================================================
# 7️⃣ Unified detector (now checks embedded links first)
# ====================================================
def phishing_detector(text: str) -> Tuple[str, List[str]]:
    text = text.strip()

    # 1) If text contains any URLs, analyze each URL and return the highest-risk verdict.
    urls = find_urls_in_text(text)
    if urls:
        # evaluate each url, choose the most severe outcome to present
        worst_score = -1.0
        worst_label = None
        worst_active = []
        severity_map = {"✅": 0, "⚠️": 1, "🚨": 2}  # simple severity ordering
        for u in urls:
            feats = extract_url_features(u)
            label, active = decide_url_label(feats, u)
            # compute score from label (try to parse the probability if present)
            if "ML" in label:
                # extract probability float
                m = re.search(r'(\d\.\d{2})', label)
                score = float(m.group(1)) if m else 0.0
            else:
                # base severity
                if "🚨" in label:
                    score = 1.0
                elif "⚠️" in label:
                    score = 0.5
                else:
                    score = 0.0
            # choose the worst by score (higher = worse)
            if score > worst_score:
                worst_score = score
                worst_label = f"{label} (source: {u})"
                worst_active = active
            # if perfect phishing found, stop early
            if "🚨" in label:
                break
        return worst_label, worst_active

    # 2) No URL found: run text (smishing) model
    return decide_text_label(text)

# ====================================================
# 8️⃣ Streamlit chat UI (with False Positive button)
# ====================================================
st.markdown("# 🛡️ PhishNet Chat – Real-Time Phishing & Smishing Detector")
st.markdown("Chat naturally — messages and links are automatically checked. Use the **Mark as False Positive** button to help improve accuracy.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for idx, m in enumerate(st.session_state.messages):
    content = m.get("display", m.get("content", ""))

    # Decide colors for each role/result
    if "✅" in content:
        bg = "#b5f5c6"   # light green
        text_col = "#0b3d1e"
    elif "🚨" in content:
        bg = "#ffb3b3"   # light red
        text_col = "#4a0000"
    elif "⚠️" in content:
        bg = "#ffeaa7"   # light amber
        text_col = "#3d2b00"
    else:
        bg = "#2b2b2b"   # neutral dark
        text_col = "#ffffff"

    # 🧑 User bubble (right-aligned)
    if m["role"] == "user":
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end; margin:6px 0;">
              <div style="max-width:80%; background:#1f2b37; color:#fff; padding:12px;
                          border-radius:12px; border:1px solid rgba(255,255,255,0.02);">
                <b>🧑 You:</b> {m['content']}
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 🤖 Assistant bubble (left-aligned)
    else:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-start; margin:6px 0;">
              <div style="max-width:90%; background:{bg}; color:{text_col};
                          padding:12px; border-radius:12px; border:1px solid rgba(0,0,0,0.08);">
                {m.get('display_html', m['content'])}
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Show active features if available
        if m.get("features"):
            st.caption(f"Active features: {', '.join(m['features'])}")

        # ⚙️ Show False Positive button for suspicious or phishing messages
        if any(tag in content for tag in ["🚨", "⚠️"]):
            if st.button("Mark as False Positive", key=f"fp_btn_{idx}"):
                append_fp_to_csv(m.get("content", ""), m.get("features", []), m.get("display", ""))
                st.success("✅ Logged as False Positive for future retraining.")

# Chat input box
user_input = st.chat_input("Type your message or paste a link...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Run detection
    result, active_feats = phishing_detector(user_input)
    assistant_text = f"🤖 **PhishNet Analysis:** {result}"

    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_text,
        "display": assistant_text,
        "features": active_feats
    })

    # rerun to show updated conversation
    st.experimental_rerun()

# ====================================================
# 9️⃣ Footer / instructions
# ====================================================
st.markdown("---")
st.caption("Tip: Test with real popular links (https://youtube.com, https://google.com) and suspicious-looking ones (secure-login.example.test). Use 'Mark as False Positive' if the model misclassifies.")
