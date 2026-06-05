import streamlit as st

st.set_page_config(
    page_title="বাংলা OCR | Bangla OCR",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── All other imports AFTER set_page_config ───────────────────────────────────
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import cv2
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Hind Siliguri', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

.hero-header {
    background: linear-gradient(90deg, #f953c6, #b91d73, #f953c6);
    background-size: 200% auto;
    animation: shine 3s linear infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    padding: 0.5rem 0;
    font-family: 'Hind Siliguri', sans-serif;
}
@keyframes shine { to { background-position: 200% center; } }

.hero-sub {
    text-align: center;
    color: #a78bfa;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
    font-family: 'Hind Siliguri', sans-serif;
}

.card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

.card-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f0abfc;
    margin-bottom: 0.8rem;
    font-family: 'Hind Siliguri', sans-serif;
}

.result-word {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 20px 60px rgba(102,126,234,0.4);
}

.result-bangla {
    font-size: 5rem;
    font-weight: 700;
    color: white;
    font-family: 'Hind Siliguri', sans-serif;
    text-shadow: 0 4px 15px rgba(0,0,0,0.3);
    line-height: 1.2;
}

.result-label {
    color: rgba(255,255,255,0.75);
    font-size: 0.9rem;
    margin-top: 0.5rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.char-card {
    background: linear-gradient(135deg, rgba(249,83,198,0.15), rgba(118,75,162,0.15));
    border: 1px solid rgba(249,83,198,0.3);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.char-display {
    font-size: 2.8rem;
    font-weight: 700;
    color: #f0abfc;
    font-family: 'Hind Siliguri', sans-serif;
}

.conf-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 99px;
    height: 8px;
    margin-top: 8px;
    overflow: hidden;
}

.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #f953c6, #b91d73);
}

.success-box {
    background: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(16,185,129,0.1));
    border: 1px solid rgba(52,211,153,0.4);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    color: #6ee7b7;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}

.warning-box {
    background: rgba(251,191,36,0.1);
    border: 1px solid rgba(251,191,36,0.4);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    color: #fcd34d;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}

.error-box {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    color: #fca5a5;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}

.gradient-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #f953c6, #764ba2, transparent);
    border: none;
    margin: 1.5rem 0;
}

.metric-tile {
    background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
    border: 1px solid rgba(102,126,234,0.3);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #a78bfa;
}

.metric-label {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'Hind Siliguri', sans-serif;
}

.rank-item {
    display: flex;
    align-items: center;
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    border: 1px solid rgba(255,255,255,0.08);
}

.rank-char {
    font-size: 1.6rem;
    color: #e2d9f3;
    flex: 1;
    font-family: 'Hind Siliguri', sans-serif;
}

.rank-conf { font-size: 0.9rem; color: #a78bfa; font-weight: 600; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a0533 0%, #0f0c29 100%);
    border-right: 1px solid rgba(249,83,198,0.2);
}
section[data-testid="stSidebar"] * { color: #e2d9f3 !important; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stButton > button {
    background: linear-gradient(135deg, #f953c6 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    font-family: 'Hind Siliguri', sans-serif !important;
    box-shadow: 0 4px 20px rgba(249,83,198,0.4) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(249,83,198,0.6) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE = 32
MODEL_PATH = "models/model.keras"
LABELS_PATH = "labels.json"

BANGLA_CHARS = {
    "1":"অ","2":"আ","3":"ই","4":"ঈ","5":"উ","6":"ঊ","7":"ঋ","8":"এ","9":"ঐ","10":"ও",
    "11":"ঔ","12":"ক","13":"খ","14":"গ","15":"ঘ","16":"ঙ","17":"চ","18":"ছ","19":"জ","20":"ঝ",
    "21":"ঞ","22":"ট","23":"ঠ","24":"ড","25":"ঢ","26":"ণ","27":"ত","28":"থ","29":"দ","30":"ধ",
    "31":"ন","32":"প","33":"ফ","34":"ব","35":"ভ","36":"ম","37":"য","38":"র","39":"ল","40":"শ",
    "41":"ষ","42":"স","43":"হ","44":"ড়","45":"ঢ়","46":"য়","47":"ৎ","48":"ং","49":"ঃ","50":"ঁ",
    "51":"্ক","52":"্ট","53":"্ত","54":"্থ","55":"্ন","56":"্প","57":"্ফ","58":"্ব","59":"্ম","60":"্য",
    "61":"্র","62":"্ল","63":"্ব","64":"্শ","65":"্স","66":"্হ","67":"ক্","68":"গ্","69":"জ্","70":"ত্",
    "71":"দ্","72":"ন্","73":"ব্","74":"ম্","75":"র্","76":"ল্","77":"স্","78":"হ্","79":"ক্ষ","80":"জ্ঞ",
    "81":"০","82":"১","83":"২","84":"৩",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_labels():
    if not Path(MODEL_PATH).exists():
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    idx_to_folder = {v: k for k, v in label_map.items()}
    return model, idx_to_folder


def preprocess_for_model(pil_img):
    gray = pil_img.convert("L")
    gray = ImageOps.invert(gray)
    gray = gray.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(gray, dtype=np.float32) / 255.0
    return arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def segment_characters(pil_img):
    gray = np.array(pil_img.convert("L"))
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    chars = []
    for lbl in range(1, num_labels):
        x, y, w, h, area = stats[lbl]
        if area < 50:
            continue
        pad = 4
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(gray.shape[1], x+w+pad), min(gray.shape[0], y+h+pad)
        chars.append((x1, pil_img.crop((x1, y1, x2, y2))))
    chars.sort(key=lambda t: t[0])
    return [img for _, img in chars]


def predict_character(model, idx_to_folder, pil_img):
    x = preprocess_for_model(pil_img)
    preds = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    conf = float(preds[top_idx])
    folder = idx_to_folder.get(top_idx, str(top_idx))
    bangla = BANGLA_CHARS.get(folder, folder)
    return bangla, conf, preds


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(model):
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:1rem 0;'>
            <div style='font-size:3rem;'>✍️</div>
            <div style='font-size:1.3rem;font-weight:700;color:#f0abfc;
                        font-family:Hind Siliguri,sans-serif;'>বাংলা OCR</div>
            <div style='font-size:0.8rem;color:#a78bfa;margin-top:4px;'>
                Bangla Handwritten Recognition</div>
        </div>
        <hr style='border-color:rgba(249,83,198,0.3);margin:0.5rem 0 1rem;'>
        """, unsafe_allow_html=True)

        if model:
            params = model.count_params()
            st.markdown(f"""
            <div class='metric-tile'>
                <div class='metric-value'>{params:,}</div>
                <div class='metric-label'>মোট প্যারামিটার | Parameters</div>
            </div>
            <div class='metric-tile'>
                <div class='metric-value'>84</div>
                <div class='metric-label'>বর্গ সংখ্যা | Classes</div>
            </div>
            <div class='metric-tile'>
                <div class='metric-value'>32×32</div>
                <div class='metric-label'>ইনপুট সাইজ | Input Size</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <hr style='border-color:rgba(249,83,198,0.3);margin:1rem 0;'>
        <div style='color:#f0abfc;font-weight:600;margin-bottom:0.5rem;
                    font-family:Hind Siliguri,sans-serif;'>💡 ব্যবহার বিধি | Tips</div>
        """, unsafe_allow_html=True)

        for num, bn, en in [
            ("১","আলাদা করে প্রতিটি বর্ণ আঁকুন","Draw chars with gaps"),
            ("২","মোটা কলম ব্যবহার করুন","Use thick pen (8–12px)"),
            ("৩","বাম থেকে ডানে লিখুন","Write left to right"),
            ("৪","আঁকা শেষে বাটন চাপুন","Click Recognise after drawing"),
        ]:
            st.markdown(f"""
            <div style='display:flex;gap:10px;margin-bottom:8px;align-items:flex-start;'>
                <div style='background:linear-gradient(135deg,#f953c6,#764ba2);color:white;
                            border-radius:50%;width:22px;height:22px;display:flex;
                            align-items:center;justify-content:center;font-size:0.7rem;
                            font-weight:700;flex-shrink:0;
                            font-family:Hind Siliguri,sans-serif;'>{num}</div>
                <div>
                    <div style='font-size:0.85rem;color:#e2d9f3;
                                font-family:Hind Siliguri,sans-serif;'>{bn}</div>
                    <div style='font-size:0.75rem;color:#a78bfa;'>{en}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <hr style='border-color:rgba(249,83,198,0.3);margin:1rem 0;'>
        <div style='font-size:0.75rem;color:#6b6b8a;text-align:center;'>
            Dataset: BanglaLekha-Isolated<br>
            Framework: TensorFlow / Keras<br>
            Tracking: MLflow<br><br>
            SICIP @ BRAC University
        </div>
        """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    model, idx_to_folder = load_model_and_labels()
    render_sidebar(model)

    st.markdown("""
    <div class='hero-header'>বাংলা হাতের লেখা সনাক্তকরণ</div>
    <div class='hero-sub'>Bangla Handwritten Word Recognition · BanglaLekha-Isolated · CNN · MLflow</div>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='gradient-divider'>", unsafe_allow_html=True)

    if model is None:
        st.markdown("""
        <div class='error-box'>
            ⚠️ <b>মডেল পাওয়া যায়নি!</b> Model not found at <code>models/model.keras</code><br>
            প্রথমে ট্রেনিং চালান: <code>python train.py --run_both</code>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown(f"""
    <div class='success-box'>
        ✅ <b>মডেল লোড সম্পন্ন!</b> Model loaded · <b>{model.count_params():,}</b> parameters · Ready ✓
    </div>
    """, unsafe_allow_html=True)

    col_canvas, col_result = st.columns([1.1, 0.9], gap="large")

    with col_canvas:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🖊️ এখানে বাংলা শব্দ আঁকুন | Draw Bangla Word Here</div>",
                    unsafe_allow_html=True)
        pen_col, _ = st.columns([1, 2])
        with pen_col:
            stroke_width = st.slider("কলমের মোটা | Pen Width", 4, 20, 10)

        canvas_result = st_canvas(
            fill_color="rgba(255,255,255,0)",
            stroke_width=stroke_width,
            stroke_color="#1a1a2e",
            background_color="#FFFFFF",
            height=200,
            width=480,
            drawing_mode="freedraw",
            key="canvas",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        recognise = st.button("🔍 সনাক্ত করুন | Recognise Word")
        st.markdown("""
        <div style='text-align:center;color:#6b6b8a;font-size:0.8rem;margin-top:0.5rem;
                    font-family:Hind Siliguri,sans-serif;'>
            ক্যানভাস রিফ্রেশ করতে পেজ রিলোড করুন · Reload page to clear canvas
        </div>
        """, unsafe_allow_html=True)

    with col_result:
        if recognise:
            if canvas_result.image_data is None or canvas_result.image_data.max() == 0:
                st.markdown("""
                <div class='warning-box'>
                    ✏️ <b>ক্যানভাস খালি!</b> Canvas is empty — draw something first.
                </div>
                """, unsafe_allow_html=True)
            else:
                pil_img = Image.fromarray(canvas_result.image_data.astype(np.uint8)[..., :3])
                chars = segment_characters(pil_img)
                if not chars:
                    st.markdown("""
                    <div class='warning-box'>
                        ⚠️ কোনো বর্ণ শনাক্ত হয়নি। No characters detected.<br>
                        মোটা কলম দিয়ে স্পষ্টভাবে আঁকুন। Try a thicker stroke.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    results = []
                    predicted_word = ""
                    for ci in chars:
                        b, c, p = predict_character(model, idx_to_folder, ci)
                        predicted_word += b
                        results.append((ci, b, c, p))

                    st.markdown(f"""
                    <div class='result-word'>
                        <div class='result-label'>🎯 Predicted Word · শনাক্তকৃত শব্দ</div>
                        <div class='result-bangla'>{predicted_word}</div>
                        <div class='result-label'>{len(results)} টি বর্ণ | {len(results)} character(s)</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<hr class='gradient-divider'>", unsafe_allow_html=True)
                    st.markdown("""
                    <div style='font-size:1rem;font-weight:600;color:#f0abfc;margin-bottom:1rem;
                                font-family:Hind Siliguri,sans-serif;'>
                        📊 প্রতিটি বর্ণের বিশ্লেষণ | Per-Character Analysis
                    </div>
                    """, unsafe_allow_html=True)

                    cols = st.columns(min(len(results), 6))
                    for i, (char_img, bangla, conf, _) in enumerate(results):
                        with cols[i % len(cols)]:
                            st.image(char_img.convert("RGB").resize((90, 90), Image.NEAREST),
                                     use_container_width=True)
                            st.markdown(f"""
                            <div class='char-card'>
                                <div class='char-display'>{bangla}</div>
                                <div style='color:#a78bfa;font-size:0.8rem;
                                            font-family:Hind Siliguri,sans-serif;'>
                                    বর্ণ {i+1}</div>
                                <div class='conf-bar-bg'>
                                    <div class='conf-bar-fill'
                                         style='width:{int(conf*100)}%'></div>
                                </div>
                                <div style='color:#f0abfc;font-size:0.8rem;
                                            font-weight:600;margin-top:4px;'>
                                    {conf*100:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)

                    # Top-3
                    st.markdown("<hr class='gradient-divider'>", unsafe_allow_html=True)
                    st.markdown("""
                    <div style='font-size:1rem;font-weight:600;color:#f0abfc;margin-bottom:0.8rem;
                                font-family:Hind Siliguri,sans-serif;'>
                        🏆 শেষ বর্ণের শীর্ষ ৩ | Top-3 for Last Character
                    </div>
                    """, unsafe_allow_html=True)
                    _, _, _, last_preds = results[-1]
                    medals = ["🥇", "🥈", "🥉"]
                    with open(LABELS_PATH, encoding="utf-8") as f:
                        lm = json.load(f)
                    i2f = {v: k for k, v in lm.items()}
                    for rank, idx in enumerate(np.argsort(last_preds)[::-1][:3]):
                        ch = BANGLA_CHARS.get(i2f.get(int(idx), str(idx)), "?")
                        st.markdown(f"""
                        <div class='rank-item'>
                            <div style='font-size:1.4rem;width:32px;'>{medals[rank]}</div>
                            <div class='rank-char'>{ch}</div>
                            <div class='rank-conf'>{last_preds[idx]*100:.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='height:200px;display:flex;flex-direction:column;
                        align-items:center;justify-content:center;
                        background:rgba(255,255,255,0.03);
                        border:2px dashed rgba(249,83,198,0.3);
                        border-radius:16px;text-align:center;'>
                <div style='font-size:3rem;margin-bottom:0.5rem;'>✍️</div>
                <div style='font-family:Hind Siliguri,sans-serif;font-size:1rem;
                            color:#a78bfa;'>বাম দিকে শব্দ এঁকে বাটন চাপুন</div>
                <div style='font-size:0.85rem;color:#6b6b8a;margin-top:4px;'>
                    Draw on the left and click Recognise</div>
            </div>
            """, unsafe_allow_html=True)


main()