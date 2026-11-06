import streamlit as st
import pandas as pd
import json
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from datetime import datetime
import plotly.graph_objects as go
from filelock import FileLock

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# --- データ保存用ファイルパス ---
DATA_FILE = "test_results.csv"
LOCK_FILE = "test_results.csv.lock"
ANSWER_KEY_FILE = "answer_keys.json"
POINTS_KEY_FILE = "points_keys.json"
TEST_CONFIG_FILE = "test_configs.json"
RULES_KEY_FILE = "special_rules.json"
SECTIONS_KEY_FILE = "sections_keys.json"
CONFIG_FILE = "config.json"
PASSWORD_FILE = "password.json"
GEMINI_KEY_FILE = "gemini_key.json"

st.set_page_config(page_title="共通テスト採点・集計システム", layout="wide")

# ---------------------------------------------------------
# カスタムCSS
# ---------------------------------------------------------
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
        font-family: 'Helvetica Neue', Arial, 'Hiragino Kaku Gothic ProN', 'Hiragino Sans', Meiryo, sans-serif;
    }
    .hero-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(30, 58, 138, 0.2);
    }
    .hero-header h1 {
        color: white !important;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .hero-header p {
        color: #e0f2fe;
        margin-top: 0.4rem;
        font-size: 0.95rem;
    }
    .section-score-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 5px solid #0284c7;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .section-score-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #0369a1;
    }
    .section-score-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0c4a6e;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 設定・データ管理関数
# ---------------------------------------------------------
def load_json_file(filepath, default_data):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return default_data

def save_json_file(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_password():
    data = load_json_file(PASSWORD_FILE, {"password": "0000"})
    return data.get("password", "0000")

def save_password(new_pass):
    save_json_file(PASSWORD_FILE, {"password": new_pass})

def load_gemini_key():
    data = load_json_file(GEMINI_KEY_FILE, {"api_key": ""})
    return data.get("api_key", "")

def save_gemini_key(api_key):
    save_json_file(GEMINI_KEY_FILE, {"api_key": api_key})

def load_global_config():
    return load_json_file(CONFIG_FILE, {"max_questions": 50, "num_options": 9})

def load_test_config(test_id):
    data = load_json_file(TEST_CONFIG_FILE, {})
    t_key = str(test_id).zfill(2)
    default_cfg = {"max_questions": 37, "total_score": 100}
    return data.get(t_key, default_cfg)

def save_test_config(test_id, max_q, total_score):
    data = load_json_file(TEST_CONFIG_FILE, {})
    t_key = str(test_id).zfill(2)
    data[t_key] = {"max_questions": int(max_q), "total_score": int(total_score)}
    save_json_file(TEST_CONFIG_FILE, data)

def load_sections_config(test_id):
    data = load_json_file(SECTIONS_KEY_FILE, {})
    t_key = str(test_id).zfill(2)
    return data.get(t_key, [])

def save_sections_config(test_id, sections_list):
    data = load_json_file(SECTIONS_KEY_FILE, {})
    t_key = str(test_id).zfill(2)
    data[t_key] = sections_list
    save_json_file(SECTIONS_KEY_FILE, data)

def load_answer_key(test_id, max_q):
    data = load_json_file(ANSWER_KEY_FILE, {})
    t_key = str(test_id).zfill(2)
    ans_data = data.get(t_key, {})
    return {f'Q{i}': int(ans_data.get(f'Q{i}', 1)) for i in range(1, max_q + 1)}

def save_answer_key(test_id, ans_dict):
    data = load_json_file(ANSWER_KEY_FILE, {})
    t_key = str(test_id).zfill(2)
    data[t_key] = {k: int(v) if str(v).isdigit() else v for k, v in ans_dict.items()}
    save_json_file(ANSWER_KEY_FILE, data)

def load_points_key(test_id, max_q):
    data = load_json_file(POINTS_KEY_FILE, {})
    t_key = str(test_id).zfill(2)
    pts_data = data.get(t_key, {})
    return {f'Q{i}': int(pts_data.get(f'Q{i}', 2)) for i in range(1, max_q + 1)}

def save_points_key(test_id, pts_dict):
    data = load_json_file(POINTS_KEY_FILE, {})
    t_key = str(test_id).zfill(2)
    data[t_key] = {k: int(v) for k, v in pts_dict.items()}
    save_json_file(POINTS_KEY_FILE, data)

def load_special_rules(test_id):
    data = load_json_file(RULES_KEY_FILE, {})
    t_key = str(test_id).zfill(2)
    return data.get(t_key, [])

def save_special_rules(test_id, rules_list):
    data = load_json_file(RULES_KEY_FILE, {})
    t_key = str(test_id).zfill(2)
    data[t_key] = rules_list
    save_json_file(RULES_KEY_FILE, data)

def safe_read_csv(filepath):
    if not os.path.exists(filepath):
        return pd.DataFrame()
    try:
        return pd.read_csv(filepath, encoding='utf-8-sig', dtype={'TEST ID': str, 'STUDENT ID': str})
    except Exception:
        try:
            return pd.read_csv(filepath, encoding='utf-8-sig', on_bad_lines='skip', dtype={'TEST ID': str, 'STUDENT ID': str})
        except Exception:
            return pd.DataFrame()

def get_clean_unique_test_ids(df):
    if df.empty or 'TEST ID' not in df.columns:
        return []
    clean_series = df['TEST ID'].dropna().astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.zfill(2)
    clean_series = clean_series[clean_series != 'NAN']
    return sorted(list(clean_series.unique()))

def save_test_result_to_csv(row_data):
    df_new = pd.DataFrame([row_data])
    lock = FileLock(LOCK_FILE, timeout=10)
    
    with lock:
        if os.path.exists(DATA_FILE):
            df_existing = safe_read_csv(DATA_FILE)
            if not df_existing.empty:
                df_all = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_all = df_new
        else:
            df_all = df_new
        df_all.to_csv(DATA_FILE, index=False, encoding='utf-8-sig')

# ---------------------------------------------------------
# 4頂点の自動正規化ソート
# ---------------------------------------------------------
def order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[3] = pts[np.argmax(s)]  # 右下

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # 右上
    rect[2] = pts[np.argmax(diff)] # 左下

    return rect

def group_12_markers_robust(markers):
    markers_by_y = sorted(markers, key=lambda m: m[1])
    top_6 = sorted(markers_by_y[:6], key=lambda m: m[0])
    bot_6 = sorted(markers_by_y[6:], key=lambda m: m[0])
    
    group_exam = [top_6[0], top_6[1], bot_6[0], bot_6[1]]
    group_q1   = [top_6[2], top_6[3], bot_6[2], bot_6[3]]
    group_q2   = [top_6[4], top_6[5], bot_6[4], bot_6[5]]
    
    return group_exam, group_q1, group_q2

def detect_square_markers(gray_img):
    norm_h, norm_w = gray_img.shape
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    thresh_methods = [
        cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5),
        cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)[1],
    ]

    for thresh in thresh_methods:
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = float(bw) / bh
            area = cv2.contourArea(c)
            solidity = area / float(bw * bh) if (bw * bh) > 0 else 0

            if 0.65 < aspect < 1.5 and (norm_w * 0.005) < bw < (norm_w * 0.05) and solidity > 0.65:
                cx, cy = x + bw/2.0, y + bh/2.0
                if not any(abs(cx - mx) < 25 and abs(cy - my) < 25 for mx, my, _ in candidates):
                    score = solidity - abs(1.0 - aspect) * 0.
