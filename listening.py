import streamlit as st
import pandas as pd
import json
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from datetime import datetime

# --- データ保存用ファイルパス ---
DATA_FILE = "test_results.csv"
ANSWER_KEY_FILE = "answer_keys.json"   # テストIDごとの正答
POINTS_KEY_FILE = "points_keys.json"   # テストIDごとの配点
TEST_CONFIG_FILE = "test_configs.json" # テストIDごとの問題数・合計点
RULES_KEY_FILE = "special_rules.json"
CONFIG_FILE = "config.json"
PASSWORD_FILE = "password.json"

st.set_page_config(page_title="共通テスト採点・集計システム", layout="wide")
st.title("共通テスト演習 自動採点・集計システム")

# ---------------------------------------------------------
# 設定・データ管理関数（テストID対応・整数配点）
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

def load_global_config():
    return load_json_file(CONFIG_FILE, {"max_questions": 50, "num_options": 9})

def load_test_config(test_id):
    data = load_json_file(TEST_CONFIG_FILE, {})
    t_key = str(test_id).zfill(2)
    default_cfg = {"max_questions": 50, "total_score": 100}
    return data.get(t_key, default_cfg)

def save_test_config(test_id, max_q, total_score):
    data = load_json_file(TEST_CONFIG_FILE, {})
    t_key = str(test_id).zfill(2)
    data[t_key] = {"max_questions": int(max_q), "total_score": int(total_score)}
    save_json_file(TEST_CONFIG_FILE, data)

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

def load_special_rules():
    return load_json_file(RULES_KEY_FILE, {"unordered": [], "all_correct": []})

# ------------------------------------------------
