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
RULES_KEY_FILE = "special_rules.json"
CONFIG_FILE = "config.json"
PASSWORD_FILE = "password.json"

st.set_page_config(page_title="共通テスト採点・集計システム", layout="wide")
st.title("共通テスト演習 自動採点・集計システム")

# ---------------------------------------------------------
# 設定・データ管理関数（テストID対応）
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

def load_config():
    return load_json_file(CONFIG_FILE, {"max_questions": 50, "num_options": 9})

# --- テストID別の正答読み込み・保存 ---
def load_answer_key(test_id, max_q):
    data = load_json_file(ANSWER_KEY_FILE, {})
    t_key = str(test_id).zfill(2)
    ans_data = data.get(t_key, {})
    default_ans = {f'Q{i}': 1 for i in range(1, max_q + 1)}
    return {f'Q{i}': ans_data.get(f'Q{i}', 1) for i in range(1, max_q + 1)}

def save_answer_key(test_id, ans_dict):
    data = load_json_file(ANSWER_KEY_FILE, {})
    t_key = str(test_id).zfill(2)
    data[t_key] = ans_dict
    save_json_file(ANSWER_KEY_FILE, data)

# --- テストID別の配点読み込み・保存 ---
def load_points_key(test_id, max_q):
    data = load_json_file(POINTS_KEY_FILE, {})
    t_key = str(test_id).zfill(2)
    pts_data = data.get(t_key, {})
    return {f'Q{i}': float(pts_data.get(f'Q{i}', 2.0)) for i in range(1, max_q + 1)}

def save_points_key(test_id, pts_dict):
    data = load_json_file(POINTS_KEY_FILE, {})
    t_key = str(test_id).zfill(2)
    data[t_key] = pts_dict
    save_json_file(POINTS_KEY_FILE, data)

def load_special_rules():
    return load_json_file(RULES_KEY_FILE, {"unordered": [], "all_correct": []})

# ---------------------------------------------------------
# 12個マーカー分離アルゴリズム
# ---------------------------------------------------------
def group_12_markers_robust(markers):
    markers_by_y = sorted(markers, key=lambda m: m[1])
    top_6 = sorted(markers_by_y[:6], key=lambda m: m[0])
    bot_6 = sorted(markers_by_y[6:], key=lambda m: m[0])
    
    group_exam = [top_6[0], top_6[1], bot_6[0], bot_6[1]]  # TEST ID + STUDENT ID
    group_q1   = [top_6[2], top_6[3], bot_6[2], bot_6[3]]  # Q1-25
    group_q2   = [top_6[4], top_6[5], bot_6[4], bot_6[5]]  # Q26-50
    
    return group_exam, group_q1, group_q2

# ---------------------------------------------------------
# マーカー自動検出
# ---------------------------------------------------------
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
                    score = solidity - abs(1.0 - aspect) * 0.2
                    candidates.append((cx, cy, score))

        if len(candidates) >= 12:
            candidates.sort(key=lambda item: item[2], reverse=True)
            top_12 = [(c[0], c[1]) for c in candidates[:12]]
            return top_12, f"マーカー12個検出成功"

    return [(c[0], c[1]) for c in candidates], f"検出数不十分 ({len(candidates)}/12個)"

def auto_orient_and_get_markers(image):
    for rot in [0, 1, 2, 3]:
        if rot == 0: test_img = image.copy()
        elif rot == 1: test_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 2: test_img = cv2.rotate(image, cv2.ROTATE_180)
        elif rot == 3: test_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w = test_img.shape[:2]
        if w < h: continue

        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        markers, msg = detect_square_markers(gray)

        if len(markers) == 12:
            try:
                g_exam, g_q1, _ = group_12_markers_robust(markers)
                if abs(g_exam[0][0] - g_exam[2][0]) < 100:
                    return test_img, markers, f"向き自動検出完了: {msg}"
            except:
                pass
    return image, markers, f"向き補正不可: {msg}"

# ---------------------------------------------------------
# メイン解析処理
# ---------------------------------------------------------
def analyze_mark_sheet(image_file, max_q=50, num_opts=9):
    logs = []
    pil_img = Image.open(image_file)
    pil_img = ImageOps.exif_transpose(pil_img)
    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    norm_w, norm_h = 3000, 2100
    image = cv2.resize(image, (norm_w, norm_h))

    oriented_img, markers, orient_msg = auto_orient_and_get_markers(image)
    logs.append(orient_msg)
    
    gray = cv2.cvtColor(oriented_img, cv2.COLOR_BGR2GRAY)
    debug_main = oriented_img.copy()

    def warp_block(pts_4, target_w, target_h):
        pts_src = np.array(pts_4, dtype="float32")
        pts_dst = np.array([[0, 0], [target_w - 1, 0], [0, target_h - 1], [target_w - 1, target_h - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped_g = cv2.warpPerspective(gray, M, (target_w, target_h))
        warped_c = cv2.warpPerspective(oriented_img, M, (target_w, target_h))
        _, bin_out = cv2.threshold(warped_g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return bin_out, warped_c

    recognized_test_id = "00"
    recognized_exam_id = "0000"
    recognized_answers = {f'Q{i}': "未マーク/無効" for i in range(1, max_q + 1)}
    img_exam_debug, img_q1_debug, img_q2_debug = None, None, None

    if len(markers) == 12:
        try:
            g_exam, g_q1, g_q2 = group_12_markers_robust(markers)

            # 1. TEST ID (2桁) & STUDENT ID (年・組・番・号 4桁)
            bin_exam, color_exam = warp_block(g_exam, 600, 1600)
            img_exam_debug = color_exam.copy()
            digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            parsed_digits = []
            
            col_w = 600.0 / 6.0
            row_h = (1600.0 - 90.0) / 10.0
            y_offset = 45.0

            for col_idx in range(6):
                best_d = "0"
                max_cnt = 0
                best_box = None
                
                cx = (col_idx + 0.5) * col_w
                for row_idx in range(10):
                    cy = y_offset + (row_idx + 0.5) * row_h
                    x1, x2 = int(cx - col_w * 0.25), int(cx + col_w * 0.25)
                    y1, y2 = int(cy - row_h * 0.25), int(cy + row_h * 0.25)
                    
                    roi = bin_exam[y1:y2, x1:x2]
                    cnt = cv2.countNonZero(roi)
                    cv2.rectangle(img_exam_debug, (x1, y1), (x2, y2), (255, 165, 0), 1)
                    
                    if cnt > max_cnt and cnt > 25:
                        max_cnt = cnt
                        best_d = digits[row_idx]
                        best_box = (x1, y1, x2, y2)
                        
                if best_box:
                    cv2.rectangle(img_exam_debug, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 255, 0), -1)
                parsed_digits.append(best_d)
                
            recognized_test_id = "".join(parsed_digits[:2])
            recognized_exam_id = "".join(parsed_digits[2:])
            logs.append(f"IDブロック読み取り完了 -> テストID: {recognized_test_id} / STUDENT ID: {recognized_exam_id}")

            # 2. 解答欄解析 (選択肢 1〜9)
            def parse_q_range(bin_img, color_img, start_q_idx):
                debug_img = color_img.copy()
                qw, qh = bin_img.shape[1], bin_img.shape[0]
                col_w = qw / 10.0
                row_h = (qh - 70.0) / 25.0
                y_offset = 35.0

                for i in range(25):
                    q_num = start_q_idx + i
                    if q_num > max_q: break
                    q_key = f'Q{q_num}'
                    
                    cy = y_offset + (i + 0.5) * row_h
                    counts = []
                    boxes = []
                    
                    for opt in range(1, num_opts + 1):
                        cx = (opt + 0.3) * col_w
                        x1, x2 = int(cx - col_w * 0.22), int(cx + col_w * 0.22)
                        y1, y2 = int(cy - row_h * 0.25), int(cy + row_h * 0.25)
                        
                        roi = bin_img[y1:y2, x1:x2]
                        cnt
