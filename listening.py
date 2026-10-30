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
ANSWER_KEY_FILE = "answer_key.json"
POINTS_KEY_FILE = "points_key.json"
RULES_KEY_FILE = "special_rules.json"

st.set_page_config(page_title="共通テスト採点・集計システム", layout="wide")
st.title("共通テスト演習 自動採点・集計システム")

# ---------------------------------------------------------
# 各種データ読み込み・保存関数
# ---------------------------------------------------------
def load_json_file(filepath, default_data):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return default_data

def save_json_file(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_answer_key():
    return load_json_file(ANSWER_KEY_FILE, {f'Q{i}': 1 for i in range(1, 51)})

def load_points_key():
    return load_json_file(POINTS_KEY_FILE, {f'Q{i}': 2.0 for i in range(1, 51)})

def load_special_rules():
    return load_json_file(RULES_KEY_FILE, {"unordered": [], "all_correct": []})

# ---------------------------------------------------------
# 12個の黒四角マーカー自動検出 ＆ 自動向き補正 ＆ 3ブロック解析
# ---------------------------------------------------------
def detect_square_markers(gray_img):
    """画像から完全に塗りつぶされた12個の黒四角マーカーを抽出"""
    norm_h, norm_w = gray_img.shape
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    square_markers = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = float(bw) / bh
        area = cv2.contourArea(c)
        solidity = area / float(bw * bh) if (bw * bh) > 0 else 0

        # 正方形かつ塗りつぶされた黒四角（Solidity > 0.82）
        if 0.75 < aspect < 1.35 and (norm_w * 0.008) < bw < (norm_w * 0.035) and (norm_h * 0.008) < bh < (norm_h * 0.035) and solidity > 0.82:
            square_markers.append((x + bw/2.0, y + bh/2.0))
            
    return square_markers

def auto_orient_and_get_markers(image):
    """画像の上下左右（0度,90度,180度,270度）から正しい向き（受験番号が左）を自動認識"""
    for rot in [0, 1, 2, 3]: # 0:そのまま, 1:90度, 2:180度, 3:270度
        if rot == 0:
            test_img = image.copy()
        elif rot == 1:
            test_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 2:
            test_img = cv2.rotate(image, cv2.ROTATE_180)
        elif rot == 3:
            test_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w = test_img.shape[:2]
        if w < h: # 横長でない場合はスキップ
            continue

        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        markers = detect_square_markers(gray)

        if len(markers) >= 10:
            # X座標でソート
            markers_sorted_x = sorted(markers, key=lambda m: m[0])
            # 左グループ4個のY範囲と、中央グループ4個のY範囲を比較
            left_ys = [m[1] for m in markers_sorted_x[:4]]
            mid_ys = [m[1] for m in markers_sorted_x[4:8]]
            
            left_h = max(left_ys) - min(left_ys)
            mid_h = max(mid_ys) - min(mid_ys)

            # 受験番号欄（左）は解答欄（中央）よりも縦幅が短い構造
            if left_h < mid_h:
                return test_img, markers

    # 自動判定できなかった場合は元の画像のまま返す
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, detect_square_markers(gray)

def analyze_mark_sheet(image_file):
    # PILで読み込んでEXIF回転を正規化
    pil_img = Image.open(image_file)
    pil_img = ImageOps.exif_transpose(pil_img)
    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    image_file.seek(0)

    # 標準解像度にリサイズ（3000 x 2100）
    norm_w, norm_h = 3000, 2100
    image = cv2.resize(image, (norm_w, norm_h))

    # 自動回転＆マーカー認識
    oriented_img, markers = auto_orient_and_get_markers(image)
    gray = cv2.cvtColor(oriented_img, cv2.COLOR_BGR2GRAY)
    debug_img = oriented_img.copy()

    def warp_block(pts_4, target_w, target_h):
        """4点のマーカーから長方形ブロックを抽出して二値化"""
        pts_src = np.array(pts_4, dtype="float32")
        pts_dst = np.array([[0, 0], [target_w - 1, 0], [0, target_h - 1], [target_w - 1, target_h - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(gray, M, (target_w, target_h))
        _, bin_out = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return bin_out

    recognized_exam_id = "0000"
    recognized_answers = {f'Q{i}': "未マーク/無効" for i in range(1, 51)}

    if len(markers) >= 12:
        # X座標順にソートして3つのブロックに分離
        markers_sorted_x = sorted(markers, key=lambda m: m[0])
        group_exam = sorted(markers_sorted_x[0:4], key=lambda m: (m[1], m[0]))  # 受験番号 (左4個)
        group_q1_25 = sorted(markers_sorted_x[4:8], key=lambda m: (m[1], m[0])) # Q1~25 (中央4個)
        group_q26_50 = sorted(markers_sorted_x[8:12], key=lambda m: (m[1], m[0]))# Q26~50 (右4個)

        # 1. 受験番号ブロックの解析 (千・百・十・一位)
        try:
            pts_exam = [group_exam[0], group_exam[1], group_exam[2], group_exam[3]]
            bin_exam = warp_block(pts_exam, 400, 600)
            digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            parsed_digits = []
            
            # 千・百・十・一の4列 (インデックス1〜4)
            for col_idx in range(1, 5):
                best_d = "0"
                max_p = 0
                x1 = int(col_idx * (400 / 6.0) + 4)
                x2 = int(x1 + (400 / 6.0) - 8)
                for row_idx in range(10):
                    y1 = int(row_idx * (600 / 10.0) + 4)
                    y2 = int(y1 + (600 / 10.0) - 8)
                    roi = bin_exam[y1:y2, x1:x2]
                    cnt = cv2.countNonZero(roi)
                    if cnt > max_p and cnt > 40:
                        max_p = cnt
                        best_d = digits[row_idx]
                parsed_digits.append(best_d)
            recognized_exam_id = "".join(parsed_digits)
        except Exception:
            pass

        # 2. 問 1 〜 25 ブロックの解析
        try:
            pts_q1 = [group_q1_25[0], group_q1_25[1], group_q1_25[2], group_q1_25[3]]
            bin_q1 = warp_block(pts_q1, 600, 1800)
            options_count = 9
            col_w = 600.0 / options_count
            row_h = 1800.0 / 25.0

            for i in range(25):
                q_key = f'Q{i+1}'
                y1 = int(i * row_h + 4)
                y2 = int(y1 + row_h - 8)
                best_opt = "未マーク/無効"
                max_p = 0
                for opt in range(1, options_count + 1):
                    x1 = int((opt - 1) * col_w + 4)
                    x2 = int(x1 + col_w - 8)
                    roi = bin_q1[y1:y2, x1:x2]
                    cnt = cv2.countNonZero(roi)
                    if cnt > max_p and cnt > 45:
                        max_p = cnt
                        best_opt = opt
                recognized_answers[q_key] = best_opt
        except Exception:
            pass

        # 3. 問 26 〜 50 ブロックの解析
        try:
            pts_q2 = [group_q26_50[0], group_q26_50[1], group_q26_50[2], group_q26_50[3]]
            bin_q2 = warp_block(pts_q2, 600, 1800)
            options_count = 9
            col_w = 600.0 / options_count
            row_h = 1800.0 / 25.0

            for i in range(25):
                q_key = f'Q{i+26}'
                y1 = int(i * row_h + 4)
                y2 = int(y1 + row_h - 8)
                best_opt = "未マーク/無効"
                max_p = 0
                for opt in range(1, options_count + 1):
                    x1 = int((opt - 1) * col_w + 4)
                    x2 = int(x1 + col_w - 8)
                    roi = bin_q2[y1:y2, x1:x2]
                    cnt = cv2.countNonZero(roi)
                    if cnt > max_p and cnt > 45:
                        max_p = cnt
                        best_opt = opt
                recognized_answers[q_key] = best_opt
        except Exception:
            pass

        # 検出マーカー12箇所を緑色の円で可視化
        for m in markers:
            cv2.circle(debug_img, (int(m[0]), int(m[1])), 20, (0, 255, 0), -1)

    debug_img_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
    return recognized_exam_id, recognized_answers, debug_img_rgb

# セッション状態初期化
if "step" not in st.session_state:
    st.session_state.step = 1

mode = st.sidebar.radio("モード選択", [
    "生徒用（マーク撮影・採点）", 
    "教員用（正解・配点・特殊ルール設定）", 
    "教員用（集計・Excel出力）"
])

# ---------------------------------------------------------
# 1. 生徒用：確認画面付き採点フロー
# ---------------------------------------------------------
if mode == "生徒用（マーク撮影・採点）":
    st.header("■ 生徒用 マークシート送信・採点")
    answer_key = load_answer_key()
    points_key = load_points_key()
    special_rules = load_special_rules()

    if st.session_state.step == 1:
        st.subheader("STEP 1: マークシート画像のアップロード")
        uploaded_file = st.file_uploader("解答用紙の全体が写るように撮影してアップロードしてください", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="撮影画像", width=450)
            if st.button("画像を解析して確認画面へ進む", type="primary"):
                with st.spinner("マークを自動読み取り中..."):
                    exam_id, answers, debug_img = analyze_mark_sheet(uploaded_file)
                    st.session_state.uploaded_image = uploaded_file
                    st.session_state.recognized_exam_id = exam_id
                    st.session_state.recognized_answers = answers
                    st.session_state.debug_image = debug_img
                    st.session_state.step = 2
                    st.rerun()

    elif st.session_state.step == 2:
        st.subheader("STEP 2: 読み取り結果の確認・修正")
        
        with st.expander("🔍 AI解析マーカー位置の確認（デバッグ表示）", expanded=True):
            st.image(st.session_state.debug_image, caption="緑色の円：自動検出された12箇所の基準点マーカー", use_container_width=True)

        col_info1, col_info2 = st.columns(2)
        with col_info1:
            student_class = st.selectbox("クラス", ["1-A", "1-B", "1-C", "2-A", "2-B", "3-A", "3-B"], index=0)
        with col_info2:
            student_id = st.text_input("出席番号 / 受験番号", value=st.session_state.recognized_exam_id)

        st.markdown("---")
        st.subheader("■ マーク判定結果 (問1 〜 問50)")

        col_left, col_right = st.columns(2)
        final_answers = {}
        options = [1, 2, 3, 4, 5, 6, 7, 8, 9, "未マーク/無効"]

        with col_left:
            st.markdown("##### ◀ 解答欄 (問1 〜 問25)")
            for i in range(1, 26):
                q_key = f'Q{i}'
                current_val = st.session_state.recognized_answers.get(q_key, "未マーク/無効")
                idx = options.index(current_val) if current_val in options else 9
                
                q_col1, q_col2 = st.columns([1, 2])
                with q_col1:
                    st.write(f"**問 {i}**")
                with q_col2:
                    final_answers[q_key] = st.selectbox(
                        f"問{i}", options=options, index=idx, key=f"select_{q_key}", label_visibility="collapsed"
                    )

        with col_right:
            st.markdown("##### ▶ 解答欄 (問26 〜 問50)")
            for i in range(26, 51):
                q_key = f'Q{i}'
                current_val = st.session_state.recognized_answers.get(q_key, "未マーク/無効")
                idx = options.index(current_val) if current_val in options else 9
                
                q_col1, q_col2 = st.columns([1, 2])
                with q_col1:
                    st.write(f"**問 {i}**")
                with q_col2:
                    final_answers[q_key] = st.selectbox(
                        f"問{i}", options=options, index=idx, key=f"select_{q_key}", label_visibility="collapsed"
                    )

        st.markdown("---")
        col_btn1, col_btn2 = st.columns([1, 2])
        with col_btn1:
            if st.button("◀ 画像を撮り直す",
