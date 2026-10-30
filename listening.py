import streamlit as st
import pandas as pd
import json
import os
import cv2
import numpy as np
from PIL import Image
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
# 横型マークシート（黒四角マーカー自動追尾）自動解析関数
# ---------------------------------------------------------
def analyze_mark_sheet(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_file.seek(0)
    
    # 1. 横長（1200x850）に標準化して歪み・回転補正
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    warp_w, warp_h = 1200, 850
    warped_color = None
    warped_gray = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]     # 左上
            rect[2] = pts[np.argmax(s)]     # 右下
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # 右上
            rect[3] = pts[np.argmax(diff)]  # 左下

            # 縦向きで撮影された場合は90度回転して横長に補正
            w_len = np.linalg.norm(rect[1] - rect[0])
            h_len = np.linalg.norm(rect[3] - rect[0])
            if h_len > w_len:
                rect = np.roll(rect, -1, axis=0)

            dst = np.array([
                [0, 0],
                [warp_w - 1, 0],
                [warp_w - 1, warp_h - 1],
                [0, warp_h - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped_color = cv2.warpPerspective(image, M, (warp_w, warp_h))
            warped_gray = cv2.warpPerspective(gray, M, (warp_w, warp_h))
            break

    if warped_gray is None:
        warped_color = cv2.resize(image, (warp_w, warp_h))
        warped_gray = cv2.resize(gray, (warp_w, warp_h))

    _, binary_sheet = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    debug_img = warped_color.copy()

    # ---------------------------------------------------------
    # 2. 受験番号欄の解析（千・百・十・一位 + 文理）
    # ---------------------------------------------------------
    exam_x_start, exam_x_end = 265, 465
    exam_y_start, exam_y_end = 460, 730
    
    digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    parsed_digits = []
    
    # 4桁（千・百・十・一位）のマーク読み取り
    col_w_exam = (exam_x_end - exam_x_start) / 5.0
    row_h_exam = (exam_y_end - exam_y_start) / 10.0
    
    for col_idx in range(4): # 0:千, 1:百, 2:十, 3:一
        best_digit = ""
        max_pixels = 0
        for row_idx in range(10):
            x1 = int(exam_x_start + col_idx * col_w_exam + 2)
            x2 = int(x1 + col_w_exam - 4)
            y1 = int(exam_y_start + row_idx * row_h_exam + 2)
            y2 = int(y1 + row_h_exam - 4)
            
            roi = binary_sheet[y1:y2, x1:x2]
            count = cv2.countNonZero(roi)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 165, 0), 1)
            
            if count > max_pixels and count > 30:
                max_pixels = count
                best_digit = digits[row_idx]
                
        parsed_digits.append(best_digit if best_digit != "" else "0")
        
    recognized_exam_id = "".join(parsed_digits)

    # ---------------------------------------------------------
    # 3. 解答欄 (1〜25, 26〜50) の解析
    # ---------------------------------------------------------
    recognized_answers = {}
    options_count = 9

    left_col_x1, left_col_x2 = 562, 755
    left_row_y_start, left_row_y_end = 168, 712

    right_col_x1, right_col_x2 = 842, 1035
    right_row_y_start, right_row_y_end = 168, 712

    row_h = (left_row_y_end - left_row_y_start) / 25.0
    col_w = (left_col_x2 - left_col_x1) / float(options_count)

    # 左列 (Q1 〜 Q25)
    for i in range(25):
        q_num = i + 1
        q_key = f'Q{q_num}'
        y1 = int(left_row_y_start + i * row_h + 3)
        y2 = int(y1 + row_h - 6)
        
        best_opt = "未マーク/無効"
        max_pixels = 0
        best_box = None
        
        for opt in range(1, options_count + 1):
            x1 = int(left_col_x1 + (opt - 1) * col_w + 2)
            x2 = int(x1 + col_w - 4)
            
            roi = binary_sheet[y1:y2, x1:x2]
            count = cv2.countNonZero(roi)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            
            if count > max_pixels and count > 35:
                max_pixels = count
                best_opt = opt
                best_box = (x1, y1, x2, y2)
                
        recognized_answers[q_key] = best_opt
        if best_box:
            cv2.rectangle(debug_img, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 255, 0), -1)

    # 右列 (Q26 〜 Q50)
    for i in range(25):
        q_num = i + 26
        q_key = f'Q{q_num}'
        y1 = int(right_row_y_start + i * row_h + 3)
        y2 = int(y1 + row_h - 6)
        
        best_opt = "未マーク/無効"
        max_pixels = 0
        best_box = None
        
        for opt in range(1, options_count + 1):
            x1 = int(right_col_x1 + (opt - 1) * col_w + 2)
            x2 = int(x1 + col_w - 4)
            
            roi = binary_sheet[y1:y2, x1:x2]
            count = cv2.countNonZero(roi)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            
            if count > max_pixels and count > 35:
                max_pixels = count
                best_opt = opt
                best_box = (x1, y1, x2, y2)
                
        recognized_answers[q_key] = best_opt
        if best_box:
            cv2.rectangle(debug_img, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 255, 0), -1)

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
        
        with st.expander("🔍 AI解析位置の確認（デバッグ表示）", expanded=True):
            st.image(st.session_state.debug_image, caption="オレンジ枠：受験番号 / 赤枠：解答欄 / 緑色：認識マーク", use_container_width=True)

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
            if st.button("◀ 画像を撮り直す", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
        with col_btn2:
            if st.button("この内容で確定して採点する ▶", type="primary", use_container_width=True):
                if not student_id:
                    st.error("出席番号/受験番号を入力してください。")
                else:
                    st.session_state.final_class = student_class
                    st.session_state.final_student_id = student_id
                    st.session_state.final_answers = final_answers
                    st.session_state.step = 3
                    st.rerun()

    elif st.session_state.step == 3:
        st.subheader("STEP 3: 採点結果")
        user_ans = st.session_state.final_answers
        student_class = st.session_state.final_class
        student_id = st.session_state.final_student_id
        
        total_score = 0
        results_dict = {}
        processed_qs = set()
        
        row_data = {
            '日時': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'クラス': student_class,
            '出席番号': student_id
        }

        for rule in special_rules.get("unordered", []):
            qs = rule["questions"]
            target_ans = set(rule["answers"])
            user_target_ans = set([user_ans.get(q) for q in qs])
            is_correct = (user_target_ans == target_ans)
            for q in qs:
                p = rule.get("point_per_q", points_key.get(q, 2.0))
                score = p if is_correct else 0
                total_score += score
                results_dict[q] = {
                    '問題': f"{q} (順不同)",
                    'あなたのマーク': user_ans.get(q),
                    '正解': "/".join(map(str, sorted(list(target_ans)))),
                    '判定': "〇" if is_correct else "✕",
                    '配点': p,
                    '獲得点数': score
                }
                processed_qs.add(q)
                row_data[q] = user_ans.get(q)
                row_data[f'{q}_正誤'] = 1 if is_correct else 0

        for rule in special_rules.get("all_correct", []):
            qs = rule["questions"]
            is_all_correct = all([user_ans.get(q) == answer_key.get(q) for q in qs])
            group_pt = rule.get("group_point", 4.0)
            for idx, q in enumerate(qs):
                if idx == 0: total_score += (group_pt if is_all_correct else 0)
                results_dict[q] = {
                    '問題': f"{q} (完答対象)",
                    'あなたのマーク': user_ans.get(q),
                    '正解': answer_key.get(q),
                    '判定': "〇" if is_all_correct else "✕",
                    '配点': f"グループ計{group_pt}" if idx == 0 else "-",
                    '獲得点数': group_pt if (is_all_correct and idx == 0) else (0 if idx == 0 else "-")
                }
                processed_qs.add(q)
                row_
