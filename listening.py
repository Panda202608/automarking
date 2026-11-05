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

# --- データ保存用ファイルパス ---
DATA_FILE = "test_results.csv"
LOCK_FILE = "test_results.csv.lock"   # 排他ロック用ファイル
ANSWER_KEY_FILE = "answer_keys.json"   # テストIDごとの正答
POINTS_KEY_FILE = "points_keys.json"   # テストIDごとの配点
TEST_CONFIG_FILE = "test_configs.json" # テストIDごとの問題数・合計点
RULES_KEY_FILE = "special_rules.json"  # テストIDごとの特殊配点ルール
SECTIONS_KEY_FILE = "sections_keys.json" # テストIDごとの大問設定
CONFIG_FILE = "config.json"
PASSWORD_FILE = "password.json"

st.set_page_config(page_title="共通テスト採点・集計システム", layout="wide")

# ---------------------------------------------------------
# カスタムCSS（生徒画面・UIデザイン改善）
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
    .custom-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
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

# --- 同時書き込み防止（排他ロック付き保存処理） ---
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
                        cnt = cv2.countNonZero(roi)
                        counts.append(cnt)
                        boxes.append((x1, y1, x2, y2))
                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (200, 200, 200), 1)

                    valid_opts = []
                    for opt_idx, cnt in enumerate(counts):
                        others = [c for j, c in enumerate(counts) if j != opt_idx]
                        avg_other = sum(others) / float(len(others)) if others else 0
                        if cnt > 30 and cnt > (avg_other * 1.6 + 10):
                            valid_opts.append(opt_idx + 1)

                    if len(valid_opts) == 1:
                        best_opt = valid_opts[0]
                        idx = valid_opts[0] - 1
                        cv2.rectangle(debug_img, (boxes[idx][0], boxes[idx][1]), (boxes[idx][2], boxes[idx][3]), (0, 255, 0), -1)
                    elif len(valid_opts) >= 2:
                        best_opt = "重複マーク"
                        for opt_num in valid_opts:
                            idx = opt_num - 1
                            cv2.rectangle(debug_img, (boxes[idx][0], boxes[idx][1]), (boxes[idx][2], boxes[idx][3]), (0, 0, 255), -1)
                    else:
                        best_opt = "未マーク/無効"

                    recognized_answers[q_key] = best_opt
                return debug_img

            bin_q1, color_q1 = warp_block(g_q1, 950, 1600)
            img_q1_debug = parse_q_range(bin_q1, color_q1, 1)

            if max_q >= 26:
                bin_q2, color_q2 = warp_block(g_q2, 950, 1600)
                img_q2_debug = parse_q_range(bin_q2, color_q2, 26)

        except Exception as e:
            logs.append(f"解析エラー: {e}")
    else:
        logs.append(f"エラー: 四角マーカーが12個正しく検出されませんでした。")

    for m in markers:
        cv2.circle(debug_main, (int(m[0]), int(m[1])), 25, (0, 255, 0), -1)

    return recognized_test_id, recognized_exam_id, recognized_answers, \
           cv2.cvtColor(debug_main, cv2.COLOR_BGR2RGB), \
           cv2.cvtColor(img_exam_debug, cv2.COLOR_BGR2RGB) if img_exam_debug is not None else None, \
           cv2.cvtColor(img_q1_debug, cv2.COLOR_BGR2RGB) if img_q1_debug is not None else None, \
           cv2.cvtColor(img_q2_debug, cv2.COLOR_BGR2RGB) if img_q2_debug is not None else None, \
           logs

# --- 設定・パスワードロード ---
global_config = load_global_config()
MAX_Q = global_config.get("max_questions", 50)
NUM_OPTS = global_config.get("num_options", 9)
teacher_password = load_password()

if "step" not in st.session_state: st.session_state.step = 1

mode = st.sidebar.radio("モード選択", ["📱 生徒用（解答・採点・統計）", "教員用（正解・配点・大問・特殊ルール設定）", "教員用（集計・Excel出力）"])

# ---------------------------------------------------------
# 教員用モード パスワード認証
# ---------------------------------------------------------
if mode in ["教員用（正解・配点・大問・特殊ルール設定）", "教員用（集計・Excel出力）"]:
    st.subheader("🔒 教員用認証")
    input_pass = st.text_input("教員用パスワードを入力してください", type="password")
    if input_pass != teacher_password:
        if input_pass:
            st.error("パスワードが正しくありません。")
        else:
            st.info("この機能を利用するには教員用パスワードの入力が必要です。")
        st.stop()
    st.success("認証に成功しました。")
    st.markdown("---")

if mode == "📱 生徒用（解答・採点・統計）":
    st.markdown("""
    <div class="hero-header">
        <h1>✨ 共通テスト演習 自動採点・統計ポータル</h1>
        <p>マークシートの撮影・自動採点や、テストごとの平均点・正答率を確認できます。</p>
    </div>
    """, unsafe_allow_html=True)

    tab_student_1, tab_student_2 = st.tabs(["📷 マークシートの撮影・採点", "📊 テスト統計・正答率の確認"])

    with tab_student_1:
        if st.session_state.step == 1:
            st.subheader("STEP 1: マークシートの撮影・選択")
            st.info("📱 下の枠をタップして「撮影」を選ぶと、カメラが起動します。")
            
            uploaded_file = st.file_uploader("撮影するか、マークシート画像を選択してください", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                st.image(uploaded_file, caption="選択した画像", width=450)
                if st.button("この画像を解析して進む ▶", type="primary", use_container_width=True):
                    with st.spinner("マークを自動読み取り中..."):
                        test_id, exam_id, answers, debug_main, img_exam, img_q1, img_q2, logs = analyze_mark_sheet(uploaded_file, MAX_Q, NUM_OPTS)
                        st.session_state.uploaded_image = uploaded_file
                        st.session_state.recognized_test_id = test_id
                        st.session_state.recognized_exam_id = exam_id
                        st.session_state.recognized_answers = answers
                        st.session_state.debug_main = debug_main
                        st.session_state.img_exam = img_exam
                        st.session_state.img_q1 = img_q1
                        st.session_state.img_q2 = img_q2
                        st.session_state.logs = logs
                        st.session_state.step = 2
                        st.rerun()

        elif st.session_state.step == 2:
            st.subheader("STEP 2: 読み取り結果の確認・修正")
            
            col_top_btn1, col_top_btn2 = st.columns([1, 2])
            with col_top_btn1:
                if st.button("🔄 再撮影（撮り直す）", type="secondary", use_container_width=True):
                    st.session_state.step = 1
                    st.rerun()

            with st.expander("🔍 デバッグ画像（検出範囲と読み取り位置）", expanded=True):
                for l in st.session_state.get("logs", []): st.info(l)
                c1, c2, c3 = st.columns([1, 1.5, 1.5])
                with c1:
                    st.caption("【TEST ID & STUDENT ID】")
                    if st.session_state.img_exam is not None: st.image(st.session_state.img_exam, use_container_width=True)
                with c2:
                    st.caption("【問 1 〜 25】")
                    if st.session_state.img_q1 is not None: st.image(st.session_state.img_q1, use_container_width=True)
                with c3:
                    st.caption("【問 26 〜 50】")
                    if st.session_state.img_q2 is not None: st.image(st.session_state.img_q2, use_container_width=True)

            raw_test_id = st.session_state.recognized_test_id.zfill(2)
            raw_student_id = st.session_state.recognized_exam_id.zfill(4)
            
            c_id1, c_id2 = st.columns(2)
            with c_id1:
                test_id_val = st.text_input("TEST ID (2桁)", value=raw_test_id)
            with c_id2:
                student_id_val = st.text_input("STUDENT ID (年・組・番・号)", value=raw_student_id)

            t_cfg = load_test_config(test_id_val)
            current_test_max_q = t_cfg.get("max_questions", MAX_Q)

            st.markdown("---")
            options = list(range(1, NUM_OPTS + 1)) + ["未マーク/無効", "重複マーク"]
            final_answers = {}

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("##### ◀ 解答欄 (問1 〜 問25)")
                for i in range(1, min(26, current_test_max_q + 1)):
                    q_key = f'Q{i}'
                    curr = st.session_state.recognized_answers.get(q_key, "未マーク/無効")
                    idx = options.index(curr) if curr in options else len(options) - 2
                    q_c1, q_c2 = st.columns([1, 2])
                    q_c1.write(f"**問 {i}**")
                    final_answers[q_key] = q_c2.selectbox(f"問{i}", options=options, index=idx, key=f"sel_{q_key}", label_visibility="collapsed")
            with col_right:
                if current_test_max_q >= 26:
                    st.markdown(f"##### ▶ 解答欄 (問26 〜 問{current_test_max_q})")
                    for i in range(26, current_test_max_q + 1):
                        q_key = f'Q{i}'
                        curr = st.session_state.recognized_answers.get(q_key, "未マーク/無効")
                        idx = options.index(curr) if curr in options else len(options) - 2
                        q_c1, q_c2 = st.columns([1, 2])
                        q_c1.write(f"**問 {i}**")
                        final_answers[q_key] = q_c2.selectbox(f"問{i}", options=options, index=idx, key=f"sel_{q_key}", label_visibility="collapsed")

            st.markdown("---")
            col_bot_btn1, col_bot_btn2 = st.columns([1, 2])
            with col_bot_btn1:
                if st.button("🔄 再撮影（撮り直す）", key="btn_re_take", use_container_width=True):
                    st.session_state.step = 1
                    st.rerun()
            with col_bot_btn2:
                if st.button("この内容で確定して採点する ▶", type="primary", use_container_width=True):
                    st.session_state.final_test_id = str(test_id_val).zfill(2)
                    st.session_state.final_student_id = str(student_id_val).zfill(4)
                    st.session_state.final_answers = final_answers

                    t_id = st.session_state.final_test_id
                    s_id = st.session_state.final_student_id
                    
                    t_cfg_calc = load_test_config(t_id)
                    t_max_q_calc = t_cfg_calc.get("max_questions", MAX_Q)

                    ans_key_calc = load_answer_key(t_id, t_max_q_calc)
                    pts_key_calc = load_points_key(t_id, t_max_q_calc)
                    spec_rules_calc = load_special_rules(t_id)
                    sections_config = load_sections_config(t_id)

                    spec_q_map = {}
                    for r in spec_rules_calc:
                        r_type = r.get("type", "range")
                        if r_type == "range":
                            for q_num in range(r.get("start", 1), r.get("end", 1) + 1):
                                spec_q_map[f'Q{q_num}'] = r
                        elif r_type == "unordered_combo":
                            for q_num in range(r.get("unordered_start", 1), r.get("unordered_end", 1) + 1):
                                spec_q_map[f'Q{q_num}'] = r
                            if r.get("extra_start", 0) > 0 and r.get("extra_end", 0) > 0:
                                for q_num in range(r.get("extra_start", 1), r.get("extra_end", 1) + 1):
                                    spec_q_map[f'Q{q_num}'] = r

                    rule_res = []
                    for r in spec_rules_calc:
                        r_type = r.get("type", "range")
                        if r_type == "range":
                            all_c = True
                            for q_num in range(r["start"], r["end"] + 1):
                                q = f'Q{q_num}'
                                if final_answers.get(q) != ans_key_calc.get(q):
                                    all_c = False
                                    break
                            rule_res.append({"rule": r, "is_achieved": all_c})
                        elif r_type == "unordered_combo":
                            u_start, u_end = r.get("unordered_start", 1), r.get("unordered_end", 1)
                            e_start, e_end = r.get("extra_start", 0), r.get("extra_end", 0)
                            
                            u_user = [final_answers.get(f'Q{i}') for i in range(u_start, u_end + 1)]
                            u_correct = [ans_key_calc.get(f'Q{i}') for i in range(u_start, u_end + 1)]
                            
                            u_ok = (sorted([str(x) for x in u_user]) == sorted([str(x) for x in u_correct]))
                            
                            e_ok = True
                            if e_start > 0 and e_end > 0:
                                for i in range(e_start, e_end + 1):
                                    if final_answers.get(f'Q{i}') != ans_key_calc.get(f'Q{i}'):
                                        e_ok = False
                                        break
                            
                            all_c = u_ok and e_ok
                            rule_res.append({"rule": r, "is_achieved": all_c})

                    t_score = 0
                    res_list = []
                    question_scores = {}
                    row_d = {
                        '日時': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'TEST ID': t_id,
                        'STUDENT ID': s_id
                    }

                    for i in range(1, t_max_q_calc + 1):
                        q = f'Q{i}'
                        u_a = final_answers.get(q)
                        c_a = ans_key_calc.get(q)
                        pts = pts_key_calc.get(q, 2)
                        
                        is_c = (u_a == c_a)
                        sc = 0 if q in spec_q_map else (pts if is_c else 0)
                        status = "〇" if is_c else "✕"

                        t_score += sc
                        question_scores[q] = sc
                        res_list.append({'問題': q, 'あなたのマーク': u_a, '正解': c_a, '判定': status, '獲得点数': int(sc)})
                        row_d[q] = u_a
                        row_d[f'{q}_正誤'] = 1 if is_c else 0

                    for rr in rule_res:
                        if rr["is_achieved"]:
                            t_score += rr["rule"]["points"]

                    row_d['合計点'] = int(t_score)

                    section_summary = []
                    if sections_config:
                        for sec in sections_config:
                            s_name = sec.get("name", "大問")
                            s_start = sec.get("start", 1)
                            s_end = sec.get("end", 1)
                            
                            sec_score = 0
                            sec_max_pts = 0
                            
                            for q_num in range(s_start, s_end + 1):
                                q_k = f'Q{q_num}'
                                sec_score += question_scores.get(q_k, 0)
                                if q_k not in spec_q_map:
                                    sec_max_pts += pts_key_calc.get(q_k, 2)
                            
                            for rr in rule_res:
                                r = rr["rule"]
                                r_type = r.get("type", "range")
                                if r_type == "range":
                                    r_s, r_e = r["start"], r["end"]
                                elif r_type == "unordered_combo":
                                    r_s = r.get("unordered_start", 1)
                                    r_e = max(r.get("unordered_end", 1), r.get("extra_end", 0))
                                
                                if r_s >= s_start and r_e <= s_end:
                                    sec_max_pts += r["points"]
                                    if rr["is_achieved"]:
                                        sec_score += r["points"]

                            sec_rate = round((sec_score / sec_max_pts * 100), 1) if sec_max_pts > 0 else 0.0

                            section_summary.append({
                                "name": s_name,
                                "range": f"問{s_start}〜問{s_end}",
                                "score": sec_score,
                                "max_score": sec_max_pts,
                                "rate": sec_rate
                            })

                    save_test_result_to_csv(row_d)

                    st.session_state.result_total_score = int(t_score)
                    st.session_state.results_list = res_list
                    st.session_state.section_summary = section_summary
                    st.session_state.step = 3
                    st.rerun()

        elif st.session_state.step == 3:
            st.subheader("STEP 3: 採点結果")
            test_id = st.session_state.final_test_id
            student_id = st.session_state.final_student_id
            total_score = st.session_state.get("result_total_score", 0)
            results_list = st.session_state.get("results_list", [])
            section_summary = st.session_state.get("section_summary", [])
            
            t_cfg = load_test_config(test_id)
            target_total = t_cfg.get("total_score", 100)

            st.balloons()
            
            st.metric(label=f"【TEST ID: {test_id} / STUDENT ID: {student_id}】 さんの総合点", value=f"{total_score} 点 / {target_total} 点")

            if section_summary:
                st.markdown("##### 📚 大問別 得点内訳（小計）")
                cols = st.columns(min(len(section_summary), 4))
                for idx, sec in enumerate(section_summary):
                    col_idx = idx % min(len(section_summary), 4)
                    with cols[col_idx]:
                        st.markdown(f"""
                        <div class="section-score-card">
                            <div class="section-score-title">{sec['name']} ({sec['range']})</div>
                            <div class="section-score-val">{sec['score']} / {sec['max_score']} 点 ({sec['rate']}%)</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                col_chart1, col_chart2 = st.columns([1.2, 1])
                
                with col_chart1:
                    st.markdown("##### 📈 大問別 得点率バランス（レーダーチャート）")
                    categories = [sec['name'] for sec in section_summary]
                    rates = [sec['rate'] for sec in section_summary]
                    
                    categories_closed = categories + [categories[0]]
                    rates_closed = rates + [rates[0]]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=rates_closed,
                        theta=categories_closed,
                        fill='toself',
                        fillcolor='rgba(59, 130, 246, 0.25)',
                        line=dict(color='#1e3a8a', width=3),
                        marker=dict(size=7, color='#3b82f6'),
                        name='得点率'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100],
                                ticksuffix="%",
                                gridcolor="#cbd5e1"
                            ),
                            angularaxis=dict(
                                gridcolor="#cbd5e1"
                            )
                        ),
                        showlegend=False,
                        margin=dict(l=50, r=50, t=30, b=30),
                        height=360,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    # staticPlot: True でタッチ・クリック操作を完全無効化（表示崩れを防止）
                    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})
                    
                with col_chart2:
                    st.markdown("##### 📝 大問別 得点率一覧")
                    rate_df = pd.DataFrame([
                        {"大問": sec['name'], "範囲": sec['range'], "得点": f"{sec['score']} / {sec['max_score']} 点", "得点率": f"{sec['rate']}%"}
                        for sec in section_summary
                    ])
                    st.dataframe(rate_df, use_container_width=True, height=280)

            st.markdown("---")
            st.write("**【設問別 正誤一覧】**")
            st.dataframe(pd.DataFrame(results_list), height=350, use_container_width=True)

            st.markdown("---")
            if st.button("次の解答を採点する ▶", type="primary", use_container_width=True):
                st.session_state.step = 1
                st.rerun()

    with tab_student_2:
        st.subheader("📊 テスト別 統計・正答率の確認")
        st.caption("※ マークシートを撮影しなくても、TEST IDを選ぶだけで全体の平均点や問題ごとの正答率を確認できます。")
        
        df_all = safe_read_csv(DATA_FILE)
        unique_test_ids = get_clean_unique_test_ids(df_all)
        
        if unique_test_ids:
            df_all['TEST ID'] = df_all['TEST ID'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.zfill(2)
            selected_stat_tid = st.selectbox("📌 確認したい TEST ID を選択してください", unique_test_ids, key="student_stat_tid")
            
            df_test = df_all[df_all['TEST ID'] == selected_stat_tid]
            
            if not df_test.empty:
                stat_t_cfg = load_test_config(selected_stat_tid)
                stat_max_q = stat_t_cfg.get("max_questions", MAX_Q)
                stat_answer_key = load_answer_key(selected_stat_tid, stat_max_q)
                stat_total_score = stat_t_cfg.get("total_score", 100)

                df_test['合計点'] = pd.to_numeric(df_test['合計点'], errors='coerce')

                s1, s2, s3 = st.columns(3)
                s1.metric("解答提出者数", f"{len(df_test)} 人")
                s2.metric("全体平均点", f"{round(df_test['合計点'].mean(), 1)} 点 / {stat_total_score}点")
                s3.metric("最高点", f"{int(df_test['合計点'].max())} 点")
                
                st.markdown("---")
                st.write("**【各問の正答率一覧】**")
                
                stat_table = []
                for i in range(1, stat_max_q + 1):
                    q = f'Q{i}'
                    correct_val = stat_answer_key.get(q, "-")
                    
                    corr_col = f'{q}_正誤'
                    if corr_col in df_test.columns:
                        acc = pd.to_numeric(df_test[corr_col], errors='coerce').mean() * 100
                    else:
                        acc = 0.0

                    stat_table.append({
                        '問題': q,
                        '正解': correct_val,
                        '全体正答率': f"{acc:.1f}%"
                    })

                st.dataframe(pd.DataFrame(stat_table), height=400, use_container_width=True)
            else:
                st.info("選択された TEST ID のデータはまだありません。")
        else:
            st.warning("まだ採点データが登録されていません。")

elif mode == "教員用（正解・配点・大問・特殊ルール設定）":
    st.header("■ 教員用 各種設定ダッシュボード")
    tab0, tab1 = st.tabs(["⚙️ システム・パスワード設定", "📝 TEST ID別 正答・配点・大問管理"])
    
    with tab0:
        st.subheader("⚙️ グローバル上限設定")
        c1, c2 = st.columns(2)
        new_max_q = c1.number_input("システム全体の最大問題数", 1, 50, MAX_Q)
        new_opts = c2.number_input("選択肢の数", 2, 9, NUM_OPTS)
        if st.button("設定保存"):
            save_json_file(CONFIG_FILE, {"max_questions": int(new_max_q), "num_options": int(new_opts)})
            st.success("基本設定を更新しました！")
            st.rerun()
            
        st.markdown("---")
        st.subheader("🔑 教員用パスワードの変更")
        curr_pass_check = st.text_input("現在のパスワードを入力してください", type="password", key="pass_curr")
        new_pass1 = st.text_input("新しいパスワード", type="password", key="pass_new1")
        new_pass2 = st.text_input("新しいパスワード（確認のため再入力）", type="password", key="pass_new2")
        
        if st.button("パスワードを変更する", type="primary"):
            if curr_pass_check != teacher_password:
                st.error("現在のパスワードが正しくありません。")
            elif not new_pass1:
                st.error("新しいパスワードを入力してください。")
            elif new_pass1 != new_pass2:
                st.error("新しいパスワードが一致しません。確認してください。")
            else:
                save_password(new_pass1)
                st.success("教員用パスワードを正常に変更しました！")
                st.rerun()

    with tab1:
        st.subheader("📝 TEST ID 別 正答・配点・大問管理")
        
        all_ans_data = load_json_file(ANSWER_KEY_FILE, {})
        existing_test_ids = sorted(list(all_ans_data.keys()))
        if not existing_test_ids:
            existing_test_ids = ["01"]

        col_tid1, col_tid2 = st.columns([2, 2])
        with col_tid1:
            selected_t_id = st.selectbox("📌 編集する TEST ID を選択", existing_test_ids)
        with col_tid2:
            new_t_input = st.text_input("新規 TEST ID を直接入力（例: 02）", value="")
            if new_t_input:
                selected_t_id = str(new_t_input).zfill(2)

        target_test_id = str(selected_t_id).zfill(2)
        t_config = load_test_config(target_test_id)
        t_max_q = int(t_config.get("max_questions", 50))
        t_total_score = int(t_config.get("total_score", 100))

        if "prev_target_test_id" not in st.session_state or st.session_state.prev_target_test_id != target_test_id:
            st.session_state.prev_target_test_id = target_test_id
            loaded_ans = load_answer_key(target_test_id, t_max_q)
            loaded_pts = load_points_key(target_test_id, t_max_q)
            for i in range(1, 51):
                q_k = f'Q{i}'
                st.session_state[f"ans_{target_test_id}_{q_k}"] = loaded_ans.get(q_k, 1)
                st.session_state[f"pts_{target_test_id}_{q_k}"] = loaded_pts.get(q_k, 2)

        st.info(f"現在 **TEST ID: {target_test_id}** の正答・配点を設定・編集しています。")
        st.markdown("---")

        st.markdown("##### ⚙️ 1. テスト基本設定（問題数・合計点）")
        col_cfg1, col_cfg2, col_cfg3 = st.columns([2, 2, 1])
        new_max_q_val = col_cfg1.number_input("問題数 (1〜50)", min_value=1, max_value=50, value=t_max_q, step=1)
        new_total_score_val = col_cfg2.number_input("合計点 (目標満点)", min_value=1, max_value=500, value=t_total_score, step=1)
        
        if col_cfg3.button("基本設定を保存"):
            save_test_config(target_test_id, new_max_q_val, new_total_score_val)
            st.success(f"TEST ID [{target_test_id}] の基本設定を保存しました！（全 {new_max_q_val} 問 / {new_total_score_val} 点満点）")
            st.rerun()

        st.markdown("---")
        st.markdown("##### 📚 2. 大問の区切り（セクション）設定")
        st.caption("※「第1問: 問1〜6」「第2問: 問7〜14」のように区切ると、生徒の採点結果画面に大問別の小計とレーダーチャートが表示されます。")
        
        existing_sections = load_sections_config(target_test_id)
        
        col_sec_num, col_sec_btn = st.columns([2, 1])
        num_sections = col_sec_num.number_input("大問の数", min_value=1, max_value=10, value=max(len(existing_sections), 1), step=1, key="num_sec_input")
        
        new_sections_list = []
        for sec_idx in range(int(num_sections)):
            default_sec = existing_sections[sec_idx] if sec_idx < len(existing_sections) else {}
            
            c_name, c_s, c_e = st.columns([2, 1.5, 1.5])
            s_name = c_name.text_input(f"大問{sec_idx+1} 名称", value=default_sec.get("name", f"第{sec_idx+1}問"), key=f"sec_name_{target_test_id}_{sec_idx}")
            
            def_start = min(int(default_sec.get("start", 1)), t_max_q)
            def_end = min(int(default_sec.get("end", min(5, t_max_q))), t_max_q)
            
            s_start = c_s.number_input(f"開始問", min_value=1, max_value=t_max_q, value=max(1, def_start), step=1, key=f"sec_start_{target_test_id}_{sec_idx}")
            s_end = c_e.number_input(f"終了問", min_value=1, max_value=t_max_q, value=max(1, def_end), step=1, key=f"sec_end_{target_test_id}_{sec_idx}")
            
            new_sections_list.append({"name": s_name, "start": int(s_start), "end": int(s_end)})

        if st.button("大問区切り設定を保存", type="secondary"):
            save_sections_config(target_test_id, new_sections_list)
            st.success(f"TEST ID [{target_test_id}] の大問設定を保存しました！")
            st.rerun()

        st.markdown("---")
        st.markdown("##### 📝 3. 正答・配点の設定")

        current_answers = load_answer_key(target_test_id, t_max_q)
        current_points = load_points_key(target_test_id, t_max_q)
        
        current_rules = load_special_rules(target_test_id)
        
        special_q_set = set()
        for r in current_rules:
            r_type = r.get("type", "range")
            if r_type == "range":
                for q_num in range(r.get("start", 1), r.get("end", 1) + 1):
                    special_q_set.add(q_num)
            elif r_type == "unordered_combo":
                for q_num in range(r.get("unordered_start", 1), r.get("unordered_end", 1) + 1):
                    special_q_set.add(q_num)
                if r.get("extra_start", 0) > 0 and r.get("extra_end", 0) > 0:
                    for q_num in range(r.get("extra_start", 1), r.get("extra_end", 1) + 1):
                        special_q_set.add(q_num)

        st.markdown("###### 💡 配点の一括変更")
        
        col_bp1, col_bp2 = st.columns([2, 1])
        bulk_val = col_bp1.number_input("全問の配点を一括変更（点）", min_value=0, max_value=50, value=2, step=1, key="bulk_all_pts")
        if col_bp2.button("全問一括適用", type="secondary"):
            new_bulk_pts = {f'Q{i}': (0 if i in special_q_set else int(bulk_val)) for i in range(1, t_max_q + 1)}
            save_points_key(target_test_id, new_bulk_pts)
            
            for i in range(1, t_max_q + 1):
                k = f"pts_{target_test_id}_Q{i}"
                st.session_state[k] = 0 if i in special_q_set else int(bulk_val)
            
            st.success(f"TEST ID [{target_test_id}] の全問配点を {int(bulk_val)} 点に変更しました！")
            st.rerun()

        st.markdown("###### 🎯 範囲を指定して配点を一括変更")
        col_r1, col_r2, col_r3, col_r4 = st.columns([1.5, 1.5, 1.5, 1.5])
        start_q = col_r1.number_input("開始問", min_value=1, max_value=t_max_q, value=1, step=1)
        end_q = col_r2.number_input("終了問", min_value=1, max_value=t_max_q, value=min(25, t_max_q), step=1)
        range_val = col_r3.number_input("配点（点）", min_value=0, max_value=50, value=2, step=1, key="bulk_range_pts")
        
        if col_r4.button("範囲一括適用", type="secondary"):
            if start_q > end_q:
                st.error("開始問は終了問以下に設定してください。")
            else:
                curr_pts = load_points_key(target_test_id, t_max_q)
                for i in range(int(start_q), int(end_q) + 1):
                    q_k = f'Q{i}'
                    if i in special_q_set:
                        curr_pts[q_k] = 0
                        st.session_state[f"pts_{target_test_id}_{q_k}"] = 0
                    else:
                        curr_pts[q_k] = int(range_val)
                        st.session_state[f"pts_{target_test_id}_{q_k}"] = int(range_val)
                save_points_key(target_test_id, curr_pts)
                st.success(f"問{int(start_q)} 〜 問{int(end_q)} の配点を {int(range_val)} 点に変更しました！")
                st.rerun()

        st.markdown("---")
        st.markdown("##### 🎯 4. 特殊配点（完答ルール）の設定")
        
        rule_type_sel = st.radio(
            "追加する完答ルールのタイプを選択してください",
            ["通常完答 (例: 問18〜問21がすべて正解で4点)", "順不同＋指定問 完答 (例: 問37・38が順不同＋問39も正解で5点)"],
            key="rule_type_radio"
        )

        if "通常完答" in rule_type_sel:
            col_sr1, col_sr2, col_sr3, col_sr4 = st.columns([1.5, 1.5, 1.5, 1.5])
            s_q = col_sr1.number_input("完答 開始問", min_value=1, max_value=t_max_q, value=min(18, t_max_q), step=1, key="sr_start_q")
            e_q = col_sr2.number_input("完答 終了問", min_value=1, max_value=t_max_q, value=min(21, t_max_q), step=1, key="sr_end_q")
            r_pts = col_sr3.number_input("完答時 得点", min_value=1, max_value=50, value=4, step=1, key="sr_pts")

            if col_sr4.button("通常ルールを追加", type="secondary"):
                if s_q > e_q:
                    st.error("開始問は終了問以下に設定してください。")
                else:
                    new_rule = {"type": "range", "start": int(s_q), "end": int(e_q), "points": int(r_pts)}
                    if new_rule not in current_rules:
                        current_rules.append(new_rule)
                        save_special_rules(target_test_id, current_rules)
                        st.success(f"問{s_q}〜問{e_q}（完答 {r_pts}点）のルールを追加しました！")
                        st.rerun()
        else:
            st.caption("※ 順不同で一致すればOKな問（例: 37〜38）と、同時正解が必要な通常の問（例: 39〜39）の組み合わせを設定します。")
            c_u1, c_u2, c_e1, c_e2, c_p = st.columns([1.2, 1.2, 1.2, 1.2, 1.2])
            
            us_q = c_u1.number_input("順不同 開始問", min_value=1, max_value=t_max_q, value=min(37, t_max_q), step=1, key="sr_us_q")
            ue_q = c_u2.number_input("順不同 終了問", min_value=1, max_value=t_max_q, value=min(38, t_max_q), step=1, key="sr_ue_q")
            es_q = c_e1.number_input("追加問 開始問", min_value=0, max_value=t_max_q, value=min(39, t_max_q), step=1, key="sr_es_q")
            ee_q = c_e2.number_input("追加問 終了問", min_value=0, max_value=t_max_q, value=min(39, t_max_q), step=1, key="sr_ee_q")
            r_pts = c_p.number_input("完答時 得点", min_value=1, max_value=50, value=5, step=1, key="sr_pts_u")

            if st.button("順不同ルールを追加", type="secondary"):
                if us_q > ue_q:
                    st.error("順不同の開始問は終了問以下に設定してください。")
                elif es_q > 0 and ee_q > 0 and es_q > ee_q:
                    st.error("追加問の開始問は終了問以下に設定してください。")
                else:
                    new_rule = {
                        "type": "unordered_combo",
                        "unordered_start": int(us_q),
                        "unordered_end": int(ue_q),
                        "extra_start": int(es_q),
                        "extra_end": int(ee_q),
                        "points": int(r_pts)
                    }
                    if new_rule not in current_rules:
                        current_rules.append(new_rule)
                        save_special_rules(target_test_id, current_rules)
                        st.success(f"問{us_q}〜問{ue_q}（順不同）＋ 問{es_q}〜問{ee_q}（完答 {r_pts}点）を追加しました！")
                        st.rerun()

        if current_rules:
            st.write("**登録済みの特殊ルール一覧:**")
            for idx, r in enumerate(current_rules):
                r_type = r.get("type", "range")
                col_list1, col_list2 = st.columns([4, 1])
                if r_type == "range":
                    col_list1.write(f"・ **問 {r['start']} 〜 問 {r['end']}** ： 全問正解で **{r['points']} 点**")
                elif r_type == "unordered_combo":
                    e_str = f" ＋ **問 {r['extra_start']} 〜 問 {r['extra_end']}**" if r.get("extra_start", 0) > 0 else ""
                    col_list1.write(f"・ **問 {r['unordered_start']} 〜 問 {r['unordered_end']} (順不同)**{e_str} ： 全正解で **{r['points']} 点**")

                if col_list2.button("削除", key=f"del_rule_{idx}"):
                    current_rules.pop(idx)
                    save_special_rules(target_test_id, current_rules)
                    st.success("ルールを削除しました。")
                    st.rerun()
        else:
            st.caption("現在設定されている特殊ルールはありません。")

        st.markdown("---")
        st.markdown(f"###### ✍️ 個別配点・正答の入力 (問1 〜 問{t_max_q})")
        
        options = list(range(1, NUM_OPTS + 1))
        new_answers = {}
        new_points = {}

        col_a_l, col_a_r = st.columns(2)
        with col_a_l:
            st.markdown("###### ◀ 問 1 〜 25")
            for i in range(1, min(26, t_max_q + 1)):
                q_k = f'Q{i}'
                idx = options.index(current_answers.get(q_k, 1)) if current_answers.get(q_k, 1) in options else 0
                
                q_c1, q_c2, q_c3 = st.columns([1, 2, 2])
                q_c1.write(f"**問 {i}**")
                new_answers[q_k] = q_c2.selectbox(f"正答_問{i}", options=options, index=idx, key=f"ans_{target_test_id}_{q_k}", label_visibility="collapsed")
                
                if i in special_q_set:
                    q_c3.caption("🔒 完答ルール対象")
                    new_points[q_k] = 0
                else:
                    init_pt = int(st.session_state.get(f"pts_{target_test_id}_{q_k}", current_points.get(q_k, 2)))
                    new_points[q_k] = q_c3.number_input(f"配点_問{i}", min_value=0, max_value=50, value=init_pt, step=1, key=f"pts_{target_test_id}_{q_k}", label_visibility="collapsed")

        with col_a_r:
            if t_max_q >= 26:
                st.markdown(f"###### ▶ 問 26 〜 {t_max_q}")
                for i in range(26, t_max_q + 1):
                    q_k = f'Q{i}'
                    idx = options.index(current_answers.get(q_k, 1)) if current_answers.get(q_k, 1) in options else 0
                    
                    q_c1, q_c2, q_c3 = st.columns([1, 2, 2])
                    q_c1.write(f"**問 {i}**")
                    new_answers[q_k] = q_c2.selectbox(f"正答_問{i}", options=options, index=idx, key=f"ans_{target_test_id}_{q_k}", label_visibility="collapsed")
                    
                    if i in special_q_set:
                        q_c3.caption("🔒 完答ルール対象")
                        new_points[q_k] = 0
                    else:
                        init_pt = int(st.session_state.get(f"pts_{target_test_id}_{q_k}", current_points.get(q_k, 2)))
                        new_points[q_k] = q_c3.number_input(f"配点_問{i}", min_value=0, max_value=50, value=init_pt, step=1, key=f"pts_{target_test_id}_{q_k}", label_visibility="collapsed")

        calculated_total = sum(new_points.values()) + sum(r["points"] for r in current_rules)

        st.markdown("---")
        if calculated_total != t_total_score:
            st.warning(f"⚠️ **配点合計の不一致**: 現在設定されている配点合計は **{calculated_total} 点** です。（基本設定の目標満点: **{t_total_score} 点**）")
        else:
            st.success(f"✅ **配点合計チェックOK**: 設定されている配点合計は **{calculated_total} 点**（目標満点: **{t_total_score} 点**）です。")

        if st.button(f"TEST ID [{target_test_id}] の設定（正答・配点・大問）を一括保存する", type="primary", use_container_width=True):
            if calculated_total != t_total_score:
                st.error(f"❌ 配点の合計（{calculated_total} 点）が基本設定の目標満点（{t_total_score} 点）と一致していません。配点を修正してから保存してください。")
            else:
                save_answer_key(target_test_id, new_answers)
                save_points_key(target_test_id, new_points)
                save_test_config(target_test_id, t_max_q, t_total_score)
                save_sections_config(target_test_id, new_sections_list)
                st.success(f"TEST ID [{target_test_id}] の設定（正答・配点・問題数・大問）を保存しました！")
                st.rerun()

elif mode == "教員用（集計・Excel出力）":
    st.header("■ 教員用 集計ダッシュボード")
    df = safe_read_csv(DATA_FILE)
    unique_test_ids = get_clean_unique_test_ids(df)
    
    if unique_test_ids:
        df['TEST ID'] = df['TEST ID'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.zfill(2)
        if 'STUDENT ID' in df.columns:
            df['STUDENT ID'] = df['STUDENT ID'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.zfill(4)
        
        selected_test_id = st.selectbox("📌 集計対象の TEST ID を選択してください", unique_test_ids)
        
        df_display = df[df['TEST ID'] == selected_test_id].copy()

        if 'STUDENT ID' in df_display.columns:
            df_display = df_display.sort_values(by='STUDENT ID').reset_index(drop=True)

        st.info(f"現在 **TEST ID: {selected_test_id}** の集計結果を表示しています。 (Student ID 順)")

        st.markdown("---")
        
        ref_t_cfg = load_test_config(selected_test_id)
        ref_total_score = ref_t_cfg.get("total_score", 100)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("提出枚数", f"{len(df_display)} 枚")
        
        df_display['合計点'] = pd.to_numeric(df_display['合計点'], errors='coerce')
        
        m2.metric("全平均点", f"{round(df_display['合計点'].mean(), 1)} 点" if not df_display.empty else "0 点")
        
        max_score = int(df_display['合計点'].max()) if not df_display.empty and not pd.isna(df_display['合計点'].max()) else 0
        m3.metric("最高点", f"{max_score} 点 / {ref_total_score} 点")
        m4.metric("最低点", f"{int(df_display['合計点'].min())} 点" if not df_display.empty and not pd.isna(df_display['合計点'].min()) else "0 点")

        if not df_display.empty and max_score > 0:
            top_students = df_display[df_display['合計点'] == max_score]['STUDENT ID'].tolist()
            top_str = ", ".join([str(s).zfill(4) for s in top_students])
            st.success(f"🏆 **最高点 ({max_score}点) を獲得した生徒 ID**: {top_str}")

        st.markdown("---")
        st.subheader("🏫 クラス別平均点")
        
        def get_class_name(sid):
            s = str(sid).zfill(4)
            if len(s) >= 4 and s[:2].isdigit():
                return f"{s[0]}年{s[1]}組"
            return f"{s[:2]}組"

        if not df_display.empty:
            df_display['クラス'] = df_display['STUDENT ID'].apply(get_class_name)
            class_summary = df_display.groupby('クラス')['合計点'].agg(
                受験人数='count',
                平均点='mean',
                最高点='max',
                最低点='min'
            ).reset_index()
            class_summary['平均点'] = class_summary['平均点'].round(1)
            st.dataframe(class_summary, use_container_width=True)

            st.markdown("---")
            st.subheader("⚠️ 得点率5割（50%）未満の生徒一覧")
            
            df_display['得点率(%)'] = (df_display['合計点'] / ref_total_score * 100).round(1)
            low_scorers = df_display[df_display['合計点'] < (ref_total_score * 0.5)]
            
            if not low_scorers.empty:
                st.error(f"⚠️ 得点率が5割を切っている生徒が **{len(low_scorers)} 名** います。")
                st.dataframe(
                    low_scorers[['STUDENT ID', 'クラス', '合計点', '得点率(%)']].reset_index(drop=True),
                    use_container_width=True
                )
            else:
                st.success("🎉 得点率が5割を切っている生徒はいません！")

            st.markdown("---")
            st.subheader("1. 個人別解答・得点一覧 (Student ID 順)")
            st.dataframe(df_display.drop(columns=['クラス', '得点率(%)'], errors='ignore'), use_container_width=True)

            st.markdown("---")
            st.subheader("🗑️ 不要データの削除（重複・誤入力データの削除）")
            with st.expander("⚠️ データ削除メニューを開く（クリックで展開）"):
                st.caption("※ 重複して送信されたデータや誤ったデータを削除できます。削除対象を選択して実行してください。")
                
                df_target = df[df['TEST ID'] == selected_test_id].copy()
                if 'STUDENT ID' in df_target.columns:
                    df_target = df_target.sort_values(by='STUDENT ID')
                
                delete_options = [
                    f"【ID: {row.get('STUDENT ID', '-') }】 日時: {row.get('日時', '-')} | 点数: {row.get('合計点', '-')}点 [元データIndex: {idx}]"
                    for idx, row in df_target.iterrows()
                ]
                
                selected_items_to_delete = st.multiselect("削除したい提出データを選択してください（複数選択可能）", delete_options)
                
                if st.button("❌ 選択したデータを完全に削除する", type="primary"):
                    if selected_items_to_delete:
                        indices_to_delete = [
                            int(item.split("[元データIndex: ")[1].split("]")[0])
                            for item in selected_items_to_delete
                        ]
                        
                        lock = FileLock(LOCK_FILE, timeout=10)
                        with lock:
                            df_updated = df.drop(index=indices_to_delete).reset_index(drop=True)
                            df_updated.to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
                        
                        st.success(f"{len(indices_to_delete)} 件のデータを正常に削除しました！")
                        st.rerun()
                    else:
                        st.warning("削除するデータを選択してください。")

            st.markdown("---")
            st.subheader("2. 問題別 正答率 (%)")
            ref_max_q = ref_t_cfg.get("max_questions", MAX_Q)
            answer_key = load_answer_key(selected_test_id, ref_max_q)
            q_cols = list(answer_key.keys())
            correct_cols = [f'{q}_正誤' for q in q_cols if f'{q}_正誤' in df_display.columns]
            
            if correct_cols:
                q_acc = df_display[correct_cols].apply(pd.to_numeric, errors='coerce').mean() * 100
                
                low_acc_questions = [f"{col.replace('_正誤', '')} ({acc:.1f}%)" for col, acc in q_acc.items() if acc < 70.0]
                if low_acc_questions:
                    st.error(f"⚠️ **正答率が7割（70%）を切っている注意問題 ({len(low_acc_questions)}問)**: {', '.join(low_acc_questions)}")
                else:
                    st.success("🎉 すべての問題で正答率7割以上を達成しています！")

                q_acc_df = q_acc.to_frame(name="正答率(%)").T
                q_acc_df.columns = [col.replace('_正誤', '') for col in correct_cols]
                
                def style_low_acc(val):
                    try:
                        v = float(val)
                        if v < 70.0:
                            return 'background-color: #ffcdd2; color: #b71c1c; font-weight: bold;'
                    except:
                        pass
                    return ''
                
                styler = q_acc_df.style
                if hasattr(styler, 'map'):
                    styled_df = styler.map(style_low_acc).format("{:.1f}%")
                else:
                    styled_df = styler.applymap(style_low_acc).format("{:.1f}%")

                st.dataframe(styled_df, use_container_width=True)

            st.markdown("---")
            excel_filename = f"集計結果_TEST_{selected_test_id}.xlsx"
            excel_path = f"/tmp/{excel_filename}" if os.path.exists("/tmp") else excel_filename
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df_display.drop(columns=['クラス', '得点率(%)'], errors='ignore').to_excel(writer, sheet_name='個人別結果', index=False)
                class_summary.to_excel(writer, sheet_name='クラス別平均点', index=False)
                if not low_scorers.empty:
                    low_scorers[['STUDENT ID', 'クラス', '合計点', '得点率(%)']].to_excel(writer, sheet_name='要フォロー生徒', index=False)
                if correct_cols:
                    q_acc_df.to_excel(writer, sheet_name='問題別正答率')

            with open(excel_path, "rb") as f:
                st.download_button(
                    label=f"📥 TEST ID [{selected_test_id}] の全集計データをExcelでダウンロード",
                    data=f,
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
    else:
        st.warning("まだ採点データが保存されていないか、データの形式が不整合です。")
