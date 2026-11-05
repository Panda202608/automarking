import streamlit as st
import pandas as pd
import json
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from datetime import datetime

# =========================================================
# セキュリティ設定
# =========================================================
TEACHER_PASSWORD = "admin123"  # 👈 教員用パスワード（必要に応じて変更してください）

# --- データ保存用ファイルパス ---
DATA_FILE = "test_results.csv"
ANSWER_KEY_FILE = "answer_key.json"
POINTS_KEY_FILE = "points_key.json"
RULES_KEY_FILE = "special_rules.json"
CONFIG_FILE = "config.json"

st.set_page_config(page_title="共通テスト採点・集計システム", layout="wide")
st.title("共通テスト演習 自動採点・集計システム")

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

def load_config():
    return load_json_file(CONFIG_FILE, {"max_questions": 50, "num_options": 9})

def load_answer_key(max_q):
    default_ans = {f'Q{i}': 1 for i in range(1, max_q + 1)}
    data = load_json_file(ANSWER_KEY_FILE, default_ans)
    return {f'Q{i}': data.get(f'Q{i}', 1) for i in range(1, max_q + 1)}

def load_points_key(max_q):
    default_pts = {f'Q{i}': 2.0 for i in range(1, max_q + 1)}
    data = load_json_file(POINTS_KEY_FILE, default_pts)
    return {f'Q{i}': float(data.get(f'Q{i}', 2.0)) for i in range(1, max_q + 1)}

def load_special_rules():
    return load_json_file(RULES_KEY_FILE, {"unordered": [], "all_correct": []})

# ---------------------------------------------------------
# 12個マーカー分離アルゴリズム
# ---------------------------------------------------------
def group_12_markers_robust(markers):
    markers_by_y = sorted(markers, key=lambda m: m[1])
    top_6 = sorted(markers_by_y[:6], key=lambda m: m[0])
    bot_6 = sorted(markers_by_y[6:], key=lambda m: m[0])
    
    group_exam = [top_6[0], top_6[1], bot_6[0], bot_6[1]]  # STUDENT ID
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

    recognized_exam_id = "0000"
    recognized_answers = {f'Q{i}': "未マーク/無効" for i in range(1, max_q + 1)}
    img_exam_debug, img_q1_debug, img_q2_debug = None, None, None

    if len(markers) == 12:
        try:
            g_exam, g_q1, g_q2 = group_12_markers_robust(markers)

            # 1. STUDENT ID (年・組・番・号) 4桁解析
            bin_exam, color_exam = warp_block(g_exam, 600, 1600)
            img_exam_debug = color_exam.copy()
            digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            parsed_digits = []
            
            col_w = 600.0 / 6.0
            row_h = (1600.0 - 90.0) / 10.0
            y_offset = 45.0

            for col_idx in range(1, 5):
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
                
            recognized_exam_id = "".join(parsed_digits)
            logs.append(f"STUDENT ID読み取り結果: {recognized_exam_id}")

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
                        cnt = cv2.countNonZero(roi)
                        counts.append(cnt)
                        boxes.append((x1, y1, x2, y2))
                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

                    best_opt = "未マーク/無効"
                    if counts:
                        max_c = max(counts)
                        avg_c = (sum(counts) - max_c) / float(len(counts) - 1) if len(counts) > 1 else 0
                        max_idx = counts.index(max_c)
                        
                        if max_c > 30 and max_c > (avg_c * 1.7 + 10):
                            best_opt = max_idx + 1
                            cv2.rectangle(debug_img, (boxes[max_idx][0], boxes[max_idx][1]), 
                                                     (boxes[max_idx][2], boxes[max_idx][3]), (0, 255, 0), -1)
                            
                    recognized_answers[q_key] = best_opt
                return debug_img

            # 問1〜25
            bin_q1, color_q1 = warp_block(g_q1, 950, 1600)
            img_q1_debug = parse_q_range(bin_q1, color_q1, 1)

            # 問26〜50
            if max_q >= 26:
                bin_q2, color_q2 = warp_block(g_q2, 950, 1600)
                img_q2_debug = parse_q_range(bin_q2, color_q2, 26)

        except Exception as e:
            logs.append(f"解析エラー: {e}")
    else:
        logs.append(f"エラー: 四角マーカーが12個正しく検出されませんでした。")

    for m in markers:
        cv2.circle(debug_main, (int(m[0]), int(m[1])), 25, (0, 255, 0), -1)

    return recognized_exam_id, recognized_answers, \
           cv2.cvtColor(debug_main, cv2.COLOR_BGR2RGB), \
           cv2.cvtColor(img_exam_debug, cv2.COLOR_BGR2RGB) if img_exam_debug is not None else None, \
           cv2.cvtColor(img_q1_debug, cv2.COLOR_BGR2RGB) if img_q1_debug is not None else None, \
           cv2.cvtColor(img_q2_debug, cv2.COLOR_BGR2RGB) if img_q2_debug is not None else None, \
           logs

# --- 設定ロード ---
config = load_config()
MAX_Q = config.get("max_questions", 50)
NUM_OPTS = config.get("num_options", 9)

if "step" not in st.session_state: st.session_state.step = 1

mode = st.sidebar.radio("モード選択", ["生徒用（マーク撮影・採点）", "教員用（正解・配点・特殊ルール設定）", "教員用（集計・Excel出力）"])

# ---------------------------------------------------------
# 教員用モード パスワード認証認証処理
# ---------------------------------------------------------
if mode in ["教員用（正解・配点・特殊ルール設定）", "教員用（集計・Excel出力）"]:
    st.subheader("🔒 教員用認証")
    input_pass = st.text_input("教員用パスワードを入力してください", type="password")
    if input_pass != TEACHER_PASSWORD:
        if input_pass:
            st.error("パスワードが正しくありません。")
        else:
            st.info("この機能を利用するには教員用パスワードの入力が必要です。")
        st.stop()
    st.success("認証に成功しました。")
    st.markdown("---")

if mode == "生徒用（マーク撮影・採点）":
    st.header(f"■ 生徒用 マークシート送信・採点 (全{MAX_Q}問 / 選択肢1〜{NUM_OPTS})")
    answer_key = load_answer_key(MAX_Q)
    points_key = load_points_key(MAX_Q)
    special_rules = load_special_rules()

    if st.session_state.step == 1:
        uploaded_file = st.file_uploader("解答用紙全体が写るように撮影してアップロードしてください", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="撮影画像", width=450)
            if st.button("画像を解析して確認画面へ進む", type="primary"):
                with st.spinner("マークを自動読み取り中..."):
                    exam_id, answers, debug_main, img_exam, img_q1, img_q2, logs = analyze_mark_sheet(uploaded_file, MAX_Q, NUM_OPTS)
                    st.session_state.uploaded_image = uploaded_file
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
        with st.expander("🔍 デバッグ画像（検出範囲と読み取り位置）", expanded=True):
            for l in st.session_state.get("logs", []): st.info(l)
            c1, c2, c3 = st.columns([1, 1.5, 1.5])
            with c1:
                st.caption("【STUDENT ID】")
                if st.session_state.img_exam is not None: st.image(st.session_state.img_exam, use_container_width=True)
            with c2:
                st.caption("【問 1 〜 25】")
                if st.session_state.img_q1 is not None: st.image(st.session_state.img_q1, use_container_width=True)
            with c3:
                st.caption("【問 26 〜 50】")
                if st.session_state.img_q2 is not None: st.image(st.session_state.img_q2, use_container_width=True)

        raw_id = st.session_state.recognized_exam_id.zfill(4)
        student_id = st.text_input("STUDENT ID (年・組・番・号)", value=raw_id)

        st.markdown("---")
        options = list(range(1, NUM_OPTS + 1)) + ["未マーク/無効"]
        final_answers = {}

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("##### ◀ 解答欄 (問1 〜 問25)")
            for i in range(1, 26):
                q_key = f'Q{i}'
                curr = st.session_state.recognized_answers.get(q_key, "未マーク/無効")
                idx = options.index(curr) if curr in options else len(options) - 1
                q_c1, q_c2 = st.columns([1, 2])
                q_c1.write(f"**問 {i}**")
                final_answers[q_key] = q_c2.selectbox(f"問{i}", options=options, index=idx, key=f"sel_{q_key}", label_visibility="collapsed")
        with col_right:
            st.markdown(f"##### ▶ 解答欄 (問26 〜 問{MAX_Q})")
            for i in range(26, MAX_Q + 1):
                q_key = f'Q{i}'
                curr = st.session_state.recognized_answers.get(q_key, "未マーク/無効")
                idx = options.index(curr) if curr in options else len(options) - 1
                q_c1, q_c2 = st.columns([1, 2])
                q_c1.write(f"**問 {i}**")
                final_answers[q_key] = q_c2.selectbox(f"問{i}", options=options, index=idx, key=f"sel_{q_key}", label_visibility="collapsed")

        st.markdown("---")
        if st.button("この内容で確定して採点する ▶", type="primary", use_container_width=True):
            st.session_state.final_student_id = student_id
            st.session_state.final_answers = final_answers
            st.session_state.step = 3
            st.rerun()

    elif st.session_state.step == 3:
        st.subheader("STEP 3: 採点結果")
        user_ans = st.session_state.final_answers
        student_id = st.session_state.final_student_id
        
        total_score = 0
        results_list = []
        row_data = {
            '日時': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'STUDENT ID': student_id
        }

        for i in range(1, MAX_Q + 1):
            q = f'Q{i}'
            u_a = user_ans.get(q)
            c_a = answer_key.get(q)
            pts = points_key.get(q, 2.0)
            is_c = (u_a == c_a)
            sc = pts if is_c else 0
            total_score += sc
            results_list.append({'問題': q, 'あなたのマーク': u_a, '正解': c_a, '判定': "〇" if is_c else "✕", '獲得点数': sc})
            row_data[q] = u_a
            row_data[f'{q}_正誤'] = 1 if is_c else 0

        row_data['合計点'] = round(total_score, 1)

        st.balloons()
        st.metric(label=f"【STUDENT ID: {student_id}】 さんの合計点", value=f"{round(total_score, 1)} 点")
        st.dataframe(pd.DataFrame(results_list), height=400, use_container_width=True)

        df_new = pd.DataFrame([row_data])
        if not os.path.exists(DATA_FILE):
            df_new.to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
        else:
            df_new.to_csv(DATA_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')

        if st.button("次の解答を採点する"):
            st.session_state.step = 1
            st.rerun()

elif mode == "教員用（正解・配点・特殊ルール設定）":
    st.header("■ 教員用 各種設定ダッシュボード")
    tab0, tab1 = st.tabs(["⚙️ 基本設定", "1. 正解データ設定"])
    with tab0:
        c1, c2 = st.columns(2)
        new_max_q = c1.number_input("問題数の上限", 1, 50, MAX_Q)
        new_opts = c2.number_input("選択肢の数", 2, 9, NUM_OPTS)
        if st.button("設定保存"):
            save_json_file(CONFIG_FILE, {"max_questions": int(new_max_q), "num_options": int(new_opts)})
            st.success("設定を更新しました！")
            st.rerun()
    with tab1:
        st.subheader("正解データ")
        ans_file = st.file_uploader("正解マークシート画像から自動入力", type=["jpg", "png"])
        if ans_file and st.button("正解データとして登録", type="primary"):
            _, extracted, _, _, _, _, _ = analyze_mark_sheet(ans_file, MAX_Q, NUM_OPTS)
            save_json_file(ANSWER_KEY_FILE, extracted)
            st.success("正解データを自動保存しました！")

elif mode == "教員用（集計・Excel出力）":
    st.header("■ 集計ダッシュボード")
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
        st.dataframe(df)
