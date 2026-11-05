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

    return recognized_test_id, recognized_exam_id, recognized_answers, \
           cv2.cvtColor(debug_main, cv2.COLOR_BGR2RGB), \
           cv2.cvtColor(img_exam_debug, cv2.COLOR_BGR2RGB) if img_exam_debug is not None else None, \
           cv2.cvtColor(img_q1_debug, cv2.COLOR_BGR2RGB) if img_q1_debug is not None else None, \
           cv2.cvtColor(img_q2_debug, cv2.COLOR_BGR2RGB) if img_q2_debug is not None else None, \
           logs

# --- 設定・パスワードロード ---
config = load_config()
MAX_Q = config.get("max_questions", 50)
NUM_OPTS = config.get("num_options", 9)
teacher_password = load_password()

if "step" not in st.session_state: st.session_state.step = 1

mode = st.sidebar.radio("モード選択", ["生徒用（マーク撮影・採点）", "教員用（正解・配点・特殊ルール設定）", "教員用（集計・Excel出力）"])

# ---------------------------------------------------------
# 教員用モード パスワード認証
# ---------------------------------------------------------
if mode in ["教員用（正解・配点・特殊ルール設定）", "教員用（集計・Excel出力）"]:
    st.subheader("🔒 教員用認証")
    input_pass = st.text_input("教員用パスワードを入力してください", type="password")
    if input_pass != teacher_password:
        if input_pass:
            st.error("パスワードが正しくありません。")
        else:
            st.info("この機能を利用するには教員用パスワードの入力が必要です。 (初期パスワード: 0000)")
        st.stop()
    st.success("認証に成功しました。")
    st.markdown("---")

if mode == "生徒用（マーク撮影・採点）":
    st.header(f"■ 生徒用 マークシート送信・採点 (全{MAX_Q}問 / 選択肢1〜{NUM_OPTS})")

    if st.session_state.step == 1:
        uploaded_file = st.file_uploader("解答用紙全体が写るように撮影してアップロードしてください", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="撮影画像", width=450)
            if st.button("画像を解析して確認画面へ進む", type="primary"):
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
            st.session_state.final_test_id = test_id_val
            st.session_state.final_student_id = student_id_val
            st.session_state.final_answers = final_answers
            st.session_state.step = 3
            st.rerun()

    elif st.session_state.step == 3:
        st.subheader("STEP 3: 採点結果")
        user_ans = st.session_state.final_answers
        test_id = str(st.session_state.final_test_id).zfill(2)
        student_id = st.session_state.final_student_id
        
        # TEST IDに対応する正答・配点を読み込み
        answer_key = load_answer_key(test_id, MAX_Q)
        points_key = load_points_key(test_id, MAX_Q)

        total_score = 0
        results_list = []
        row_data = {
            '日時': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'TEST ID': test_id,
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

        # データのファイル保存
        df_new = pd.DataFrame([row_data])
        if not os.path.exists(DATA_FILE):
            df_new.to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
        else:
            df_new.to_csv(DATA_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')

        st.balloons()
        st.metric(label=f"【TEST ID: {test_id} / STUDENT ID: {student_id}】 さんの合計点", value=f"{round(total_score, 1)} 点")
        st.dataframe(pd.DataFrame(results_list), height=350, use_container_width=True)

        # 全体統計
        if os.path.exists(DATA_FILE):
            df_all = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
            df_all['TEST ID'] = df_all['TEST ID'].astype(str).str.zfill(2)
            df_test = df_all[df_all['TEST ID'] == test_id]
            
            if not df_test.empty:
                st.markdown("---")
                st.subheader(f"📊 全体統計 (TEST ID: {test_id})")
                
                s1, s2, s3 = st.columns(3)
                s1.metric("受験人数", f"{len(df_test)} 人")
                s2.metric("平均点", f"{round(df_test['合計点'].mean(), 1)} 点")
                s3.metric("最高点", f"{round(df_test['合計点'].max(), 1)} 点")
                
                st.write("**【問題別正答率 (%)】**")
                correct_cols = [f'{q}_正誤' for q in list(answer_key.keys()) if f'{q}_正誤' in df_test.columns]
                if correct_cols:
                    q_acc = (df_test[correct_cols].mean() * 100).to_frame(name="正答率(%)").T
                    q_acc.columns = [col.replace('_正誤', '') for col in correct_cols]
                    st.dataframe(q_acc.style.format("{:.1f}%"), use_container_width=True)

        st.markdown("---")
        if st.button("次の解答を採点する", type="primary"):
            st.session_state.step = 1
            st.rerun()

elif mode == "教員用（正解・配点・特殊ルール設定）":
    st.header("■ 教員用 各種設定ダッシュボード")
    tab0, tab1 = st.tabs(["⚙️ 基本設定・パスワード変更", "📝 正答・配点設定 (TEST ID別)"])
    
    with tab0:
        st.subheader("⚙️ 試験形式・問題数・選択肢数の上限設定")
        c1, c2 = st.columns(2)
        new_max_q = c1.number_input("問題数の上限", 1, 50, MAX_Q)
        new_opts = c2.number_input("選択肢の数", 2, 9, NUM_OPTS)
        if st.button("基本設定を保存"):
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
        st.subheader("📝 TEST ID 別 正答・配点管理")
        
        # TEST ID の選択 / 新規追加入力
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
        st.info(f"現在 **TEST ID: {target_test_id}** の正答・配点を設定・編集しています。")
        st.markdown("---")

        # 設定方法の選択
        input_method = st.radio("入力方法を選択", ["✍️ 正答・配点の手入力", "📷 正解マークシート画像から正答を自動読み取り"], horizontal=True)

        current_answers = load_answer_key(target_test_id, MAX_Q)
        current_points = load_points_key(target_test_id, MAX_Q)

        if input_method == "📷 正解マークシート画像から正答を自動読み取り":
            ans_file = st.file_uploader(f"TEST ID [{target_test_id}] の模範解答マークシート画像をアップロード", type=["jpg", "jpeg", "png"])
            if ans_file and st.button(f"画像から正答を登録 (TEST ID: {target_test_id})", type="primary"):
                _, _, extracted, _, _, _, _, _ = analyze_mark_sheet(ans_file, MAX_Q, NUM_OPTS)
                save_answer_key(target_test_id, extracted)
                st.success(f"TEST ID [{target_test_id}] の正答データを登録しました！")
                st.rerun()
        else:
            # 1. 一括配点変更ツール
            st.markdown("##### 💡 配点の一括設定")
            col_bp1, col_bp2 = st.columns([2, 1])
            bulk_val = col_bp1.number_input("全問の配点を一括変更（点）", min_value=0.5, max_value=20.0, value=2.0, step=0.5)
            if col_bp2.button("全問一括適用"):
                new_bulk_pts = {f'Q{i}': bulk_val for i in range(1, MAX_Q + 1)}
                save_points_key(target_test_id, new_bulk_pts)
                st.success(f"TEST ID [{target_test_id}] の全問配点を {bulk_val} 点に変更しました！")
                st.rerun()

            st.markdown("---")
            st.markdown(f"##### ✍️ 正答・個別の配点設定 (問1 〜 問{MAX_Q})")
            
            options = list(range(1, NUM_OPTS + 1))
            new_answers = {}
            new_points = {}

            col_a_l, col_a_r = st.columns(2)
            with col_a_l:
                st.markdown("###### ◀ 問 1 〜 25")
                for i in range(1, min(26, MAX_Q + 1)):
                    q_k = f'Q{i}'
                    idx = options.index(current_answers.get(q_k, 1)) if current_answers.get(q_k, 1) in options else 0
                    
                    q_c1, q_c2, q_c3 = st.columns([1, 2, 2])
                    q_c1.write(f"**問 {i}**")
                    new_answers[q_k] = q_c2.selectbox(f"正答_問{i}", options=options, index=idx, key=f"ans_{q_k}", label_visibility="collapsed")
                    new_points[q_k] = q_c3.number_input(f"配点_問{i}", min_value=0.0, max_value=20.0, value=float(current_points.get(q_k, 2.0)), step=0.5, key=f"pts_{q_k}", label_visibility="collapsed")

            with col_a_r:
                if MAX_Q >= 26:
                    st.markdown("###### ▶ 問 26 〜 50")
                    for i in range(26, MAX_Q + 1):
                        q_k = f'Q{i}'
                        idx = options.index(current_answers.get(q_k, 1)) if current_answers.get(q_k, 1) in options else 0
                        
                        q_c1, q_c2, q_c3 = st.columns([1, 2, 2])
                        q_c1.write(f"**問 {i}**")
                        new_answers[q_k] = q_c2.selectbox(f"正答_問{i}", options=options, index=idx, key=f"ans_{q_k}", label_visibility="collapsed")
                        new_points[q_k] = q_c3.number_input(f"配点_問{i}", min_value=0.0, max_value=20.0, value=float(current_points.get(q_k, 2.0)), step=0.5, key=f"pts_{q_k}", label_visibility="collapsed")

            st.markdown("---")
            if st.button(f"TEST ID [{target_test_id}] の正答・配点を保存する", type="primary", use_container_width=True):
                save_answer_key(target_test_id, new_answers)
                save_points_key(target_test_id, new_points)
                st.success(f"TEST ID [{target_test_id}] の正答・配点データを保存しました！")
                st.rerun()

elif mode == "教員用（集計・Excel出力）":
    st.header("■ 教員用 集計ダッシュボード")
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
        df['TEST ID'] = df['TEST ID'].astype(str).str.zfill(2)
        
        # --- TEST ID フィルター機能 ---
        unique_test_ids = sorted(list(df['TEST ID'].unique()))
        filter_options = ["すべて"] + unique_test_ids
        selected_test_id = st.selectbox("📌 集計対象の TEST ID を選択してください", filter_options)
        
        if selected_test_id != "すべて":
            df_display = df[df['TEST ID'] == selected_test_id].copy()
            st.info(f"現在 **TEST ID: {selected_test_id}** の結果を表示しています。")
        else:
            df_display = df.copy()
            st.info("現在 **すべての TEST ID** の結果を表示しています。")

        st.markdown("---")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("提出枚数", f"{len(df_display)} 枚")
        m2.metric("平均点", f"{round(df_display['合計点'].mean(), 1)} 点" if not df_display.empty else "0 点")
        m3.metric("最高点", f"{round(df_display['合計点'].max(), 1)} 点" if not df_display.empty else "0 点")
        m4.metric("最低点", f"{round(df_display['合計点'].min(), 1)} 点" if not df_display.empty else "0 点")

        st.markdown("---")
        st.subheader("1. 個人別解答・得点一覧")
        st.dataframe(df_display, use_container_width=True)

        st.subheader("2. 問題別 正答率 (%)")
        ref_test_id = selected_test_id if selected_test_id != "すべて" else (unique_test_ids[0] if unique_test_ids else "01")
        answer_key = load_answer_key(ref_test_id, MAX_Q)
        q_cols = list(answer_key.keys())
        correct_cols = [f'{q}_正誤' for q in q_cols if f'{q}_正誤' in df_display.columns]
        
        if correct_cols and not df_display.empty:
            q_acc = (df_display[correct_cols].mean() * 100).to_frame(name="正答率(%)").T
            q_acc.columns = [col.replace('_正誤', '') for col in correct_cols]
            st.dataframe(q_acc.style.format("{:.1f}%"), use_container_width=True)

        # TEST ID「すべて」選択時の比較テーブル
        if selected_test_id == "すべて" and len(unique_test_ids) > 1:
            st.markdown("---")
            st.subheader("3. TEST ID 別 統計・比較")
            test_summary = df.groupby('TEST ID')['合計点'].agg(
                提出件数='count',
                平均点='mean',
                最高点='max',
                最低点='min'
            ).reset_index()
            test_summary['平均点'] = test_summary['平均点'].round(1)
            st.dataframe(test_summary, use_container_width=True)

        # Excel出力機能
        st.markdown("---")
        excel_filename = f"集計結果_TEST_{selected_test_id}.xlsx"
        excel_path = f"/tmp/{excel_filename}" if os.path.exists("/tmp") else excel_filename
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_display.to_excel(writer, sheet_name='個人別結果', index=False)
            if correct_cols and not df_display.empty:
                q_acc.to_excel(writer, sheet_name='問題別正答率')
            if selected_test_id == "すべて":
                df.groupby('TEST ID')['合計点'].describe().to_excel(writer, sheet_name='TEST_ID別統計')

        with open(excel_path, "rb") as f:
            st.download_button(
                label=f"📥 選択中({selected_test_id})のデータをExcelでダウンロード",
                data=f,
                file_name=excel_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
    else:
        st.warning("まだ採点データが保存されていません。")
