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

        if 0.75 < aspect < 1.35 and (norm_w * 0.008) < bw < (norm_w * 0.035) and (norm_h * 0.008) < bh < (norm_h * 0.035) and solidity > 0.82:
            square_markers.append((x + bw/2.0, y + bh/2.0))
            
    return square_markers

def auto_orient_and_get_markers(image):
    for rot in [0, 1, 2, 3]:
        if rot == 0:
            test_img = image.copy()
        elif rot == 1:
            test_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 2:
            test_img = cv2.rotate(image, cv2.ROTATE_180)
        elif rot == 3:
            test_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w = test_img.shape[:2]
        if w < h:
            continue

        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        markers = detect_square_markers(gray)

        if len(markers) >= 10:
            markers_sorted_x = sorted(markers, key=lambda m: m[0])
            left_ys = [m[1] for m in markers_sorted_x[:4]]
            mid_ys = [m[1] for m in markers_sorted_x[4:8]]
            
            left_h = max(left_ys) - min(left_ys)
            mid_h = max(mid_ys) - min(mid_ys)

            if left_h < mid_h:
                return test_img, markers

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, detect_square_markers(gray)

def analyze_mark_sheet(image_file):
    pil_img = Image.open(image_file)
    pil_img = ImageOps.exif_transpose(pil_img)
    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    image_file.seek(0)

    norm_w, norm_h = 3000, 2100
    image = cv2.resize(image, (norm_w, norm_h))

    oriented_img, markers = auto_orient_and_get_markers(image)
    gray = cv2.cvtColor(oriented_img, cv2.COLOR_BGR2GRAY)
    debug_img = oriented_img.copy()

    def warp_block(pts_4, target_w, target_h):
        pts_src = np.array(pts_4, dtype="float32")
        pts_dst = np.array([[0, 0], [target_w - 1, 0], [0, target_h - 1], [target_w - 1, target_h - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(gray, M, (target_w, target_h))
        _, bin_out = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return bin_out

    recognized_exam_id = "0000"
    recognized_answers = {f'Q{i}': "未マーク/無効" for i in range(1, 51)}

    if len(markers) >= 12:
        markers_sorted_x = sorted(markers, key=lambda m: m[0])
        group_exam = sorted(markers_sorted_x[0:4], key=lambda m: (m[1], m[0]))
        group_q1_25 = sorted(markers_sorted_x[4:8], key=lambda m: (m[1], m[0]))
        group_q26_50 = sorted(markers_sorted_x[8:12], key=lambda m: (m[1], m[0]))

        try:
            pts_exam = [group_exam[0], group_exam[1], group_exam[2], group_exam[3]]
            bin_exam = warp_block(pts_exam, 400, 600)
            digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            parsed_digits = []
            
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

        for m in markers:
            cv2.circle(debug_img, (int(m[0]), int(m[1])), 20, (0, 255, 0), -1)

    debug_img_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
    return recognized_exam_id, recognized_answers, debug_img_rgb

if "step" not in st.session_state:
    st.session_state.step = 1

mode = st.sidebar.radio("モード選択", [
    "生徒用（マーク撮影・採点）", 
    "教員用（正解・配点・特殊ルール設定）", 
    "教員用（集計・Excel出力）"
])

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
                row_data[q] = user_ans.get(q)
                row_data[f'{q}_正誤'] = 1 if is_all_correct else 0

        for q, correct_ans in answer_key.items():
            if q in processed_qs: continue
            u_a = user_ans.get(q)
            q_p = points_key.get(q, 2.0)
            is_c = (u_a == correct_ans)
            score = q_p if is_c else 0
            total_score += score
            
            results_dict[q] = {
                '問題': q, 'あなたのマーク': u_a, '正解': correct_ans,
                '判定': "〇" if is_c else "✕", '配点': q_p, '獲得点数': score
            }
            row_data[q] = u_a
            row_data[f'{q}_正誤'] = 1 if is_c else 0

        results_list = [results_dict[f'Q{i}'] for i in range(1, 51) if f'Q{i}' in results_dict]
        
        max_possible_score = sum([points_key[q] for q in points_key if q not in processed_qs])
        for r in special_rules.get("unordered", []):
            max_possible_score += sum([r.get("point_per_q", 2.0) for _ in r["questions"]])
        for r in special_rules.get("all_correct", []):
            max_possible_score += r.get("group_point", 4.0)

        row_data['合計点'] = round(total_score, 1)

        st.balloons()
        st.success("採点が完了しました！")
        st.metric(
            label=f"【{student_class} / 番号: {student_id}】 さんの合計点", 
            value=f"{round(total_score, 1)} / {round(max_possible_score, 1)} 点"
        )
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
    tab1, tab2, tab3 = st.tabs(["1. 正解データ設定", "2. 基本配点設定", "3. 特殊ルール設定"])
    
    with tab1:
        input_method = st.radio("入力方法を選択", ["📷 マークシート画像から自動読み取り", "✍️ 手入力"], horizontal=True)
        if input_method == "📷 マークシート画像から自動読み取り":
            ans_file = st.file_uploader("正解マークシート画像をアップロード", type=["jpg", "jpeg", "png"])
            if ans_file and st.button("正解データとして登録", type="primary"):
                _, extracted_answers, _ = analyze_mark_sheet(ans_file)
                save_json_file(ANSWER_KEY_FILE, extracted_answers)
                st.success("画像から正解データを登録しました！")
                st.rerun()
        else:
            current_answers = load_answer_key()
            options = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            new_answers = {}
            col_a_l, col_a_r = st.columns(2)
            with col_a_l:
                for i in range(1, 26):
                    q_k = f'Q{i}'
                    idx = options.index(current_answers.get(q_k, 1)) if current_answers.get(q_k, 1) in options else 0
                    c1, c2 = st.columns([1, 2])
                    c1.write(f"**問 {i}**")
                    new_answers[q_k] = c2.selectbox(f"問{i}", options=options, index=idx, key=f"ans_{q_k}", label_visibility="collapsed")
            with col_a_r:
                for i in range(26, 51):
                    q_k = f'Q{i}'
                    idx = options.index(current_answers.get(q_k, 1)) if current_answers.get(q_k, 1) in options else 0
                    c1, c2 = st.columns([1, 2])
                    c1.write(f"**問 {i}**")
                    new_answers[q_k] = c2.selectbox(f"問{i}", options=options, index=idx, key=f"ans_{q_k}", label_visibility="collapsed")
            if st.button("手入力した正解データを保存", type="primary"):
                save_json_file(ANSWER_KEY_FILE, new_answers)
                st.success("正解データを保存しました！")
                st.rerun()

    with tab2:
        current_points = load_points_key()
        col_b1, col_b2 = st.columns([2, 1])
        bulk_val = col_b1.number_input("全問の配点を一括変更（点）", min_value=0.5, max_value=20.0, value=2.0, step=0.5)
        if col_b2.button("全問一括適用"):
            save_json_file(POINTS_KEY_FILE, {f'Q{i}': bulk_val for i in range(1, 51)})
            st.success(f"全問の配点を {bulk_val} 点に変更しました！")
            st.rerun()

        st.markdown("---")
        col_p_l, col_p_r = st.columns(2)
        new_points = {}
        with col_p_l:
            for i in range(1, 26):
                q_k = f'Q{i}'
                new_points[q_k] = st.number_input(f"問 {i} の配点", min_value=0.0, max_value=20.0, value=float(current_points.get(q_k, 2.0)), step=0.5, key=f"p_{q_k}")
        with col_p_r:
            for i in range(26, 51):
                q_k = f'Q{i}'
                new_points[q_k] = st.number_input(f"問 {i} の配点", min_value=0.0, max_value=20.0, value=float(current_points.get(q_k, 2.0)), step=0.5, key=f"p_{q_k}")
        if st.button("個別配点設定を保存", type="primary"):
            save_json_file(POINTS_KEY_FILE, new_points)
            st.success("配点設定を保存しました！")

    with tab3:
        st.subheader("■ 特殊配点ルールの設定")
        special_rules = load_special_rules()
        col_u1, col_u2, col_u3 = st.columns([2, 2, 1])
        unorder_qs = col_u1.multiselect("対象の問題を選択", [f'Q{i}' for i in range(1, 51)], key="sel_unorder_qs")
        unorder_ans = col_u2.multiselect("正解となる数字の組み合わせ", [1, 2, 3, 4, 5, 6, 7, 8, 9], key="sel_unorder_ans")
        unorder_pt = col_u3.number_input("各問の配点", value=2.0, step=0.5, key="unorder_pt")
        
        if st.button("＋ 順不同グループを追加"):
            if len(unorder_qs) >= 2 and len(unorder_qs) == len(unorder_ans):
                special_rules["unordered"].append({"questions": unorder_qs, "answers": unorder_ans, "point_per_q": unorder_pt})
                save_json_file(RULES_KEY_FILE, special_rules)
                st.success("順不同ルールを追加しました！")
                st.rerun()

        st.markdown("---")
        col_c1, col_c2 = st.columns([3, 1])
        all_correct_qs = col_c1.multiselect("完答対象の問題を選択", [f'Q{i}' for i in range(1, 51)], key="sel_ac_qs")
        group_pt = col_c2.number_input("完答時の合計配点", value=4.0, step=0.5, key="ac_pt")
        
        if st.button("＋ 完答グループを追加"):
            if len(all_correct_qs) >= 2:
                special_rules["all_correct"].append({"questions": all_correct_qs, "group_point": group_pt})
                save_json_file(RULES_KEY_FILE, special_rules)
                st.success("完答ルールを追加しました！")
                st.rerun()

        st.markdown("---")
        st.write("**【現在登録されている特殊ルール】**")
        st.json(special_rules)
        if st.button("特殊ルールをすべてクリア（初期化）"):
            save_json_file(RULES_KEY_FILE, {"unordered": [], "all_correct": []})
            st.success("特殊ルールをクリアしました。")
            st.rerun()

elif mode == "教員用（集計・Excel出力）":
    st.header("■ 教員用 集計ダッシュボード")
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
        st.subheader("1. 個人別結果")
        st.dataframe(df)
        
        st.subheader("2. クラス別・問題別正答率 (%)")
        answer_key = load_answer_key()
        q_cols = list(answer_key.keys())
        correct_cols = [f'{q}_正誤' for q in q_cols if f'{q}_正誤' in df.columns]
        
        if correct_cols:
            class_summary = df.groupby('クラス')[correct_cols].mean() * 100
            class_summary.columns = [col.replace('_正誤', '') for col in correct_cols]
            st.dataframe(class_summary.style.format("{:.1f}%"))
        
        excel_path = "共通テスト集計結果.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='個人別結果', index=False)
            if correct_cols:
                class_summary.to_excel(writer, sheet_name='クラス別正答率')
            
        with open(excel_path, "rb") as f:
            st.download_button("📥 Excel形式でダウンロード", f, file_name="共通テスト集計結果.xlsx")
    else:
        st.warning("まだ提出データがありません。")
