FROM python:3.9-slim

WORKDIR /app

# 必要な部品リストをコピーしてインストール
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# アプリのプログラム本体をすべてコピー
COPY . ./

# Streamlit用のポートを開放
EXPOSE 8501

# アプリの起動コマンド（※ main.py の部分は先生のメインファイル名に合わせてください）
CMD ["streamlit", "run", "listening.py", "--server.port=8501", "--server.address=0.0.0.0"]
