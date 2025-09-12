"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################

# --- sqlite fallback for Streamlit Cloud（最上部に置く）---
try:
    import pysqlite3   # requirements.txt に pysqlite3-binary を入れてある
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
# --- ここまで ---

import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import constants as ct
import re
import csv
from langchain.schema import Document as LCDocument 


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()

# Cloud では .env は使わないため、Secrets を優先して環境変数に流し込む
import os
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # RAGのRetrieverを作成
    initialize_retriever()


def initialize_logger():
    """
    ログ出力の設定
    """
    # 指定のログフォルダが存在すれば読み込み、存在しなければ新規作成
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    
    # 引数に指定した名前のロガー（ログを記録するオブジェクト）を取得
    # 再度別の箇所で呼び出した場合、すでに同じ名前のロガーが存在していれば読み込む
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにロガーにハンドラー（ログの出力先を制御するもの）が設定されている場合、同じログ出力が複数回行われないよう処理を中断する
    if logger.hasHandlers():
        return

    # 1日単位でログファイルの中身をリセットし、切り替える設定
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    # 出力するログメッセージのフォーマット定義
    # - 「levelname」: ログの重要度（INFO, WARNING, ERRORなど）
    # - 「asctime」: ログのタイムスタンプ（いつ記録されたか）
    # - 「lineno」: ログが出力されたファイルの行番号
    # - 「funcName」: ログが出力された関数名
    # - 「session_id」: セッションID（誰のアプリ操作か分かるように）
    # - 「message」: ログメッセージ
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )

    # 定義したフォーマッターの適用
    log_handler.setFormatter(formatter)

    # ログレベルを「INFO」に設定
    logger.setLevel(logging.INFO)

    # 作成したハンドラー（ログ出力先を制御するオブジェクト）を、
    # ロガー（ログメッセージを実際に生成するオブジェクト）に追加してログ出力の最終設定
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        # ランダムな文字列（セッションID）を、ログ出力用に作成
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    """
    画面読み込み時にRAGのRetriever（ベクターストアから検索するオブジェクト）を作成
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    if "retriever" in st.session_state:
        return
    
    # 参照データ読み込み
    docs_all = load_data_sources()

    # 文字化け対策
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    # 埋め込みモデル
    embeddings = OpenAIEmbeddings(model=ct.EMBEDDING_MODEL)
    
    # 分割器
    from constants import CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_TOP_K
    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator="\n"
    )

    # ==== ここがポイント ====
    # 分割禁止ドキュメントは取り分け、分割対象のみsplitする
    no_split_docs = [d for d in docs_all if d.metadata.get("no_split")]
    split_targets = [d for d in docs_all if not d.metadata.get("no_split")]

    splitted_docs = text_splitter.split_documents(split_targets)
    # 分割禁止のものはそのまま追加（部署ごと1ドキュメントの名簿など）
    splitted_docs.extend(no_split_docs)
    # ==== ここまで ====

    # ベクターストア作成
    db = Chroma(embedding_function=embeddings)

    # 小分けで安全に追加（トークン上限対策）
    BATCH = ct.EMBEDDING_BATCH_SIZE  # 例: 64
    for i in range(0, len(splitted_docs), BATCH):
        db.add_documents(splitted_docs[i:i+BATCH])

    # Retriever
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})




def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        # 「表示用」の会話ログを順次格納するリストを用意
        st.session_state.messages = []
        # 「LLMとのやりとり用」の会話ログを順次格納するリストを用意
        st.session_state.chat_history = []


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み

    Returns:
        読み込んだ通常データソース
    """
    # データソースを格納する用のリスト
    docs_all = []
    # ファイル読み込みの実行（渡した各リストにデータが格納される）
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    web_docs_all = []
    # ファイルとは別に、指定のWebページ内のデータも読み込み
    # 読み込み対象のWebページ一覧に対して処理
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        # 指定のWebページを読み込み
        loader = WebBaseLoader(web_url)
        web_docs = loader.load()
        # for文の外のリストに読み込んだデータソースを追加
        web_docs_all.extend(web_docs)
    # 通常読み込みのデータソースにWebページのデータを追加
    docs_all.extend(web_docs_all)

    return docs_all


def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    # パスがフォルダかどうかを確認
    if os.path.isdir(path):
        # フォルダの場合、フォルダ内のファイル/フォルダ名の一覧を取得
        files = os.listdir(path)
        # 各ファイル/フォルダに対して処理
        for file in files:
            # ファイル/フォルダ名だけでなく、フルパスを取得
            full_path = os.path.join(path, file)
            # フルパスを渡し、再帰的にファイル読み込みの関数を実行
            recursive_file_check(full_path, docs_all)
    else:
        # パスがファイルの場合、ファイル読み込み
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み
    """
    file_extension = os.path.splitext(path)[1]
    file_name = os.path.basename(path)

    # --- 部署名の正規化（人事/人事課/HR → 人事部 など） ---
    def _normalize_dept(s: str) -> str:
        if s is None:
            return "不明"
        s = str(s).strip()
        # 半角/全角スペース除去
        s = re.sub(r"[ \u3000]", "", s)
        # 代表的な表記ゆれを吸収（必要に応じて増やしてください）
        if any(x in s for x in ["人事", "人事課", "HR", "ＨＲ", "HumanResources"]):
            return "人事部"
        if "総務" in s:
            return "総務部"
        return s or "不明"

    # --- 特別扱い: 社員名簿.csv は『部署ごとに1ドキュメント』へ統合し、分割禁止（no_split） ---
    if file_extension == ".csv" and file_name == "社員名簿.csv":
        try:
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames or (rows[0].keys() if rows else [])

            if not rows:
                return

            # 列名のゆらぎ対応
            def pick(colnames, candidates):
                for c in candidates:
                    if c in colnames:
                        return c
                return None

            dept_key = pick(fieldnames, ["部署", "部門", "所属", "部署名", "Department"])
            id_key   = pick(fieldnames, ["社員ID", "従業員ID", "社員番号", "ID"])
            name_key = pick(fieldnames, ["氏名（フルネーム）", "氏名", "名前", "従業員名", "社員名", "Name"])
            mail_key = pick(fieldnames, ["メールアドレス", "メール", "Email", "E-mail"])
            role_key = pick(fieldnames, ["役職", "職位", "ポジション", "役割"])
            skill_key= pick(fieldnames, ["スキルセット", "スキル", "Skill", "Skills"])

            # 部署ごとにグルーピング（正規化してから）
            groups = {}
            for r in rows:
                raw_dept = (r.get(dept_key) or "不明") if dept_key else "不明"
                dept = _normalize_dept(raw_dept)
                groups.setdefault(dept, []).append(r)

            # 件数をログへ（統合できているか確認用）
            logging.getLogger(ct.LOGGER_NAME).info(
                {"roster_by_dept_counts": {k: len(v) for k, v in groups.items()}}
            )

            # 各部署ごとに1ドキュメント（no_split=True）
            for dept, members in groups.items():
                lines = []
                lines.append(f"【部署:{dept} の社員一覧】（{len(members)}名）")

                # 検索用タグ（人事部は表記ゆれも付与）
                if dept == "人事部":
                    lines.append("以下は社員名簿の統合ビューです。検索用タグ: [部署:人事部] [部署:人事] [部署:HR]")
                else:
                    lines.append("以下は社員名簿の統合ビューです。検索用タグ: " + f"[部署:{dept}]")
                lines.append("")

                for r in members:
                    parts = []
                    def add(label, key):
                        if key and r.get(key):
                            parts.append(f"{label}:{str(r.get(key)).strip()}")

                    add("社員ID", id_key)
                    add("氏名",   name_key)
                    add("メール", mail_key)
                    add("役職",   role_key)
                    add("スキル", skill_key)

                    # すべての列も残してヒット率UP
                    for k in fieldnames:
                        v = (r.get(k) or "").strip()
                        if v != "" and k not in {id_key, name_key, mail_key, role_key, skill_key}:
                            parts.append(f"{k}:{v}")

                    # 検索タグ
                    parts.append(f"[部署:{dept}]")
                    if dept == "人事部":
                        parts.append("[部署:人事]")
                        parts.append("[部署:HR]")
                    if name_key and r.get(name_key):
                        parts.append(f"[氏名:{str(r.get(name_key)).strip()}]")

                    lines.append("- " + " | ".join(parts))

                content = "\n".join(lines)
                docs_all.append(
                    LCDocument(
                        page_content=content,
                        metadata={"source": path, "dept": dept, "no_split": True}
                    )
                )
            return  # ここで終了（通常処理に落とさない）

        except Exception:
            # 失敗時は従来ローダーにフォールバック
            loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
            docs_all.extend(loader.load())
            return

    # --- 通常処理（PDF/DOCX/TXT/その他CSV） ---
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs = loader.load()

        # PDFなどで "page" が別名のときに正規化
        for d in docs:
            if "page" not in d.metadata:
                if "page_number" in d.metadata and isinstance(d.metadata["page_number"], int):
                    d.metadata["page"] = d.metadata["page_number"]

        docs_all.extend(docs)


    # ▲ ここまで追記
    
    """
    ファイル内のデータ読み込み

    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    # ファイルの拡張子を取得
    file_extension = os.path.splitext(path)[1]
    # ファイル名（拡張子を含む）を取得
    file_name = os.path.basename(path)

    # 想定していたファイル形式の場合のみ読み込む
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        # ファイルの拡張子に合ったdata loaderを使ってデータ読み込み
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs = loader.load()
        docs_all.extend(docs)
        


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s