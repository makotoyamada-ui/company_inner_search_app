# utils.py （フル置換）

"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import re
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import constants as ct

############################################################
# 設定関連
############################################################
load_dotenv()

############################################################
# 共通ユーティリティ
############################################################

def get_source_icon(source):
    if source.startswith("http"):
        return ct.LINK_SOURCE_ICON
    return ct.DOC_SOURCE_ICON


def build_error_message(message):
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])

############################################################
# 人事名簿 用の決定論的パーサ & 描画
############################################################

# ラベルを正規化（別名 → 正式名）
_LABEL_ALIASES = {
    "氏名": {"氏名", "氏名（フルネーム）", "フルネーム", "名前", "Name", "Full Name"},
    "役職": {"役職", "職位", "ポジション", "役割"},
    "メール": {"メール", "メールアドレス", "Email", "E-mail"},
    "社員ID": {"社員ID", "従業員ID", "社員番号", "ID"},
    "スキル": {"スキル", "スキルセット", "Skill", "Skills"},
}

def _canon_label(label: str) -> str:
    s = str(label).strip()
    for k, al in _LABEL_ALIASES.items():
        if s in al:
            return k
    return s  # そのまま返す（その他の列は補足情報として使う）

def _is_hr_list_query(text: str) -> bool:
    if not text:
        return False
    t = str(text)
    return ("人事" in t or "HR" in t or "ＨＲ" in t) and ("一覧" in t or "リスト" in t or "一覧化" in t)

def _is_roster_doc(doc) -> bool:
    # initialize.py で no_split=True, metadata["dept"] を付与している
    return bool(doc.metadata.get("no_split")) and "社員名簿" in doc.metadata.get("source", "")

def _parse_roster_doc(doc):
    """
    initialize.py で生成した統合ビュー（- ラベル:値 | ... の行）をパースして
    [ {氏名, 役職, メール, 社員ID, スキル, ...}, ... ] を返す
    """
    emps = []
    for line in doc.page_content.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        line = line[2:]  # 先頭の "- " を除去
        cols = [c.strip() for c in line.split(" | ")]
        rec = {}
        for col in cols:
            if not col or col.startswith("["):  # 検索タグは無視
                continue
            if ":" not in col:
                continue
            label, val = col.split(":", 1)
            label = _canon_label(label)
            rec[label] = val.strip()
        if rec:
            emps.append(rec)
    return emps

def _dedup_emps(emps):
    seen = set()
    uniq = []
    for r in emps:
        key = r.get("社員ID") or r.get("メール") or r.get("氏名")
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return uniq

def _format_emps_table(emps):
    if not emps:
        return "該当者は見つかりませんでした。"

    # 安定した順序（社員ID→氏名）でソート
    def _sort_key(r):
        return (str(r.get("社員ID", "")), str(r.get("氏名", "")))
    emps = sorted(emps, key=_sort_key)

    header = ["氏名", "役職", "メール", "社員ID", "スキル"]
    lines = [
        "### 人事部に所属している従業員情報",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for r in emps:
        lines.append(
            "| {氏名} | {役職} | {メール} | {社員ID} | {スキル} |".format(
                氏名=r.get("氏名", ""),
                役職=r.get("役職", ""),
                メール=r.get("メール", ""),
                社員ID=r.get("社員ID", ""),
                スキル=r.get("スキル", ""),
            )
        )
    lines.append("")
    lines.append("この表は、**社員名簿（人事部・統合ビュー）** を元に機械的に抽出しています。")
    return "\n".join(lines)

############################################################
# LLM 応答（人事部のときは決定論的に）
############################################################

def get_llm_response(chat_message):
    """
    LLMからの回答取得。人事部の一覧は決定論で処理し、それ以外は従来のRAGにフォールバック。
    """
    # まず「人事部の一覧」なら決定論モード
    if _is_hr_list_query(chat_message):
        # 関連ドキュメントを取得
        retriever = st.session_state.retriever
        # 人事部／名簿に強くヒットするようクエリを少しリッチに
        query = "人事部 社員名簿 部署:人事部 HR 一覧"
        docs = retriever.get_relevant_documents(query)

        # 人事部の統合名簿ドキュメントだけを拾う
        roster_docs = [d for d in docs if _is_roster_doc(d) and d.metadata.get("dept") == "人事部"]

        # なければ全部から人事部だけ抽出してみる
        if not roster_docs:
            roster_docs = [d for d in docs if _is_roster_doc(d)]

        # まだ無ければ通常RAGへフォールバック
        if roster_docs:
            # 複数見つかったら最初のものを採用
            doc = roster_docs[0]
            emps = _parse_roster_doc(doc)
            emps = _dedup_emps(emps)
            # 最大10名（実データは9名）
            emps = emps[:10]
            answer = _format_emps_table(emps)

            # 画面側のロジックと整合させるため、RAGの戻り値形式に合わせて返す
            llm_response = {"answer": answer, "context": roster_docs}
            st.session_state.chat_history.extend([HumanMessage(content=chat_message), answer])
            return llm_response
        # （フォールバックに続行）

    # —— 通常のRAG（従来どおり） ——
    llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE)

    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [("system", question_generator_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")]
    )

    if st.session_state.mode == ct.ANSWER_MODE_1:
        question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        question_answer_template = ct.SYSTEM_PROMPT_INQUIRY

    question_answer_prompt = ChatPromptTemplate.from_messages(
        [("system", question_answer_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, question_generator_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    llm_response = chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.extend([HumanMessage(content=chat_message), llm_response["answer"]])
    return llm_response
