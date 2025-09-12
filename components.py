"""
このファイルは、画面表示に特化した関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import streamlit as st
import utils
import constants as ct


############################################################
# 関数定義
############################################################

def _label_with_page(path: str, page: int | None):
    """画面表示用ラベル: ファイルパス + （ページNo.X）"""
    if page is None:
        return path
    return f"{path}（ページNo.{page + 1}）"


def display_app_title():
    """
    タイトル表示（メインエリア）
    """
    st.markdown(f"## {ct.APP_NAME}")


def display_select_mode():
    """
    サイドバーに「利用目的」ラジオと説明カードを表示し、選択結果を session_state に格納
    既存の呼び出し側から関数名は変えずに使えるようにしている。
    """
    with st.sidebar:
        st.header("利用目的")

        # 既定値の初期化
        if "mode" not in st.session_state:
            st.session_state.mode = ct.ANSWER_MODE_1

        st.session_state.mode = st.radio(
            label="",
            options=[ct.ANSWER_MODE_1, ct.ANSWER_MODE_2],
            index=0 if st.session_state.mode == ct.ANSWER_MODE_1 else 1
        )

        st.markdown("---")

        # --- 「社内文書検索」の説明 ---
        st.markdown("**「社内文書検索」を選択した場合**")
        st.info("入力内容と関連性が高い社内文書のありかを検索できます。")
        with st.expander("【入力例】"):
            st.write("社員の育成方針に関するMTGの議事録")

        st.markdown("---")

        # --- 「社内問い合わせ」の説明 ---
        st.markdown("**「社内問い合わせ」を選択した場合**")
        st.info("質問・要望に対して、社内文書の情報をもとに回答を得られます。")
        with st.expander("【入力例】"):
            st.write("人事部に所属している従業員情報を一覧化して")


def display_initial_ai_message():
    """
    初期の案内メッセージ（メインエリア）
    旧: chat_message の吹き出し → 新: 緑/黄の案内ボックス
    """
    st.success(
        "こんにちは。私は社内文書の情報をもとに回答する生成AIチャットボットです。"
        " サイドバーで利用目的を選択し、画面下部のチャット欄からメッセージを送信してください。"
    )
    st.warning("具体的に入力したほうが期待通りの回答を得やすいです。")


def display_conversation_log():
    """
    会話ログの一覧表示（ページ番号表示に対応）
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
                continue

            if message["content"]["mode"] == ct.ANSWER_MODE_1:
                # 社内文書検索
                if not "no_file_path_flg" in message["content"]:
                    st.markdown(message["content"]["main_message"])

                    main_path = message["content"]["main_file_path"]
                    main_page = message["content"].get("main_page_number")
                    icon = utils.get_source_icon(main_path)
                    st.success(_label_with_page(main_path, main_page), icon=icon)

                    if "sub_message" in message["content"]:
                        st.markdown(message["content"]["sub_message"])
                        for sub_choice in message["content"]["sub_choices"]:
                            sub_path = sub_choice["source"]
                            sub_page = sub_choice.get("page_number")
                            icon = utils.get_source_icon(sub_path)
                            st.info(_label_with_page(sub_path, sub_page), icon=icon)
                else:
                    st.markdown(message["content"]["answer"])

            else:
                # 社内問い合わせ
                st.markdown(message["content"]["answer"])
                if "file_info_list" in message["content"]:
                    st.divider()
                    st.markdown(f"##### {message['content']['message']}")
                    for item in message["content"]["file_info_list"]:
                        # 新形式（辞書）
                        if isinstance(item, dict) and "path" in item:
                            path = item["path"]
                            page = item.get("page_number")
                            icon = utils.get_source_icon(path)
                            st.info(_label_with_page(path, page), icon=icon)
                        else:
                            # 後方互換（文字列のみ）
                            label = str(item)
                            base = label.split("（ページNo.")[0]
                            icon = utils.get_source_icon(base)
                            st.info(label, icon=icon)


def display_search_llm_response(llm_response):
    """
    「社内文書検索」モードにおけるLLMレスポンスを表示
    """
    if llm_response["context"] and llm_response["answer"] != ct.NO_DOC_MATCH_ANSWER:
        main_file_path = llm_response["context"][0].metadata["source"]
        main_message = "入力内容に関する情報は、以下のファイルに含まれている可能性があります。"
        st.markdown(main_message)

        icon = utils.get_source_icon(main_file_path)
        content = {"mode": ct.ANSWER_MODE_1, "main_message": main_message, "main_file_path": main_file_path}

        # メイン
        main_page_number = llm_response["context"][0].metadata.get("page")
        st.success(_label_with_page(main_file_path, main_page_number), icon=icon)
        if main_page_number is not None:
            content["main_page_number"] = main_page_number

        # サブ
        sub_choices, seen = [], set([main_file_path])
        for document in llm_response["context"][1:]:
            sub_path = document.metadata["source"]
            if sub_path in seen:
                continue
            seen.add(sub_path)

            sub_page = document.metadata.get("page")
            sub_choice = {"source": sub_path}
            if sub_page is not None:
                sub_choice["page_number"] = sub_page
            sub_choices.append(sub_choice)

        if sub_choices:
            sub_message = "その他、ファイルありかの候補を提示します。"
            st.markdown(sub_message)
            for sub in sub_choices:
                icon = utils.get_source_icon(sub["source"])
                st.info(_label_with_page(sub["source"], sub.get("page_number")), icon=icon)

            content["sub_message"] = sub_message
            content["sub_choices"] = sub_choices

    else:
        st.markdown(ct.NO_DOC_MATCH_MESSAGE)
        content = {"mode": ct.ANSWER_MODE_1, "answer": ct.NO_DOC_MATCH_MESSAGE, "no_file_path_flg": True}

    return content

def display_contact_llm_response(llm_response):
    """
    「社内問い合わせ」モードにおけるLLMレスポンスを表示
    """
    st.markdown(llm_response["answer"])

    file_info_list = []
    if llm_response["answer"] != ct.INQUIRY_NO_MATCH_ANSWER:
        st.divider()
        message = "情報源"
        st.markdown(f"##### {message}")

        seen = set()
        for document in llm_response["context"]:
            path = document.metadata["source"]
            if path in seen:
                continue
            seen.add(path)

            page = document.metadata.get("page")
            icon = utils.get_source_icon(path)
            st.info(_label_with_page(path, page), icon=icon)

            file_info_list.append({"path": path, "page_number": page})

    content = {"mode": ct.ANSWER_MODE_2, "answer": llm_response["answer"]}
    if llm_response["answer"] != ct.INQUIRY_NO_MATCH_ANSWER:
        content["message"] = "情報源"
        content["file_info_list"] = file_info_list
    return content