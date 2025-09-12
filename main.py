"""
このファイルは、Webアプリのメイン処理が記述されたファイルです。
"""

############################################################
# 1. ライブラリの読み込み
############################################################
from dotenv import load_dotenv
import logging
import streamlit as st
import utils
from initialize import initialize
import components as cn
import constants as ct

############################################################
# 2. 設定関連
############################################################
st.set_page_config(page_title=ct.APP_NAME)

logger = logging.getLogger(ct.LOGGER_NAME)

############################################################
# 3. 初期化処理
############################################################
try:
    initialize()
except Exception as e:
    logging.getLogger(ct.LOGGER_NAME).exception(e)  # ログにスタック
    st.exception(e)  # 画面にもスタックを出す（暫定）
    st.stop()

# アプリ起動時のログファイルへの出力
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    logger.info(ct.APP_BOOT_MESSAGE)

############################################################
# 4. 初期表示
############################################################
cn.display_app_title()
cn.display_select_mode()
cn.display_initial_ai_message()

############################################################
# 5. 会話ログの表示
############################################################
try:
    cn.display_conversation_log()
except Exception as e:
    logging.getLogger(ct.LOGGER_NAME).exception(e)
    st.exception(e)   # 画面に詳細
    st.stop()

############################################################
# 6. チャット入力の受け付け
############################################################
chat_message = st.chat_input(ct.CHAT_INPUT_HELPER_TEXT)

############################################################
# 7. チャット送信時の処理
############################################################
if chat_message:
    # 7-1. ユーザーメッセージの表示
    logger.info({"message": chat_message, "application_mode": st.session_state.mode})
    with st.chat_message("user"):
        st.markdown(chat_message)

    # 7-2. LLMからの回答取得（元のスピナー版）
    res_box = st.empty()
    with st.spinner(ct.SPINNER_TEXT):
        try:
            llm_response = utils.get_llm_response(chat_message)
        except Exception as e:
            logger.error(f"{ct.GET_LLM_RESPONSE_ERROR_MESSAGE}\n{e}")
            st.error(utils.build_error_message(ct.GET_LLM_RESPONSE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            st.stop()

    # 7-3. LLMからの回答表示
    with st.chat_message("assistant"):
        try:
            if st.session_state.mode == ct.ANSWER_MODE_1:
                content = cn.display_search_llm_response(llm_response)
            else:
                content = cn.display_contact_llm_response(llm_response)
            logger.info({"message": content, "application_mode": st.session_state.mode})
        except Exception as e:
            logging.getLogger(ct.LOGGER_NAME).exception(e)
            st.exception(e)
            st.stop()

    # 7-4. 会話ログへの追加
    st.session_state.messages.append({"role": "user", "content": chat_message})
    st.session_state.messages.append({"role": "assistant", "content": content})
