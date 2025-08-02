from dotenv import load_dotenv
load_dotenv()

import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# ✅ 引入 RAG 功能
from rag_pdf import create_vector_store, ask_question

# ✅ 建立 PDF 向量資料庫（建一次即可）
vectorstore = create_vector_store("課程介紹.pdf")


# ✅ 載入 LINE 憑證
line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))

# ✅ Flask 啟動
app = Flask(__name__)

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK",200

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    question = event.message.text
    answer = ask_question(vectorstore, question)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=answer)
    )



if __name__ == "__main__":
    app.run(port=8000)
    
