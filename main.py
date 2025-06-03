
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import pandas as pd
import os
import traceback

app = Flask(__name__)
CORS(app)

openai.api_key = os.environ.get("OPENAI_API_KEY")

EXCEL_PATH = "toptop.xlsx"

try:
    df = pd.read_excel(EXCEL_PATH)
except Exception:
    df = None

def search_excel(question):
    if df is None:
        return None
    for _, row in df.iterrows():
        if str(row["질문키"]) in question:
            return row["답변"]
    return None

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "")
        if not question:
            return jsonify({"answer": "질문을 입력해 주세요 !", "image": None}), 400

        excel_info = search_excel(question)
        system_prompt = "넌 청주탑병원의 안내 도우미 '탑탑이'란다. 친절하게 한글로 대답해줘. 100자 이내로 요약하면 좋아."
        if excel_info:
            system_prompt += f" 아래 내용을 반드시 반영해서 정확히 대답해:\n\"{excel_info}\""

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        )
        answer = response.choices[0].message.content

    except Exception:
        traceback.print_exc()
        answer = "탑탑이가 대답할 수 없어요 !"

    image_url = None
    try:
        if any(keyword in question for keyword in ["원무과", "수납", "접수"]):
            image_url = "https://res.cloudinary.com/duvoimzkv/image/upload/v1747505265/toptop_admdepart_rm36ov.png"
    except Exception:
        pass

    return jsonify({"answer": answer, "image": image_url})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
