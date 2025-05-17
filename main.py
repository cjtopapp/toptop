from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os

app = Flask(__name__)
CORS(app)

openai.api_key = os.environ.get("OPENAI_API_KEY")

@app.route('/ask', methods=['POST'])
def ask():
    print("✅ /ask 진입됨")

    try:
        data = request.get_json()
        print("📦 Raw data:", data)

        question = data.get("question", "")
        print("👉 받은 질문:", question)

        print("🚀 GPT 호출 시도")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 친절한 병원 AI입니다."},
                {"role": "user", "content": question}
            ]
        )
        answer = response['choices'][0]['message']['content']
        print("✅ GPT 응답:", answer)
    except Exception as e:
        print("❌ GPT 호출 에러:", e)
        answer = "탑탑이가 이해하기 어려운 질문입니다."
        
    image_url = None
    if any(keyword in question for keyword in ["원무과", "수납", "접수"]):
        image_url = "https://res.cloudinary.com/duvoimzkv/image/upload/v1747505265/toptop_admdepart_rm36ov.png"

    return jsonify({"answer": answer, "image": image_url})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
