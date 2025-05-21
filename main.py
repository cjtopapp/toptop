from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
import traceback

app = Flask(__name__)
CORS(app)

# Load API Key
api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = api_key

# Debugging output for API Key state
print("🔧 [DEBUG] OPENAI_API_KEY 존재 여부:", "✅ 있음" if api_key else "❌ 없음")
if api_key:
    print("🔧 [DEBUG] API Key 앞 5자리:", api_key[:5])

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        print("📩 [DEBUG] 수신된 JSON 데이터:", data)

        question = data.get("question", "")
        print("🧾 [DEBUG] 질문 내용:", question)

        if not question:
            print("⚠️ [WARNING] 질문이 비어 있습니다.")
            return jsonify({"answer": "질문이 비어 있습니다", "image": None}), 400

        if not api_key:
            print("❌ [ERROR] API 키가 설정되어 있지 않습니다.")
            return jsonify({"answer": "서버 오류: API 키가 누락되었습니다", "image": None}), 500

        # GPT 호출
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "넌 청주탑병원의 안내 도우미 '탑탑이' 란다. 환자의 질문에 간결하고 친절하게 한글로 답해줘. 100자 이내로 답변해주고, 답변은 200자를 넘지 않았으면 해."},
                {"role": "user", "content": question}
            ]
        )

        answer = response['choices'][0]['message']['content']
        print("✅ [DEBUG] GPT 응답:", answer)

    except Exception as e:
        print("❌ [EXCEPTION] 탑탑이 응답 중 오류 발생:")
        traceback.print_exc()
        answer = "탑탑이가 이해하기 어려운 질문입니다."

    image_url = None
    try:
        if any(keyword in question for keyword in ["원무과", "수납", "접수"]):
            image_url = "https://res.cloudinary.com/duvoimzkv/image/upload/v1747505265/toptop_admdepart_rm36ov.png"
            print("🖼️ [DEBUG] 관련 이미지 URL 설정됨:", image_url)
    except Exception as img_err:
        print("⚠️ [ERROR] 이미지 처리 중 오류:", img_err)

    return jsonify({"answer": answer, "image": image_url})

if __name__ == '__main__':
    print("🚀 [INFO] 서버 실행 중 (포트 10000)...")
    app.run(host='0.0.0.0', port=10000)
