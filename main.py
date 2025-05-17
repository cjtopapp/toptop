from flask import Flask, request, jsonify
import openai
import os

app = Flask(__name__)
openai.api_key = os.environ.get("OPENAI_API_KEY")

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"answer": "질문이 비어 있습니다.", "image": None}), 400

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "넌 청주탑병원의 안내 도우미 '탑탑이' 란다. 환자의 질문에 간결하고 친절하게 한글로 답해줘. 100자 이내로 답변해주고, 답변은 200자를 넘지 않았으면 해."},
            {"role": "user", "content": question}
        ]
    )

    answer = response['choices'][0]['message']['content']

# cloudinary

    image_url = None

    if any(keyword in question for keyword in ["원무과", "수납", "접수"]):
        image_url = "https://res.cloudinary.com/duvoimzkv/image/upload/v1747505265/toptop_admdepart_rm36ov.png"

    return jsonify({"answer": answer, "image": image_url})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
