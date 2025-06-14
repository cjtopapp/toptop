# [코드 시작]
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import pandas as pd
import os
import traceback
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# --- 설정 부분 ---
# 1. OpenAI API 키 설정
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 2. 시맨틱 서치를 위한 설정
EMBEDDED_FILE_PATH = "toptop_with_embeddings.pkl" # 임베딩이 저장된 파일
EMBEDDING_MODEL = "text-embedding-3-small"      # OpenAI 임베딩 모델
SIMILARITY_THRESHOLD = 0.8  # 유사도 임계값 (0.8 이상일 때만 관련 정보로 판단)

# --- 데이터 로딩 ---
# 서버 시작 시, 미리 시맨틱 서치용 데이터를 로드합니다.
try:
    df = pd.read_pickle(EMBEDDED_FILE_PATH)
    print(f"시맨틱 서치용 데이터 파일('{EMBEDDED_FILE_PATH}')을 성공적으로 로드했습니다.")
except Exception as e:
    print(f"오류: 시맨틱 서치용 데이터 파일('{EMBEDDED_FILE_PATH}')을 로드할 수 없습니다.")
    print("솔루션: create_embeddings.py 스크립트를 먼저 실행하여 데이터 파일을 생성해주세요.")
    df = None

# --- 함수 정의 ---
def get_embedding(text, model=EMBEDDING_MODEL):
    """텍스트를 OpenAI 임베딩 모델을 사용해 벡터로 변환하는 함수"""
    text = str(text).replace("\n", " ")
    return openai.embeddings.create(input=[text], model=model).data[0].embedding

def search_semantic(user_question, dataframe):
    """시맨틱 서치를 통해 가장 유사한 답변을 찾는 함수"""
    if dataframe is None:
        return None

    # 1. 사용자 질문을 벡터로 변환
    question_embedding = get_embedding(user_question)

    # 2. 미리 저장된 모든 '질문키' 벡터와 코사인 유사도 계산
    all_embeddings = np.array(list(dataframe['임베딩']))
    similarities = cosine_similarity([question_embedding], all_embeddings)[0]

    # 3. 가장 높은 유사도를 가진 정보 찾기
    best_match_index = np.argmax(similarities)
    best_similarity = similarities[best_match_index]

    # 4. 유사도가 임계값 이상일 경우에만 해당 답변 반환
    if best_similarity >= SIMILARITY_THRESHOLD:
        return dataframe.iloc[best_match_index]["답변"]
    else:
        return None

# --- Flask 라우트 ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "")
        if not question:
            return jsonify({"answer": "질문을 입력해 주세요 !", "image": None}), 400

        # 6) 시맨틱 서치 기능으로 정보 검색
        excel_info = search_semantic(question, df)

        # 2) 정보가 없을 경우의 답변 처리
        if not excel_info:
            answer = "죄송하지만 탑탑이가 모르는 내용이에요, 병원에 직접 문의해주세요!"
        else:
            # 1), 3), 5) 피드백 반영
            system_prompt = """너는 청주탑병원의 안내 도우미 '탑탑이'다.
너의 임무는 주어진 '참고 자료'를 바탕으로만 질문에 답변하는 것이다.
- '참고 자료'에 질문에 대한 내용이 있으면, 해당 내용을 바탕으로 정확하게 답변해야 한다.
- '참고 자료'에 질문에 대한 내용이 없으면, 절대로 답변을 지어내지 말고 "죄송하지만 탑탑이가 모르는 내용이에요, 병원에 직접 문의해주세요!"라고만 대답해야 한다.
- 답변은 항상 친절한 말투를 사용하며 한국어로 해야 한다.
- 200자 이내로 대답하도록 한다."""

            # 참고 자료를 메시지에 포함
            final_user_content = f"""[참고 자료]
"{excel_info}"

위 참고 자료를 바탕으로 다음 질문에 답변해줘: {question}"""

            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_user_content}
                ],
                temperature=0.1  # 5) temperature 설정
            )
            answer = response.choices[0].message.content

    except Exception:
        traceback.print_exc()
        answer = "탑탑이가 대답할 수 없어요 !"
    
    # 4) 이미지 관련 로직은 그대로 유지
    image_url = None [cite: 5]
    try:
        if any(keyword in question for keyword in ["원무과", "수납", "접수"]):
            image_url = "https://res.cloudinary.com/duvoimzkv/image/upload/v1747505265/toptop_admdepart_rm36ov.png" [cite: 5]
    except Exception:
        pass

    return jsonify({"answer": answer, "image": image_url})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
# [코드 끝]
