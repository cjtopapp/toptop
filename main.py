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
openai.api_key = os.environ.get("OPENAI_API_KEY")
EMBEDDED_FILE_PATH = "toptop_with_embeddings.pkl"
EMBEDDING_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.8

# --- 데이터 로딩 ---
try:
    df = pd.read_pickle(EMBEDDED_FILE_PATH)
    print(f"시맨틱 서치용 데이터 파일('{EMBEDDED_FILE_PATH}')을 성공적으로 로드했습니다.")
except Exception as e:
    print(f"오류: 시맨틱 서치용 데이터 파일('{EMBEDDED_FILE_PATH}')을 로드할 수 없습니다.")
    print(f"솔루션: create_embeddings.py 스크립트를 먼저 실행하여 데이터 파일을 생성해주세요. 오류: {e}")
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

    question_embedding = get_embedding(user_question)
    all_embeddings = np.array(list(dataframe['임베딩']))
    similarities = cosine_similarity([question_embedding], all_embeddings)[0]
    best_match_index = np.argmax(similarities)
    best_similarity = similarities[best_match_index]
    
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

        question_normalized = question.strip().lower().replace(" ", "")
        question_for_search = question.replace(" ", "")

        greetings = ["안녕", "하이", "ㅎㅇ", "hi", "hello"]
        thanks = ["고마워", "감사합니다", "땡큐", "thank"]
        farewells = ["잘가", "바이", "수고했어", "수고하셨습니다", "안녕히가세요"]
        identity = ["누구야", "넌누구니", "이름이뭐야", "뭐하는애야", "정체가뭐야"]
        compliments = ["잘한다", "똑똑하다", "최고야", "좋았어", "훌륭해"]
        # 필요에 따라 부적절한 단어 목록을 추가할 수 있습니다.
        profanity = ["바보", "멍청이"] 

        # 키워드 포함 여부 확인 및 즉시 응답
        if any(keyword in question_normalized for keyword in greetings):
            answer = "안녕하세요! 청주탑병원 AI 안내원 탑탑이입니다. 무엇을 도와드릴까요?"
            return jsonify({"answer": answer, "image": None})

        if any(keyword in question_normalized for keyword in thanks):
            answer = "천만에요! 더 궁금한 점이 있으시면 언제든지 물어보세요."
            return jsonify({"answer": answer, "image": None})

        if any(keyword in question_normalized for keyword in farewells):
            answer = "네, 안녕히 가세요. 추가적으로 궁금한 점이 생기시면 언제든지 다시 찾아주세요!"
            return jsonify({"answer": answer, "image": None})
            
        if any(keyword in question_normalized for keyword in identity):
            answer = "저는 청주탑병원의 궁금한 점을 해결해드리는 AI 안내원 탑탑이입니다."
            return jsonify({"answer": answer, "image": None})

        if any(keyword in question_normalized for keyword in compliments):
            answer = "칭찬해주셔서 감사합니다! 더 도움이 될 수 있도록 노력할게요."
            return jsonify({"answer": answer, "image": None})

        if any(keyword in question_normalized for keyword in profanity):
            answer = "바르고 고운 말을 사용해주세요. 도움이 필요하시면 다시 질문해주시기 바랍니다."
            return jsonify({"answer": answer, "image": None})
       
        # --- 질의 확장(Query Expansion) 로직 시작 ---
        excel_info = None
        try:
            # 1. 먼저 사용자의 원본 질문으로 검색 시도
            excel_info = search_semantic(question_for_search, df)

            # 2. 원본 질문으로 못 찾았을 경우, GPT를 이용해 질문 확장
            if not excel_info:
                print("1차 검색 실패. 질의 확장을 시도합니다...")
                expansion_prompt = f"""너는 검색어 확장 전문가야. 사용자의 질문 '{question}'을 받아서, 우리 병원 데이터베이스에서 검색하기 좋은, 의미적으로 유사한 질문 3개를 목록으로 만들어줘. 각 질문은 줄바꿈으로 구분해줘.
예시: '이형준' -> '이형준 원장님의 전문 진료 분야는 무엇인가요?\n이형준 원장님의 약력이 궁금합니다.\n이형준 원장님 진료 시간 알려주세요.'"""

                expansion_response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": expansion_prompt}],
                    temperature=0.5
                )
                expanded_questions = expansion_response.choices[0].message.content.strip().split('\n')
                
                print(f"확장된 질문: {expanded_questions}")

                # 3. 확장된 질문들로 다시 검색 시도
                for q in expanded_questions:
                    excel_info = search_semantic(q, df)
                    if excel_info:
                        print(f"'{q}' 질문으로 검색 성공!")
                        break # 정보를 찾으면 루프 중단
        except Exception as e:
            print(f"질의 확장 또는 검색 중 오류 발생: {e}")
            excel_info = None # 오류 발생 시 excel_info를 None으로 초기화
        # --- 질의 확장 로직 끝 ---

        if not excel_info:
            answer = "죄송하지만 탑탑이가 모르는 내용이에요, 병원에 직접 문의해주세요!"
        else:
            # (이하 답변 생성 부분은 위의 3번에서 수정한 프롬프트로 동일하게 적용)
            system_prompt = """너는 청주탑병원의 안내 도우미 '탑탑이'다.
너의 임무는 주어진 '참고 자료'를 바탕으로 사용자의 질문에 답변하는 것이다.
참고 자료의 내용을 딱딱하게 그대로 읽어주지 말고, '탑탑이'의 역할에 맞게 친절하고 자연스러운 대화체로 정보를 재구성해서 설명해줘.
단, 참고 자료에 없는 사실을 지어내거나 추가해서는 절대로 안 된다.
답변은 항상 친절한 말투를 사용하며 한국어로 해야 하고, 200자 이내로 간결하게 요약해줘."""

            final_user_content = f"""너는 지금부터 병원 안내원 '탑탑이'야. 아래 [참고 자료]는 너의 지식이고, [사용자 질문]은 너에게 온 문의사항이야.
[참고 자료]를 바탕으로, '탑탑이'의 입장에서 친절하고 명확하게 답변해줘.

[참고 자료]:
"{excel_info}"

[사용자 질문]:
"{question}"
"""

            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_user_content}
                ],
                temperature=0.1
            )
            answer = response.choices[0].message.content

    except Exception:
        traceback.print_exc()
        answer = "탑탑이가 대답할 수 없어요 !"
    
    # (이미지 처리 로직은 그대로 유지)
    image_url = None
    try:
        if any(keyword in question for keyword in ["원무과", "수납", "접수"]):
            image_url = "https://res.cloudinary.com/duvoimzkv/image/upload/v1747505265/toptop_admdepart_rm36ov.png"
    except Exception:
        pass

    return jsonify({"answer": answer, "image": image_url})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
