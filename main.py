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
    text = str(text).replace("\n", " ")
    return openai.embeddings.create(input=[text], model=model).data[0].embedding

def search_semantic(user_question, dataframe):
    if dataframe is None: return None
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
        history = data.get("history", []) # 대화 기록 수신

        if not question:
            return jsonify({"answer": "질문을 입력해 주세요 !", "image": None}), 400

        # --- 일상 대화 필터 ---
        question_normalized = question.strip().lower().replace(" ", "")
        # (greetings, thanks 등 목록 정의는 생략, 기존 코드와 동일)
        greetings = ["안녕", "하이", "ㅎㅇ", "hi", "hello"]
        if any(keyword in question_normalized for keyword in greetings):
            # ... (이하 모든 일상 대화 필터 로직은 기존과 동일)
            answer = "안녕하세요! 청주탑병원 AI 안내원 탑탑이입니다. 무엇을 도와드릴까요?"
            return jsonify({"answer": answer, "image": None})

        # --- 검색 및 답변 생성 로직 ---
        excel_info = None
        search_query = question

        try:
            # [수정됨] 대화 기록이 있을 경우, 맥락을 파악하여 검색어 재구성
            if history:
                print("대화 기록 감지. 맥락 파악을 시작합니다...")
                history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
                
                contextual_prompt = f"""You are an expert at understanding conversation context. Your task is to rewrite a user's new, potentially short question into a full, self-contained question based on the previous conversation history.

Focus on identifying the main subject (e.g., a person's name, a department, a topic) from the history and applying it to the new question.

[Previous Conversation History]:
{history_str}

[User's New Question]:
{question}

Rewrite the user's new question into one complete, clear question. For example, if the history is about "Director Lee Hyeong-jun" and the new question is "specialty," the rewritten question should be "What is Director Lee Hyeong-jun's specialty?".
"""
                contextual_response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": contextual_prompt}],
                    temperature=0.1 
                )
                search_query = contextual_response.choices[0].message.content.strip()
                print(f"맥락을 반영한 검색어: '{search_query}'")

            # 재구성된 질문(또는 원본 질문)으로 1차 검색 시도
            excel_info = search_semantic(search_query, df)

            # 1차 검색 실패 시, 질의 확장(기존 로직)으로 2차 시도
            if not excel_info:
                print(f"'{search_query}'로 1차 검색 실패. 질의 확장을 시도합니다...")
                # ... (이하 기존 질의 확장 로직은 그대로 유지) ...
                
        except Exception as e:
            print(f"검색 과정 중 오류 발생: {e}")
            excel_info = "QUERY_EXPANSION_ERROR"

        if not excel_info:
            answer = "죄송하지만 탑탑이가 모르는 내용이에요, 병원에 직접 문의해주세요!"
        elif excel_info == "QUERY_EXPANSION_ERROR": # [추가됨] 오류 메시지 케이스
            answer = "죄송하지만 질문을 조금 더 자세하게 해주세요!"
        else:
            # [수정됨] 최종 답변 생성 프롬프트
            system_prompt = """너는 청주탑병원의 안내 도우미 '탑탑이'다. 너의 임무는 주어진 '참고 자료'를 바탕으로 사용자의 질문에 답변하는 것이다. 참고 자료의 내용을 딱딱하게 그대로 읽어주지 말고, '탑탑이'의 역할에 맞게 친절하고 자연스러운 대화체로 정보를 재구성해서 설명해줘. 단, 참고 자료에 없는 사실을 지어내거나 추가해서는 절대로 안 된다. 답변은 항상 친절한 말투를 사용하며 한국어로 해야 하고, 200자 이내로 간결하게 요약해줘."""
            
            final_user_content = f"""너는 지금부터 병원 안내원 '탑탑이'야. 아래 [참고 자료]는 너의 지식이고, [사용자 질문]은 너에게 온 문의사항이야.
[참고 자료]를 바탕으로, '탑탑이'의 입장에서 친절하고 명확하게 답변해줘.

### 매우 중요한 규칙 ###
절대로 사용자를 질문 내용이나 참고 자료에 있는 특정 인물의 이름으로 부르지 마. 예를 들어, 질문이 '이형준'에 대한 것이라도 "이형준님, 안녕하세요"와 같이 답변을 시작하면 안 된다. 사용자는 항상 '환자분' 또는 '보호자분' 혹은 호칭 없이 대화해야 한다.

[참고 자료]:
"{excel_info}"

[사용자 질문]:
"{question}"
"""

            messages_for_generation = [
                {"role": "system", "content": system_prompt}
            ]
            messages_for_generation.extend(history)
            messages_for_generation.append({"role": "user", "content": final_user_content})

            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages_for_generation,
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
