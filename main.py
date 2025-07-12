# toptop_1.0.1

from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import pandas as pd
import os
import traceback
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

app = Flask(__name__)
CORS(app)

openai.api_key = os.environ.get("OPENAI_API_KEY")
EMBEDDED_FILE_PATH = "toptop_with_embeddings.pkl"
EMBEDDING_MODEL = "text-embedding-ada-002"
SIMILARITY_THRESHOLD = 0.8

try:
    df = pd.read_pickle(EMBEDDED_FILE_PATH)
    print(f"시맨틱 서치용 데이터 파일('{EMBEDDED_FILE_PATH}')을 성공적으로 로드했습니다.")
    question_keys = df['질문키'].tolist()
    tokenized_corpus = [doc.split(" ") for doc in question_keys]
    bm25 = BM25Okapi(tokenized_corpus)
    syn_df = pd.read_excel("toptop.xlsx", sheet_name="synonyms")
    syn_map = dict(zip(syn_df["variant"], syn_df["canonical_key"]))    
    print("키워드 검색(BM25) 모델을 성공적으로 준비했습니다.")

except Exception as e:
    print(f"오류: 데이터 파일을 로드하거나 BM25 모델을 준비할 수 없습니다. 오류: {e}")
    df = None
    bm25 = None

def preprocess(query: str) -> str:
    for variant, key in syn_map.items():
        if variant in query:
            return key
    return query

def get_embedding(text, model=EMBEDDING_MODEL):
    text = str(text).replace("\n", " ")
    return openai.embeddings.create(input=[text], model=model).data[0].embedding

def search_hybrid(user_question, dataframe, bm25_model):
    if dataframe is None or bm25_model is None: return None
    
    question_embedding = get_embedding(user_question)
    all_embeddings = np.array(list(dataframe['임베딩']))
    semantic_scores = cosine_similarity([question_embedding], all_embeddings)[0]

    tokenized_query = user_question.split(" ")
    keyword_scores = bm25_model.get_scores(tokenized_query)
    normalized_keyword_scores = keyword_scores / (np.max(keyword_scores) + 1e-8)

    alpha = 0.5
    hybrid_scores = (alpha * semantic_scores) + ((1 - alpha) * normalized_keyword_scores)
    best_match_index = np.argmax(hybrid_scores)
    
    if hybrid_scores[best_match_index] < 0.4:   # tunning
        return None
    return dataframe.iloc[best_match_index]["답변"]

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "")
        history = data.get("history", [])

        if not question:
            return jsonify({"answer": "질문을 입력해 주세요 !", "image": None}), 400

        question_normalized = question.strip().lower().replace(" ", "")
        greetings = ["안녕", "하이", "ㅎㅇ", "hi", "hello"]
        if any(keyword in question_normalized for keyword in greetings):
            answer = "안녕하세요! 청주탑병원 AI 안내원 탑탑이입니다. 무엇을 도와드릴까요?"
            return jsonify({"answer": answer, "image": None})

        excel_info = None
        search_query = preprocess(question)

        try:
            if history:
                print("대화 기록이 있습니다. 맥락을 파악하여 질문을 재구성합니다...")
                history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
                search_query = preprocess(search_query)
                contextual_prompt = f"""You are a helpful assistant who understands conversation context. Based on the [Previous Conversation], rewrite the user's [New Question] into a single, complete, and self-contained question.

[Previous Conversation]:
{history_str}

[New Question]:
{question}
"""
                contextual_response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": contextual_prompt}],
                    temperature=0.2
                )
                search_query = contextual_response.choices[0].message.content.strip()
                search_query = preprocess(search_query)
                print(f"맥락을 반영한 검색어: '{search_query}'")

            excel_info = search_hybrid(search_query, df, bm25)
            
        except Exception as e:
            print(f"검색 또는 질의 확장 중 오류 발생: {e}")
            excel_info = None

        if not excel_info:
            answer = "죄송하지만 탑탑이가 모르는 내용이에요, 병원에 직접 문의해주세요!"
        else:
            system_prompt = """너는 청주탑병원의 안내 도우미 '탑탑이'다. 너의 임무는 주어진 '참고 자료'를 바탕으로 사용자의 질문에 답변하는 것이다. 참고 자료의 내용을 딱딱하게 그대로 읽어주지 말고, '탑탑이'의 역할에 맞게 친절하고 자연스러운 대화체로 정보를 재구성해서 설명해줘. 단, 참고 자료에 없는 사실을 지어내거나 추가해서는 절대로 안 된다. 답변은 항상 친절한 말투를 사용하며 한국어로 해야 하고, 200자 이내로 간결하게 요약해줘."""
            
            final_user_content = f"""[참고 자료]:
"{excel_info}"

위 참고 자료를 바탕으로, '탑탑이'의 입장에서 아래 [사용자 질문]에 대해 친절하고 명확하게 답변해줘."""

            messages_for_generation = [
                {"role": "system", "content": system_prompt}
            ]
            messages_for_generation.extend(history)
            messages_for_generation.append({"role": "user", "content": f"{final_user_content}\n\n[사용자 질문]:\n{question}"})

            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages_for_generation,
                temperature=0.1
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