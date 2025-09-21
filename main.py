# toptop_250923   # 250919 버전으로 롤백   # excel 내용 추가

from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import pandas as pd
import os
import re
import traceback
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

app = Flask(__name__)
CORS(app)

# --- 기본 설정 ---
openai.api_key = os.environ.get("OPENAI_API_KEY")
EMBEDDED_FILE_PATH = "toptop_with_embeddings.pkl"
EMBEDDING_MODEL = "text-embedding-ada-002"

# --- 하이브리드 검색 하이퍼파라미터(환경변수로 조정) ---
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.65"))          # 0~1, 임베딩 비중
TOPK = int(os.getenv("TOPK", "5"))                               # 재랭킹 후보 개수
HYBRID_THRESHOLD = float(os.getenv("HYBRID_THRESHOLD", "0.38"))  # 최종 컷오프

# --- 유틸리티 ---
def tokenize_kor(s: str):
    s = str(s)
    toks = s.split()
    return toks if toks else list(s)

def _normalize_text(x: str) -> str:
    x = str(x).lower()
    x = re.sub(r'\s+', '', x)
    x = re.sub(r'[^\w가-힣]', '', x)
    return x

# --- 데이터 로드 ---
try:
    df = pd.read_pickle(EMBEDDED_FILE_PATH)
    print(f"임베딩 데이터('{EMBEDDED_FILE_PATH}') 로드 완료.")
    # 정확일치용 정규화 키
    df['_질문키_norm'] = df['질문키'].astype(str).map(_normalize_text)

    question_keys = df['질문키'].astype(str).tolist()
    tokenized_corpus = [tokenize_kor(doc) for doc in question_keys]
    bm25 = BM25Okapi(tokenized_corpus)

    try:
        syn_df = pd.read_excel("toptop.xlsx", sheet_name="synonyms")
        syn_df['variant'] = syn_df['variant'].astype(str)
        syn_df['canonical_key'] = syn_df['canonical_key'].astype(str)
        syn_map = dict(zip(syn_df["variant"], syn_df["canonical_key"]))
    except Exception as e:
        print(f"synonyms 시트 로드 실패, 동의어 치환 비활성화: {e}")
        syn_map = {}

    print("BM25 준비 완료.")
except Exception as e:
    print(f"오류: 데이터/모델 준비 실패: {e}")
    df = None
    bm25 = None
    syn_map = {}

# --- 전처리 ---
def preprocess(query: str) -> str:
    if not query:
        return ""
    q = str(query)
    # 긴 variant부터 치환(부분일치 충돌 방지)
    try:
        for variant, key in sorted(syn_map.items(), key=lambda kv: len(kv[0]), reverse=True):
            if variant and key and variant in q:
                q = q.replace(variant, key)
    except Exception:
        pass
    return q.strip()

# --- 임베딩 ---
def get_embedding(text, model=EMBEDDING_MODEL):
    try:
        text = str(text).replace("\n", " ")
        # v1 스타일 API
        return openai.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        print(f"임베딩 오류: {e}")
        return None

# --- 검색 ---
def search_hybrid(user_question: str, dataframe: pd.DataFrame, bm25_model: BM25Okapi):
    if dataframe is None or bm25_model is None:
        return None

    # 0) 규칙 기반 정확일치(가장 안전)
    nq = _normalize_text(user_question)
    exact = dataframe.loc[dataframe['_질문키_norm'] == nq]
    if not exact.empty:
        return exact.iloc[0]["답변"]

    # 1) 의미론 점수
    question_embedding = get_embedding(user_question)
    if question_embedding is not None:
        try:
            all_embeddings = np.array(list(dataframe['임베딩']))
            semantic_scores = cosine_similarity([question_embedding], all_embeddings)[0]
        except Exception as e:
            print(f"코사인 유사도 오류: {e}")
            semantic_scores = np.zeros(len(dataframe), dtype=float)
    else:
        semantic_scores = np.zeros(len(dataframe), dtype=float)

    # 2) 키워드 점수(BM25)
    tokenized_query = tokenize_kor(user_question)
    try:
        keyword_scores = bm25_model.get_scores(tokenized_query)
    except Exception as e:
        print(f"BM25 점수 계산 오류: {e}")
        keyword_scores = np.zeros(len(dataframe), dtype=float)

    # 스케일 노멀라이즈(0~1)
    max_kw = np.max(keyword_scores) if len(keyword_scores) else 0.0
    if max_kw > 0:
        keyword_scores = keyword_scores / (max_kw + 1e-8)
    else:
        keyword_scores = np.zeros(len(dataframe), dtype=float)

    # 3) 하이브리드 스코어 및 TOP-K 재랭킹
    hybrid = (HYBRID_ALPHA * semantic_scores) + ((1 - HYBRID_ALPHA) * keyword_scores)
    if len(hybrid) == 0:
        return None

    top_idx = np.argsort(hybrid)[-TOPK:][::-1]
    top_df = dataframe.iloc[top_idx].copy()
    top_df["hybrid_score"] = hybrid[top_idx]
    top_df["semantic_score"] = semantic_scores[top_idx]
    top_df["keyword_score"] = keyword_scores[top_idx]

    # 4) 보수적 컷오프 + 키워드 구제 휴리스틱
    best = top_df.iloc[0]
    if best["hybrid_score"] >= HYBRID_THRESHOLD:
        return best["답변"]

    kw_best = top_df.sort_values("keyword_score", ascending=False).iloc[0]
    if kw_best["keyword_score"] >= 0.75:
        return kw_best["답변"]

    return None

# --- API ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "")
        history = data.get("history", [])

        if not question:
            return jsonify({"answer": "질문을 입력해 주세요 !", "image": None}), 400

        # 간단 인사 처리
        question_normalized = question.strip().lower().replace(" ", "")
        greetings = ["안녕", "하이", "ㅎㅇ", "hi", "hello"]
        if any(keyword in question_normalized for keyword in greetings):
            answer = "안녕하세요! 청주탑병원 AI 안내원 탑탑이입니다. 무엇을 도와드릴까요?"
            return jsonify({"answer": answer, "image": None})

        # 동의어 전처리
        search_query = preprocess(question)

        # 맥락 기반 재작성(있을 때만)
        try:
            if history:
                history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
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
                search_query = preprocess(contextual_response.choices[0].message.content.strip())
        except Exception as e:
            print(f"질의 재작성 실패: {e}")

        # 검색
        try:
            excel_info = search_hybrid(search_query, df, bm25)
        except Exception as e:
            print(f"검색 오류: {e}")
            excel_info = None

        # 응답 생성
        if not excel_info:
            answer = "죄송하지만 탑탑이가 모르는 내용이에요, 병원에 직접 문의해주세요!"
        else:
            system_prompt = (
                "너는 청주탑병원의 안내 도우미 '탑탑이'다. 주어진 '참고 자료'만 근거로 답하라. "
                "지어내지 말고, 한국어로 200자 이내로 간결하게."
            )

            final_user_content = f"""[참고 자료]:
"{excel_info}"

위 자료를 바탕으로 아래 [사용자 질문]에 답변해줘."""

            messages_for_generation = [{"role": "system", "content": system_prompt}]
            if isinstance(history, list):
                messages_for_generation.extend(history)
            messages_for_generation.append(
                {"role": "user", "content": f"{final_user_content}\n\n[사용자 질문]:\n{question}"}
            )

            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages_for_generation,
                    temperature=0.1
                )
                answer = response.choices[0].message.content
            except Exception as e:
                print(f"생성모델 오류: {e}")
                answer = str(excel_info)

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

# 헬스체크
@app.get("/")
def healthz():
    return "ok", 200

if __name__ == '__main__':
    # Render 호환: $PORT 우선 사용
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", "10000")))
