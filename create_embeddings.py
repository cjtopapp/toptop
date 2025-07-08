# toptop_1.0.1

import pandas as pd # type: ignore
import openai # type: ignore
import os

# --- 설정 ---
# 1. OpenAI API 키 설정
# 터미널 환경 변수에 OPENAI_API_KEY가 설정되어 있어야 합니다.
# 예: export OPENAI_API_KEY='sk-...'
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. API 키를 설정해주세요.")
openai.api_key = api_key

# 2. 사용할 모델 및 파일 경로 지정
EMBEDDING_MODEL = "text-embedding-ada-002"
EXCEL_PATH = "toptop.xlsx"
OUTPUT_PATH = "toptop_with_embeddings.pkl" # 최종 결과물이 저장될 파일

# --- 함수 정의 ---
def get_embedding(text, model=EMBEDDING_MODEL):
    """주어진 텍스트를 OpenAI 임베딩 모델을 사용해 벡터로 변환하는 함수"""
    # API 요청 시 비어있는 문자열은 오류를 유발할 수 있으므로 처리
    text = str(text).replace("\n", " ")
    if not text.strip():
        return None
    try:
        # OpenAI API를 호출하여 임베딩 생성
        return openai.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        print(f"오류 발생: 텍스트 '{text}'의 임베딩 생성에 실패했습니다 - {e}")
        return None

# --- 메인 실행 로직 ---
if __name__ == '__main__':
    print(f"1. 원본 데이터 파일인 '{EXCEL_PATH}'을(를) 읽어옵니다.")
    try:
        df = pd.read_excel(EXCEL_PATH)
    except FileNotFoundError:
        print(f"[오류] '{EXCEL_PATH}' 파일을 찾을 수 없습니다. 스크립트와 같은 폴더에 있는지 확인해주세요.")
        exit()

    print("2. '질문키' 열의 각 항목에 대한 벡터 임베딩을 생성합니다. (데이터 양에 따라 시간이 소요될 수 있습니다)")
    
    # '질문키' 열의 모든 텍스트에 대해 get_embedding 함수를 적용하여 새로운 '임베딩' 열을 생성
    df['임베딩'] = df['질문키'].apply(get_embedding)

    # 임베딩 생성에 실패한 행(결과가 None인 경우)이 있는지 확인하고 제거
    failed_count = df['임베딩'].isnull().sum()
    if failed_count > 0:
        print(f"[경고] {failed_count}개의 항목에 대한 임베딩 생성에 실패했습니다. 해당 항목은 제외됩니다.")
        df.dropna(subset=['임베딩'], inplace=True)

    print(f"3. 생성된 임베딩 데이터를 '{OUTPUT_PATH}' 파일로 저장합니다.")
    
    # 최종 데이터프레임을 pickle 형태로 저장 (리스트 형태의 벡터 데이터를 안전하게 보존)
    df.to_pickle(OUTPUT_PATH)
    
    print("\n[성공] 작업이 완료되었습니다!")
    print(f"'{OUTPUT_PATH}' 파일이 생성되었으며, 이제 'main.py' 서버를 실행할 수 있습니다.")
