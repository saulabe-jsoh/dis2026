import pandas as pd
import numpy as np
import re
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sdv.tabular import CTGAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

# ==========================================
# 1. 하이브리드 비식별화 (Regex + NER)
# ==========================================
# BERT 기반 NER 모델 로드 (HuggingFace)
ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

def hybrid_deidentification(text):
    # Step 1: 정규표현식 (Regex) - IP 및 이메일 형태 탐지
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP_REDACTED]', text)
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}', '[USER_REDACTED]', text)
    
    # Step 2: NER (Named Entity Recognition) - 미탐지된 특수 개체명(서버명, 조직) 탐지
    ner_results = ner_model(text)
    for entity in ner_results:
        if entity['entity'] in ['I-PER', 'I-ORG', 'B-ORG', 'I-LOC']:
            text = text.replace(entity['word'], '[NER_REDACTED]')
    return text

# ==========================================
# 2. Llama-3 8B 기반 합성데이터 생성 함수
# ==========================================
def generate_synthetic_logs_llama3(prompt_examples, num_samples=1000):
    # 온프레미스(폐쇄망) 환경 Llama-3 모델 로드 
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    
    system_prompt = "You are a defense cybersecurity AI generating de-identified synthetic Red Team logs for N2SF."
    user_prompt = f"Maintain the logical sequence of an attack based on the examples and generate {num_samples} JSON records.\n\nExamples:\n{prompt_examples}"
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    
    outputs = model.generate(input_ids, max_new_tokens=2048, temperature=0.7, top_p=0.9, do_sample=True)
    generated_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    
    return generated_text

# ==========================================
# 3. TSTR 성능 평가 (XGBoost Classifier)
# ==========================================
def run_tstr_evaluation(X_train, y_train, X_test_real, y_test_real, model_name):
    # 레드팀 공격 데이터 불균형 가중치 보정
    scale_weight = sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) > 0 else 1
    
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=scale_weight, random_state=42)
    xgb.fit(X_train, y_train)
    
    y_pred = xgb.predict(X_test_real)
    f1 = f1_score(y_test_real, y_pred)
    
    print(f"[{model_name}] TSTR F1-Score: {f1:.4f}")
    return f1

# 데이터셋 분리 및 모델 비교 실행 예시
# f1_baseline = run_tstr_evaluation(X_train_real, y_train_real, X_test_real, y_test_real, "TRTR (Real Baseline)")
# f1_sdv = run_tstr_evaluation(X_train_sdv, y_train_sdv, X_test_real, y_test_real, "TSTR (SDV-CTGAN)")
# f1_llama = run_tstr_evaluation(X_train_llama, y_train_llama, X_test_real, y_test_real, "TSTR (Llama-3 sLLM)")
# print(f"SDV 보존율: {(f1_sdv/f1_baseline)*100:.2f}%, Llama-3 보존율: {(f1_llama/f1_baseline)*100:.2f}%")
