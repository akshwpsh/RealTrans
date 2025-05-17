#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import google.generativeai as genai
from pathlib import Path
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Gemini API를 사용한 답변 생성 클래스"""
    
    def __init__(self, api_key=None, model_name="gemini-pro"):
        """Gemini API 초기화
        
        Args:
            api_key (str): Gemini API 키 (None: 환경 변수에서 가져옴)
            model_name (str): 사용할 모델 이름
        """
        self.model_name = model_name
        
        # API 키 설정
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("Gemini API 키가 설정되지 않았습니다. 설정 메뉴에서 API 키를 설정해 주세요.")
        
        # Gemini API 설정
        self._setup_gemini()
        
        # 언어 코드 매핑 (ISO 639-1 -> 언어 이름)
        self.language_names = {
            "ja": "일본어",
            "zh": "중국어",
            "en": "영어",
            "ko": "한국어",
            "th": "태국어",
            "vi": "베트남어",
            "id": "인도네시아어",
            "ms": "말레이시아어"
        }
    
    def _setup_gemini(self):
        """Gemini API 설정"""
        try:
            if self.api_key:
                logger.info(f"Gemini API 키 설정 중... (길이: {len(self.api_key)} 문자)")
                genai.configure(api_key=self.api_key)
                
                # 모델 생성
                self.model = genai.GenerativeModel(self.model_name)
                logger.info(f"Gemini API 모델 '{self.model_name}'이 준비되었습니다.")
                
                # 간단한 테스트 요청으로 API 키 확인
                try:
                    test_response = self.model.generate_content("안녕하세요")
                    logger.info("Gemini API 테스트 요청 성공!")
                    return True
                except Exception as e:
                    logger.error(f"Gemini API 테스트 요청 실패: {str(e)}")
                    self.model = None
                    return False
            else:
                self.model = None
                logger.warning("API 키가 없어 Gemini 모델이 초기화되지 않았습니다.")
                return False
                
        except Exception as e:
            logger.error(f"Gemini API 설정 중 오류 발생: {str(e)}")
            self.model = None
            return False
    
    def generate_response(self, original_text, translated_text, language_code, max_tokens=150):
        """원본 텍스트에 대한 번역 및 답변 생성
        
        Args:
            original_text (str): 원본 외국어 텍스트
            translated_text (str): 이전 번역된 텍스트 (이제 사용하지 않음, 호환성을 위해 유지)
            language_code (str): 원본 언어 코드 ('ja', 'zh', 'en' 등)
            max_tokens (int): 최대 토큰 수
            
        Returns:
            dict: {
                'response_text': 생성된 답변 텍스트 (한국어),
                'translated_response': 번역된 답변 (한국어 -> 외국어),
                'pronunciation': 발음 가이드 (한글 발음)
            }
        """
        if not self.model:
            return {
                'response_text': "Gemini API 키가 설정되지 않았습니다.",
                'translated_response': "",
                'pronunciation': ""
            }
        
        if not original_text:
            return {
                'response_text': "",
                'translated_response': "",
                'pronunciation': ""
            }
        
        try:
            # 언어 이름 가져오기
            language_name = self.language_names.get(language_code, "외국어")
            
            # 프롬프트 구성
            prompt = f"""
다음은 게임 중 외국어({language_name}) 대화입니다:
원문: {original_text}

다음 단계를 수행해주세요:
1. 위 원문을 한국어로 번역해주세요
2. 번역된 내용에 대한 짧고 적절한 답변을 한국어로 생성해주세요
3. 한국어 답변을 {language_name}로 번역해주세요
4. 발음 가이드를 한글로 제공해주세요

응답은 다음 형식으로 정확히 제공해주세요:
1. 번역: (원문을 한국어로 번역)
2. 한국어 답변: (게임 상황에 맞는 간결한 응답)
3. {language_name} 번역: (한국어 답변을 {language_name}로 번역)
4. 발음 가이드: (한글로 된 발음 가이드)

답변은 간결하게, 최대 2-3문장으로 제한해 주세요. 게임 상황에 어울리는 실용적인 응답이어야 합니다.
"""
            
            # 답변 생성
            start_time = time.time()
            response = self.model.generate_content(prompt)
            generation_time = time.time() - start_time
            
            logger.info(f"Gemini API 응답 생성 시간: {generation_time:.2f}초")
            
            if not response or not response.text:
                return {
                    'response_text': "응답 생성에 실패했습니다.",
                    'translated_response': "",
                    'pronunciation': ""
                }
            
            # 응답 텍스트 파싱
            response_text = response.text.strip()
            
            # 응답에서 번역, 한국어 답변, 번역, 발음 부분 추출
            gemini_translation = ""
            korean_response = ""
            translated_response = ""
            pronunciation = ""
            
            lines = response_text.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("1. 번역:"):
                    gemini_translation = line.replace("1. 번역:", "").strip()
                elif line.startswith("2. 한국어 답변:"):
                    korean_response = line.replace("2. 한국어 답변:", "").strip()
                elif line.startswith("3. ") and "번역:" in line:
                    translated_response = line.split("번역:", 1)[1].strip()
                elif line.startswith("4. 발음 가이드:"):
                    pronunciation = line.replace("4. 발음 가이드:", "").strip()
            
            return {
                'gemini_translation': gemini_translation,  # Gemini가 생성한 번역
                'response_text': korean_response,
                'translated_response': translated_response,
                'pronunciation': pronunciation
            }
            
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}")
            return {
                'gemini_translation': "",
                'response_text': f"응답 생성 중 오류가 발생했습니다: {str(e)}",
                'translated_response': "",
                'pronunciation': ""
            }
    
    def set_api_key(self, api_key):
        """API 키 설정
        
        Args:
            api_key (str): Gemini API 키
            
        Returns:
            bool: 성공 여부
        """
        if not api_key:
            logger.warning("ResponseGenerator.set_api_key: API 키가 비어 있습니다.")
            return False
        
        logger.info(f"ResponseGenerator.set_api_key: API 키 설정 시도 (길이: {len(api_key)} 문자)")
        
        # 이전 API 키와 동일하면 다시 설정할 필요 없음
        if self.api_key == api_key and self.model is not None:
            logger.info("ResponseGenerator.set_api_key: 이전과 동일한 API 키, 설정 유지")
            return True
        
        self.api_key = api_key
        # 현재 세션 환경 변수에도 설정 (임시 저장)
        os.environ["GEMINI_API_KEY"] = api_key
        
        # Gemini API 다시 설정
        setup_result = self._setup_gemini()
        logger.info(f"ResponseGenerator.set_api_key: API 설정 결과: {'성공' if setup_result else '실패'}")
        return self.model is not None 