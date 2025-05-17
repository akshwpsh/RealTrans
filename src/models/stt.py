#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
import torch
from faster_whisper import WhisperModel
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeechToText:
    """음성 인식 및 언어 식별 클래스 - Faster Whisper 기반"""
    
    def __init__(self, 
                 model_size="medium",
                 device="auto",
                 compute_type="auto",
                 download_root=None,
                 beam_size=5,
                 vad_filter=True,
                 vad_parameters=None):
        """STT 모델 초기화
        
        Args:
            model_size (str): 모델 크기 ('tiny', 'base', 'small', 'medium', 'large')
            device (str): 실행 장치 ('cpu', 'cuda', 'auto')
            compute_type (str): 계산 타입 ('default', 'auto', 'int8', 'int8_float16', 'int16', 'float16')
            download_root (str): 모델 다운로드 위치 (None: 기본 캐시 디렉토리)
            beam_size (int): 빔 검색 크기
            vad_filter (bool): VAD 필터 사용 여부
            vad_parameters (dict): VAD 매개변수 (None: 기본값)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.vad_parameters = vad_parameters or {}
        
        # 모델 로드
        self._load_model()
        
        # 한국어 언어 코드
        self.korean_code = "ko"
        
        # 지원하는 언어 목록 (주요 아시아 언어)
        self.supported_languages = {
            "ja": "일본어",
            "zh": "중국어 (표준)",
            "en": "영어",
            "ko": "한국어",
            "th": "태국어",
            "vi": "베트남어",
            "id": "인도네시아어",
            "ms": "말레이시아어",
            "tl": "타갈로그어",
            "hi": "힌디어",
            "ta": "타밀어"
        }
    
    def _load_model(self):
        """Faster Whisper 모델 로드"""
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root
            )
            
            logger.info(f"Faster Whisper '{self.model_size}' 모델을 로드했습니다. 장치: {self.device}, 계산 타입: {self.compute_type}")
        except Exception as e:
            logger.error(f"STT 모델 로드 중 오류 발생: {str(e)}")
            raise
    
    def transcribe(self, audio_data, sample_rate=16000, language=None, task="transcribe"):
        """오디오 데이터를 텍스트로 변환 (언어 감지 및 STT)
        
        Args:
            audio_data (numpy.ndarray): 오디오 데이터
            sample_rate (int): 샘플링 레이트
            language (str): 언어 코드 (None: 자동 감지)
            task (str): 작업 유형 ('transcribe', 'translate')
            
        Returns:
            dict: {
                'text': 변환된 텍스트,
                'language': 감지된 언어 코드,
                'language_name': 언어 이름,
                'is_korean': 한국어 여부,
                'segments': 세그먼트 목록
            }
        """
        try:
            # 오디오 데이터가 없으면 빈 결과 반환
            if audio_data is None or len(audio_data) == 0:
                return {
                    'text': '',
                    'language': None,
                    'language_name': None,
                    'is_korean': False,
                    'segments': []
                }
            
            # Numpy 배열로 변환
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)
            
            # 필요한 경우 데이터 정규화
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # 로깅 추가
            logger.info(f"STT 실행: language={language}, task={task}, 오디오 길이={len(audio_data)}, 최대 진폭={np.max(np.abs(audio_data)):.4f}")
            
            # Faster Whisper로 음성 인식 수행
            segments, info = self.model.transcribe(
                audio_data,
                beam_size=self.beam_size,
                language=language,
                task=task,
                vad_filter=self.vad_filter,
                vad_parameters=self.vad_parameters
            )
            
            # 세그먼트 목록 추출
            segments_list = list(segments)
            
            # 결과 텍스트 결합
            text = ' '.join([segment.text for segment in segments_list]).strip()
            
            # 언어 정보
            detected_language = info.language
            language_name = self.supported_languages.get(detected_language, "알 수 없음")
            is_korean = detected_language == self.korean_code
            
            # 결과 로깅
            logger.info(f"STT 결과: 감지 언어={detected_language}({language_name}), 텍스트 길이={len(text)}")
            logger.info(f"STT 텍스트: {text[:100]}{'...' if len(text) > 100 else ''}")
            
            return {
                'text': text,
                'language': detected_language,
                'language_name': language_name,
                'is_korean': is_korean,
                'segments': segments_list
            }
        
        except Exception as e:
            logger.error(f"음성 인식 중 오류 발생: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'text': '',
                'language': None,
                'language_name': None,
                'is_korean': False,
                'segments': [],
                'error': str(e)
            }
    
    def detect_language(self, audio_data, sample_rate=16000):
        """오디오 데이터에서 언어만 감지
        
        Args:
            audio_data (numpy.ndarray): 오디오 데이터
            sample_rate (int): 샘플링 레이트
            
        Returns:
            dict: {
                'language': 감지된 언어 코드,
                'language_name': 언어 이름,
                'is_korean': 한국어 여부
            }
        """
        try:
            # 오디오 데이터가 없으면 빈 결과 반환
            if audio_data is None or len(audio_data) == 0:
                return {
                    'language': None,
                    'language_name': None,
                    'is_korean': False
                }
            
            # 오디오 데이터 정보 로깅
            logger.info(f"언어 감지 시작: 오디오 길이={len(audio_data)}, 최대 진폭={np.max(np.abs(audio_data)):.4f}")
            
            # 언어 감지
            segments, info = self.model.transcribe(
                audio_data,
                language=None,
                beam_size=1,  # 속도를 위해 빔 크기 축소
                task="transcribe",
                vad_filter=self.vad_filter,
                word_timestamps=False
            )
            
            # 언어 정보
            detected_language = info.language
            language_name = self.supported_languages.get(detected_language, "알 수 없음")
            is_korean = detected_language == self.korean_code
            
            # 언어 감지 결과 로깅
            logger.info(f"언어 감지 결과: {detected_language} ({language_name})")
            
            return {
                'language': detected_language,
                'language_name': language_name,
                'is_korean': is_korean
            }
        
        except Exception as e:
            logger.error(f"언어 감지 중 오류 발생: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'language': None,
                'language_name': None,
                'is_korean': False,
                'error': str(e)
            } 