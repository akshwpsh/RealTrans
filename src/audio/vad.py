#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import os
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoiceActivityDetector:
    """음성 활성 감지(VAD) 클래스 - Silero VAD 사용"""
    
    def __init__(self, 
                 sample_rate=16000, 
                 model_path=None,
                 threshold=0.2,
                 min_speech_duration_ms=100,
                 min_silence_duration_ms=400):
        """VAD 초기화
        
        Args:
            sample_rate (int): 오디오 샘플링 레이트 (Hz)
            model_path (str): 사전 다운로드된 모델 경로 (None이면 자동 다운로드)
            threshold (float): 음성 감지 임계값 (0.0~1.0)
            min_speech_duration_ms (int): 최소 음성 지속 시간 (ms)
            min_silence_duration_ms (int): 최소 무음 지속 시간 (ms)
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        
        # 추가 VAD 매개변수
        self.speech_pad_ms = 50  # 음성 구간 앞뒤 패딩 (ms)
        self.window_size_samples = 512  # VAD 윈도우 크기
        
        # GPU 사용 가능 여부 확인
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"VAD 모델 실행 장치: {self.device}")
        
        # 모델 로드
        self._load_model(model_path)
    
    def _load_model(self, model_path=None):
        """Silero VAD 모델 로드
        
        Args:
            model_path (str): 모델 경로 (None이면 자동 다운로드)
        """
        try:
            # 로컬 모델이 없으면 Torch Hub에서 자동 다운로드
            # 먼저 선택적 매개변수 설정
            torch.hub.set_dir(str(Path.home() / ".cache" / "torch" / "hub"))
            
            # Torch Hub에서 Silero VAD 모델 다운로드
            logger.info("Silero VAD 모델 로드 중...")
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                verbose=False
            )
            
            self.model = model.to(self.device)
            self.get_speech_timestamps = utils[0]
            self.get_speech_ts_stream = utils[2]
            self.collect_chunks = utils[4]
            
            logger.info(f"Silero VAD 모델을 로드했습니다. 장치: {self.device}")
                
        except Exception as e:
            logger.error(f"VAD 모델 로드 중 오류 발생: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def detect_speech(self, audio_chunk):
        """오디오 청크에서 음성 구간 감지
        
        Args:
            audio_chunk (numpy.ndarray): 오디오 데이터 (형태: [samples])
            
        Returns:
            list: 음성 구간 타임스탬프 목록 (start, end)
            numpy.ndarray: 음성 구간만 추출한 오디오 (없으면 None)
        """
        try:
            # Numpy 배열을 PyTorch 텐서로 변환
            if isinstance(audio_chunk, np.ndarray):
                # 필요한 경우 값 범위 조정 (-1.0~1.0)
                max_amplitude = np.max(np.abs(audio_chunk))
                logger.info(f"VAD 입력 오디오 최대 진폭: {max_amplitude:.6f}, 길이: {len(audio_chunk)}")
                
                if max_amplitude > 1.0:
                    audio_chunk = audio_chunk / max_amplitude
                
                tensor = torch.from_numpy(audio_chunk).float()
            else:
                tensor = audio_chunk
                
            # VAD 모델에 입력하기 위해 텐서 차원 확인 및 조정
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)  # [samples] -> [1, samples]
            
            # 장치 이동
            tensor = tensor.to(self.device)
            
            # Silero VAD로 음성 타임스탬프 추출
            speech_timestamps = self.get_speech_timestamps(
                tensor,
                self.model,
                threshold=self.threshold,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms,
                window_size_samples=self.window_size_samples
            )
            
            # 결과 로깅
            if speech_timestamps:
                logger.info(f"VAD 결과: {len(speech_timestamps)}개 음성 구간 감지 - 첫 구간: {speech_timestamps[0]}")
                for i, ts in enumerate(speech_timestamps):
                    logger.info(f"  구간 {i+1}: 시작={ts['start']}, 종료={ts['end']}, 길이={ts['end']-ts['start']} 샘플")
            else:
                logger.warning(f"VAD 결과: 음성 구간 감지 실패 (임계값: {self.threshold}, 입력 길이: {len(audio_chunk)})")
            
            # 결과가 없으면 빈 배열과 None 반환
            if not speech_timestamps:
                return [], None
            
            # 타임스탬프로 음성 구간만 추출 (collect_chunks 대신 직접 구현)
            # CPU로 텐서 이동 및 차원 축소
            audio_np = tensor.cpu().numpy().squeeze()
            
            # 모든 음성 구간 병합
            all_speech_segments = []
            total_speech_samples = 0
            
            logger.info(f"오디오 구간 추출 시작: 전체 오디오 길이={len(audio_np)}, 감지된 구간={len(speech_timestamps)}개")
            
            for i, ts in enumerate(speech_timestamps):
                start_idx = max(0, ts['start'])
                end_idx = min(len(audio_np), ts['end'])
                
                if start_idx >= end_idx or start_idx >= len(audio_np):
                    logger.warning(f"  구간 {i+1} 추출 실패: 유효하지 않은 인덱스 (시작={start_idx}, 종료={end_idx}, 전체 길이={len(audio_np)})")
                    continue
                
                segment = audio_np[start_idx:end_idx]
                segment_length = len(segment)
                total_speech_samples += segment_length
                
                if segment_length > 0:
                    all_speech_segments.append(segment)
                    logger.info(f"  구간 {i+1} 추출 성공: 길이={segment_length}, 최대 진폭={np.max(np.abs(segment)):.4f}")
                else:
                    logger.warning(f"  구간 {i+1} 추출 실패: 빈 구간")
            
            # 음성 구간이 없으면 처리 중단
            if not all_speech_segments:
                logger.warning("추출할 음성 구간이 없습니다.")
                return speech_timestamps, None
            
            # 모든 음성 구간 병합
            speech_np = np.concatenate(all_speech_segments)
            
            # 추출 결과 검증
            if speech_np.size == 0:
                logger.warning("추출된 음성 구간이 비어 있습니다 (크기: 0)")
                return speech_timestamps, None
            
            # 안전하게 최대 진폭 계산 (빈 배열인 경우 처리)
            max_amplitude = np.max(np.abs(speech_np))
            logger.info(f"추출된 음성 구간 길이: {len(speech_np)} 샘플, 최대 진폭: {max_amplitude:.6f}")
            
            # 텐서를 Numpy 배열로 변환하여 반환
            return speech_timestamps, speech_np
            
        except Exception as e:
            logger.error(f"음성 감지 중 오류 발생: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return [], None
    
    def has_speech(self, audio_chunk):
        """오디오 청크에 음성이 포함되어 있는지 빠르게 확인
        
        Args:
            audio_chunk (numpy.ndarray): 오디오 데이터
            
        Returns:
            bool: 음성 포함 여부
        """
        try:
            # 직접 감지 진행 (detect_speech 함수 재사용하지 않고)
            # Numpy 배열을 PyTorch 텐서로 변환
            if isinstance(audio_chunk, np.ndarray):
                # 필요한 경우 값 범위 조정 (-1.0~1.0)
                if np.max(np.abs(audio_chunk)) > 1.0:
                    audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
                
                tensor = torch.from_numpy(audio_chunk).float()
            else:
                tensor = audio_chunk
                
            # VAD 모델에 입력하기 위해 텐서 차원 확인 및 조정
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)  # [samples] -> [1, samples]
            
            # 장치 이동
            tensor = tensor.to(self.device)
            
            # Silero VAD로 음성 타임스탬프 추출 - 빠른 감지를 위해 최소값 설정
            speech_timestamps = self.get_speech_timestamps(
                tensor,
                self.model,
                threshold=self.threshold,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=30,  # 더 짧은 음성도 빠르게 감지
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=10,  # 패딩 축소
                window_size_samples=512
            )
            
            return len(speech_timestamps) > 0
        except Exception as e:
            logger.error(f"음성 포함 여부 확인 중 오류 발생: {str(e)}")
            return False 