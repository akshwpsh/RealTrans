#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import soundcard as sc
import sounddevice as sd
from threading import Thread
import time
from queue import Queue
import logging
import wave
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioCapture:
    """시스템 오디오 출력 캡처 클래스"""
    
    def __init__(self, 
                 sample_rate=16000, 
                 chunk_size=16000,  # 약 1초 분량의 오디오
                 buffer_size=20,    # 약 20초 분량의 버퍼
                 overlap_ratio=0.0): # 청크 간 겹치는 비율 (0.0~1.0) - 겹침 제거를 위해 기본값 0.0으로 변경
        """오디오 캡처 클래스 초기화
        
        Args:
            sample_rate (int): 샘플링 레이트 (Hz)
            chunk_size (int): 한 번에 처리할 샘플 수
            buffer_size (int): 버퍼에 저장할 최대 청크 수
            overlap_ratio (float): 청크 간 겹치는 비율 (0.0~1.0)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.overlap_ratio = 0.0  # 겹침 비율을 0으로 고정하여 겹침 제거
        self.overlap_samples = 0  # 겹치는 샘플 수를 0으로 설정
        self.audio_queue = Queue(maxsize=buffer_size)
        self.running = False
        self.capture_thread = None
        
        # 이전 청크 데이터 저장용 버퍼 (겹침 제거로 불필요)
        self.previous_data = None
        
        # 무음 필터링 설정
        self.noise_threshold = 0.003  # 0.3% 이하의 진폭은 무음으로 간주 (기존 0.005에서 낮춤)
        self.min_activity_ratio = 0.05  # 전체 오디오 중 최소 5% 이상이 활성화되어야 함 (기존 0.1에서 낮춤)
          # 디버깅용 변수 추가
        self.save_debug_chunks = False  # 기본적으로 디버깅 파일 저장 비활성화
        self.debug_chunks_saved = 0
        self.max_debug_chunks = 10
        self.debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "debug_audio")
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
        
        # 사용 가능한 오디오 출력 장치 확인
        self.output_devices = sc.all_speakers()
        logger.info(f"사용 가능한 출력 장치: {self.output_devices}")
        
        # 기본 출력 장치 설정
        if self.output_devices:
            self.current_device = self.output_devices[0]
        else:
            self.current_device = None
            logger.error("사용 가능한 출력 장치가 없습니다.")
    
    def start_capture(self):
        """오디오 캡처 시작"""
        if self.running:
            logger.warning("오디오 캡처가 이미 실행 중입니다.")
            return
        
        if not self.current_device:
            logger.error("사용 가능한 출력 장치가 없어 캡처를 시작할 수 없습니다.")
            return
        
        self.running = True
        self.capture_thread = Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        logger.info("오디오 캡처를 시작했습니다.")
    
    def stop_capture(self):
        """오디오 캡처 중지"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
            self.capture_thread = None
        logger.info("오디오 캡처를 중지했습니다.")
    
    def get_devices(self):
        """사용 가능한 모든 오디오 출력 장치 반환"""
        return self.output_devices
    
    def set_device(self, device_index):
        """사용할 오디오 출력 장치 설정
        
        Args:
            device_index (int): 출력 장치 인덱스
        """
        if 0 <= device_index < len(self.output_devices):
            self.current_device = self.output_devices[device_index]
            logger.info(f"오디오 출력 장치를 변경했습니다: {self.current_device.name}")
            
            # 이미 실행 중이면 재시작
            if self.running:
                self.stop_capture()
                self.start_capture()
        else:
            logger.error(f"유효하지 않은 장치 인덱스입니다: {device_index}")
    
    def get_audio_chunk(self, timeout=1.0):
        """오디오 큐에서 다음 청크를 가져옴
        
        Args:
            timeout (float): 대기 시간 (초)
            
        Returns:
            numpy.ndarray: 오디오 데이터 (None: 타임아웃 또는 비어있음)
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except:
            return None
    
    def _is_silent(self, audio_data):
        """오디오 데이터가 무음인지 확인
        
        Args:
            audio_data (numpy.ndarray): 오디오 데이터
            
        Returns:
            bool: 무음 여부
        """
        if audio_data is None or len(audio_data) == 0:
            return True
        
        # 최대 진폭이 임계값보다 작으면 무음으로 간주
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude < self.noise_threshold:
            return True
        
        # 임계값 이상의 활성 샘플 비율이 너무 적으면 무음으로 간주
        activity_ratio = np.mean(np.abs(audio_data) > self.noise_threshold)
        if activity_ratio < self.min_activity_ratio:
            return True
        
        return False
    
    def _capture_loop(self):
        """오디오 캡처 쓰레드 메인 루프"""
        try:
            # with 문 대신 직접 마이크 객체 생성
            mic = sc.get_microphone(id=str(self.current_device.id), include_loopback=True)
            logger.info(f"마이크 객체 생성 성공: {mic}")
            
            # 오버랩을 고려한 실제 캡처 크기 계산 (겹침 제거로 전체 크기 캡처)
            capture_size = self.chunk_size
            
            while self.running:
                try:
                    # 오디오 캡처 (스테레오 -> 모노로 변환)
                    data = mic.record(capture_size, self.sample_rate)
                    
                    # 데이터 유효성 검사
                    if data is None or len(data) == 0:
                        logger.warning("오디오 데이터가 비어 있습니다.")
                        time.sleep(0.1)  # 잠시 대기 후 재시도
                        continue
                    
                    # 스테레오인 경우 모노로 변환
                    if data.ndim > 1 and data.shape[1] > 1:
                        data = np.mean(data, axis=1)
                    
                    # 데이터 정규화
                    if np.max(np.abs(data)) > 1.0:
                        data = data / np.max(np.abs(data))
                    
                    # 겹침 제거: 이전 데이터와 결합하지 않고 현재 데이터만 사용
                    combined_data = data
                    
                    # 겹침 제거: 이전 청크를 위한 데이터 저장하지 않음
                    self.previous_data = None
                    
                    # 무음 필터링 - 무음이면 처리하지 않음
                    if self._is_silent(combined_data):
                        logger.debug("무음 감지: 오디오 청크 무시")
                        continue
                    
                    # 디버깅용 오디오 청크 저장
                    if self.save_debug_chunks and self.debug_chunks_saved < self.max_debug_chunks:
                        self._save_debug_chunk(combined_data)
                        self.debug_chunks_saved += 1
                        logger.info(f"디버깅용 오디오 청크 저장: {self.debug_chunks_saved}/{self.max_debug_chunks} (활성 비율: {np.mean(np.abs(combined_data) > self.noise_threshold):.4f}, 길이: {len(combined_data)})")
                    
                    # 디버깅 (오디오 활성 확인)
                    if np.max(np.abs(combined_data)) > 0.01:
                        logger.debug(f"오디오 신호 감지: 최대 진폭 {np.max(np.abs(combined_data)):.4f}, 활성 비율 {np.mean(np.abs(combined_data) > self.noise_threshold):.4f}")
                    
                    # 오디오 버퍼에 추가 (큐가 가득 차면 가장 오래된 데이터 버림)
                    if self.audio_queue.full():
                        try:
                            self.audio_queue.get_nowait()
                        except:
                            pass
                    
                    try:
                        self.audio_queue.put_nowait(combined_data)
                    except:
                        pass
                    
                except Exception as e:
                    logger.error(f"오디오 캡처 루프 내 오류 발생: {str(e)}")
                    time.sleep(0.5)  # 오류 발생 시 잠시 대기
            
        except Exception as e:
            logger.error(f"오디오 캡처 쓰레드 오류 발생: {str(e)}")
            self.running = False
    
    def _save_debug_chunk(self, data):
        """디버깅용 오디오 청크 저장
        
        Args:
            data (numpy.ndarray): 오디오 데이터
        """
        try:
            # float32 데이터를 int16로 변환 (WAV 파일 형식)
            scaled_data = np.int16(data * 32767)
            
            # WAV 파일로 저장
            filename = os.path.join(self.debug_dir, f"audio_chunk_{self.debug_chunks_saved}.wav")
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)  # 모노
                wf.setsampwidth(2)  # 16비트
                wf.setframerate(self.sample_rate)
                wf.writeframes(scaled_data.tobytes())
            
            # 최대 진폭 정보도 기록
            info_file = os.path.join(self.debug_dir, f"audio_chunk_{self.debug_chunks_saved}.txt")
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"최대 진폭: {np.max(np.abs(data)):.6f}\n")
                f.write(f"RMS 값: {np.sqrt(np.mean(data**2)):.6f}\n")
                f.write(f"샘플 수: {len(data)}\n")
                f.write(f"0.01 이상 샘플 비율: {np.mean(np.abs(data) > 0.01):.6f}\n")
            
        except Exception as e:
            logger.error(f"오디오 청크 저장 중 오류 발생: {str(e)}") 