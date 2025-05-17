#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
from pathlib import Path
import threading
from PySide6.QtCore import QObject, Signal, Qt

from src.utils.pipeline import ProcessingPipeline

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UIController(QObject):
    """UI 컨트롤러 클래스 - UI와 백엔드 처리 파이프라인 연결"""
    
    # 시그널 정의
    translationUpdated = Signal(dict)  # 번역 결과 업데이트 시그널
    statusUpdated = Signal(str)  # 상태 메시지 업데이트 시그널
    audioDevicesUpdated = Signal(list)  # 오디오 장치 목록 업데이트 시그널
    
    def __init__(self, config_path=None):
        """UI 컨트롤러 초기화
        
        Args:
            config_path (str): 설정 파일 경로 (None: 기본 설정 사용)
        """
        super().__init__()
        
        # 설정 경로 로깅
        logger.info(f"UIController 초기화 - 설정 파일 경로: {config_path}")
        
        # 설정 파일 경로 확인 및 변환
        if config_path:
            # 상대 경로를 절대 경로로 변환
            config_path_obj = Path(config_path)
            if not config_path_obj.is_absolute():
                config_path_obj = config_path_obj.absolute()
            config_path = str(config_path_obj)
            logger.info(f"절대 경로로 변환된 설정 파일 경로: {config_path}")
        
        # 처리 파이프라인 초기화 (설정 파일 경로 직접 전달)
        if config_path and Path(config_path).exists():
            # 설정 파일이 있으면 넘겨서 초기화
            logger.info(f"설정 파일로 파이프라인 초기화: {config_path}")
            self.pipeline = ProcessingPipeline(config_path=config_path)
        else:
            # 설정 파일이 없으면 기본 설정으로 초기화
            logger.info("기본 설정으로 파이프라인 초기화")
            self.pipeline = ProcessingPipeline()
        
        # 응답 생성기에 대한 참조 (쉬운 접근용)
        self.response_generator = self.pipeline.response_generator
        
        # 상태 변수
        self.running = False
        self.result_polling_thread = None
        self.polling_interval = 0.1  # 초
    
    def start_translation(self):
        """번역 처리 시작"""
        if self.running:
            logger.warning("번역 처리가 이미 실행 중입니다.")
            return False
        
        # 처리 파이프라인 시작
        try:
            self.pipeline.start()
            
            # 결과 폴링 스레드 시작
            self.running = True
            self.result_polling_thread = threading.Thread(target=self._result_polling_loop)
            self.result_polling_thread.daemon = True
            self.result_polling_thread.start()
            
            # 상태 업데이트
            self.statusUpdated.emit("번역 중...")
            logger.info("번역 처리를 시작했습니다.")
            
            return True
        except Exception as e:
            logger.error(f"번역 처리 시작 중 오류 발생: {str(e)}")
            self.statusUpdated.emit(f"오류: {str(e)}")
            return False
    
    def stop_translation(self):
        """번역 처리 중지"""
        if not self.running:
            logger.warning("번역 처리가 실행 중이 아닙니다.")
            return False
        
        # 플래그 변경 (스레드 종료 신호)
        self.running = False
        
        # 처리 파이프라인 중지
        try:
            self.pipeline.stop()
            
            # 결과 폴링 스레드 종료 대기
            if self.result_polling_thread:
                self.result_polling_thread.join(timeout=1.0)
                self.result_polling_thread = None
            
            # 상태 업데이트
            self.statusUpdated.emit("준비됨")
            logger.info("번역 처리를 중지했습니다.")
            
            return True
        except Exception as e:
            logger.error(f"번역 처리 중지 중 오류 발생: {str(e)}")
            self.statusUpdated.emit(f"오류: {str(e)}")
            return False
    
    def refresh_audio_devices(self):
        """오디오 장치 목록 갱신"""
        try:
            # 오디오 캡처에서 장치 목록 가져오기
            devices = self.pipeline.audio_capture.get_devices()
            
            # 장치 이름 목록 생성
            device_names = [device.name for device in devices]
            
            # 시그널 발생
            self.audioDevicesUpdated.emit(device_names)
            logger.info(f"{len(device_names)}개의 오디오 장치를 감지했습니다.")
            
            return device_names
        except Exception as e:
            logger.error(f"오디오 장치 목록 갱신 중 오류 발생: {str(e)}")
            self.statusUpdated.emit(f"오디오 장치 오류: {str(e)}")
            return []
    
    def set_audio_device(self, device_index):
        """사용할 오디오 장치 설정
        
        Args:
            device_index (int): 장치 인덱스
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 오디오 캡처 장치 설정
            self.pipeline.audio_capture.set_device(device_index)
            
            # 설정에 오디오 장치 인덱스 저장
            if 'audio' not in self.pipeline.config:
                self.pipeline.config['audio'] = {}
            self.pipeline.config['audio']['device_index'] = device_index
            
            # 설정 저장
            self.save_config()
            logger.info(f"오디오 장치 인덱스 {device_index}를 설정 파일에 저장했습니다.")
            
            # 일시적으로 중지 중이었다면 재시작
            was_running = self.running
            if was_running:
                self.stop_translation()
                time.sleep(0.5)  # 잠시 대기
                self.start_translation()
            
            return True
        except Exception as e:
            logger.error(f"오디오 장치 설정 중 오류 발생: {str(e)}")
            self.statusUpdated.emit(f"오디오 장치 설정 오류: {str(e)}")
            return False
    
    def set_gemini_api_key(self, api_key):
        """Gemini API 키 설정
        
        Args:
            api_key (str): API 키
            
        Returns:
            bool: 성공 여부
        """
        try:
            return self.pipeline.set_gemini_api_key(api_key)
        except Exception as e:
            logger.error(f"API 키 설정 중 오류 발생: {str(e)}")
            self.statusUpdated.emit(f"API 키 설정 오류: {str(e)}")
            return False
    
    def save_config(self, config_path=None):
        """현재 설정 저장
        
        Args:
            config_path (str): 설정 파일 경로 (None: 기본 경로 사용)
            
        Returns:
            bool: 성공 여부
        """
        return self.pipeline.save_config(config_path)
    
    def get_current_settings(self):
        """현재 설정 가져오기
        
        Returns:
            dict: 현재 설정
        """
        return self.pipeline.config
    
    def apply_model_settings(self, settings):
        """모델 설정 적용
        
        Args:
            settings (dict): 적용할 설정 딕셔너리
            
        Returns:
            bool: 성공 여부 (True: 재시작 필요, False: 설정 변경 실패)
        """
        # 설정 저장
        try:
            # 각 카테고리별 설정 업데이트
            for category, category_settings in settings.items():
                if category in self.pipeline.config:
                    self.pipeline.config[category].update(category_settings)
            
            # 모델 설정이 바뀌었으므로 재시작 필요
            self.statusUpdated.emit("모델 설정이 변경되었습니다. 재시작이 필요합니다.")
            return True
            
        except Exception as e:
            logger.error(f"모델 설정 적용 중 오류 발생: {str(e)}")
            self.statusUpdated.emit(f"모델 설정 적용 오류: {str(e)}")
            return False
    
    def restart_pipeline(self):
        """처리 파이프라인 재시작"""
        was_running = self.running
        
        # 실행 중이면 중지
        if was_running:
            self.stop_translation()
        
        # 파이프라인 다시 생성
        try:
            old_config = self.pipeline.config.copy()
            self.pipeline = ProcessingPipeline(config=old_config)
            
            # 응답 생성기 참조 업데이트
            self.response_generator = self.pipeline.response_generator
            
            # 다시 시작
            if was_running:
                self.start_translation()
            
            self.statusUpdated.emit("처리 파이프라인이 재시작되었습니다.")
            return True
            
        except Exception as e:
            logger.error(f"파이프라인 재시작 중 오류 발생: {str(e)}")
            self.statusUpdated.emit(f"재시작 오류: {str(e)}")
            return False
    
    def _result_polling_loop(self):
        """결과 폴링 루프 (별도 스레드에서 실행)"""
        try:
            while self.running:
                # 결과 큐에서 다음 결과 가져오기
                result = self.pipeline.get_result(timeout=self.polling_interval)
                
                # 결과가 있으면 시그널 발생
                if result is not None:
                    self.translationUpdated.emit(result)
                
                # 짧은 대기 (CPU 사용량 감소)
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"결과 폴링 루프 중 오류 발생: {str(e)}")
            if self.running:
                self.statusUpdated.emit(f"내부 오류: {str(e)}")
                self.running = False 