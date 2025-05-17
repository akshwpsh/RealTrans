#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import logging
import json
import soundcard as sc
from pathlib import Path
from src.ui.main_window import MainWindow
from PySide6.QtWidgets import QApplication

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_config_test(config_path):
    """설정 파일 로드 테스트"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 설정 내용 검증
        logger.info("===== 설정 파일 내용 테스트 =====")
        logger.info(f"STT 모델 크기 (설정 파일): {config.get('stt', {}).get('model_size', '없음')}")
        
        return True
    except Exception as e:
        logger.error(f"설정 파일 테스트 중 오류: {str(e)}")
        return False

def print_audio_devices():
    """사용 가능한 오디오 장치 출력"""
    try:
        output_devices = sc.all_speakers()
        logger.info("===== 사용 가능한 오디오 출력 장치 =====")
        for i, device in enumerate(output_devices):
            logger.info(f"장치 {i}: {device.name}")
        return True
    except Exception as e:
        logger.error(f"오디오 장치 확인 중 오류: {str(e)}")
        return False

def main():
    """애플리케이션 메인 진입점"""
    app = QApplication(sys.argv)
    
    # 사용 가능한 오디오 장치 출력
    print_audio_devices()
    
    # 설정 파일 경로 확인
    config_path = Path(__file__).parent / "config.json"
    
    # 로깅 추가
    if config_path.exists():
        logger.info(f"설정 파일을 찾았습니다: {config_path}")
        # 설정 파일 테스트
        print_config_test(config_path)
    else:
        logger.warning(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    # 메인 윈도우 생성
    if config_path.exists():
        # 설정 파일이 있으면 해당 설정으로 초기화
        logger.info(f"설정 파일을 로드합니다: {config_path}")
        main_window = MainWindow(config_path=str(config_path))
    else:
        # 설정 파일이 없으면 기본 설정으로 초기화
        logger.info("기본 설정으로 초기화합니다.")
        main_window = MainWindow()
    
    main_window.show()
    
    # 애플리케이션 실행
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

# 주요 특징:
# 1. 오디오 입력 감지 및 음성 인식 (Whisper 모델 사용)
# 2. 실시간 번역 및 응답 생성 (Gemini API 사용)
# 3. Gemini API를 통한 외국어 번역, 한국어 응답 생성, 원어민 응답 및 발음 가이드 제공
# 4. 다양한 언어 지원 (일본어, 중국어, 영어 등)
# 5. 간편한 오디오 장치 설정
