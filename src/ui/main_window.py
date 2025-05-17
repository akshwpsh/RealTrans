#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox, 
                              QCheckBox, QLineEdit, QHBoxLayout, QGroupBox, QMessageBox)
from PySide6.QtCore import Qt, QSize, QTimer
import sys
import logging
from pathlib import Path
import os

from src.ui.controller import UIController
from src.ui.model_settings_dialog import ModelSettingsDialog

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """메인 애플리케이션 윈도우"""
    
    def __init__(self, config_path=None):
        super().__init__()
        
        # 윈도우 기본 설정
        self.setWindowTitle("RealTrans - 실시간 게임 음성 번역")
        self.setMinimumSize(QSize(800, 600))
        
        # config_path 디버깅 로그
        logger.info(f"MainWindow 초기화 - 설정 파일 경로: {config_path}")
        
        # UI 컨트롤러 초기화
        self.controller = UIController(config_path)
        
        # 설정 로드 확인
        current_settings = self.controller.get_current_settings()
        if current_settings:
            logger.info("===== MainWindow에서 확인한 설정 =====")
            logger.info(f"STT 모델 크기 (컨트롤러): {current_settings.get('stt', {}).get('model_size', '없음')}")
        
        # 컨트롤러 시그널 연결
        self.controller.translationUpdated.connect(self.on_translation_updated)
        self.controller.statusUpdated.connect(self.on_status_updated)
        self.controller.audioDevicesUpdated.connect(self.on_audio_devices_updated)
        
        # 메인 위젯 및 레이아웃 설정
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        self.layout = QVBoxLayout(self.main_widget)
        
        # 상태 레이블
        self.status_label = QLabel("준비됨")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)
        
        # 설정 그룹
        self.settings_group = QGroupBox("설정")
        self.settings_layout = QVBoxLayout(self.settings_group)
        
        # 오디오 소스 선택
        self.audio_source_label = QLabel("오디오 소스:")
        self.settings_layout.addWidget(self.audio_source_label)
        
        self.audio_source_combo = QComboBox()
        self.audio_source_combo.currentIndexChanged.connect(self.on_audio_source_changed)
        self.settings_layout.addWidget(self.audio_source_combo)
        
        # API 키 입력
        self.api_key_layout = QHBoxLayout()
        self.api_key_label = QLabel("Gemini API 키:")
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)  # 보안을 위해 암호 모드
        self.api_key_input.setPlaceholderText("API 키를 입력하세요 (환경 변수 GEMINI_API_KEY 사용 가능)")
        self.api_key_button = QPushButton("설정")
        self.api_key_button.clicked.connect(self.on_api_key_set)
        
        self.api_key_layout.addWidget(self.api_key_label)
        self.api_key_layout.addWidget(self.api_key_input)
        self.api_key_layout.addWidget(self.api_key_button)
        self.settings_layout.addLayout(self.api_key_layout)
        
        # 모델 설정 버튼 추가
        self.model_settings_button = QPushButton("모델 설정")
        self.model_settings_button.clicked.connect(self.open_model_settings)
        self.settings_layout.addWidget(self.model_settings_button)
        
        # 설정 옵션
        self.always_on_top_checkbox = QCheckBox("항상 위에 표시")
        self.always_on_top_checkbox.setChecked(True)
        self.always_on_top_checkbox.stateChanged.connect(self.toggle_always_on_top)
        self.settings_layout.addWidget(self.always_on_top_checkbox)
        
        self.layout.addWidget(self.settings_group)
        
        # 시작/중지 버튼
        self.start_button = QPushButton("번역 시작")
        self.start_button.clicked.connect(self.toggle_translation)
        self.layout.addWidget(self.start_button)
        
        # 번역 결과 표시 영역
        self.translation_group = QGroupBox("번역 결과")
        self.translation_layout = QVBoxLayout(self.translation_group)
        
        self.translation_label = QLabel("번역 결과가 여기에 표시됩니다")
        self.translation_label.setAlignment(Qt.AlignCenter)
        self.translation_label.setStyleSheet("""
            font-size: 18px; 
            background-color: #f0f0f0; 
            color: #000000;
            padding: 20px; 
            border-radius: 10px;
            min-height: 80px;
            border: 1px solid #d0d0d0;
        """)
        self.translation_label.setWordWrap(True)
        self.translation_layout.addWidget(self.translation_label)
        
        self.layout.addWidget(self.translation_group)
        
        # 응답 표시 영역
        self.response_group = QGroupBox("응답")
        self.response_layout = QVBoxLayout(self.response_group)
        
        # 한국어 응답
        self.response_label = QLabel("한국어 응답")
        self.response_label.setAlignment(Qt.AlignCenter)
        self.response_label.setStyleSheet("""
            font-size: 18px; 
            background-color: #e0f0ff; 
            color: #000066;
            padding: 15px; 
            border-radius: 10px;
            min-height: 60px;
            border: 1px solid #c0d0e0;
            margin-bottom: 10px;
        """)
        self.response_label.setWordWrap(True)
        self.response_layout.addWidget(self.response_label)
        
        # 원어 응답
        self.translated_response_label = QLabel("원어 응답")
        self.translated_response_label.setAlignment(Qt.AlignCenter)
        self.translated_response_label.setStyleSheet("""
            font-size: 18px; 
            background-color: #ffe0e0; 
            color: #660000;
            padding: 15px; 
            border-radius: 10px;
            min-height: 60px;
            border: 1px solid #e0c0c0;
            margin-bottom: 10px;
        """)
        self.translated_response_label.setWordWrap(True)
        self.response_layout.addWidget(self.translated_response_label)
        
        # 발음 가이드
        self.pronunciation_label = QLabel("발음 가이드")
        self.pronunciation_label.setAlignment(Qt.AlignCenter)
        self.pronunciation_label.setStyleSheet("""
            font-size: 18px; 
            background-color: #e0ffe0; 
            color: #006600;
            padding: 15px; 
            border-radius: 10px;
            min-height: 60px;
            border: 1px solid #c0e0c0;
        """)
        self.pronunciation_label.setWordWrap(True)
        self.response_layout.addWidget(self.pronunciation_label)
        
        self.layout.addWidget(self.response_group)
        
        # 초기 상태 설정
        self.is_translating = False
        self.set_always_on_top()
        
        # 컨트롤러에서 API 키 가져오기 - 이미 설정 파일에서 로드됨
        api_key = self.controller.get_current_settings().get('gemini', {}).get('api_key')
        if api_key:
            logger.info(f"설정 파일에서 API 키를 발견했습니다. (길이: {len(api_key)} 문자)")
            self.api_key_input.setText(api_key)
        else:
            # 환경 변수에서 API 키 가져오기 (설정 파일에 없는 경우)
            env_api_key = os.environ.get("GEMINI_API_KEY")
            if env_api_key:
                logger.info(f"환경 변수에서 API 키를 발견했습니다. (길이: {len(env_api_key)} 문자)")
                self.api_key_input.setText(env_api_key)
                
                # 컨트롤러에 API 키 설정
                set_result = self.controller.set_gemini_api_key(env_api_key)
                logger.info(f"환경 변수에서 가져온 API 키 설정 결과: {'성공' if set_result else '실패'}")
                
                # 설정 저장
                if set_result:
                    self.controller.save_config()
                    logger.info("환경 변수에서 가져온 API 키를 설정 파일에 저장했습니다.")
            else:
                logger.info("API 키를 찾을 수 없습니다. 수동으로 설정이 필요합니다.")
        
        # 장치 목록 가져오기 (0.5초 후 실행 -> 즉시 실행으로 변경)
        logger.info("오디오 장치 목록을 즉시 불러옵니다.")
        self.refresh_audio_devices()
    
    def toggle_translation(self):
        """번역 시작/중지 전환"""
        self.is_translating = not self.is_translating
        
        if self.is_translating:
            # API 키 확인
            if not self.controller.response_generator.api_key:
                api_key = self.api_key_input.text().strip()
                if api_key:
                    self.controller.set_gemini_api_key(api_key)
                else:
                    QMessageBox.warning(self, "API 키 필요", "Gemini API 키가 설정되지 않았습니다. API 키를 입력하거나 환경 변수 GEMINI_API_KEY를 설정하세요.")
                    self.is_translating = False
                    return
            
            self.start_button.setText("번역 중지")
            self.status_label.setText("번역 중...")
            self.controller.start_translation()
        else:
            self.start_button.setText("번역 시작")
            self.status_label.setText("준비됨")
            self.controller.stop_translation()
    
    def toggle_always_on_top(self, state):
        """항상 위 표시 전환"""
        self.set_always_on_top()
    
    def set_always_on_top(self):
        """항상 위 플래그 설정"""
        if self.always_on_top_checkbox.isChecked():
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show()  # 윈도우 플래그 적용을 위해 show 호출 필요
    
    def refresh_audio_devices(self):
        """오디오 장치 목록 갱신"""
        logger.info("오디오 장치 목록 갱신 시작...")
        self.controller.refresh_audio_devices()
    
    def on_audio_devices_updated(self, device_names):
        """오디오 장치 목록 업데이트 이벤트 처리"""
        # 콤보박스 초기화
        self.audio_source_combo.clear()
        
        # 장치 목록 추가
        if device_names:
            self.audio_source_combo.addItems(device_names)
            
            # 설정 파일에서 저장된 오디오 장치 인덱스 불러오기
            saved_device_index = self.controller.get_current_settings().get('audio', {}).get('device_index')
            
            if saved_device_index is not None and 0 <= saved_device_index < len(device_names):
                logger.info(f"설정 파일에서 오디오 장치 인덱스를 불러왔습니다: {saved_device_index}")
                
                # 콤보박스에서 저장된 장치 선택 (signal 방출 방지)
                self.audio_source_combo.blockSignals(True)
                self.audio_source_combo.setCurrentIndex(saved_device_index)
                self.audio_source_combo.blockSignals(False)
                
                # 명시적으로 오디오 소스 변경 메서드 호출
                logger.info(f"저장된 오디오 장치로 설정: 인덱스 {saved_device_index}, 이름: {device_names[saved_device_index]}")
                self.on_audio_source_changed(saved_device_index)
            else:
                logger.info("설정 파일에 저장된 오디오 장치 인덱스가 없거나 범위를 벗어납니다.")
        else:
            self.audio_source_combo.addItem("사용 가능한 장치 없음")
    
    def on_audio_source_changed(self, index):
        """오디오 소스 변경 이벤트 처리"""
        if index >= 0:
            device_name = self.audio_source_combo.currentText()
            logger.info(f"오디오 소스 변경: 인덱스 {index}, 장치명: {device_name}")
            
            # 컨트롤러를 통해 장치 설정 (이 과정에서 config.json에도 저장됨)
            set_result = self.controller.set_audio_device(index)
            if set_result:
                self.status_label.setText(f"오디오 장치가 '{device_name}'(으)로 변경되었습니다.")
                logger.info(f"오디오 장치 변경 성공: {device_name}")
            else:
                self.status_label.setText(f"오디오 장치 변경 실패")
                logger.warning(f"오디오 장치 변경 실패: 인덱스 {index}")
    
    def on_api_key_set(self):
        """API 키 설정 버튼 클릭 이벤트 처리"""
        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "API 키 필요", "API 키를 입력하세요.")
            return
        
        logger.info(f"API 키 설정 시도 (길이: {len(api_key)} 문자)")
        
        # 환경 변수에도 저장
        os.environ["GEMINI_API_KEY"] = api_key
        
        # 컨트롤러를 통해 API 키 설정
        if self.controller.set_gemini_api_key(api_key):
            # 설정 저장 (API 키를 포함하여 저장)
            self.controller.save_config()
            
            QMessageBox.information(self, "API 키 설정 완료", 
                                    "Gemini API 키가 성공적으로 설정되었습니다.\n\n" +
                                    "API 키는 설정 파일(config.json)에 저장됩니다. " +
                                    "다음에 프로그램을 시작할 때 자동으로 로드됩니다.")
        else:
            QMessageBox.warning(self, "API 키 설정 실패", 
                               "Gemini API 키 설정에 실패했습니다.\n\n" +
                               "키가 올바른지 확인하세요.")
    
    def open_model_settings(self):
        """모델 설정 대화 상자 열기"""
        # 현재 설정 가져오기
        current_settings = self.controller.get_current_settings()
        
        # 모델 설정 대화 상자 열기
        dialog = ModelSettingsDialog(self, current_settings)
        result = dialog.exec()
        
        # 사용자가 OK를 클릭한 경우
        if result == ModelSettingsDialog.Accepted:
            # 새 설정 가져오기
            new_settings = dialog.get_settings()
            
            # 설정 적용 및 저장
            if self.controller.apply_model_settings(new_settings):
                # 설정 저장
                self.controller.save_config()
                
                # 재시작 필요 여부 확인
                reply = QMessageBox.question(
                    self, 
                    "모델 설정 변경", 
                    "모델 설정이 변경되었습니다. 변경사항을 적용하려면 처리 파이프라인을 재시작해야 합니다.\n\n지금 재시작하시겠습니까?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    self.controller.restart_pipeline()
    
    def on_translation_updated(self, result):
        """번역 결과 업데이트 이벤트 처리"""
        # 원본 및 번역 텍스트 표시
        original_text = result.get('original_text', '')
        translated_text = result.get('translated_text', '')
        language_name = result.get('language_name', '알 수 없음')
        
        # 번역 결과 표시
        self.translation_label.setText(f"{language_name}: {original_text}\n\n{translated_text}")
        
        # 응답 표시
        response = result.get('response', {})
        
        response_text = response.get('response_text', '')
        translated_response = response.get('translated_response', '')
        pronunciation = response.get('pronunciation', '')
        
        if response_text:
            self.response_label.setText(response_text)
        else:
            self.response_label.setText("응답이 생성되지 않았습니다.")
        
        if translated_response:
            self.translated_response_label.setText(translated_response)
        else:
            self.translated_response_label.setText("번역된 응답이 없습니다.")
        
        if pronunciation:
            self.pronunciation_label.setText(f"발음 가이드: {pronunciation}")
        else:
            self.pronunciation_label.setText("발음 가이드가 없습니다.")
    
    def on_status_updated(self, status_message):
        """상태 메시지 업데이트 이벤트 처리"""
        self.status_label.setText(status_message)
    
    def closeEvent(self, event):
        """윈도우 종료 이벤트 처리"""
        # 번역 처리 중이면 중지
        if self.is_translating:
            self.controller.stop_translation()
        
        # 기본 종료 이벤트 처리
        super().closeEvent(event) 