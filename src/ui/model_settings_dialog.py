#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QPushButton, QGroupBox, QFormLayout, QDialogButtonBox)
from PySide6.QtCore import Qt
import logging

logger = logging.getLogger(__name__)

class ModelSettingsDialog(QDialog):
    """모델 설정 대화 상자"""
    
    def __init__(self, parent=None, settings=None):
        """모델 설정 대화 상자 초기화
        
        Args:
            parent: 부모 위젯
            settings: 현재 설정 딕셔너리
        """
        super().__init__(parent)
        
        self.settings = settings or {}
        
        # 대화 상자 기본 설정
        self.setWindowTitle("모델 설정")
        self.setMinimumWidth(500)
        
        # 레이아웃 설정
        self.layout = QVBoxLayout(self)
        
        # STT 모델 설정 그룹
        self.stt_group = QGroupBox("음성 인식 모델 (Fast Whisper)")
        self.stt_layout = QFormLayout(self.stt_group)
        
        # STT 모델 크기 선택
        self.stt_model_size_label = QLabel("모델 크기:")
        self.stt_model_size_combo = QComboBox()
        self.whisper_models = [
            "small (244M, ~2GB VRAM)",
            "medium (769M, ~5GB VRAM)",
            "large-v3 (1550M, ~10GB VRAM)",
            "distil-large-v3 (809M, ~6GB VRAM)"
        ]
        self.stt_model_size_combo.addItems(self.whisper_models)
        
        # 모델에서 크기 부분만 추출하는 함수 (예: "small-v3 (244M, ~2GB VRAM)" -> "small-v3")
        self.get_model_size = lambda model_text: model_text.split(" ")[0]
        
        # STT 장치 선택
        self.stt_device_label = QLabel("실행 장치:")
        self.stt_device_combo = QComboBox()
        self.stt_device_combo.addItems(["auto", "cpu", "cuda"])
        
        # STT 계산 타입
        self.stt_compute_type_label = QLabel("계산 타입:")
        self.stt_compute_type_combo = QComboBox()
        self.stt_compute_type_combo.addItems(["auto", "float16", "int8", "int8_float16"])
        
        # 폼에 추가
        self.stt_layout.addRow("모델 크기:", self.stt_model_size_combo)
        self.stt_layout.addRow("실행 장치:", self.stt_device_combo)
        self.stt_layout.addRow("계산 타입:", self.stt_compute_type_combo)
        
        self.layout.addWidget(self.stt_group)
        
        # Gemini 모델 설정 그룹
        self.gemini_group = QGroupBox("Gemini 번역 및 응답 생성 모델")
        self.gemini_layout = QFormLayout(self.gemini_group)
        
        # Gemini 모델 선택
        self.gemini_model_label = QLabel("모델:")
        self.gemini_model_combo = QComboBox()
        self.gemini_model_combo.addItems([
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.0-flash"
        ])
        
        # Gemini 최대 토큰
        self.gemini_max_tokens_label = QLabel("최대 토큰:")
        self.gemini_max_tokens_combo = QComboBox()
        self.gemini_max_tokens_combo.addItems(["100", "150", "200", "300", "500"])
        
        # 폼에 추가
        self.gemini_layout.addRow("모델:", self.gemini_model_combo)
        self.gemini_layout.addRow("최대 토큰:", self.gemini_max_tokens_combo)
        
        self.layout.addWidget(self.gemini_group)
        
        # 버튼
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)
        
        # 현재 설정 값으로 UI 초기화
        self.load_settings()
    
    def load_settings(self):
        """현재 설정 값으로 UI 초기화"""
        # STT 설정
        stt_settings = self.settings.get('stt', {})
        model_size = stt_settings.get('model_size', 'small')  # 기본값을 small로 변경
        
        # 이전 버전 모델 이름을 새 이름으로 변환
        model_mapping = {
            'tiny': 'small',  # tiny는 small로 매핑
            'base': 'small',  # base는 small로 매핑
            'small-v3': 'small', # small-v3는 small로 매핑
            'medium-v3': 'medium', # medium-v3는 medium으로 매핑
            'turbo': 'distil-large-v3' # turbo는 distil-large-v3로 매핑
        }
        
        # 이전 버전 모델 이름이면 새 이름으로 변환
        if model_size in model_mapping:
            model_size = model_mapping[model_size]
        
        # 모델 이름에 해당하는 항목 찾기
        found = False
        for i, model_text in enumerate(self.whisper_models):
            if model_text.startswith(model_size + " ") or model_size in model_text:
                self.stt_model_size_combo.setCurrentIndex(i)
                found = True
                break
        
        # 일치하는 항목이 없을 경우 기본값으로 설정
        if not found:
            # 커스텀 항목 추가 대신 첫 번째 항목(small-v3)으로 설정
            self.stt_model_size_combo.setCurrentIndex(0)
        
        self.set_combo_value(self.stt_device_combo, stt_settings.get('device', 'auto'))
        self.set_combo_value(self.stt_compute_type_combo, stt_settings.get('compute_type', 'auto'))
        
        # Gemini 설정
        gemini_settings = self.settings.get('gemini', {})
        model_name = gemini_settings.get('model_name', 'gemini-2.5-flash-preview-04-17')
        
        # 이전 버전 모델 이름을 새 이름으로 변환
        gemini_model_mapping = {
            'gemini-pro': 'gemini-2.5-flash-preview-04-17',
            'gemini-1.0-pro': 'gemini-2.5-flash-preview-04-17',
            'gemini-1.5-pro': 'gemini-2.5-flash-preview-04-17',
            'gemini-1.5-flash': 'gemini-2.0-flash'
        }
        
        # 이전 버전 모델 이름이면 새 이름으로 변환
        if model_name in gemini_model_mapping:
            model_name = gemini_model_mapping[model_name]
        
        # 모델 이름에 해당하는 항목 찾기
        found = False
        for i in range(self.gemini_model_combo.count()):
            if self.gemini_model_combo.itemText(i) == model_name:
                self.gemini_model_combo.setCurrentIndex(i)
                found = True
                break
        
        # 일치하는 항목이 없을 경우 기본값으로 설정
        if not found:
            self.gemini_model_combo.setCurrentIndex(0)
            
        max_tokens = str(gemini_settings.get('max_tokens', 150))
        self.set_combo_value(self.gemini_max_tokens_combo, max_tokens)
    
    def set_combo_value(self, combo, value):
        """콤보 박스 값 설정 (목록에 없으면 추가)"""
        index = combo.findText(str(value))
        if index >= 0:
            combo.setCurrentIndex(index)
        else:
            combo.addItem(str(value))
            combo.setCurrentIndex(combo.count() - 1)
    
    def get_settings(self):
        """사용자가 선택한 설정 값을 딕셔너리로 반환"""
        # 모델 크기 텍스트에서 실제 모델 크기만 추출 (예: "small-v3 (244M, ~2GB VRAM)" -> "small-v3")
        model_text = self.stt_model_size_combo.currentText()
        model_size = self.get_model_size(model_text)
        
        settings = {
            'stt': {
                'model_size': model_size,
                'device': self.stt_device_combo.currentText(),
                'compute_type': self.stt_compute_type_combo.currentText()
            },
            'gemini': {
                'model_name': self.gemini_model_combo.currentText(),
                'max_tokens': int(self.gemini_max_tokens_combo.currentText())
            }
        }
        return settings 