#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import logging
import threading
import queue
from pathlib import Path
import json
import numpy as np
import os
import wave

from src.audio.capture import AudioCapture
from src.audio.vad import VoiceActivityDetector
from src.models.stt import SpeechToText
from src.data.phrases import ResponseGenerator

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProcessingPipeline:
    """메인 처리 파이프라인 클래스"""
    
    def __init__(self, config=None, config_path=None):
        """처리 파이프라인 초기화
        
        Args:
            config (dict): 설정 (None: 기본값 사용)
            config_path (str): 설정 파일 경로 (config보다 우선순위 높음)
        """
        # 설정 파일 경로가 있으면 먼저 파일에서 로드
        if config_path and Path(config_path).exists():
            logger.info(f"설정 파일에서 초기 설정 로드: {config_path}")
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info(f"설정 파일을 로드했습니다: {config_path}")
            except Exception as e:
                logger.error(f"설정 파일 로드 실패, 기본 설정 사용: {str(e)}")
                self.config = self._get_default_config()
                if config:
                    # 인라인 설정으로 기본 설정 업데이트
                    for category, settings in config.items():
                        if category in self.config:
                            self.config[category].update(settings)
                        else:
                            self.config[category] = settings
        else:
            # 설정 파일이 없으면 기본 설정 또는 인라인 설정 사용
            self.config = config or self._get_default_config()
        
        # STT 모델 크기 확인
        logger.info(f"초기화된 STT 모델 크기: {self.config.get('stt', {}).get('model_size', '없음')}")
        
        # 결과 큐 (UI 업데이트용)
        self.result_queue = queue.Queue(maxsize=10)
        
        # 오디오 캡처 초기화
        self.audio_capture = AudioCapture(
            sample_rate=self.config['audio']['sample_rate'],
            chunk_size=self.config['audio']['chunk_size'],
            buffer_size=self.config['audio']['buffer_size'],
            overlap_ratio=0.0  # 청크 간 겹침 제거
        )
        
        # 저장된 오디오 장치 인덱스가 있으면 설정
        if 'device_index' in self.config['audio']:
            device_index = self.config['audio']['device_index']
            devices = self.audio_capture.get_devices()
            
            if 0 <= device_index < len(devices):
                device_name = devices[device_index].name if devices else "Unknown"
                logger.info(f"설정 파일에서 오디오 장치 설정을 적용합니다: 인덱스 {device_index}, 장치명: {device_name}")
                try:
                    self.audio_capture.set_device(device_index)
                    logger.info(f"오디오 장치 설정 성공: {device_name}")
                except Exception as e:
                    logger.error(f"오디오 장치 설정 중 오류 발생: {str(e)}")
            else:
                logger.warning(f"설정 파일의 오디오 장치 인덱스({device_index})가 유효하지 않습니다. 기본 장치를 사용합니다.")
                logger.info(f"사용 가능한 장치 수: {len(devices)}")
        else:
            logger.info("설정 파일에 저장된 오디오 장치 인덱스가 없습니다. 기본 장치를 사용합니다.")
        
        # VAD 초기화
        self.vad = VoiceActivityDetector(
            sample_rate=self.config['audio']['sample_rate'],
            threshold=0.05,  # 임계값을 0.3에서 0.05로 낮춤 (음성 감지 민감도 향상)
            min_speech_duration_ms=50,  # 더 짧은 음성도 감지하도록 50ms로 수정 (기존 150ms)
            min_silence_duration_ms=300
        )
        
        # STT 모델 초기화
        self.stt = SpeechToText(
            model_size=self.config['stt']['model_size'],
            device=self.config['stt']['device'],
            compute_type=self.config['stt']['compute_type'],
            beam_size=self.config['stt']['beam_size'],
            vad_filter=self.config['stt']['vad_filter']
        )
        
        # 번역 모델 부분 제거 (Gemini로 대체)
        self.translator = None
        
        # 응답 생성기 초기화
        try:
            self.response_generator = ResponseGenerator(
                api_key=self.config['gemini']['api_key'],
                model_name=self.config['gemini']['model_name']
            )
        except Exception as e:
            logger.error(f"응답 생성기 초기화 중 오류 발생, 프로그램은 계속 실행됩니다: {str(e)}")
            self.response_generator = None
        
        # 음성 버퍼링 관련 변수
        self.speech_buffer = []
        self.max_buffer_chunks = 5  # 최대 5개 청크까지 버퍼링 (약 5초)
        self.buffering_active = False
        self.buffer_timeout = 1.5  # 마지막 음성 감지 후 1.5초 대기
        self.last_buffer_time = 0

        # 상태 변수
        self.running = False
        self.processing_thread = None
        
        # 최근 검출된 언어 코드 (반복 검출 방지)
        self.last_detected_language = None
        self.last_speech_timestamp = 0
        self.language_reset_delay = 2.0  # 2초 이상 지나면 언어 코드 리셋
    
    def _get_default_config(self):
        """기본 설정 반환
        
        Returns:
            dict: 기본 설정
        """
        return {
            'audio': {
                'sample_rate': 16000,
                'chunk_size': 16000,
                'buffer_size': 20,
                'overlap_ratio': 0.0  # 청크 간 겹침 제거 (기존 0.5에서 0.0으로 변경)
            },
            'vad': {
                'threshold': 0.05,  # 임계값을 0.3에서 0.05로 낮춤 (음성 감지 민감도 향상)
                'min_speech_duration_ms': 50,  # 더 짧은 음성도 감지하도록 50ms로 수정 (기존 150ms)
                'min_silence_duration_ms': 300
            },
            'stt': {
                'model_size': 'small',
                'device': 'auto',
                'compute_type': 'auto',
                'beam_size': 3,
                'vad_filter': True
            },
            'gemini': {
                'api_key': None,  # 환경 변수 또는 설정에서 가져옴
                'model_name': 'gemini-pro',
                'max_tokens': 150
            }
        }
    
    def save_config(self, config_path=None):
        """설정 저장
        
        Args:
            config_path (str): 설정 파일 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            save_path = config_path or Path(__file__).parent.parent.parent / "config.json"
            
            # 모든 설정을 그대로 저장 (API 키 포함)
            save_config = self.config.copy()
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"설정을 저장했습니다: {save_path}")
            return True
        
        except Exception as e:
            logger.error(f"설정 저장 중 오류 발생: {str(e)}")
            return False
    
    def load_config(self, config_path):
        """설정 로드
        
        Args:
            config_path (str): 설정 파일 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            config_path = str(config_path)  # Path 객체인 경우 문자열로 변환
            logger.info(f"파이프라인 설정 로드 시도: {config_path}")
            
            if not Path(config_path).exists():
                logger.warning(f"설정 파일이 존재하지 않습니다: {config_path}")
                return False
            
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            
            logger.info(f"설정 파일 내용: {json.dumps(loaded_config, ensure_ascii=False)[:100]}...")
            
            # 기존 설정과 병합
            for category, settings in loaded_config.items():
                if category in self.config:
                    logger.info(f"카테고리 '{category}' 설정 병합")
                    self.config[category].update(settings)
                else:
                    logger.info(f"새 카테고리 '{category}' 추가")
                    self.config[category] = settings
            
            logger.info(f"설정을 로드했습니다: {config_path}")
            logger.info(f"병합된 설정 내용: {json.dumps(self.config, ensure_ascii=False)[:100]}...")
            return True
        
        except Exception as e:
            logger.error(f"설정 로드 중 오류 발생: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def set_gemini_api_key(self, api_key):
        """Gemini API 키 설정
        
        Args:
            api_key (str): API 키
            
        Returns:
            bool: 성공 여부
        """
        if not api_key:
            logger.warning("API 키가 비어 있습니다.")
            return False
        
        logger.info(f"파이프라인에서 API 키 설정 중... (길이: {len(api_key)} 문자)")
        self.config['gemini']['api_key'] = api_key
        
        # 응답 생성기가 없는 경우 생성
        if self.response_generator is None:
            logger.info("응답 생성기가 없어 새로 생성합니다.")
            try:
                self.response_generator = ResponseGenerator(
                    api_key=api_key,
                    model_name=self.config['gemini']['model_name']
                )
                logger.info("새 응답 생성기 생성 성공")
                return True
            except Exception as e:
                logger.error(f"새 응답 생성기 생성 실패: {str(e)}")
                return False
        
        # 기존 응답 생성기에 API 키 설정
        result = self.response_generator.set_api_key(api_key)
        logger.info(f"응답 생성기에 API 키 설정 결과: {'성공' if result else '실패'}")
        return result
    
    def start(self):
        """처리 파이프라인 시작"""
        if self.running:
            logger.warning("처리 파이프라인이 이미 실행 중입니다.")
            return
        
        # 오디오 캡처 시작
        self.audio_capture.start_capture()
        
        # 처리 스레드 시작
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("처리 파이프라인을 시작했습니다.")
    
    def stop(self):
        """처리 파이프라인 중지"""
        self.running = False
        
        # 오디오 캡처 중지
        self.audio_capture.stop_capture()
        
        # 처리 스레드 종료 대기
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None
        
        logger.info("처리 파이프라인을 중지했습니다.")
    
    def get_result(self, timeout=0.1):
        """결과 큐에서 다음 결과 가져오기
        
        Args:
            timeout (float): 대기 시간 (초)
            
        Returns:
            dict: 결과 (None: 타임아웃 또는 비어있음)
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _processing_loop(self):
        """메인 처리 루프"""
        try:
            logger.info("처리 파이프라인 루프 시작")
            audio_chunk_count = 0
            speech_detect_count = 0
            first_speech_detected = False  # 첫 번째 음성 감지 플래그
            
            while self.running:
                try:
                    # 오디오 캡처에서 다음 청크 가져오기
                    audio_chunk = self.audio_capture.get_audio_chunk(timeout=0.5)
                    
                    if audio_chunk is None:
                        # 버퍼링 중이고 타임아웃이 지났으면 버퍼 처리
                        current_time = time.time()
                        if (self.buffering_active and 
                            self.speech_buffer and 
                            current_time - self.last_buffer_time > self.buffer_timeout):
                            logger.info(f"버퍼 타임아웃: 버퍼링된 {len(self.speech_buffer)}개 청크 처리")
                            self._process_buffered_speech()
                        continue
                    
                    audio_chunk_count += 1
                    if audio_chunk_count % 5 == 0:  # 5개 청크마다 로그 출력
                        logger.info(f"오디오 청크 {audio_chunk_count}개 처리, 음성 감지 {speech_detect_count}회, 버퍼링 {'활성' if self.buffering_active else '비활성'}")
                    
                    # 데이터 유효성 검사
                    if len(audio_chunk) == 0 or np.max(np.abs(audio_chunk)) < 0.001:
                        continue  # 무음이면 건너뛰기
                    
                    # VAD로 음성 감지
                    has_speech = self.vad.has_speech(audio_chunk)
                    
                    if has_speech:
                        speech_detect_count += 1
                        logger.info(f"음성 감지됨! (총 {speech_detect_count}회, 청크 길이: {len(audio_chunk)})")
                        
                        # 첫 번째 음성 감지 처리
                        if not first_speech_detected:
                            first_speech_detected = True
                            self.buffering_active = True
                            # 첫 번째 음성 청크부터 확실히 버퍼에 추가되도록 하기 위해 버퍼 초기화
                            self.speech_buffer = []
                            logger.info("첫 번째 음성 감지: 버퍼 초기화 및 버퍼링 활성화")
                        
                        # 음성 버퍼에 추가
                        self.speech_buffer.append(audio_chunk)
                        self.last_buffer_time = time.time()
                        
                        # 버퍼가 최대 크기에 도달하면 처리
                        if len(self.speech_buffer) >= self.max_buffer_chunks:
                            logger.info(f"최대 버퍼 크기 도달: {len(self.speech_buffer)}개 청크 처리")
                            self._process_buffered_speech()
                            first_speech_detected = False  # 처리 후 첫 음성 감지 플래그 초기화
                    
                    elif self.buffering_active:
                        # 음성이 감지되지 않았지만 버퍼링 중이면 타이머 확인
                        current_time = time.time()
                        if current_time - self.last_buffer_time > self.buffer_timeout:
                            # 타임아웃이 지나면 현재까지 버퍼링된 내용 처리
                            if self.speech_buffer:
                                logger.info(f"버퍼 타임아웃: 버퍼링된 {len(self.speech_buffer)}개 청크 처리")
                                self._process_buffered_speech()
                                first_speech_detected = False  # 처리 후 첫 음성 감지 플래그 초기화
                    
                except Exception as e:
                    logger.error(f"파이프라인 루프 내부 처리 중 오류: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    time.sleep(1.0)  # 오류 발생 시 잠시 대기
        
        except Exception as e:
            logger.error(f"처리 파이프라인 루프 중 오류 발생: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.running = False
    
    def _process_buffered_speech(self):
        """버퍼링된 음성 처리"""
        try:
            if not self.speech_buffer:
                self.buffering_active = False
                return
            
            # 버퍼 크기 및 청크 번호 로깅
            buffer_size = len(self.speech_buffer)
            logger.info(f"버퍼링된 청크 처리 시작: 총 {buffer_size}개 청크")
            
            # 각 청크 정보 로깅
            for i, chunk in enumerate(self.speech_buffer):
                logger.info(f"  청크 {i}: 길이={len(chunk)} 샘플, 최대 진폭={np.max(np.abs(chunk)):.4f}")
            
            # 버퍼링된 청크 병합
            combined_audio = np.concatenate(self.speech_buffer)
            logger.info(f"버퍼링된 {buffer_size}개 청크 병합 완료, 총 길이: {len(combined_audio)} 샘플")
            
            # 버퍼 초기화
            self.speech_buffer = []
            self.buffering_active = False
            
            # 병합된 오디오 저장 (디버깅용)
            self._save_merged_audio(combined_audio, buffer_size)
            
            # 음성 구간 추출
            speech_timestamps, speech_audio = self.vad.detect_speech(combined_audio)
            
            # 안전한 검사: timestamps는 있지만 speech_audio가 None인 경우 처리
            if speech_timestamps and speech_audio is None:
                logger.warning("음성 구간 타임스탬프는 감지되었으나 오디오 추출에 실패했습니다.")
                return
                
            # 음성 구간이 없거나 비어있으면 처리 중단
            if speech_audio is None or (isinstance(speech_audio, np.ndarray) and speech_audio.size == 0):
                logger.warning("버퍼링된 오디오에서 음성 구간이 추출되지 않았습니다.")
                return
            
            # 음성 오디오 증폭 (최대 진폭이 0.8이 되도록)
            try:
                max_amplitude = np.max(np.abs(speech_audio))
                if max_amplitude > 0:
                    target_amplitude = 0.8
                    gain = target_amplitude / max_amplitude
                    speech_audio = speech_audio * gain
                    logger.info(f"STT 처리를 위한 오디오 볼륨 증폭: 게인 {gain:.2f}배")
            except Exception as e:
                logger.error(f"오디오 증폭 중 오류 발생: {str(e)}")
                if speech_audio is None or (isinstance(speech_audio, np.ndarray) and speech_audio.size == 0):
                    logger.error("증폭할 오디오가 없어 처리를 중단합니다.")
                    return
            
            # 추출된 음성 구간 오디오 저장 (디버깅용)
            self._save_speech_audio(speech_audio)
            
            # 언어 감지 및 필터링
            lang_result = self.stt.detect_language(speech_audio)
            logger.info(f"감지된 언어: {lang_result.get('language_name', '알 수 없음')} (코드: {lang_result.get('language', '없음')})")
            
            # 한국어일 경우 처리 건너뛰기
            if lang_result['is_korean']:
                logger.info("한국어 감지됨, 처리 건너뛰기")
                return
            
            # 마지막 언어와 동일하고, 시간이 짧게 지났으면 건너뛰기 (중복 방지)
            current_time = time.time()
            if (self.last_detected_language == lang_result['language'] and 
                current_time - self.last_speech_timestamp < self.language_reset_delay):
                return
            
            # 언어 정보 업데이트
            self.last_detected_language = lang_result['language']
            self.last_speech_timestamp = current_time
            
            # STT로 음성을 텍스트로 변환
            # 확실히 원본 언어 그대로 인식되도록 task를 명시적으로 "transcribe"로 지정
            stt_result = self.stt.transcribe(
                speech_audio, 
                language=lang_result['language'],
                task="transcribe"  # 원본 언어를 유지하기 위해 명시적으로 transcribe 지정
            )
            
            if not stt_result['text']:
                logger.warning("음성 인식 결과가 없습니다.")
                return
            
            logger.info(f"인식된 텍스트: {stt_result['text']}")
            
            # 번역 및 응답 생성 - Gemini API로 한 번에 처리
            if self.response_generator is not None:
                language_code = stt_result['language']
                if hasattr(self.stt, 'reverse_language_map'):
                    language_code = self.stt.reverse_language_map.get(language_code, 'en')
                
                logger.info(f"Gemini로 번역 및 응답 생성 중... (언어: {language_code})")
                generated_response = self.response_generator.generate_response(
                    stt_result['text'],
                    "",  # 번역된 텍스트는 이제 사용하지 않음
                    language_code,
                    max_tokens=self.config['gemini']['max_tokens']
                )
                
                # Gemini가 생성한 번역 결과 사용
                translated_text = generated_response.get('gemini_translation', "")
                
                logger.info(f"Gemini 번역 결과: '{translated_text}'")
                logger.info(f"Gemini 응답: '{generated_response.get('response_text', '')}'")
            else:
                # 응답 생성기가 없는 경우
                logger.warning("응답 생성기가 초기화되지 않았습니다.")
                translated_text = f"[번역 불가] {stt_result['text']}"
                generated_response = {
                    'gemini_translation': translated_text,
                    'response_text': "[응답 생성 불가]",
                    'translated_response': "",
                    'pronunciation': ""
                }
            
            # 결과 구성
            result = {
                'original_text': stt_result['text'],
                'translated_text': translated_text,
                'language': stt_result['language'],
                'language_name': stt_result['language_name'],
                'response': generated_response,
                'timestamp': current_time
            }
            
            logger.info("결과 생성 완료, UI에 전송")
            
            # 결과 큐에 추가 (가득 차면 가장 오래된 결과 버림)
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except:
                    pass
            
            try:
                self.result_queue.put_nowait(result)
            except:
                pass
            
        except Exception as e:
            logger.error(f"버퍼링된 음성 처리 중 오류: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # 오류 발생 시 버퍼 초기화
            self.speech_buffer = []
            self.buffering_active = False
    
    def _save_merged_audio(self, audio_data, buffer_chunks_count):
        """병합된 오디오 저장 (디버깅용)
        
        Args:
            audio_data (numpy.ndarray): 오디오 데이터
            buffer_chunks_count (int): 병합에 사용된 청크 수
        """
        try:
            # 데이터 유효성 검사
            if audio_data is None or len(audio_data) == 0:
                logger.warning("저장할 병합 오디오가 비어 있습니다.")
                return
                
            # 디버그 디렉토리 확인
            debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "debug_audio")
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            
            # 타임스탬프로 파일명 생성
            timestamp = int(time.time())
            filename = os.path.join(debug_dir, f"merged_audio_{timestamp}.wav")
            
            # float32 데이터를 int16으로 변환 (WAV 파일 형식)
            scaled_data = np.int16(audio_data * 32767)
            
            # WAV 파일로 저장
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)  # 모노
                wf.setsampwidth(2)  # 16비트
                wf.setframerate(self.config['audio']['sample_rate'])
                wf.writeframes(scaled_data.tobytes())
            
            # 오디오 정보 텍스트 파일 저장
            info_file = os.path.join(debug_dir, f"merged_audio_{timestamp}.txt")
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"병합된 청크 수: {buffer_chunks_count}\n")
                f.write(f"샘플 수: {len(audio_data)}\n")
                
                # 안전하게 최대/평균값 계산
                max_amplitude = np.max(np.abs(audio_data)) if len(audio_data) > 0 else 0.0
                mean_amplitude = np.mean(np.abs(audio_data)) if len(audio_data) > 0 else 0.0
                
                f.write(f"최대 진폭: {max_amplitude:.6f}\n")
                f.write(f"평균 진폭: {mean_amplitude:.6f}\n")
            
            logger.info(f"병합된 오디오를 저장했습니다: {filename}")
            
        except Exception as e:
            logger.error(f"병합된 오디오 저장 중 오류 발생: {str(e)}")
    
    def _save_speech_audio(self, audio_data):
        """VAD로 검출된 음성 구간 오디오 저장 (디버깅용)
        
        Args:
            audio_data (numpy.ndarray): 음성 구간 오디오 데이터
        """
        try:
            # 데이터 유효성 검사 (더 엄격하게)
            if audio_data is None:
                logger.warning("저장할 음성 구간 오디오가 None입니다.")
                return
                
            if not isinstance(audio_data, np.ndarray):
                logger.warning(f"음성 구간 오디오가 numpy 배열이 아닙니다: {type(audio_data)}")
                return
                
            if audio_data.size == 0:
                logger.warning("저장할 음성 구간 오디오가 비어 있습니다. (크기: 0)")
                return
            
            # 볼륨이 너무 작은지 확인
            max_amplitude = np.max(np.abs(audio_data))
            logger.info(f"추출된 음성 오디오 원본 최대 진폭: {max_amplitude:.6f}")
            
            # 볼륨 증폭 (최대 진폭이 0.8이 되도록)
            if max_amplitude > 0:
                target_amplitude = 0.8
                gain = target_amplitude / max_amplitude
                amplified_audio = audio_data * gain
                logger.info(f"오디오 볼륨 증폭: 원본 진폭 {max_amplitude:.6f} -> 증폭 진폭 {np.max(np.abs(amplified_audio)):.6f} (게인: {gain:.2f})")
                audio_data = amplified_audio
            
            # 디버그 디렉토리 확인
            debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "debug_audio")
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            
            # 타임스탬프로 파일명 생성
            timestamp = int(time.time())
            filename = os.path.join(debug_dir, f"speech_audio_{timestamp}.wav")
            
            # float32 데이터를 int16으로 변환 (WAV 파일 형식)
            scaled_data = np.int16(audio_data * 32767)
            
            # WAV 파일로 저장
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)  # 모노
                wf.setsampwidth(2)  # 16비트
                wf.setframerate(self.config['audio']['sample_rate'])
                wf.writeframes(scaled_data.tobytes())
            
            # 오디오 정보 텍스트 파일 저장
            info_file = os.path.join(debug_dir, f"speech_audio_{timestamp}.txt")
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"샘플 수: {len(audio_data)}\n")
                
                # 안전하게 최대/평균값 계산
                max_amplitude = np.max(np.abs(audio_data)) if len(audio_data) > 0 else 0.0
                mean_amplitude = np.mean(np.abs(audio_data)) if len(audio_data) > 0 else 0.0
                active_ratio = np.mean(np.abs(audio_data) > 0.01) if len(audio_data) > 0 else 0.0
                
                f.write(f"최대 진폭: {max_amplitude:.6f}\n")
                f.write(f"평균 진폭: {mean_amplitude:.6f}\n")
                f.write(f"활성 비율: {active_ratio:.6f}\n")
            
            logger.info(f"음성 구간 오디오를 저장했습니다: {filename}")
            
        except Exception as e:
            logger.error(f"음성 구간 오디오 저장 중 오류 발생: {str(e)}")
            import traceback
            logger.error(traceback.format_exc()) 