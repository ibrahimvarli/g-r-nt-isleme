import speech_recognition as sr
import threading
import time
import queue

class VoiceCommands:
    def __init__(self):
        # Tanıyıcı nesnesi
        self.recognizer = sr.Recognizer()
        
        # Mikrofon nesnesi
        self.microphone = None
        
        # Komut listesi ve karşılık gelen işlevler
        self.commands = {
            "başlat": "start",
            "durdur": "stop",
            "kaydet": "capture",
            "filtre": "filter",
            "normal": "filter_normal",
            "siyah beyaz": "filter_bw",
            "sepya": "filter_sepia",
            "negatif": "filter_negative",
            "bulanık": "filter_blur",
            "kenar": "filter_edge",
            "yüz noktaları": "toggle_landmarks",
            "duygu analizi": "toggle_emotions",
            "yaş cinsiyet": "toggle_age_gender",
            "yüz tanıma": "toggle_face_recognition",
            "yüz haritası": "toggle_face_mesh",
            "ölçümler": "toggle_measurements",
            "dudak okuma": "toggle_lip_reading",
            "gözlük": "filter_glasses",
            "şapka": "filter_hat",
            "maske": "filter_mask",
            "sakal": "filter_beard",
            "kulaklar": "filter_ears",
            "ışık": "filter_light",
            "döndür": "rotate_model",
            "büyüt": "scale_up",
            "küçült": "scale_down",
            "derinlik artır": "depth_up",
            "derinlik azalt": "depth_down",
            "animasyon": "toggle_animation",
            "tel kafes": "toggle_wireframe",
            "doku": "toggle_texture",
            "gölgelendirme": "toggle_shading",
            "gülümse": "expression_smile",
            "üzgün": "expression_sad",
            "şaşkın": "expression_surprise",
            "kızgın": "expression_angry",
            "normal ifade": "expression_neutral"
        }
        
        # Dinleme durumu
        self.is_listening = False
        
        # Dinleme iş parçacığı
        self.listen_thread = None
        
        # Komut kuyruğu
        self.command_queue = queue.Queue()
        
        # Dil ayarı
        self.language = "tr-TR"
        
        # Gürültü eşiği
        self.energy_threshold = 4000
        
        # Komut algılama hassasiyeti (0.0 - 1.0)
        self.command_threshold = 0.6
    
    def initialize(self):
        """Mikrofonu başlatır"""
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.recognizer.energy_threshold = self.energy_threshold
            return True
        except Exception as e:
            print(f"Mikrofon başlatılamadı: {e}")
            return False
    
    def start_listening(self):
        """Ses komutlarını dinlemeye başlar"""
        if self.is_listening:
            return False
        
        if self.microphone is None and not self.initialize():
            return False
        
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        return True
    
    def stop_listening(self):
        """Ses komutlarını dinlemeyi durdurur"""
        self.is_listening = False
        if self.listen_thread is not None:
            self.listen_thread.join(timeout=1)
            self.listen_thread = None
        return True
    
    def _listen_loop(self):
        """Sürekli dinleme döngüsü"""
        while self.is_listening:
            try:
                with self.microphone as source:
                    print("Dinleniyor...")
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                
                try:
                    text = self.recognizer.recognize_google(audio, language=self.language)
                    print(f"Algılanan: {text}")
                    
                    # Algılanan metni küçük harfe çevir
                    text = text.lower()
                    
                    # Komut eşleştirme
                    for command, action in self.commands.items():
                        if command in text:
                            print(f"Komut algılandı: {command} -> {action}")
                            self.command_queue.put(action)
                            break
                    
                except sr.UnknownValueError:
                    # Konuşma anlaşılamadı
                    pass
                except sr.RequestError as e:
                    print(f"Google Speech Recognition servis hatası: {e}")
            
            except Exception as e:
                print(f"Dinleme hatası: {e}")
                time.sleep(0.5)
    
    def get_next_command(self):
        """Sıradaki komutu döndürür, yoksa None döner"""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None
    
    def set_language(self, language_code):
        """Dil ayarını değiştirir"""
        self.language = language_code
        return True
    
    def set_energy_threshold(self, threshold):
        """Gürültü eşiğini ayarlar"""
        if 0 < threshold < 10000:
            self.energy_threshold = threshold
            if self.recognizer:
                self.recognizer.energy_threshold = threshold
            return True
        return False
    
    def set_command_threshold(self, threshold):
        """Komut algılama hassasiyetini ayarlar"""
        if 0.0 <= threshold <= 1.0:
            self.command_threshold = threshold
            return True
        return False
    
    def add_command(self, command_text, action_name):
        """Yeni bir komut ekler"""
        if command_text and action_name:
            self.commands[command_text.lower()] = action_name
            return True
        return False
    
    def remove_command(self, command_text):
        """Bir komutu kaldırır"""
        if command_text.lower() in self.commands:
            del self.commands[command_text.lower()]
            return True
        return False
    
    def get_all_commands(self):
        """Tüm komutları döndürür"""
        return self.commands.copy()
    
    def is_available(self):
        """Ses tanıma özelliğinin kullanılabilir olup olmadığını kontrol eder"""
        try:
            sr.Microphone()
            return True
        except:
            return False