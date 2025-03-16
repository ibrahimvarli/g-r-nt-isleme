import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from PIL import Image, ImageTk
from math import hypot
import time
import pickle
import json
from datetime import datetime
import uuid
import threading
# Import the improved lip reading module
try:
    from improved_lip_reading import ImprovedLipReading as LipReading
except ImportError:
    from lip_reading import LipReading  # Fallback to basic lip reading
# Import voice commands module
from voice_commands import VoiceCommands
# Import AR filters module
from ar_filters import ARFilters
# Import advanced features module
from advanced_features import AdvancedFeatures

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Profesyonel Yüz Tarama ve Modelleme Uygulaması")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        
        # Uygulama versiyonu
        self.app_version = "2.3.0"  # Updated version number
        
        # Sesli komut modülünü başlat
        self.voice_commands = VoiceCommands()
        self.voice_command_active = False
        
        # AR filtreleri modülünü başlat
        self.ar_filters = ARFilters()
        
        # Gelişmiş özellikleri başlat
        self.advanced_features = AdvancedFeatures()
        
        # Ana çerçeve
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sol panel - Kamera görüntüsü ve kontroller
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Kamera görüntüsü için çerçeve
        self.camera_frame = ttk.LabelFrame(self.left_panel, text="Kamera Görüntüsü")
        self.camera_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Kontrol paneli
        self.control_frame = ttk.LabelFrame(self.left_panel, text="Kontroller")
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Başlat/Durdur düğmesi
        self.is_running = False
        self.start_stop_button = ttk.Button(self.control_frame, text="Başlat", command=self.toggle_camera)
        self.start_stop_button.grid(row=0, column=0, padx=5, pady=5)
        
        # Görüntü kaydet düğmesi
        self.capture_button = ttk.Button(self.control_frame, text="Görüntü Kaydet", command=self.capture_image)
        self.capture_button.grid(row=0, column=1, padx=5, pady=5)
        
        # Filtre seçimi
        ttk.Label(self.control_frame, text="Filtre:").grid(row=0, column=2, padx=5, pady=5)
        self.filter_var = tk.StringVar(value="Normal")
        self.filter_combo = ttk.Combobox(self.control_frame, textvariable=self.filter_var, 
                                        values=["Normal", "Siyah-Beyaz", "Sepya", "Negatif", "Bulanık", "Kenar Algılama"])
        self.filter_combo.grid(row=0, column=3, padx=5, pady=5)
        self.filter_combo.bind("<<ComboboxSelected>>", self.update_filter)
        
        # Özellik seçimi
        self.features_frame = ttk.LabelFrame(self.control_frame, text="Özellikler")
        self.features_frame.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky="ew")
        
        # Özellik onay kutuları
        self.show_landmarks_var = tk.BooleanVar(value=True)
        self.show_landmarks_check = ttk.Checkbutton(self.features_frame, text="Yüz Noktaları", 
                                                  variable=self.show_landmarks_var)
        self.show_landmarks_check.grid(row=0, column=0, padx=5, pady=5)
        
        self.show_emotions_var = tk.BooleanVar(value=False)
        self.show_emotions_check = ttk.Checkbutton(self.features_frame, text="Duygu Analizi", 
                                                 variable=self.show_emotions_var)
        self.show_emotions_check.grid(row=0, column=1, padx=5, pady=5)
        
        self.show_age_gender_var = tk.BooleanVar(value=False)
        self.show_age_gender_check = ttk.Checkbutton(self.features_frame, text="Yaş/Cinsiyet Tahmini", 
                                                   variable=self.show_age_gender_var)
        self.show_age_gender_check.grid(row=0, column=2, padx=5, pady=5)
        
        self.show_face_recognition_var = tk.BooleanVar(value=False)
        self.show_face_recognition_check = ttk.Checkbutton(self.features_frame, text="Yüz Tanıma", 
                                                        variable=self.show_face_recognition_var)
        self.show_face_recognition_check.grid(row=0, column=3, padx=5, pady=5)
        
        self.show_face_mesh_var = tk.BooleanVar(value=False)
        self.show_face_mesh_check = ttk.Checkbutton(self.features_frame, text="Yüz Haritası", 
                                                 variable=self.show_face_mesh_var)
        self.show_face_mesh_check.grid(row=1, column=0, padx=5, pady=5)
        
        self.show_measurements_var = tk.BooleanVar(value=False)
        self.show_measurements_check = ttk.Checkbutton(self.features_frame, text="Yüz Ölçümleri", 
                                                    variable=self.show_measurements_var)
        self.show_measurements_check.grid(row=1, column=1, padx=5, pady=5)
        
        # Dudak okuma özelliği
        self.show_lip_reading_var = tk.BooleanVar(value=False)
        self.show_lip_reading_check = ttk.Checkbutton(self.features_frame, text="Dudak Okuma", 
                                                   variable=self.show_lip_reading_var)
        self.show_lip_reading_check.grid(row=1, column=2, padx=5, pady=5)
        
        # AR filtreleri özelliği
        self.ar_filter_var = tk.StringVar(value="Yok")
        ttk.Label(self.features_frame, text="AR Filtresi:").grid(row=2, column=0, padx=5, pady=5)
        self.ar_filter_combo = ttk.Combobox(self.features_frame, textvariable=self.ar_filter_var, 
                                          values=["Yok", "Gözlük", "Şapka", "Maske", "Sakal", "Hayvan Kulakları", "Işık Efekti"])
        self.ar_filter_combo.grid(row=2, column=1, padx=5, pady=5)
        
        # Sesli komut kontrolü
        self.voice_command_var = tk.BooleanVar(value=False)
        self.voice_command_check = ttk.Checkbutton(self.features_frame, text="Sesli Komut Kontrolü", 
                                                variable=self.voice_command_var, command=self.toggle_voice_commands)
        self.voice_command_check.grid(row=2, column=2, padx=5, pady=5)
        
        # Avatar animasyonu
        self.show_avatar_var = tk.BooleanVar(value=False)
        self.show_avatar_check = ttk.Checkbutton(self.features_frame, text="Avatar Animasyonu", 
                                              variable=self.show_avatar_var)
        self.show_avatar_check.grid(row=2, column=3, padx=5, pady=5)
        
        # Gelişmiş özellikler çerçevesi
        self.advanced_frame = ttk.LabelFrame(self.control_frame, text="Gelişmiş Özellikler")
        self.advanced_frame.grid(row=3, column=0, columnspan=4, padx=5, pady=5, sticky="ew")
        
        # Göz takibi ve yorgunluk tespiti
        self.eye_tracking_var = tk.BooleanVar(value=False)
        self.eye_tracking_check = ttk.Checkbutton(self.advanced_frame, text="Göz Takibi", 
                                               variable=self.eye_tracking_var)
        self.eye_tracking_check.grid(row=0, column=0, padx=5, pady=5)
        
        self.fatigue_detection_var = tk.BooleanVar(value=False)
        self.fatigue_detection_check = ttk.Checkbutton(self.advanced_frame, text="Yorgunluk Tespiti", 
                                                    variable=self.fatigue_detection_var)
        self.fatigue_detection_check.grid(row=0, column=1, padx=5, pady=5)
        
        # Yaşlandırma/Gençleştirme
        ttk.Label(self.advanced_frame, text="Yaş Efekti:").grid(row=1, column=0, padx=5, pady=5)
        self.age_effect_var = tk.IntVar(value=0)
        self.age_effect_scale = ttk.Scale(self.advanced_frame, from_=-10, to=10, orient="horizontal",
                                        variable=self.age_effect_var, length=200)
        self.age_effect_scale.grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(self.advanced_frame, text="-10: Gençleştirme, +10: Yaşlandırma").grid(row=1, column=2, padx=5, pady=5)
        
        # Sanal makyaj
        ttk.Label(self.advanced_frame, text="Sanal Makyaj:").grid(row=2, column=0, padx=5, pady=5)
        self.makeup_type_var = tk.StringVar(value="none")
        self.makeup_combo = ttk.Combobox(self.advanced_frame, textvariable=self.makeup_type_var, 
                                       values=["none", "light", "medium", "heavy"])
        self.makeup_combo.grid(row=2, column=1, padx=5, pady=5)
        
        # Makyaj rengi seçimi
        self.makeup_color_button = ttk.Button(self.advanced_frame, text="Renk Seç", command=self.choose_makeup_color)
        self.makeup_color_button.grid(row=2, column=2, padx=5, pady=5)
        self.makeup_color = (0, 0, 255)  # Varsayılan kırmızı (BGR)
        
        # Sağ panel - 3D model ve derinlik görselleştirme
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 3D model görüntüsü için çerçeve
        self.model_frame = ttk.LabelFrame(self.right_panel, text="3D Yüz Modeli")
        self.model_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.model_label = ttk.Label(self.model_frame)
        self.model_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Derinlik görselleştirme için çerçeve
        self.depth_frame = ttk.LabelFrame(self.right_panel, text="Derinlik Görselleştirme")
        self.depth_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.depth_label = ttk.Label(self.depth_frame)
        self.depth_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Yeni: Dudak okuma sonuçları için çerçeve
        self.lip_reading_frame = ttk.LabelFrame(self.right_panel, text="Dudak Okuma Sonuçları")
        self.lip_reading_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.lip_reading_label = ttk.Label(self.lip_reading_frame, text="Henüz bir kelime algılanmadı")
        self.lip_reading_label.pack(fill=tk.X, padx=5, pady=5)
        
        self.lip_reading_confidence = ttk.Progressbar(self.lip_reading_frame, orient="horizontal", length=200, mode="determinate")
        self.lip_reading_confidence.pack(fill=tk.X, padx=5, pady=5)
        
        # Yüz tanıma sonuçları için çerçeve
        self.recognition_frame = ttk.LabelFrame(self.right_panel, text="Yüz Tanıma Sonuçları")
        self.recognition_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Yüz tanıma sonuçları için liste kutusu
        self.recognition_listbox = tk.Listbox(self.recognition_frame, height=5)
        self.recognition_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Yüz kaydetme düğmesi
        self.save_face_button = ttk.Button(self.recognition_frame, text="Yüzü Kaydet", command=self.save_face_data)
        self.save_face_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Yüz veritabanını temizle düğmesi
        self.clear_db_button = ttk.Button(self.recognition_frame, text="Veritabanını Temizle", command=self.clear_face_database)
        self.clear_db_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Durum çubuğu
        self.status_var = tk.StringVar(value="Hazır")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Kamera ve görüntü işleme değişkenleri
        self.cap = None
        self.current_frame = None
        self.current_filter = "Normal"
        self.face_cascade = None
        self.eye_cascade = None
        self.initialize_face_detector()
        
        # Duygu analizi için basit sözlük (gerçek uygulamada ML modeli kullanılabilir)
        self.emotions = ["Mutlu", "Üzgün", "Kızgın", "Şaşkın", "Nötr"]
        
        # Yaş ve cinsiyet için basit tahmin (gerçek uygulamada ML modeli kullanılabilir)
        self.age_ranges = ["18-25", "26-35", "36-45", "46-60", "60+"]
        self.genders = ["Erkek", "Kadın"]
        
        # Yüz tanıma için veritabanı
        self.face_database = {}
        self.load_face_database()
        
        # Yüz ölçümleri için referans değerler
        self.face_measurements = {
            "göz_arası_mesafe": 0,
            "burun_uzunluğu": 0,
            "ağız_genişliği": 0,
            "yüz_genişliği": 0,
            "yüz_yüksekliği": 0
        }
        
        # Yüz haritası için renk paleti
        self.face_mesh_colors = [
            (0, 255, 0),    # Yeşil - çene
            (255, 0, 0),    # Mavi - kaşlar
            (0, 0, 255),    # Kırmızı - burun
            (255, 255, 0),  # Turkuaz - gözler
            (255, 0, 255)   # Mor - dudaklar
        ]
        
        # Dudak okuma sınıfını başlat
        self.lip_reader = LipReading()
        self.last_predicted_word = ""
        self.lip_reading_history = []
        
        # 3D model parametreleri
        self.model_rotation = 0
        self.model_scale = 1.0
        self.model_depth_factor = 1.0
        
        # 3D model kontrolleri
        self.model_controls_frame = ttk.LabelFrame(self.model_frame, text="3D Model Kontrolleri")
        self.model_controls_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        
        ttk.Label(self.model_controls_frame, text="Döndürme:").grid(row=0, column=0, padx=5, pady=5)
        self.rotation_scale = ttk.Scale(self.model_controls_frame, from_=0, to=360, orient="horizontal", 
                                      command=self.update_model_rotation)
        self.rotation_scale.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(self.model_controls_frame, text="Ölçek:").grid(row=1, column=0, padx=5, pady=5)
        self.scale_scale = ttk.Scale(self.model_controls_frame, from_=0.5, to=2.0, orient="horizontal", 
                                   command=self.update_model_scale)
        self.scale_scale.set(1.0)
        self.scale_scale.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(self.model_controls_frame, text="Derinlik:").grid(row=2, column=0, padx=5, pady=5)
        self.depth_scale = ttk.Scale(self.model_controls_frame, from_=0.5, to=2.0, orient="horizontal", 
                                   command=self.update_model_depth)
        self.depth_scale.set(1.0)
        self.depth_scale.grid(row=2, column=1, padx=5, pady=5, sticky="ew")


    def initialize_face_detector(self):
        try:
            # OpenCV'nin yüz ve göz dedektörlerini yükle
            face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            eye_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')
            
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            
            if self.face_cascade.empty() or self.eye_cascade.empty():
                messagebox.showerror("Hata", "Yüz veya göz dedektörü yüklenemedi!")
                return False
                
            self.status_var.set("Yüz ve göz dedektörleri başarıyla yüklendi")
            return True
        except Exception as e:
            messagebox.showerror("Hata", f"Dedektörler yüklenirken hata oluştu: {e}")
            return False
    
    def load_face_database(self):
        """Yüz veritabanını yükle"""
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_database.pkl")
        if os.path.exists(db_path):
            try:
                with open(db_path, 'rb') as f:
                    self.face_database = pickle.load(f)
                self.status_var.set(f"{len(self.face_database)} yüz veritabanından yüklendi")
                self.update_recognition_list()
            except Exception as e:
                print(f"Yüz veritabanı yüklenirken hata: {e}")
                self.status_var.set("Yüz veritabanı yüklenemedi!")
    
    def toggle_camera(self):
        """Toggle camera on/off and update UI accordingly"""
        try:
            if not self.is_running:
                # Initialize video capture
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception("Could not open camera")
                    
                self.is_running = True
                self.start_stop_button.config(text="Durdur")
                self.status_var.set("Kamera başlatıldı")
                self.update_frame()
            else:
                # Release camera resources
                if self.cap is not None:
                    self.cap.release()
                self.is_running = False
                self.start_stop_button.config(text="Başlat")
                self.status_var.set("Kamera durduruldu")
                self.camera_label.config(image="")
                
        except Exception as e:
            messagebox.showerror("Hata", f"Kamera başlatılırken hata oluştu: {e}")
            self.is_running = False
            self.start_stop_button.config(text="Başlat")
            if self.cap is not None:
                self.cap.release()
            self.status_var.set("Kamera görüntüsü alınamadı!")
    
    def update_frame(self):
        """Update camera frame and process it"""
        if not self.is_running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.status_var.set("Kamera görüntüsü alınamadı!")
            return
        
        # Görüntüyü işle
        self.current_frame = frame.copy()
        processed_frame = self.process_frame(frame)
        
        # Dudak okuma işlemi
        if self.show_lip_reading_var.get():
            self.process_lip_reading(frame)
        
        # Görüntüyü Tkinter'da göstermek için dönüştür
        camera_img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        camera_img = Image.fromarray(camera_img)
        camera_img = ImageTk.PhotoImage(image=camera_img)
        
        # Görüntüyü güncelle
        self.camera_label.configure(image=camera_img)
        self.camera_label.image = camera_img
        
        # Tekrar çağır
        self.root.after(10, self.update_frame)
    
    def choose_makeup_color(self):
        # Renk seçici iletişim kutusu
        color = colorchooser.askcolor(title="Makyaj Rengi Seç", initialcolor="#FF0000")
        if color[0] is not None:
            # RGB'den BGR'ye dönüştür (OpenCV için)
            r, g, b = [int(c) for c in color[0]]
            self.makeup_color = (b, g, r)
    
    def process_frame(self, frame):
        # Filtre uygula
        filtered_frame = self.apply_filter(frame)
        
        # Yüz tespiti yap
        face_rect, points = self.detect_face(filtered_frame)
        
        if face_rect is not None:
            # Yüz dikdörtgenini çiz
            x, y, w, h = face_rect
            cv2.rectangle(filtered_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Yüz noktalarını çiz
            if self.show_landmarks_var.get() and points is not None:
                self.draw_landmarks(filtered_frame, points)
                
                # Göz takibi ve yorgunluk tespiti
                if self.eye_tracking_var.get() or self.fatigue_detection_var.get():
                    # Göz noktaları
                    left_eye_points = points[36:42]
                    right_eye_points = points[42:48]
                    mouth_points = points[48:68]
                    
                    if self.eye_tracking_var.get():
                        # Göz kırpma tespiti
                        blinked, ear, blink_count = self.advanced_features.detect_blinks(left_eye_points, right_eye_points)
                        if blinked:
                            cv2.putText(filtered_frame, "Göz Kırpma Algılandı!", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Bakış yönü tespiti
                        gaze_direction, gaze_duration, eye_center = self.advanced_features.detect_gaze(points, points)
                        cv2.putText(filtered_frame, f"Bakış: {gaze_direction}", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    if self.fatigue_detection_var.get():
                        # Yorgunluk tespiti
                        fatigue_level, yawning, yawn_count = self.advanced_features.detect_fatigue(ear, mouth_points)
                        
                        # Yorgunluk seviyesini göster
                        fatigue_color = (0, 255, 0)  # Yeşil (düşük yorgunluk)
                        if fatigue_level > 0.3:
                            fatigue_color = (0, 165, 255)  # Turuncu (orta yorgunluk)
                        if fatigue_level > 0.7:
                            fatigue_color = (0, 0, 255)  # Kırmızı (yüksek yorgunluk)
                            
                        cv2.putText(filtered_frame, f"Yorgunluk: {fatigue_level:.2f}", (10, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, fatigue_color, 2)
                        
                        if yawning:
                            cv2.putText(filtered_frame, "Esneme Algılandı!", (10, 120), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Yaşlandırma/Gençleştirme efekti
                age_effect_level = self.age_effect_var.get()
                if age_effect_level != 0:
                    # Yüz bölgesini kırp
                    x, y, w, h = face_rect
                    face_img = filtered_frame[y:y+h, x:x+w].copy()
                    
                    # Yüz noktalarını yüz bölgesine göre ayarla
                    face_points = points.copy()
                    face_points[:, 0] -= x
                    face_points[:, 1] -= y
                    
                    # Yaş efektini uygula
                    aged_face = self.advanced_features.apply_age_effect(face_img, face_points, age_effect_level)
                    
                    # Efekti uygula
                    filtered_frame[y:y+h, x:x+w] = aged_face
                
                # Sanal makyaj
                makeup_type = self.makeup_type_var.get()
                if makeup_type != "none":
                    # Yüz bölgesini kırp
                    x, y, w, h = face_rect
                    face_img = filtered_frame[y:y+h, x:x+w].copy()
                    
                    # Yüz noktalarını yüz bölgesine göre ayarla
                    face_points = points.copy()
                    face_points[:, 0] -= x
                    face_points[:, 1] -= y
                    
                    # Makyaj uygula
                    makeup_face = self.advanced_features.apply_virtual_makeup(face_img, face_points, 
                                                                           makeup_type, self.makeup_color)
                    
                    # Efekti uygula
                    filtered_frame[y:y+h, x:x+w] = makeup_face
            
            # Yüz haritası göster
            if self.show_face_mesh_var.get() and points is not None:
                self.draw_face_mesh(filtered_frame, points)
            
            # Yüz ölçümlerini göster
            if self.show_measurements_var.get() and points is not None:
                self.calculate_face_measurements(points)
                self.display_measurements(filtered_frame, x, y)
            
            # AR filtreleri uygula
            if hasattr(self, 'ar_filter_var') and self.ar_filter_var.get() != "Yok":
                self.ar_filters.set_active_filter(self.ar_filter_var.get())
                filtered_frame = self.apply_ar_filter(filtered_frame, points, face_rect)
            
            # Gelişmiş 3D model oluştur
            try:
                # Enhanced3DFaceModel sınıfını kullan
                from enhanced_3d_model import Enhanced3DFaceModel
                
                if not hasattr(self, 'face_model_3d'):
                    self.face_model_3d = Enhanced3DFaceModel()
                    self.face_model_3d.rotation_y = self.model_rotation
                    self.face_model_3d.scale = self.model_scale
                    self.face_model_3d.depth_factor = self.model_depth_factor
                    self.face_model_3d.render_mode = "solid"  # wireframe, solid, textured
                
                # 3D model parametrelerini güncelle
                self.face_model_3d.rotation_y = self.model_rotation
                self.face_model_3d.scale = self.model_scale
                self.face_model_3d.depth_factor = self.model_depth_factor
                
                # 3D modeli oluştur - avatar modu kontrolü
                if hasattr(self, 'show_avatar_var') and self.show_avatar_var.get():
                    # Gelişmiş avatar modeli kullan
                    model_img = self.face_model_3d.create_avatar(points, frame)
                else:
                    # Normal 3D model oluştur
                    model_img = self.face_model_3d.create_3d_model(points, frame)
                
                model_img = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)
                model_img = Image.fromarray(model_img)
                model_img = ImageTk.PhotoImage(image=model_img)
                self.model_label.configure(image=model_img)
                self.model_label.image = model_img
            except Exception as e:
                # Hata durumunda basit modele geri dön
                print(f"3D model hatası: {e}")
                model_img = self.create_face_model(points)
                model_img = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)
                model_img = Image.fromarray(model_img)
                model_img = ImageTk.PhotoImage(image=model_img)
                self.model_label.configure(image=model_img)
                self.model_label.image = model_img
            
            # Derinlik görselleştirme - gelişmiş hesaplama kullan
            try:
                if hasattr(self, 'face_model_3d'):
                    depth_points = self.face_model_3d.calculate_depth(points)
                else:
                    depth_points = self.calculate_depth(points)
                    
                depth_img = np.zeros((500, 500, 3), np.uint8)
                
                if depth_points:
                    for i, (x, y, z) in enumerate(depth_points):
                        # Koordinatları derinlik görüntüsüne sığacak şekilde ölçekle
                        x_scaled = int((x / frame.shape[1]) * 500)
                        y_scaled = int((y / frame.shape[0]) * 500)
                        # Derinliğe göre renk (mavi-kırmızı)
                        color = (255 - int(z) * 8, 0, int(z) * 8)
                        cv2.circle(depth_img, (x_scaled, y_scaled), 2, color, -1)
            except Exception as e:
                print(f"Derinlik hesaplama hatası: {e}")
                depth_img = np.zeros((500, 500, 3), np.uint8)
            
            depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)
            depth_img = Image.fromarray(depth_img)
            depth_img = ImageTk.PhotoImage(image=depth_img)
            self.depth_label.configure(image=depth_img)
            self.depth_label.image = depth_img
            
            # Duygu analizi göster
            if self.show_emotions_var.get():
                emotion = self.analyze_emotion(points)
                cv2.putText(filtered_frame, f"Duygu: {emotion}", (x, y - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Yaş ve cinsiyet tahmini göster
            if self.show_age_gender_var.get():
                age, gender = self.estimate_age_gender(points)
                cv2.putText(filtered_frame, f"Yaş: {age}, Cinsiyet: {gender}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Yüz tanıma göster
            if self.show_face_recognition_var.get():
                face_id = self.recognize_face(points)
                if face_id:
                    cv2.putText(filtered_frame, f"Tanındı: {face_id}", (x, y - 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(filtered_frame, "Tanınmadı", (x, y - 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return filtered_frame
    
    def apply_filter(self, frame):
        if self.current_filter == "Siyah-Beyaz":
            return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        elif self.current_filter == "Sepya":
            kernel = np.array([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])
            return cv2.transform(frame, kernel)
        elif self.current_filter == "Negatif":
            return cv2.bitwise_not(frame)
        elif self.current_filter == "Bulanık":
            return cv2.GaussianBlur(frame, (15, 15), 0)
        else:  # Normal
            return frame
    
    def update_filter(self, event=None):
        """Filtre değişikliğini günceller"""
        self.current_filter = self.filter_var.get()
        self.status_var.set(f"Filtre: {self.current_filter}")
        print(f"Filtre değiştirildi: {self.current_filter}")
    
    def detect_face(self, frame):
        # Yüz tespiti için gri tonlamaya dönüştür
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        try:
            # Yüzleri tespit et
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None, None
            
            # İlk tespit edilen yüzü al
            (x, y, w, h) = faces[0]
            
            # Yüz bölgesini kırp
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Gözleri tespit et (daha doğru landmark tespiti için)
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            # Gelişmiş landmark noktaları oluştur
            points = []
            
            # Çene noktaları (0-16)
            for i in range(17):
                jaw_x = x + int(i * w / 16)
                jaw_y = y + h - int(h / 8)
                points.append((jaw_x, jaw_y))
            
            # Kaş noktaları (17-26)
            for i in range(5):
                # Sol kaş
                left_eyebrow_x = x + int(w / 4) + int(i * w / 10)
                left_eyebrow_y = y + int(h / 4)
                points.append((left_eyebrow_x, left_eyebrow_y))
            
            for i in range(5):
                # Sağ kaş
                right_eyebrow_x = x + int(w / 2) + int(i * w / 10)
                right_eyebrow_y = y + int(h / 4)
                points.append((right_eyebrow_x, right_eyebrow_y))
            
            # Burun noktaları (27-35)
            nose_center_x = x + int(w / 2)
            for i in range(9):
                nose_x = nose_center_x
                nose_y = y + int(h / 3) + int(i * h / 15)
                points.append((nose_x, nose_y))
            
            # Göz noktaları (36-47)
            # Eğer gözler tespit edildiyse, gerçek göz konumlarını kullan
            if len(eyes) >= 2:
                # Gözleri sol ve sağ olarak sırala
                eyes = sorted(eyes, key=lambda e: e[0])
                
                # Sol göz
                ex, ey, ew, eh = eyes[0]
                left_eye_center_x = x + ex + ew // 2
                left_eye_center_y = y + ey + eh // 2
                
                # Sağ göz
                ex, ey, ew, eh = eyes[1]
                right_eye_center_x = x + ex + ew // 2
                right_eye_center_y = y + ey + eh // 2
            else:
                # Gözler tespit edilmediyse, tahmin et
                left_eye_center_x = x + int(w / 3)
                left_eye_center_y = y + int(h / 3)
                right_eye_center_x = x + int(2 * w / 3)
                right_eye_center_y = y + int(h / 3)
            
            # Sol göz noktaları
            for i in range(6):
                angle = i * 60
                radius = w / 12
                eye_x = left_eye_center_x + int(radius * np.cos(np.radians(angle)))
                eye_y = left_eye_center_y + int(radius * np.sin(np.radians(angle)))
                points.append((eye_x, eye_y))
            
            # Sağ göz noktaları
            for i in range(6):
                angle = i * 60
                radius = w / 12
                eye_x = right_eye_center_x + int(radius * np.cos(np.radians(angle)))
                eye_y = right_eye_center_y + int(radius * np.sin(np.radians(angle)))
                points.append((eye_x, eye_y))
            
            # Ağız noktaları (48-67)
            mouth_center_x = x + int(w / 2)
            mouth_center_y = y + int(3 * h / 4)
            
            # Dış dudak
            for i in range(12):
                angle = i * 30
                radius = w / 6
                lip_x = mouth_center_x + int(radius * np.cos(np.radians(angle)))
                lip_y = mouth_center_y + int(radius * np.sin(np.radians(angle)))
                points.append((lip_x, lip_y))
            
            # İç dudak
            for i in range(8):
                angle = i * 45
                radius = w / 10
                lip_x = mouth_center_x + int(radius * np.cos(np.radians(angle)))
                lip_y = mouth_center_y + int(radius * np.sin(np.radians(angle)))
                points.append((lip_x, lip_y))
            
            return (x, y, w, h), points
        
        except Exception as e:
            print(f"Yüz tespitinde hata: {e}")
            return None, None
            
    def toggle_camera(self):
        """Toggle camera on/off and update UI accordingly"""
        try:
            if not self.is_running:
                # Initialize video capture
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception("Could not open camera")
                    
                self.is_running = True
                self.start_stop_button.config(text="Durdur")
                self.status_var.set("Kamera başlatıldı")
                self.update_frame()
            else:
                # Release camera resources
                if self.cap is not None:
                    self.cap.release()
                self.is_running = False
                self.start_stop_button.config(text="Başlat")
                self.status_var.set("Kamera durduruldu")
                self.camera_label.config(image="")
                
        except Exception as e:
            messagebox.showerror("Hata", f"Kamera başlatılırken hata oluştu: {e}")
            self.is_running = False
            self.start_stop_button.config(text="Başlat")
            if self.cap is not None:
                self.cap.release()
            self.status_var.set("Kamera görüntüsü alınamadı!")
    
    def update_frame(self):
        """Update camera frame and process it"""
        if not self.is_running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.status_var.set("Kamera görüntüsü alınamadı!")
            return
        
        # Görüntüyü işle
        self.current_frame = frame.copy()
        processed_frame = self.process_frame(frame)
        
        # Dudak okuma işlemi
        if self.show_lip_reading_var.get():
            self.process_lip_reading(frame)
        
        # Görüntüyü Tkinter'da göstermek için dönüştür
        camera_img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        camera_img = Image.fromarray(camera_img)
        camera_img = ImageTk.PhotoImage(image=camera_img)
        
        # Görüntüyü güncelle
        self.camera_label.configure(image=camera_img)
        self.camera_label.image = camera_img
        
        # Tekrar çağır
        self.root.after(10, self.update_frame)
    
    def draw_landmarks(self, frame, points):
        if points is None:
            return
        
        for point in points:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)
    
    def create_face_model(self, points):
        if points is None:
            return np.zeros((500, 500, 3), np.uint8)
        
        # 3D model görselleştirmesi için boş bir görüntü oluştur
        model_img = np.zeros((500, 500, 3), np.uint8)
        
        # Yüz hatlarını çiz
        for i in range(16):
            cv2.line(model_img, points[i], points[i+1], (0, 255, 0), 2)
        
        # Kaşları çiz
        for i in range(17, 21):
            cv2.line(model_img, points[i], points[i+1], (0, 255, 0), 2)
        for i in range(22, 26):
            cv2.line(model_img, points[i], points[i+1], (0, 255, 0), 2)
        
        # Burnu çiz
        for i in range(27, 35):
            cv2.line(model_img, points[i], points[i+1], (0, 255, 0), 2)
        
        # Gözleri çiz
        for i in range(36, 41):
            cv2.line(model_img, points[i], points[i+1], (0, 255, 0), 2)
        cv2.line(model_img, points[41], points[36], (0, 255, 0), 2)
        
        for i in range(42, 47):
            cv2.line(model_img, points[i], points[i+1], (0, 255, 0), 2)
        cv2.line(model_img, points[47], points[42], (0, 255, 0), 2)
        
        # Dudakları çiz
        for i in range(48, 59):
            cv2.line(model_img, points[i], points[i+1], (0, 255, 0), 2)
        cv2.line(model_img, points[59], points[48], (0, 255, 0), 2)
        
        for i in range(60, 67):
            cv2.line(model_img, points[i], points[i+1], (0, 255, 0), 2)
        cv2.line(model_img, points[67], points[60], (0, 255, 0), 2)
        
        return model_img
    
    def calculate_depth(self, points):
        if points is None:
            return []
        
        # Gözler arası mesafeyi referans olarak hesapla
        left_eye = (sum([points[36][0], points[37][0], points[38][0], points[39][0], points[40][0], points[41][0]]) / 6,
                    sum([points[36][1], points[37][1], points[38][1], points[39][1], points[40][1], points[41][1]]) / 6)
        right_eye = (sum([points[42][0], points[43][0], points[44][0], points[45][0], points[46][0], points[47][0]]) / 6,
                     sum([points[42][1], points[43][1], points[44][1], points[45][1], points[46][1], points[47][1]]) / 6)
        
        eye_distance = hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
        
        # Her nokta için derinlik hesapla (basitleştirilmiş model)
        depth_points = []
        for point in points:
            # Burun ve yüz merkezi noktaları daha fazla derinliğe sahip
            if point in points[27:36]:
                depth = 30
            # Gözler ve kaşlar orta derinliğe sahip
            elif point in points[17:27] or point in points[36:48]:
                depth = 15
            # Çene ve dudaklar daha az derinliğe sahip
            else:
                depth = 5
                
            depth_points.append((point[0], point[1], depth))
        
        return depth_points
    
    def analyze_emotion(self, points):
        # Basit bir duygu analizi simülasyonu
        # Gerçek uygulamada, bu bir makine öğrenimi modeli kullanılarak yapılır
        # Burada rastgele bir duygu döndürüyoruz
        import random
        return random.choice(self.emotions)
    
    def estimate_age_gender(self, points):
        # Basit bir yaş ve cinsiyet tahmini simülasyonu
        # Gerçek uygulamada, bu bir makine öğrenimi modeli kullanılarak yapılır
        # Burada rastgele bir yaş aralığı ve cinsiyet döndürüyoruz
        import random
        return random.choice(self.age_ranges), random.choice(self.genders)
    
    def draw_face_mesh(self, frame, points):
        """Yüz haritası çizimi - farklı yüz bölgelerini renkli çizgilerle gösterir"""
        if points is None or len(points) < 68:
            return
        
        # Çene çizgisi (0-16)
        for i in range(16):
            cv2.line(frame, points[i], points[i+1], self.face_mesh_colors[0], 2)
        
        # Sol kaş (17-21)
        for i in range(17, 21):
            cv2.line(frame, points[i], points[i+1], self.face_mesh_colors[1], 2)
        
        # Sağ kaş (22-26)
        for i in range(22, 26):
            cv2.line(frame, points[i], points[i+1], self.face_mesh_colors[1], 2)
        
        # Burun köprüsü (27-30)
        for i in range(27, 30):
            cv2.line(frame, points[i], points[i+1], self.face_mesh_colors[2], 2)
        
        # Burun alt kısmı (31-35)
        for i in range(31, 35):
            cv2.line(frame, points[i], points[i+1], self.face_mesh_colors[2], 2)
        cv2.line(frame, points[35], points[31], self.face_mesh_colors[2], 2)
        
        # Sol göz (36-41)
        for i in range(36, 41):
            cv2.line(frame, points[i], points[i+1], self.face_mesh_colors[3], 2)
        cv2.line(frame, points[41], points[36], self.face_mesh_colors[3], 2)
        
        # Sağ göz (42-47)
        for i in range(42, 47):
            cv2.line(frame, points[i], points[i+1], self.face_mesh_colors[3], 2)
        cv2.line(frame, points[47], points[42], self.face_mesh_colors[3], 2)
        
        # Dış dudak (48-59)
        for i in range(48, 59):
            cv2.line(frame, points[i], points[i+1], self.face_mesh_colors[4], 2)
        cv2.line(frame, points[59], points[48], self.face_mesh_colors[4], 2)
        
        # İç dudak (60-67)
        for i in range(60, 67):
            cv2.line(frame, points[i], points[i+1], self.face_mesh_colors[4], 2)
        cv2.line(frame, points[67], points[60], self.face_mesh_colors[4], 2)
    
    def calculate_face_measurements(self, points):
        """Yüz ölçümlerini hesapla"""
        if points is None or len(points) < 68:
            return
        
        # Göz arası mesafe
        left_eye_center = (sum([points[36][0], points[37][0], points[38][0], points[39][0], points[40][0], points[41][0]]) / 6,
                          sum([points[36][1], points[37][1], points[38][1], points[39][1], points[40][1], points[41][1]]) / 6)
        right_eye_center = (sum([points[42][0], points[43][0], points[44][0], points[45][0], points[46][0], points[47][0]]) / 6,
                           sum([points[42][1], points[43][1], points[44][1], points[45][1], points[46][1], points[47][1]]) / 6)
        self.face_measurements["göz_arası_mesafe"] = hypot(right_eye_center[0] - left_eye_center[0], right_eye_center[1] - left_eye_center[1])
        
        # Burun uzunluğu
        nose_top = points[27]
        nose_bottom = points[33]
        self.face_measurements["burun_uzunluğu"] = hypot(nose_bottom[0] - nose_top[0], nose_bottom[1] - nose_top[1])
        
        # Ağız genişliği
        mouth_left = points[48]
        mouth_right = points[54]
        self.face_measurements["ağız_genişliği"] = hypot(mouth_right[0] - mouth_left[0], mouth_right[1] - mouth_left[1])
        
        # Yüz genişliği
        face_left = points[0]
        face_right = points[16]
        self.face_measurements["yüz_genişliği"] = hypot(face_right[0] - face_left[0], face_right[1] - face_left[1])
        
        # Yüz yüksekliği
        face_top = points[27]
        face_bottom = points[8]
        self.face_measurements["yüz_yüksekliği"] = hypot(face_bottom[0] - face_top[0], face_bottom[1] - face_top[1])
    
    def display_measurements(self, frame, x, y):
        """Yüz ölçümlerini ekranda göster"""
        offset = 70
        for i, (key, value) in enumerate(self.face_measurements.items()):
            text = f"{key}: {value:.1f} piksel"
            cv2.putText(frame, text, (x, y - offset - i*20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def apply_ar_filter(self, frame, points, face_rect):
        """AR filtrelerini uygula"""
        if points is None or face_rect is None:
            return frame
        
        try:
            # Yüz dikdörtgeni ve noktalarını kullanarak filtreyi uygula
            x, y, w, h = face_rect
            
            # Aktif filtreyi al
            filter_name = self.ar_filter_var.get()
            
            if filter_name == "Yok":
                return frame
            
            # Göz merkezlerini hesapla (gözlük filtresi için)
            left_eye_center = (sum([points[36][0], points[37][0], points[38][0], points[39][0], points[40][0], points[41][0]]) // 6,
                              sum([points[36][1], points[37][1], points[38][1], points[39][1], points[40][1], points[41][1]]) // 6)
            right_eye_center = (sum([points[42][0], points[43][0], points[44][0], points[45][0], points[46][0], points[47][0]]) // 6,
                               sum([points[42][1], points[43][1], points[44][1], points[45][1], points[46][1], points[47][1]]) // 6)
            
            # Burun noktası (şapka filtresi için)
            nose_top = points[27]
            
            # Ağız merkezi (maske ve sakal filtresi için)
            mouth_center = (sum([points[48][0], points[54][0]]) // 2,
                           sum([points[48][1], points[54][1]]) // 2)
            
            # Filtreyi uygula
            if filter_name == "Gözlük":
                # Gözlük filtresini gözlerin üzerine yerleştir
                eye_distance = right_eye_center[0] - left_eye_center[0]
                filter_width = int(eye_distance * 2.5)
                filter_height = int(filter_width * 0.5)
                filter_x = left_eye_center[0] - int(filter_width * 0.2)
                filter_y = left_eye_center[1] - int(filter_height * 0.5)
                
                frame = self.ar_filters.apply_filter(frame, filter_name, (filter_x, filter_y), (filter_width, filter_height))
                
            elif filter_name == "Şapka":
                # Şapka filtresini başın üzerine yerleştir
                filter_width = w * 2
                filter_height = h
                filter_x = x - w // 2
                filter_y = y - h
                
                frame = self.ar_filters.apply_filter(frame, filter_name, (filter_x, filter_y), (filter_width, filter_height))
                
            elif filter_name == "Maske":
                # Maske filtresini ağız ve burun üzerine yerleştir
                filter_width = w
                filter_height = h // 2
                filter_x = x
                filter_y = nose_top[1]
                
                frame = self.ar_filters.apply_filter(frame, filter_name, (filter_x, filter_y), (filter_width, filter_height))
                
            elif filter_name == "Sakal":
                # Sakal filtresini çene üzerine yerleştir
                filter_width = w
                filter_height = h // 2
                filter_x = x
                filter_y = mouth_center[1]
                
                frame = self.ar_filters.apply_filter(frame, filter_name, (filter_x, filter_y), (filter_width, filter_height))
                
            elif filter_name == "Hayvan Kulakları":
                # Hayvan kulaklarını başın üzerine yerleştir
                filter_width = w * 2
                filter_height = h
                filter_x = x - w // 2
                filter_y = y - h // 2
                
                frame = self.ar_filters.apply_filter(frame, filter_name, (filter_x, filter_y), (filter_width, filter_height))
                
            elif filter_name == "Işık Efekti":
                # Işık efektini yüzün etrafına yerleştir
                filter_width = w * 2
                filter_height = h * 2
                filter_x = x - w // 2
                filter_y = y - h // 2
                
                frame = self.ar_filters.apply_filter(frame, filter_name, (filter_x, filter_y), (filter_width, filter_height))
            
            return frame
            
        except Exception as e:
            print(f"AR filtresi uygulanırken hata: {e}")
            return frame
    
    def recognize_face(self, points):
        """Basit yüz tanıma - yüz noktalarının konumlarını kullanarak"""
        if not self.face_database or points is None or len(points) < 68:
            return None
        
        # Yüz özelliklerini çıkar
        face_features = self.extract_face_features(points)
        
        # En yakın eşleşmeyi bul
        best_match = None
        min_distance = float('inf')
        
        for face_id, stored_features in self.face_database.items():
            distance = self.calculate_feature_distance(face_features, stored_features)
            if distance < min_distance and distance < 100:  # Eşik değeri
                min_distance = distance
                best_match = face_id
        
        return best_match
    
    def extract_face_features(self, points):
        """Yüz noktalarından özellik vektörü çıkar"""
        features = []
        
        # Göz arası mesafe
        left_eye_center = (sum([points[36][0], points[37][0], points[38][0], points[39][0], points[40][0], points[41][0]]) / 6,
                          sum([points[36][1], points[37][1], points[38][1], points[39][1], points[40][1], points[41][1]]) / 6)
        right_eye_center = (sum([points[42][0], points[43][0], points[44][0], points[45][0], points[46][0], points[47][0]]) / 6,
                           sum([points[42][1], points[43][1], points[44][1], points[45][1], points[46][1], points[47][1]]) / 6)
        eye_distance = hypot(right_eye_center[0] - left_eye_center[0], right_eye_center[1] - left_eye_center[1])
        
        # Normalize edilmiş noktalar (göz mesafesine göre)
        normalized_points = []
        for point in points:
            normalized_points.append((point[0] / eye_distance, point[1] / eye_distance))
        
        # Özellik vektörü oluştur
        for point in normalized_points:
            features.append(point[0])
            features.append(point[1])
        
        return features
    
    def calculate_feature_distance(self, features1, features2):
        """İki özellik vektörü arasındaki mesafeyi hesapla"""
        if len(features1) != len(features2):
            return float('inf')
        
        sum_squared_diff = 0
        for i in range(len(features1)):
            sum_squared_diff += (features1[i] - features2[i]) ** 2
        
        return np.sqrt(sum_squared_diff)
    
    def save_face_data(self):
        """Mevcut yüzü veritabanına kaydet"""
        if self.current_frame is None:
            messagebox.showinfo("Bilgi", "Kaydedilecek yüz yok!")
            return
        
        # Yüz tespiti yap
        face_rect, points = self.detect_face(self.current_frame)
        if face_rect is None or points is None:
            messagebox.showinfo("Bilgi", "Kaydedilecek yüz tespit edilemedi!")
            return
        
        # Yüz özelliklerini çıkar
        face_features = self.extract_face_features(points)
        
        # Yüz ID'si oluştur
        face_id = f"Kişi_{len(self.face_database) + 1}"
        name = tk.simpledialog.askstring("İsim Girin", "Bu yüz için bir isim girin:")
        if name:
            face_id = name
        
        # Veritabanına ekle
        self.face_database[face_id] = face_features
        
        # Veritabanını kaydet
        self.save_face_database()
        
        # Listeyi güncelle
        self.update_recognition_list()
        
        messagebox.showinfo("Başarılı", f"{face_id} veritabanına kaydedildi!")
    
    def save_face_database(self):
        """Yüz veritabanını dosyaya kaydet"""
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_database.pkl")
        try:
            with open(db_path, 'wb') as f:
                pickle.dump(self.face_database, f)
            self.status_var.set(f"Veritabanı kaydedildi: {len(self.face_database)} yüz")
        except Exception as e:
            messagebox.showerror("Hata", f"Veritabanı kaydedilirken hata oluştu: {e}")
    
    def update_recognition_list(self):
        """Tanıma listesini güncelle"""
        self.recognition_listbox.delete(0, tk.END)
        for face_id in self.face_database.keys():
            self.recognition_listbox.insert(tk.END, face_id)
    
    def clear_face_database(self):
        """Yüz veritabanını temizle"""
        if messagebox.askyesno("Onay", "Tüm yüz veritabanını silmek istediğinizden emin misiniz?"):
            self.face_database = {}
            self.save_face_database()
            self.update_recognition_list()
            messagebox.showinfo("Bilgi", "Veritabanı temizlendi!")
    
    def process_lip_reading(self, frame):
        """Dudak okuma işlemini gerçekleştir"""
        # Yüz tespiti yap
        face_rect, points = self.detect_face(frame)
        
        if face_rect is None or points is None:
            return
        
        try:
            # Dudak bölgesini çıkar
            lip_result = self.lip_reader.extract_lip_region(frame, points)
            
            if lip_result is None:
                return
                
            if len(lip_result) == 3:  # ImprovedLipReading kullanılıyor
                lip_region, lip_bbox, lip_points = lip_result
                
                # Dudak şeklini analiz et
                lip_shape, confidence = self.lip_reader.analyze_lip_shape(lip_points)
                
                # Dudak özelliklerini çıkar
                lip_features = self.lip_reader.extract_lip_features(lip_region, lip_points)
                
                # Kelime tahmin et
                current_time = time.time()
                if current_time - self.lip_reader.last_prediction_time > self.lip_reader.prediction_cooldown:
                    predicted_word, word_confidence = self.lip_reader.predict_word(lip_features)
                    
                    if predicted_word and word_confidence > 0.6:  # Güven eşiği
                        self.lip_reader.last_prediction_time = current_time
                        self.lip_reader.word_buffer = predicted_word
                        self.lip_reader.confidence = word_confidence
                        
                        # Sonuçları göster
                        self.lip_reading_label.config(text=f"Okunan: {predicted_word}")
                        self.lip_reading_confidence["value"] = word_confidence * 100
                        
                        # Geçmişe ekle
                        if hasattr(self.lip_reader, 'lip_reading_history'):
                            self.lip_reader.lip_reading_history.append((predicted_word, word_confidence))
                            if len(self.lip_reader.lip_reading_history) > 10:  # Son 10 tahmini tut
                                self.lip_reader.lip_reading_history.pop(0)
            else:  # Temel LipReading kullanılıyor
                lip_region, lip_bbox = lip_result
                
                # Dudak şeklini analiz et
                lip_shape, confidence = self.lip_reader.analyze_lip_shape(points[48:68])
                
                # Kelime tahmin et
                phonemes, phoneme_confidence = self.lip_reader.predict_phoneme(lip_shape, confidence)
                
                if phonemes:
                    # Fonem geçmişini güncelle
                    self.lip_reader.lip_history.append((phonemes[0], phoneme_confidence))
                    
                    # Kelime tahmin et
                    predicted_word = self.lip_reader.predict_word_from_phonemes()
                    
                    if predicted_word:
                        # Sonuçları göster
                        self.lip_reading_label.config(text=f"Okunan: {predicted_word}")
                        self.lip_reading_confidence["value"] = self.lip_reader.confidence * 100
            
            # Dudak bölgesini çerçeve içine al
            if lip_bbox:
                x_min, y_min, x_max, y_max = lip_bbox
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                
                # Dudak şeklini göster
                cv2.putText(frame, f"Dudak: {lip_shape} ({confidence:.2f})", 
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        except Exception as e:
            print(f"Dudak okuma hatası: {e}")
    
    def update_model_rotation(self, value):
        """3D model döndürme değerini güncelle"""
        self.model_rotation = float(value)
        # Eğer yüz modeli oluşturulduysa, döndürme değerini güncelle
        if hasattr(self, 'face_model_3d'):
            self.face_model_3d.rotation_y = self.model_rotation
        # Eğer kamera çalışıyorsa, görüntüyü güncelle
        if self.is_running and hasattr(self, 'current_frame') and self.current_frame is not None:
            self.process_frame(self.current_frame)
    
    def update_model_scale(self, value):
        """3D model ölçek değerini güncelle"""
        self.model_scale = float(value)
        # Eğer yüz modeli oluşturulduysa, ölçek değerini güncelle
        if hasattr(self, 'face_model_3d'):
            self.face_model_3d.scale = self.model_scale
        # Eğer kamera çalışıyorsa, görüntüyü güncelle
        if self.is_running and hasattr(self, 'current_frame') and self.current_frame is not None:
            self.process_frame(self.current_frame)
    
    def update_model_depth(self, value):
        """3D model derinlik değerini güncelle"""
        self.model_depth_factor = float(value)
        # Eğer yüz modeli oluşturulduysa, derinlik değerini güncelle
        if hasattr(self, 'face_model_3d'):
            self.face_model_3d.depth_factor = self.model_depth_factor
        # Eğer kamera çalışıyorsa, görüntüyü güncelle
        if self.is_running and hasattr(self, 'current_frame') and self.current_frame is not None:
            self.process_frame(self.current_frame)
    
    def toggle_voice_commands(self):
        """Sesli komut dinlemeyi aç/kapat"""
        if self.voice_command_var.get():
            # Sesli komut dinlemeyi başlat
            if self.voice_commands.initialize():
                if self.voice_commands.start_listening():
                    self.voice_command_active = True
                    self.status_var.set("Sesli komut dinleme başlatıldı")
                    # Komut işleme döngüsünü başlat
                    self.process_voice_commands()
                else:
                    messagebox.showerror("Hata", "Sesli komut dinleme başlatılamadı!")
                    self.voice_command_var.set(False)
            else:
                messagebox.showerror("Hata", "Mikrofon başlatılamadı!")
                self.voice_command_var.set(False)
        else:
            # Sesli komut dinlemeyi durdur
            if self.voice_command_active:
                self.voice_commands.stop_listening()
                self.voice_command_active = False
                self.status_var.set("Sesli komut dinleme durduruldu")
    
    def process_voice_commands(self):
        """Sesli komutları işle"""
        if not self.voice_command_active:
            return
        
        # Sıradaki komutu al
        command = self.voice_commands.get_next_command()
        
        if command:
            # Komutu işle
            if command == "start":
                if not self.is_running:
                    self.toggle_camera()
            elif command == "stop":
                if self.is_running:
                    self.toggle_camera()
            elif command == "capture":
                self.capture_image()
            elif command == "filter":
                # Bir sonraki filtreye geç
                filters = self.filter_combo["values"]
                current_index = filters.index(self.current_filter) if self.current_filter in filters else 0
                next_index = (current_index + 1) % len(filters)
                self.filter_var.set(filters[next_index])
                self.update_filter()
            elif command.startswith("filter_"):
                # Belirli bir filtreyi seç
                filter_name = command.replace("filter_", "")
                if filter_name == "normal":
                    self.filter_var.set("Normal")
                elif filter_name == "bw":
                    self.filter_var.set("Siyah-Beyaz")
                elif filter_name == "sepia":
                    self.filter_var.set("Sepya")
                elif filter_name == "negative":
                    self.filter_var.set("Negatif")
                elif filter_name == "blur":
                    self.filter_var.set("Bulanık")
                elif filter_name == "edge":
                    self.filter_var.set("Kenar Algılama")
                self.update_filter()
            elif command.startswith("toggle_"):
                # Özellikleri aç/kapat
                feature = command.replace("toggle_", "")
                if feature == "landmarks":
                    self.show_landmarks_var.set(not self.show_landmarks_var.get())
                elif feature == "emotions":
                    self.show_emotions_var.set(not self.show_emotions_var.get())
                elif feature == "age_gender":
                    self.show_age_gender_var.set(not self.show_age_gender_var.get())
                elif feature == "face_recognition":
                    self.show_face_recognition_var.set(not self.show_face_recognition_var.get())
                elif feature == "face_mesh":
                    self.show_face_mesh_var.set(not self.show_face_mesh_var.get())
                elif feature == "measurements":
                    self.show_measurements_var.set(not self.show_measurements_var.get())
                elif feature == "lip_reading":
                    self.show_lip_reading_var.set(not self.show_lip_reading_var.get())
            elif command.startswith("filter_") and command.replace("filter_", "") in ["glasses", "hat", "mask", "beard", "ears", "light"]:
                # AR filtrelerini ayarla
                ar_filter = command.replace("filter_", "")
                if ar_filter == "glasses":
                    self.ar_filter_var.set("Gözlük")
                elif ar_filter == "hat":
                    self.ar_filter_var.set("Şapka")
                elif ar_filter == "mask":
                    self.ar_filter_var.set("Maske")
                elif ar_filter == "beard":
                    self.ar_filter_var.set("Sakal")
                elif ar_filter == "ears":
                    self.ar_filter_var.set("Hayvan Kulakları")
                elif ar_filter == "light":
                    self.ar_filter_var.set("Işık Efekti")
                else:
                    self.ar_filter_var.set("Yok")
            
            # Durum çubuğunu güncelle
            self.status_var.set(f"Sesli komut uygulandı: {command}")
        
        # Tekrar çağır
        self.root.after(100, self.process_voice_commands)
    
    def capture_image(self):
        if self.current_frame is None:
            messagebox.showinfo("Bilgi", "Görüntü yok!")
            return
        
        # Dosya kaydetme iletişim kutusu
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Görüntüyü kaydet
            cv2.imwrite(file_path, self.current_frame)
            self.status_var.set(f"Görüntü kaydedildi: {file_path}")
            messagebox.showinfo("Başarılı", "Görüntü başarıyla kaydedildi!")
        except Exception as e:
            messagebox.showerror("Hata", f"Görüntü kaydedilirken hata oluştu: {e}")

# Ana uygulama başlatma
def main():
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()