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
# Import enhanced 3D model
from enhanced_3d_model import Enhanced3DFaceModel

class TabbedFaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gelişmiş Yüz Tarama ve Modelleme Uygulaması")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        
        # Uygulama versiyonu
        self.app_version = "3.0.0"  # Updated version number
        
        # Sesli komut modülünü başlat
        self.voice_commands = VoiceCommands()
        self.voice_command_active = False
        
        # AR filtreleri modülünü başlat
        self.ar_filters = ARFilters()
        
        # Gelişmiş özellikleri başlat
        self.advanced_features = AdvancedFeatures()
        
        # 3D model modülünü başlat
        self.face_model = Enhanced3DFaceModel()
        
        # Ana çerçeve
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Üst panel - Kamera görüntüsü
        self.top_panel = ttk.Frame(self.main_frame)
        self.top_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Sol panel - Kamera görüntüsü
        self.left_panel = ttk.Frame(self.top_panel)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Kamera görüntüsü için çerçeve
        self.camera_frame = ttk.LabelFrame(self.left_panel, text="Kamera Görüntüsü")
        self.camera_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Sağ panel - 3D model ve derinlik görselleştirme
        self.right_panel = ttk.Frame(self.top_panel)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 3D model görüntüsü için çerçeve
        self.model_frame = ttk.LabelFrame(self.right_panel, text="3D Yüz Modeli")
        self.model_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.model_label = ttk.Label(self.model_frame)
        self.model_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Alt panel - Sekmeli kontrol paneli
        self.bottom_panel = ttk.Frame(self.main_frame)
        self.bottom_panel.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        # Sekmeli arayüz oluştur
        self.tab_control = ttk.Notebook(self.bottom_panel)
        
        # Temel kontroller sekmesi
        self.basic_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.basic_tab, text="Temel Kontroller")
        
        # Filtreler sekmesi
        self.filters_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.filters_tab, text="Filtreler")
        
        # Özellikler sekmesi
        self.features_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.features_tab, text="Özellikler")
        
        # AR Filtreleri sekmesi
        self.ar_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.ar_tab, text="AR Filtreleri")
        
        # 3D Model sekmesi
        self.model_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.model_tab, text="3D Model Ayarları")
        
        # Gelişmiş sekmesi
        self.advanced_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.advanced_tab, text="Gelişmiş Özellikler")
        
        # Sekmeleri yerleştir
        self.tab_control.pack(expand=1, fill=tk.BOTH)
        
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
        self.model_rotation_x = 0
        self.model_rotation_y = 0
        self.model_rotation_z = 0
        self.model_scale = 1.0
        self.model_depth_factor = 1.5
        self.model_mesh_quality = "high"
        self.model_render_mode = "solid"
        
        # Makyaj rengi
        self.makeup_color = (0, 0, 255)  # Varsayılan kırmızı (BGR)
    
    def setup_basic_tab(self):
        """Temel kontroller sekmesini ayarla"""
        # Başlat/Durdur düğmesi
        self.is_running = False
        self.start_stop_button = ttk.Button(self.basic_tab, text="Başlat", command=self.toggle_camera)
        self.start_stop_button.grid(row=0, column=0, padx=10, pady=10)
        
        # Görüntü kaydet düğmesi
        self.capture_button = ttk.Button(self.basic_tab, text="Görüntü Kaydet", command=self.capture_image)
        self.capture_button.grid(row=0, column=1, padx=10, pady=10)
        
        # 3D Model kaydet düğmesi
        self.save_model_button = ttk.Button(self.basic_tab, text="3D Modeli Kaydet", command=self.save_3d_model)
        self.save_model_button.grid(row=0, column=2, padx=10, pady=10)
        
        # Sesli komut kontrolü
        self.voice_command_var = tk.BooleanVar(value=False)
        self.voice_command_check = ttk.Checkbutton(self.basic_tab, text="Sesli Komut Kontrolü", 
                                                variable=self.voice_command_var, command=self.toggle_voice_commands)
        self.voice_command_check.grid(row=0, column=3, padx=10, pady=10)
        
        # Yardım düğmesi
        self.help_button = ttk.Button(self.basic_tab, text="Yardım", command=self.show_help)
        self.help_button.grid(row=0, column=4, padx=10, pady=10)
    
    def setup_filters_tab(self):
        """Filtreler sekmesini ayarla"""
        # Görüntü filtreleri
        ttk.Label(self.filters_tab, text="Görüntü Filtresi:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.filter_var = tk.StringVar(value="Normal")
        self.filter_combo = ttk.Combobox(self.filters_tab, textvariable=self.filter_var, 
                                        values=["Normal", "Siyah-Beyaz", "Sepya", "Negatif", "Bulanık", "Kenar Algılama"])
        self.filter_combo.grid(row=0, column=1, padx=10, pady=10)
        self.filter_combo.bind("<<ComboboxSelected>>", self.update_filter)
        
        # Filtre yoğunluğu
        ttk.Label(self.filters_tab, text="Filtre Yoğunluğu:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.filter_intensity_var = tk.DoubleVar(value=1.0)
        self.filter_intensity_scale = ttk.Scale(self.filters_tab, from_=0.1, to=2.0, orient="horizontal",
                                              variable=self.filter_intensity_var, length=200)
        self.filter_intensity_scale.grid(row=1, column=1, padx=10, pady=10)
    
    def setup_features_tab(self):
        """Özellikler sekmesini ayarla"""
        # Özellik onay kutuları - 1. sütun
        self.show_landmarks_var = tk.BooleanVar(value=True)
        self.show_landmarks_check = ttk.Checkbutton(self.features_tab, text="Yüz Noktaları", 
                                                  variable=self.show_landmarks_var)
        self.show_landmarks_check.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.show_emotions_var = tk.BooleanVar(value=False)
        self.show_emotions_check = ttk.Checkbutton(self.features_tab, text="Duygu Analizi", 
                                                 variable=self.show_emotions_var)
        self.show_emotions_check.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.show_age_gender_var = tk.BooleanVar(value=False)
        self.show_age_gender_check = ttk.Checkbutton(self.features_tab, text="Yaş/Cinsiyet Tahmini", 
                                                   variable=self.show_age_gender_var)
        self.show_age_gender_check.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        # Özellik onay kutuları - 2. sütun
        self.show_face_recognition_var = tk.BooleanVar(value=False)
        self.show_face_recognition_check = ttk.Checkbutton(self.features_tab, text="Yüz Tanıma", 
                                                        variable=self.show_face_recognition_var)
        self.show_face_recognition_check.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        self.show_face_mesh_var = tk.BooleanVar(value=False)
        self.show_face_mesh_check = ttk.Checkbutton(self.features_tab, text="Yüz Haritası", 
                                                 variable=self.show_face_mesh_var)
        self.show_face_mesh_check.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        self.show_measurements_var = tk.BooleanVar(value=False)
        self.show_measurements_check = ttk.Checkbutton(self.features_tab, text="Yüz Ölçümleri", 
                                                    variable=self.show_measurements_var)
        self.show_measurements_check.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        
        # Özellik onay kutuları - 3. sütun
        self.show_lip_reading_var = tk.BooleanVar(value=False)
        self.show_lip_reading_check = ttk.Checkbutton(self.features_tab, text="Dudak Okuma", 
                                                   variable=self.show_lip_reading_var)
        self.show_lip_reading_check.grid(row=0, column=2, padx=10, pady=5, sticky="w")
        
        self.show_avatar_var = tk.BooleanVar(value=False)
        self.show_avatar_check = ttk.Checkbutton(self.features_tab, text="Avatar Animasyonu", 
                                              variable=self.show_avatar_var)
        self.show_avatar_check.grid(row=1, column=2, padx=10, pady=5, sticky="w")
    
    def setup_ar_tab(self):
        """AR Filtreleri sekmesini ayarla"""
        # AR filtresi seçimi
        ttk.Label(self.ar_tab, text="AR Filtresi:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.ar_filter_var = tk.StringVar(value="Yok")
        self.ar_filter_combo = ttk.Combobox(self.ar_tab, textvariable=self.ar_filter_var, 
                                          values=["Yok", "Gözlük", "Şapka", "Maske", "Sakal", "Hayvan Kulakları", "Işık Efekti"])
        self.ar_filter_combo.grid(row=0, column=1, padx=10, pady=10)
        
        # Filtre boyutu ayarı
        ttk.Label(self.ar_tab, text="Filtre Boyutu:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.ar_size_var = tk.DoubleVar(value=1.0)
        self.ar_size_scale = ttk.Scale(self.ar_tab, from_=0.5, to=2.0, orient="horizontal",
                                     variable=self.ar_size_var, length=200)
        self.ar_size_scale.grid(row=1, column=1, padx=10, pady=10)
        
        # Filtre konumu ayarı
        ttk.Label(self.ar_tab, text="Filtre Konumu (X):").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.ar_pos_x_var = tk.IntVar(value=0)
        self.ar_pos_x_scale = ttk.Scale(self.ar_tab, from_=-50, to=50, orient="horizontal",
                                      variable=self.ar_pos_x_var, length=200)
        self.ar_pos_x_scale.grid(row=2, column=1, padx=10, pady=10)
        
        ttk.Label(self.ar_tab, text="Filtre Konumu (Y):").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.ar_pos_y_var = tk.IntVar(value=0)
        self.ar_pos_y_scale = ttk.Scale(self.ar_tab, from_=-50, to=50, orient="horizontal",
                                      variable=self.ar_pos_y_var, length=200)
        self.ar_pos_y_scale.grid(row=3, column=1, padx=10, pady=10)
    
    def setup_model_tab(self):
        """3D Model Ayarları sekmesini ayarla"""
        # Döndürme ayarları
        ttk.Label(self.model_tab, text="X Ekseni Döndürme:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.rotation_x_var = tk.IntVar(value=0)
        self.rotation_x_scale = ttk.Scale(self.model_tab, from_=0, to=360, orient="horizontal",
                                        variable=self.rotation_x_var, length=200)
        self.rotation_x_scale.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(self.model_tab, text="Y Ekseni Döndürme:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.rotation_y_var = tk.IntVar(value=0)
        self.rotation_y_scale = ttk.Scale(self.model_tab, from_=0, to=360, orient="horizontal",
                                        variable=self.rotation_y_var, length=200)
        self.rotation_y_scale.grid(row=1, column=1, padx=10, pady=10)
        
        ttk.Label(self.model_tab, text="Z Ekseni Döndürme:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.rotation_z_var = tk.IntVar(value=0)
        self.rotation_z_scale = ttk.Scale(self.model_tab, from_=0, to=360, orient="horizontal",
                                        variable=self.rotation_z_var, length=200)
        self.rotation_z_scale.grid(row=2, column=1, padx=10, pady=10)
        
        # Ölçek ayarı
        ttk.Label(self.model_tab, text="Model Ölçeği:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.scale_var = tk.DoubleVar(value=1.0)
        self.scale_scale = ttk.Scale(self.model_tab, from_=0.5, to=2.0, orient="horizontal",
                                   variable=self.scale_var, length=200)
        self.scale_scale.grid(row=3, column=1, padx=10, pady=10)
        
        # Derinlik faktörü
        ttk.Label(self.model_tab, text="Derinlik Faktörü:").grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.depth_var = tk.DoubleVar(value=1.5)
        self.depth_scale = ttk.Scale(self.model_tab, from_=0.5, to=3.0, orient="horizontal",
                                   variable=self.depth_var, length=200)
        self.depth_scale.grid(row=4, column=1, padx=10, pady=10)
        
        # Render modu
        ttk.Label(self.model_tab, text="Render Modu:").grid(row=5, column=0, padx=10, pady=10, sticky="w")
        self.render_mode_var = tk.StringVar(value="solid")
        self.render_mode_combo = ttk.Combobox(self.model_tab, textvariable=self.render_mode_var, 
                                            values=["wireframe", "solid", "textured"])
        self.render_mode_combo.grid(row=5, column=1, padx=10, pady=10)
        
        # Mesh kalitesi
        ttk.Label(self.model_tab, text="Mesh Kalitesi:").grid(row=6, column=0, padx=10, pady=10, sticky="w")
        self.mesh_quality_var = tk.StringVar(value="high")
        self.mesh_quality_combo = ttk.Combobox(self.model_tab, textvariable=self.mesh_quality_var, 
                                             values=["low", "medium", "high"])
        self.mesh_quality_combo.grid(row=6, column=1, padx=10, pady=10)
    
    def setup_advanced_tab(self):
        """Gelişmiş Özellikler sekmesini ayarla"""
        # Göz takibi ve yorgunluk tespiti
        self.eye_tracking_var = tk.BooleanVar(value=False)
        self.eye_tracking_check = ttk.Checkbutton(self.advanced_tab, text="Göz Takibi", 
                                               variable=self.eye_tracking_var)
        self.eye_tracking_check.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.fatigue_detection_var = tk.BooleanVar(value=False)
        self.fatigue_detection_check = ttk.Checkbutton(self.advanced_tab, text="Yorgunluk Tespiti", 
                                                    variable=self.fatigue_detection_var)
        self.fatigue_detection_check.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        # Yaşlandırma/Gençleştirme
        ttk.Label(self.advanced_tab, text="Yaş Efekti:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.age_effect_var = tk.IntVar(value=0)
        self.age_effect_scale = ttk.Scale(self.advanced_tab, from_=-10, to=10, orient="horizontal",
                                        variable=self.age_effect_var, length=200)
        self.age_effect_scale.grid(row=1, column=1, padx=10, pady=10)
        ttk.Label(self.advanced_tab, text="-10: Gençleştirme, +10: Yaşlandırma").grid(row=1, column=2, padx=10, pady=10)
        
        # Sanal makyaj
        ttk.Label(self.advanced_tab, text="Sanal Makyaj:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.makeup_type_var = tk.StringVar(value="none")
        self.makeup_combo = ttk.Combobox(self.advanced_tab, textvariable=self.makeup_type_var, 
                                       values=["none", "light", "medium", "heavy"])
        self.makeup_combo.grid(row=2, column=1, padx=10, pady=10)
        
        # Makyaj rengi seçimi
        self.makeup_color_button = ttk.Button(self.advanced_tab, text="Renk Seç", command=self.choose_makeup_color)
        self.makeup_color_button.grid(row=2, column=2, padx=10, pady=10)
        
        # Yüz hareketleriyle kontrol
        self.face_control_var = tk.BooleanVar(value=False)
        self.face_control_check = ttk.Checkbutton(self.advanced_tab, text="Yüz Hareketleriyle Kontrol", 
                                               variable=self.face_control_var)
        self.face_control_check.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        
        # Göz izleme tabanlı kontrol
        self.eye_control_var = tk.BooleanVar(value=False)
        self.eye_control_check = ttk.Checkbutton(self.advanced_tab, text="Göz İzleme Tabanlı Kontrol", 
                                              variable=self.eye_control_var)
        self.eye_control_check.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        
        # Filtre yoğunluğu
        ttk.Label(self.filters_tab, text="Filtre Yoğunluğu:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.filter_intensity_var = tk.DoubleVar(value=1.0)
        self.filter_intensity_scale = ttk.Scale(self.filters_tab, from_=0.1, to=2.0, orient="horizontal",
                                              variable=self.filter_intensity_var, length=200)
        self.filter_intensity_scale.grid(row=1, column=1, padx=10, pady=10)
    
    def setup_features_tab(self):
        """Özellikler sekmesini ayarla"""
        # Özellik onay kutuları - 1. sütun
        self.show_landmarks_var = tk.BooleanVar(value=True)
        self.show_landmarks_check = ttk.Checkbutton(self.features_tab, text="Yüz Noktaları", 
                                                  variable=self.show_landmarks_var)
        self.show_landmarks_check.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.show_emotions_var = tk.BooleanVar(value=False)
        self.show_emotions_check = ttk.Checkbutton(self.features_tab, text="Duygu Analizi", 
                                                 variable=self.show_emotions_var)
        self.show_emotions_check.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.show_age_gender_var = tk.BooleanVar(value=False)
        self.show_age_gender_check = ttk.Checkbutton(self.features_tab, text="Yaş/Cinsiyet Tahmini", 
                                                   variable=self.show_age_gender_var)
        self.show_age_gender_check.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        # Özellik onay kutuları - 2. sütun
        self.show_face_recognition_var = tk.BooleanVar(value=False)
        self.show_face_recognition_check = ttk.Checkbutton(self.features_tab, text="Yüz Tanıma", 
                                                        variable=self.show_face_recognition_var)
        self.show_face_recognition_check.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        self.show_face_mesh_var = tk.BooleanVar(value=False)
        self.show_face_mesh_check = ttk.Checkbutton(self.features_tab, text="Yüz Haritası", 
                                                 variable=self.show_face_mesh_var)
        self.show_face_mesh_check.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        self.show_measurements_var = tk.BooleanVar(value=False)
        self.show_measurements_check = ttk.Checkbutton(self.features_tab, text="Yüz Ölçümleri", 
                                                    variable=self.show_measurements_var)
        self.show_measurements_check.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        
        # Özellik onay kutuları - 3. sütun
        self.show_lip_reading_var = tk.BooleanVar(value=False)
        self.show_lip_reading_check = ttk.Checkbutton(self.features_tab, text="Dudak Okuma", 
                                                   variable=self.show_lip_reading_var)
        self.show_lip_reading_check.grid(row=0, column=2, padx=10, pady=5, sticky="w")
        
        self.show_avatar_var = tk.BooleanVar(value=False)
        self.show_avatar_check = ttk.Checkbutton(self.features_tab, text="Avatar Animasyonu", 
                                              variable=self.show_avatar_var)
        self.show_avatar_check.grid(row=1, column=2, padx=10, pady=5, sticky="w")
    
    def setup_ar_tab(self):
        """AR Filtreleri sekmesini ayarla"""
        # AR filtresi seçimi
        ttk.Label(self.ar_tab, text="AR Filtresi:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.ar_filter_var = tk.StringVar(value="Yok")
        self.ar_filter_combo = ttk.Combobox(self.ar_tab, textvariable=self.ar_filter_var, 
                                          values=["Yok", "Gözlük", "Şapka", "Maske", "Sakal", "Hayvan Kulakları", "Işık Efekti"])
        self.ar_filter_combo.grid(row=0, column=1, padx=10, pady=10)
        
        # Filtre boyutu ayarı
        ttk.Label(self.ar_tab, text="Filtre Boyutu:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.ar_size_var = tk.DoubleVar(value=1.0)
        self.ar_size_scale = ttk.Scale(self.ar_tab, from_=0.5, to=2.0, orient="horizontal",
                                     variable=self.ar_size_var, length=200)
        self.ar_size_scale.grid(row=1, column=1, padx=10, pady=10)
        
        # Filtre konumu ayarı
        ttk.Label(self.ar_tab, text="Filtre Konumu (X):").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.ar_pos_x_var = tk.IntVar(value=0)
        self.ar_pos_x_scale = ttk.Scale(self.ar_tab, from_=-50, to=50, orient="horizontal",
                                      variable=self.ar_pos_x_var, length=200)
        self.ar_pos_x_scale.grid(row=2, column=1, padx=10, pady=10)
        
        ttk.Label(self.ar_tab, text="Filtre Konumu (Y):").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.ar_pos_y_var = tk.IntVar(value=0)
        self.ar_pos_y_scale = ttk.Scale(self.ar_tab, from_=-50, to=50, orient="horizontal",
                                      variable=self.ar_pos_y_var, length=200)
        self.ar_pos_y_scale.grid(row=3, column=1, padx=10, pady=10)
    
    def setup_model_tab(self):
        """3D Model Ayarları sekmesini ayarla"""
        # Döndürme ayarları
        ttk.Label(self.model_tab, text="X Ekseni Döndürme:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.rotation_x_var = tk.IntVar(value=0)
        self.rotation_x_scale = ttk.Scale(self.model_tab, from_=0, to=360, orient="horizontal",
                                        variable=self.rotation_x_var, length=200)
        self.rotation_x_scale.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(self.model_tab, text="Y Ekseni Döndürme:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.rotation_y_var = tk.IntVar(value=0)
        self.rotation_y_scale = ttk.Scale(self.model_tab, from_=0, to=360, orient="horizontal",
                                        variable=self.rotation_y_var, length=200)
        self.rotation_y_scale.grid(row=1, column=1, padx=10, pady=10)
        
        ttk.Label(self.model_tab, text="Z Ekseni Döndürme:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.rotation_z_var = tk.IntVar(value=0)
        self.rotation_z_scale = ttk.Scale(self.model_tab, from_=0, to=360, orient="horizontal",
                                        variable=self.rotation_z_var, length=200)
        self.rotation_z_scale.grid(row=2, column=1, padx=10, pady=10)
        
        # Ölçek ayarı
        ttk.Label(self.model_tab, text="Model Ölçeği:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.scale_var = tk.DoubleVar(value=1.0)
        self.scale_scale = ttk.Scale(self.model_tab, from_=0.5, to=2.0, orient="horizontal",
                                   variable=self.scale_var, length=200)
        self.scale_scale.grid(row=3, column=1, padx=10, pady=10)
        
        # Derinlik faktörü
        ttk.Label(self.model_tab, text="Derinlik Faktörü:").grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.depth_var = tk.DoubleVar(value=1.5)
        self.depth_scale = ttk.Scale(self.model_tab, from_=0.5, to=3.0, orient="horizontal",
                                   variable=self.depth_var, length=200)
        self.depth_scale.grid(row=4, column=1, padx=10, pady=10)
        
        # Render modu
        ttk.Label(self.model_tab, text="Render Modu:").grid(row=5, column=0, padx=10, pady=10, sticky="w")
        self.render_mode_var = tk.StringVar(value="solid")
        self.render_mode_combo = ttk.Combobox(self.model_tab, textvariable=self.render_mode_var, 
                                            values=["wireframe", "solid", "textured"])
        self.render_mode_combo.grid(row=5, column=1, padx=10, pady=10)
        
        # Mesh kalitesi
        ttk.Label(self.model_tab, text="Mesh Kalitesi:").grid(row=6, column=0, padx=10, pady=10, sticky="w")
        self.mesh_quality_var = tk.StringVar(value="high")
        self.mesh_quality_combo = ttk.Combobox(self.model_tab, textvariable=self.mesh_quality_var, 
                                             values=["low", "medium", "high"])
        self.mesh_quality_combo.grid(row=6, column=1, padx=10, pady=10)
    
    def setup_advanced_tab(self):
        """Gelişmiş Özellikler sekmesini ayarla"""
        # Göz takibi ve yorgunluk tespiti
        self.eye_tracking_var = tk.BooleanVar(value=False)
        self.eye_tracking_check = ttk.Checkbutton(self.advanced_tab, text="Göz Takibi", 
                                               variable=self.eye_tracking_var)
        self.eye_tracking_check.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.fatigue_detection_var = tk.BooleanVar(value=False)
        self.fatigue_detection_check = ttk.Checkbutton(self.advanced_tab, text="Yorgunluk Tespiti", 
                                                    variable=self.fatigue_detection_var)
        self.fatigue_detection_check.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        # Yaşlandırma/Gençleştirme
        ttk.Label(self.advanced_tab, text="Yaş Efekti:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.age_effect_var = tk.IntVar(value=0)
        self.age_effect_scale = ttk.Scale(self.advanced_tab, from_=-10, to=10, orient="horizontal",
                                        variable=self.age_effect_var, length=200)
        self.age_effect_scale.grid(row=1, column=1, padx=10, pady=10)
        ttk.Label(self.advanced_tab, text="-10: Gençleştirme, +10: Yaşlandırma").grid(row=1, column=2, padx=10, pady=10)
        
        # Sanal makyaj
        ttk.Label(self.advanced_tab, text="Sanal Makyaj:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.makeup_type_var = tk.StringVar(value="none")
        self.makeup_combo = ttk.Combobox(self.advanced_tab, textvariable=self.makeup_type_var, 
                                       values=["none", "light", "medium", "heavy"])
        self.makeup_combo.grid(row=2, column=1, padx=10, pady=10)
        
        # Makyaj rengi seçimi
        self.makeup_color_button = ttk.Button(self.advanced_tab, text="Renk Seç", command=self.choose_makeup_color)
        self.makeup_color_button.grid(row=2, column=2, padx=10, pady=10)
        
        # Yüz hareketleriyle kontrol
        self.face_control_var = tk.BooleanVar(value=False)
        self.face_control_check = ttk.Checkbutton(self.advanced_tab, text="Yüz Hareketleriyle Kontrol", 
                                               variable=self.face_control_var)
        self.face_control_check.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        
        # Göz izleme tabanlı kontrol
        self.eye_control_var = tk.BooleanVar(value=False)
        self.eye_control_check = ttk.Checkbutton(self.advanced_tab, text="Göz İzleme Tabanlı Kontrol", 
                                              variable=self.eye_control_var)
        self.eye_control_check.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        
        # Filtre yoğunluğu
        ttk.Label(self.filters_tab, text="Filtre Yoğunluğu:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.filter_intensity_var = tk.DoubleVar(value=1.0)
        self.filter_intensity_scale = ttk.Scale(self.filters_tab, from_=0.1, to=2.0, orient="horizontal",
                                              variable=self.filter_intensity_var, length=200)
        self.filter_intensity_scale.grid(row=1, column=1, padx=10, pady=10)
    
    def setup_features_tab(self):
        """Özellikler sekmesini ayarla"""
        # Özellik onay kutuları - 1. sütun
        self.show_landmarks_var = tk.BooleanVar(value=True)
        self.show_landmarks_check = ttk.Checkbutton(self.features_tab, text="Yüz Noktaları", 
                                                  variable=self.show_landmarks_var)
        self.show_landmarks_check.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.show_emotions_var = tk.BooleanVar(value=False)
        self.show_emotions_check = ttk.Checkbutton(self.features_tab, text="Duygu Analizi", 
                                                 variable=self.show_emotions_var)
        self.show_emotions_check.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.show_age_gender_var = tk.BooleanVar(value=False)
        self.show_age_gender_check = ttk.Checkbutton(self.features_tab, text="Yaş/Cinsiyet Tahmini", 
                                                   variable=self.show_age_gender_var)
        self.show_age_gender_check.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        # Özellik onay kutuları - 2. sütun
        self.show_face_recognition_var = tk.BooleanVar(value=False)
        self.show_face_recognition_check = ttk.Checkbutton(self.features_tab, text="Yüz Tanıma", 
                                                        variable=self.show_face_recognition_var)
        self.show_face_recognition_check.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        self.show_face_mesh_var = tk.BooleanVar(value=False)
        self.show_face_mesh_check = ttk.Checkbutton(self.features_tab, text="Yüz Haritası", 
                                                 variable=self.show_face_mesh_var)
        self.show_face_mesh_check.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        self.show_measurements_var = tk.BooleanVar(value=False)
        self.show_measurements_check = ttk.Checkbutton(self.features_tab, text="Yüz Ölçümleri", 
                                                    variable=self.show_measurements_var)
        self.show_measurements_check.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        
        # Özellik onay kutuları - 3. sütun
        self.show_lip_reading_var = tk.BooleanVar(value=False)
        self.show_lip_reading_check = ttk.Checkbutton(self.features_tab, text="Dudak Okuma", 
                                                   variable=self.show_lip_reading_var)
        self.show_lip_reading_check.grid(row=0, column=2, padx=10, pady=5, sticky="w")
        
        self.show_avatar_var = tk.BooleanVar(value=False)
        self.show_avatar_check = ttk.Checkbutton(self.features_tab, text="Avatar Animasyonu", 
                                              variable=self.show_avatar_var)
        self.show_avatar_check.grid(row=1, column=2, padx=10, pady=5, sticky="w")
    
    def setup_ar_tab(self):
        """AR Filtreleri sekmesini ayarla"""
        # AR filtresi seçimi
        ttk.Label(self.ar_tab, text="AR Filtresi:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.ar_filter_var = tk.StringVar(value="Yok")
        self.ar_filter_combo = ttk.Combobox(self.ar_tab, textvariable=self.ar_filter_var, 
                                          values=["Yok", "Gözlük", "Şapka", "Maske", "Sakal", "Hayvan Kulakları", "Işık Efekti"])
        self.ar_filter_combo.grid(row=0, column=1, padx=10, pady=10)
        
        # Filtre boyutu ayarı
        ttk.Label(self.ar_tab, text="Filtre Boyutu:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.ar_size_var = tk.DoubleVar(value=1.0)
        self.ar_size_scale = ttk.Scale(self.ar_tab, from_=0.5, to=2.0, orient="horizontal",
                                     variable=self.ar_size_var, length=200)
        self.ar_size_scale.grid(row=1, column=1, padx=10, pady=10)
        
        # Filtre konumu ayarı
        ttk.Label(self.ar_tab, text="Filtre Konumu (X):").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.ar_pos_x_var = tk.IntVar(value=0)
        self.ar_pos_x_scale = ttk.Scale(self.ar_tab, from_=-50, to=50, orient="horizontal",
                                      variable=self.ar_pos_x_var, length=200)
        self.ar_pos_x_scale.grid(row=2, column=1, padx=10, pady=10)
        
        ttk.Label(self.ar_tab, text="Filtre Konumu (Y):").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.ar_pos_y_var = tk.IntVar(value=0)
        self.ar_pos_y_scale = ttk.Scale(self.ar_tab, from_=-50, to=50, orient="horizontal",
                                      variable=self.ar_pos_y_var, length=200)
        self.ar_pos_y_scale.grid(row=3, column=1, padx=10, pady=10)
    
    def setup_model_tab(self):
        """3D Model Ayarları sekmesini ayarla"""
        # Döndürme ayarları
        ttk.Label(self.model_tab, text="X Ekseni Döndürme:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.rotation_x_var = tk.IntVar(value=0)
        self.rotation_x_scale = ttk.Scale(self.model_tab, from_=0, to=360, orient="horizontal",
                                        variable=self.rotation_x_var, length=200)
        self.rotation_x_scale.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(self.model_tab, text="Y Ekseni Döndürme:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.rotation_y_var = tk.IntVar(value=0)
        self.rotation_y_scale = ttk.Scale(self.model_tab, from_=0, to=360, orient="horizontal",
                                        variable=self.rotation_y_var, length=200)
        self.rotation_y_scale.grid(row=1, column=1, padx=10, pady=10)
        
        ttk.Label(self.model_tab, text="Z Ekseni Döndürme:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.rotation_z_var = tk.IntVar(value=0)
        self.rotation_z_scale = ttk.Scale(self.model_tab, from_=0, to=360, orient="horizontal",
                                        variable=self.rotation_z_var, length=200)
        self.rotation_z_scale.grid(row=2, column=1, padx=10, pady=10)
        
        # Ölçek ayarı
        ttk.Label(self.model_tab, text="Model Ölçeği:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.scale_var = tk.DoubleVar(value=1.0)
        self.scale_scale = ttk.Scale(self.model_tab, from_=0.5, to=2.0, orient="horizontal",
                                   variable=self.scale_var, length=200)
        self.scale_scale.grid(row=3, column=1, padx=10, pady=10)
        
        # Derinlik faktörü
        ttk.Label(self.model_tab, text="Derinlik Faktörü:").grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.depth_var = tk.DoubleVar(value=1.5)
        self.depth_scale = ttk.Scale(self.model_tab, from_=0.5, to=3.0, orient="horizontal",
                                   variable=self.depth_var, length=200)
        self.depth_scale.grid(row=4, column=1, padx=10, pady=10)
        
        # Render modu
        ttk.Label(self.model_tab, text="Render Modu:").grid(row=5, column=0, padx=10, pady=10, sticky="w")
        self.render_mode_var = tk.StringVar(value="solid")
        self.render_mode_combo = ttk.Combobox(self.model_tab, textvariable=self.render_mode_var, 
                                            values=["wireframe", "solid", "textured"])
        self.render_mode_combo.grid(row=5, column=1, padx=10, pady=10)
        
        # Mesh kalitesi
        ttk.Label(self.model_tab, text="Mesh Kalitesi:").grid(row=6, column=0, padx=10, pady=10, sticky="w")
        self.mesh_quality_var = tk.StringVar(value="high")
        self.mesh_quality_combo = ttk.Combobox(self.model_tab, textvariable=self.mesh_quality_var, 
                                             values=["low", "medium", "high"])
        self.mesh_quality_combo.grid(row=6, column=1, padx=10, pady=10)
    
    def setup_advanced_tab(self):
        """Gelişmiş Özellikler sekmesini ayarla"""
        # Göz takibi ve yorgunluk tespiti
        self.eye_tracking_var = tk.BooleanVar(value=False)
        self.eye_tracking_check = ttk.Checkbutton(self.advanced_tab, text="Göz Takibi", 
                                               variable=self.eye_tracking_var)
        self.eye_tracking_check.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.fatigue_detection_var = tk.BooleanVar(value=False)
        self.fatigue_detection_check = ttk.Checkbutton(self.advanced_tab, text="Yorgunluk Tespiti", 
                                                    variable=self.fatigue_detection_var)
        self.fatigue_detection_check.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        # Yaşlandırma/Gençleştirme
        ttk.Label(self.advanced_tab, text="Yaş Efekti:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.age_effect_var = tk.IntVar(value=0)
        self.age_effect_scale = ttk.Scale(self.advanced_tab, from_=-10, to=10, orient="horizontal",
                                        variable=self.age_effect_var, length=200)
        self.age_effect_scale.grid(row=1, column=1, padx=10, pady=10)
        ttk.Label(self.advanced_tab, text="-10: Gençleştirme, +10: Yaşlandırma").grid(row=1, column=2, padx=10, pady=10)
        
        # Sanal makyaj
        ttk.Label(self.advanced_tab, text="Sanal Makyaj:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.makeup_type_var = tk.StringVar(value="none")
        self.makeup_combo = ttk.Combobox(self.advanced_tab, textvariable=self.makeup_type_var, 
                                       values=["none", "light", "medium", "heavy"])
        self.makeup_combo.grid(row=2, column=1, padx=10, pady=10)
        
        # Makyaj rengi seçimi
        self.makeup_color_button = ttk.Button(self.advanced_tab, text="Renk Seç", command=self.choose_makeup_color)
        self.makeup_color_button.grid(row=2, column=2, padx=10, pady=10)
        
        # Yüz hareketleriyle kontrol
        self.face_control_var = tk.BooleanVar(value=False)
        self.face_control_check = ttk.Checkbutton(self.advanced_tab, text="Yüz Hareketleriyle Kontrol", 
                                               variable=self.face_control_var)
        self.face_control_check.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        
        # Göz izleme tabanlı kontrol
        self.eye_control_var = tk.BooleanVar(value=False)
        self.eye_control_check = ttk.Checkbutton(self.advanced_tab, text="Göz İzleme Tabanlı Kontrol", 
                                              variable=self.eye_control_var)
        self.eye_control_check.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        
        # Filtre yoğunluğu
        ttk.Label(self.filters_tab, text="Filtre Yoğunluğu:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.filter_intensity_var = tk.DoubleVar(value=1.0)
        self.filter_intensity_scale = ttk.Scale(self.filters_tab, from_=0.1, to=2.0, orient="horizontal",
                                              variable=self.filter_intensity_var, length=200)
        self.filter_intensity_scale.grid(row=1, column=1, padx=10, pady=10)
    
    def setup_features_tab(self):
        """Özellikler sekmesini ayarla"""
        # Özellik onay kutuları - 1. sütun
        self.show_landmarks_var = tk.BooleanVar(value=True)
        self.show_landmarks_check = ttk.Checkbutton(self.features_tab, text="Yüz Noktaları", 
                                                  variable=self.show_landmarks_var)
        self.show_landmarks_check.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.show_emotions_var = tk.BooleanVar(value=False)
        self.show_emotions_check = ttk.Checkbutton(self.features_tab, text="Duygu Analizi", 
                                                 variable=self.show_emotions_var)
        self.show_emotions_check.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.show_age_gender_var = tk.BooleanVar(value=False)
        self.show_age_gender_check = ttk.Checkbutton(self.features_tab, text="Yaş/Cinsiyet Tahmini", 
                                                   variable=self.show_age_gender_var)
        self.show_age_gender_check.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        # Özellik onay kutuları - 2. sütun
        self.show_face_recognition_var = tk.BooleanVar(value=False)
        self.show_face_recognition_check = ttk.Checkbutton(self.features_tab, text="Yüz Tanıma", 
                                                        variable=self.show_face_recognition_var)
        self.show_face_recognition_check.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        self.show_face_mesh_var = tk.BooleanVar(value=False)
        self.show_face_mesh_check = ttk.Checkbutton(self.features_tab, text="Yüz Haritası", 
                                                 variable=self.show_face_mesh_var)
        self.show_face_mesh_check.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        self.show_measurements_var = tk.BooleanVar(value=False)
        self.show_measurements_check = ttk.Checkbutton(self.features_tab, text="Yüz Ölçümleri", 
                                                    variable=self.show_measurements_var)
        self.show_measurements_check.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        
        # Özellik onay kutuları - 3. sütun
        self.show_lip_reading_var = tk.BooleanVar(value=False)
        self.show_lip_reading_check = ttk.Checkbutton(self.features_tab, text="Dudak Okuma", 
                                                   variable=self.show_lip_reading_var)
        self.show_lip_reading_check.grid(row=0, column=2, padx=10, pady=5, sticky="w")
        
        self.show_avatar_var = tk.BooleanVar(value=False)
        self.show_avatar_check = ttk.Checkbutton(self.features_tab, text="Avatar Animasyonu", 
                                              variable=self.show_avatar_var)
        self.show_avatar_check.grid(row=1, column=2, padx=10, pady=5, sticky="w")
    
    def setup_ar_tab(self):
        """AR Filtreleri sekmesini ayarla"""
        # AR filtresi seçimi
        ttk.Label(self.ar_tab, text="AR Filtresi:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.ar_filter_var = tk.StringVar(value="Yok")
        self.ar_filter_combo = ttk.Combobox(self.ar_tab, textvariable=self.ar_filter_var, 
                                          values=["Yok", "Gözlük", "Şapka", "Maske", "Sakal", "Hayvan Kulakları", "Işık Efekti"])
        self.ar_filter_combo.grid(row=0, column=1, padx=10, pady=10)
        
        # Filtre boyutu ayarı
        ttk.Label(self.ar_tab, text="Filtre Boyutu:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.ar_size_var = tk.DoubleVar(value=1.0)
        self.ar_size_scale = ttk.Scale(self.ar_tab, from_=0.5, to=2.0, orient="horizontal",
                                     variable=self.ar_size_var, length=200)
        self.ar_size_scale.grid(row=1, column=1, padx=10, pady=10)
        
        # Filtre konumu ayarı
        ttk.Label(self.ar_tab, text="Filtre Konumu (X):").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.ar_pos_x_var = tk.IntVar(value=0)
        self.ar_pos_x_scale = ttk.Scale(self.ar_tab, from_=-50, to=50, orient="horizontal",
                                      variable=self.ar_pos_x_var, length=200)
        self.ar_pos_x_scale.grid(row=2, column=1, padx=10, pady=10)
        
        ttk.Label(self.ar_tab, text="Filtre Konumu (Y):").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.ar_pos_y_var = tk.IntVar(value=0)
        self.ar_pos_y_scale = ttk.Scale(self.ar_tab, from_=-50, to=50, orient="horizontal",
                                      variable=self.ar_pos_y_var, length=200)
        self.ar_pos_y_scale.grid(row=3, column=1, padx=10, pady=10)
    
    def setup_model_tab(self):
        """3D Model Ayarları sekmesini ayarla"""
        # Döndürme ayarları
        ttk.Label(self.model_tab, text="X Ekseni Döndürme:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.rotation_x_var = tk.IntVar(value=0)
        self.rotation_x_scale = ttk.Scale(self.model_tab, from_=0, to=360, orient="horizontal",
                                        variable=self.rotation_x_var, length=200)
        self.rotation_x_scale.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(self.model_tab, text="Y Ekseni Döndürme:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.rotation_y_var = tk.IntVar(value=0)
        self.rotation_y_scale = ttk.Scale(self.model_tab, from_=0, to=360, orient="horizontal",
                                        variable=self.rotation_y_var, length=200)
        self.rotation_y_scale.grid(row=1, column=1, padx=10, pady=10)
        
        ttk.Label(self.model_tab, text="Z Ekseni Döndürme:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.rotation_z_var = tk.IntVar(value=0)
        self.rotation_z_scale = ttk.Scale(self.model_tab, from_=0, to=360, orient="horizontal",
                                        variable=self.rotation_z_var, length=200)
        self.rotation_z_scale.grid(row=2, column=1, padx=10, pady=10)
        
        # Ölçek ayarı
        ttk.Label(self.model_tab, text="Model Ölçeği:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.scale_var = tk.DoubleVar(value=1.0)
        self.scale_scale = ttk.Scale(self.model_tab, from_=0.5, to=2.0, orient="horizontal",
                                   variable=self.scale_var, length=200)
        self.scale_scale.grid(row=3, column=1, padx=10, pady=10)
        
        # Derinlik faktörü
        ttk.Label(self.model_tab, text="Derinlik Faktörü:").grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.depth_var = tk.DoubleVar(value=1.5)
        self.depth_scale = ttk.Scale(self.model_tab, from_=0.5, to=3.0, orient="horizontal",
                                   variable=self.depth_var, length=200)
        self.depth_scale.grid(row=4, column=1, padx=10, pady=10)
        
        # Render modu
        ttk.Label(self.model_tab, text="Render Modu:").grid(row=5, column=0, padx=10, pady=10, sticky="w")
        self.render_mode_var = tk.StringVar(value="solid")
        self.render_mode_combo = ttk.Combobox(self.model_tab, textvariable=self.render_mode_var, 
                                            values=["wireframe", "solid", "textured"])
        self.render_mode_combo.grid(row=5, column=1, padx=10, pady=10)
        
        # Mesh kalitesi
        ttk.Label(self.model_tab, text="Mesh Kalitesi:").grid(row=6, column=0, padx=10, pady=10, sticky="w")
        self.mesh_quality_var = tk.StringVar(value="high")
        self.mesh_quality_combo = ttk.Combobox(self.model_tab, textvariable=self.mesh_quality_var, 
                                             values=["low", "medium", "high"])
        self.mesh_quality_combo.grid(row=6, column=1, padx=10, pady=10)
    
    def setup_advanced_tab(self):
        """Gelişmiş Özellikler sekmesini ayarla"""
        # Göz takibi ve yorgunluk tespiti
        self.eye_tracking_var = tk.BooleanVar(value=False)
        self.eye_tracking_check = ttk.Checkbutton(self.advanced_tab, text="Göz Takibi", 
                                               variable=self.eye_tracking_var)
        self.eye_tracking_check.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.fatigue_detection_var = tk.BooleanVar(value=False)
        self.fatigue_detection_check = ttk.Checkbutton(self.advanced_tab, text="Yorgunluk Tespiti", 
                                                    variable=self.fatigue_detection_var)
        self.fatigue_detection_check.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        # Yaşlandırma/Gençleştirme
        ttk.Label(self.advanced_tab, text="Yaş Efekti:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.age_effect_var = tk.IntVar(value=0)
        self.age_effect_scale = ttk.Scale(self.advanced_tab, from_=-10, to=10, orient="horizontal",
                                        variable=self.age_effect_var, length=200)
        self.age_effect_scale.grid(row=1, column=1, padx=10, pady=10)
        ttk.Label(self.advanced_tab, text="-10: Gençleştirme, +10: Yaşlandırma").grid(row=1, column=2, padx=10, pady=10)
        
        # Sanal makyaj
        ttk.Label(self.advanced_tab, text="Sanal Makyaj:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.makeup_type_var = tk.StringVar(value="none")
        self.makeup_combo = ttk.Combobox(self.advanced_tab