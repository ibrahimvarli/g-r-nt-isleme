import cv2
import numpy as np
import time
from datetime import datetime

class AdvancedFeatures:
    def __init__(self):
        # Göz takibi için değişkenler
        self.blink_threshold = 0.2  # Göz kırpma eşiği
        self.blink_counter = 0
        self.blink_time = time.time()
        self.eye_aspect_ratios = []
        self.eye_closed_time = 0
        self.last_blink_time = time.time()
        
        # Yorgunluk tespiti için değişkenler
        self.fatigue_threshold = 0.5  # Yorgunluk eşiği
        self.fatigue_level = 0.0
        self.fatigue_start_time = None
        self.yawn_counter = 0
        self.yawn_threshold = 0.6  # Esneme eşiği
        self.yawn_time = 0
        
        # Yaşlandırma/gençleştirme için değişkenler
        self.age_effect_level = 0  # -10 (gençleştirme) ile 10 (yaşlandırma) arası
        
        # Sanal makyaj için değişkenler
        self.makeup_type = "none"  # none, light, medium, heavy
        self.makeup_color = (0, 0, 0)  # BGR renk değeri
        
        # Yüz hareketleriyle kontrol için değişkenler
        self.gesture_history = []
        self.last_gesture_time = time.time()
        
        # Göz izleme tabanlı kontrol için değişkenler
        self.gaze_points = []
        self.gaze_direction = "center"  # left, right, up, down, center
        self.gaze_duration = 0
        
    def calculate_eye_aspect_ratio(self, eye_points):
        """Göz açıklık oranını hesaplar (EAR - Eye Aspect Ratio)"""
        # Dikey mesafeler
        # Tuple'ları numpy array'e dönüştür
        p1 = np.array(eye_points[1])
        p2 = np.array(eye_points[2])
        p3 = np.array(eye_points[3])
        p4 = np.array(eye_points[4])
        p5 = np.array(eye_points[5])
        p0 = np.array(eye_points[0])
        
        # Dikey mesafeler
        v1 = np.linalg.norm(p1 - p5)
        v2 = np.linalg.norm(p2 - p4)
        
        # Yatay mesafe
        h = np.linalg.norm(p0 - p3)
        
        # EAR hesaplama
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def detect_blinks(self, left_eye_points, right_eye_points):
        """Göz kırpma tespiti yapar"""
        # Sol ve sağ göz için EAR hesapla
        left_ear = self.calculate_eye_aspect_ratio(left_eye_points)
        right_ear = self.calculate_eye_aspect_ratio(right_eye_points)
        
        # Ortalama EAR
        ear = (left_ear + right_ear) / 2.0
        
        # EAR geçmişini güncelle
        self.eye_aspect_ratios.append(ear)
        if len(self.eye_aspect_ratios) > 30:  # Son 30 kareyi tut
            self.eye_aspect_ratios.pop(0)
        
        # Göz kırpma tespiti
        blinked = False
        if ear < self.blink_threshold:
            self.eye_closed_time += 1
            if self.eye_closed_time >= 3:  # En az 3 kare boyunca kapalı olmalı
                blinked = True
                self.blink_counter += 1
                self.last_blink_time = time.time()
        else:
            self.eye_closed_time = 0
        
        return blinked, ear, self.blink_counter
    
    def detect_fatigue(self, ear, mouth_points):
        """Yorgunluk tespiti yapar"""
        current_time = time.time()
        
        # Göz yorgunluğu tespiti (uzun süre düşük EAR)
        avg_ear = sum(self.eye_aspect_ratios) / max(len(self.eye_aspect_ratios), 1)
        
        # Esneme tespiti (ağız açıklığı)
        # Tuple'ları numpy array'e dönüştür
        try:
            mp2 = np.array(mouth_points[2])
            mp6 = np.array(mouth_points[6])
            mp0 = np.array(mouth_points[0])
            mp4 = np.array(mouth_points[4])
            
            mouth_height = np.linalg.norm(mp2 - mp6)
            mouth_width = np.linalg.norm(mp0 - mp4)
            mouth_ratio = mouth_height / max(mouth_width, 0.001)
        except (TypeError, IndexError) as e:
            # Hata durumunda varsayılan değerler kullan
            print(f"Ağız noktaları işlenirken hata: {e}")
            mouth_ratio = 0
        
        yawning = False
        if mouth_ratio > self.yawn_threshold:
            self.yawn_time += 1
            if self.yawn_time > 10:  # En az 10 kare boyunca esneme
                yawning = True
                if current_time - self.last_blink_time > 3.0:  # Son esneme üzerinden 3 saniye geçtiyse
                    self.yawn_counter += 1
                    self.last_blink_time = current_time  # Yawn time'ı güncelle
        else:
            self.yawn_time = 0
        
        # Yorgunluk seviyesi hesaplama
        blink_rate = self.blink_counter / max((current_time - self.blink_time) / 60, 0.1)  # Dakikadaki kırpma sayısı
        
        # Normal kırpma hızı: dakikada 15-20 kez
        # Düşük kırpma hızı veya çok yüksek kırpma hızı yorgunluk belirtisi olabilir
        if blink_rate < 10 or blink_rate > 30 or avg_ear < 0.25 or self.yawn_counter > 3:
            if self.fatigue_start_time is None:
                self.fatigue_start_time = current_time
            
            fatigue_duration = current_time - self.fatigue_start_time
            self.fatigue_level = min(1.0, fatigue_duration / 60.0)  # 1 dakika sürekli yorgunluk belirtisi -> %100 yorgun
        else:
            self.fatigue_level = max(0.0, self.fatigue_level - 0.01)  # Yavaşça azalt
            if self.fatigue_level < 0.3:
                self.fatigue_start_time = None
        
        return self.fatigue_level, yawning, self.yawn_counter
    
    def apply_age_effect(self, face_img, landmarks, effect_level):
        """Yaşlandırma veya gençleştirme efekti uygular"""
        # Efekt seviyesini -10 (gençleştirme) ile 10 (yaşlandırma) arasında sınırla
        effect_level = max(-10, min(10, effect_level))
        self.age_effect_level = effect_level
        
        # Temel görüntü işleme
        result = face_img.copy()
        
        # Yaşlandırma efekti
        if effect_level > 0:
            # Kontrast azaltma
            alpha = 1.0 - (effect_level / 20.0)
            result = cv2.convertScaleAbs(result, alpha=alpha, beta=10)
            
            # Bulanıklaştırma (cilt pürüzsüzlüğünü azaltma)
            blur_amount = int(effect_level / 2)
            if blur_amount > 0:
                result = cv2.GaussianBlur(result, (2*blur_amount+1, 2*blur_amount+1), 0)
            
            # Gri tonları artırma (saç beyazlatma efekti)
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            gray_intensity = effect_level / 20.0
            result = cv2.addWeighted(result, 1 - gray_intensity, gray, gray_intensity, 0)
        
        # Gençleştirme efekti
        elif effect_level < 0:
            # Pozitif değere çevir
            youth_level = -effect_level
            
            # Kontrast artırma
            alpha = 1.0 + (youth_level / 30.0)
            result = cv2.convertScaleAbs(result, alpha=alpha, beta=-5)
            
            # Bulanıklaştırma (cilt pürüzsüzleştirme)
            blur_amount = int(youth_level / 2)
            if blur_amount > 0:
                face_mask = np.zeros(result.shape[:2], dtype=np.uint8)
                hull = cv2.convexHull(landmarks.astype(np.int32))
                cv2.fillConvexPoly(face_mask, hull, 255)
                
                blurred = cv2.GaussianBlur(result, (2*blur_amount+1, 2*blur_amount+1), 0)
                result = np.where(face_mask[:,:,np.newaxis] == 255, blurred, result)
        
        return result
    
    def apply_virtual_makeup(self, face_img, landmarks, makeup_type="light", makeup_color=(0, 0, 255)):
        """Sanal makyaj uygular"""
        self.makeup_type = makeup_type
        self.makeup_color = makeup_color
        
        result = face_img.copy()
        
        # Yüz maskesi oluştur
        face_mask = np.zeros(result.shape[:2], dtype=np.uint8)
        hull = cv2.convexHull(landmarks.astype(np.int32))
        cv2.fillConvexPoly(face_mask, hull, 255)
        
        # Dudak bölgesi
        mouth_points = landmarks[48:60]  # Dudak noktaları
        mouth_mask = np.zeros(result.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mouth_mask, mouth_points.astype(np.int32), 255)
        
        # Göz bölgeleri
        left_eye = landmarks[36:42]  # Sol göz noktaları
        right_eye = landmarks[42:48]  # Sağ göz noktaları
        
        eyes_mask = np.zeros(result.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(eyes_mask, left_eye.astype(np.int32), 255)
        cv2.fillConvexPoly(eyes_mask, right_eye.astype(np.int32), 255)
        
        # Makyaj tipine göre uygulama
        if makeup_type == "none":
            return result
        
        # Ruj uygulama
        lip_color = makeup_color
        lip_alpha = 0.0
        if makeup_type == "light":
            lip_alpha = 0.3
        elif makeup_type == "medium":
            lip_alpha = 0.5
        elif makeup_type == "heavy":
            lip_alpha = 0.7
        
        lip_layer = np.zeros_like(result)
        lip_layer[mouth_mask == 255] = lip_color
        result = cv2.addWeighted(result, 1.0, lip_layer, lip_alpha, 0)
        
        # Göz makyajı (eyeliner/far)
        if makeup_type in ["medium", "heavy"]:
            eye_color = (makeup_color[0], makeup_color[1], min(255, makeup_color[2] + 50))
            eye_alpha = 0.4 if makeup_type == "medium" else 0.6
            
            # Göz çevresini genişlet
            kernel = np.ones((3, 3), np.uint8)
            eyes_mask_dilated = cv2.dilate(eyes_mask, kernel, iterations=2)
            eyes_mask_outline = eyes_mask_dilated - eyes_mask
            
            eye_layer = np.zeros_like(result)
            eye_layer[eyes_mask_outline == 255] = eye_color
            result = cv2.addWeighted(result, 1.0, eye_layer, eye_alpha, 0)
        
        # Allık (yanak)
        if makeup_type in ["medium", "heavy"]:
            cheek_color = (makeup_color[0], min(255, makeup_color[1] + 50), makeup_color[2])
            cheek_alpha = 0.2 if makeup_type == "medium" else 0.3
            
            # Yanak bölgelerini belirle (basit yaklaşım)
            left_cheek_center = landmarks[29].astype(np.int32)  # Burun ucu
            left_cheek_center[0] -= 30  # Sola kaydır
            
            right_cheek_center = landmarks[29].astype(np.int32)
            right_cheek_center[0] += 30  # Sağa kaydır
            
            cheek_radius = int(np.linalg.norm(landmarks[0] - landmarks[16]) / 8)
            
            cheek_mask = np.zeros(result.shape[:2], dtype=np.uint8)
            cv2.circle(cheek_mask, (left_cheek_center[0], left_cheek_center[1]), cheek_radius, 255, -1)
            cv2.circle(cheek_mask, (right_cheek_center[0], right_cheek_center[1]), cheek_radius, 255, -1)
            
            # Göz ve ağız bölgelerini çıkar
            cheek_mask = cheek_mask & ~eyes_mask & ~mouth_mask
            
            cheek_layer = np.zeros_like(result)
            cheek_layer[cheek_mask == 255] = cheek_color
            result = cv2.addWeighted(result, 1.0, cheek_layer, cheek_alpha, 0)
        
        # Cilt tonu düzeltme (foundation)
        if makeup_type in ["medium", "heavy"]:
            foundation_alpha = 0.15 if makeup_type == "medium" else 0.25
            
            # Yüz maskesini kullan, göz ve ağız bölgelerini çıkar
            foundation_mask = face_mask & ~eyes_mask & ~mouth_mask
            
            # Cilt tonunu düzeltme (hafif bulanıklaştırma ve renk düzeltme)
            foundation = cv2.GaussianBlur(result, (5, 5), 0)
            
            # Cilt tonunu hafifçe düzelt
            foundation_layer = np.zeros_like(result)
            foundation_layer[foundation_mask == 255] = foundation[foundation_mask == 255]
            
            result = cv2.addWeighted(result, 1.0, foundation_layer, foundation_alpha, 0)
        
        return result
    
    def detect_gaze(self, eye_points, face_points):
        """Göz bakış yönünü tespit eder"""
        # Göz merkezini hesapla
        left_eye_center = np.mean(eye_points[36:42], axis=0).astype(np.int32)
        right_eye_center = np.mean(eye_points[42:48], axis=0).astype(np.int32)
        
        # Yüz merkezini hesapla
        face_center = np.mean(face_points, axis=0).astype(np.int32)
        
        # Göz merkezlerinin ortalaması
        eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2, 
                      (left_eye_center[1] + right_eye_center[1]) // 2)
        
        # Göz-yüz merkezi vektörü
        gaze_vector = (eye_center[0] - face_center[0], eye_center[1] - face_center[1])
        
        # Bakış yönünü belirle
        threshold = 5  # Piksel cinsinden eşik değeri
        
        if abs(gaze_vector[0]) < threshold and abs(gaze_vector[1]) < threshold:
            direction = "center"
        elif gaze_vector[0] < -threshold:
            direction = "left"
        elif gaze_vector[0] > threshold:
            direction = "right"
        elif gaze_vector[1] < -threshold:
            direction = "up"
        elif gaze_vector[1] > threshold:
            direction = "down"
        else:
            direction = "center"
        
        # Bakış yönü aynıysa süreyi artır, değilse sıfırla
        if direction == self.gaze_direction:
            self.gaze_duration += 1
        else:
            self.gaze_duration = 0
            self.gaze_direction = direction
        
        # Bakış noktalarını kaydet (son 10 nokta)
        self.gaze_points.append(eye_center)
        if len(self.gaze_points) > 10:
            self.gaze_points.pop(0)
        
        return direction, self.gaze_duration, eye_center
    
    def detect_face_gesture(self, landmarks, prev_landmarks=None):
        """Yüz hareketlerini tespit eder"""
        if prev_landmarks is None or len(prev_landmarks) == 0:
            return "none", 0
        
        # Hareket vektörlerini hesapla
        motion_vectors = landmarks - prev_landmarks
        
        # Toplam hareket miktarı
        total_motion = np.sum(np.abs(motion_vectors))
        
        # Yatay ve dikey hareket bileşenleri
        horizontal_motion = np.sum(motion_vectors[:, 0])
        vertical_motion = np.sum(motion_vectors[:, 1])
        
        # Hareket eşik değerleri
        motion_threshold = 50
        direction_threshold = 30
        
        # Hareket yönünü belirle
        gesture = "none"
        if total_motion > motion_threshold:
            if abs(horizontal_motion) > abs(vertical_motion) and abs(horizontal_motion) > direction_threshold:
                gesture = "head_turn_left" if horizontal_motion < 0 else "head_turn_right"
            elif abs(vertical_motion) > abs(horizontal_motion) and abs(vertical_motion) > direction_threshold:
                gesture = "head_nod_up" if vertical_motion < 0 else "head_nod_down"
        
        # Hareket geçmişini güncelle
        current_time = time.time()
        if gesture != "none":
            self.gesture_history.append((gesture, current_time))
        
        # Son 3 saniyedeki hareketleri tut
        self.gesture_history = [(g, t) for g, t in self.gesture_history if current_time - t < 3.0]
        
        # En sık tekrarlanan hareketi bul
        if self.gesture_history:
            gestures = [g for g, _ in self.gesture_history]
            most_common_gesture = max(set(gestures), key=gestures.count)
            gesture_count = gestures.count(most_common_gesture)
        else:
            most_common_gesture = "none"
            gesture_count = 0
        
        return most_common_gesture, gesture_count
    
    def reset_counters(self):
        """Sayaçları sıfırlar"""
        self.blink_counter = 0
        self.blink_time = time.time()
        self.yawn_counter = 0
        self.fatigue_level = 0.0
        self.fatigue_start_time = None
        self.eye_aspect_ratios = []
        self.gesture_history = []
        self.gaze_points = []
        self.gaze_direction = "center"
        self.gaze_duration = 0