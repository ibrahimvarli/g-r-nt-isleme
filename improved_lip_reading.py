import cv2
import numpy as np
import time
from collections import deque

class ImprovedLipReading:
    def __init__(self):
        # Dudak indeksleri (48-67 arası noktalar dudakları temsil eder)
        self.lip_indices = list(range(48, 68))
        
        # Kelime tahmin etme için gerekli değişkenler
        self.lip_history = deque(maxlen=30)  # Son 30 dudak şekli
        self.word_buffer = ""
        self.last_prediction = ""
        self.confidence = 0.0
        self.last_prediction_time = 0
        self.prediction_cooldown = 1.0  # saniye
        
        # Kelime kalıpları (gerçek uygulamada makine öğrenimi modeli kullanılır)
        self.word_patterns = {
            "merhaba": [(0.2, 0.3, 0.5), (0.3, 0.4, 0.6), (0.4, 0.3, 0.5), (0.3, 0.2, 0.4)],
            "tamam": [(0.3, 0.2, 0.4), (0.4, 0.3, 0.5), (0.3, 0.4, 0.6)],
            "evet": [(0.4, 0.3, 0.5), (0.3, 0.2, 0.4), (0.2, 0.3, 0.5)],
            "hayır": [(0.2, 0.4, 0.6), (0.3, 0.3, 0.5), (0.4, 0.2, 0.4)],
            "teşekkürler": [(0.3, 0.4, 0.6), (0.4, 0.3, 0.5), (0.3, 0.2, 0.4), (0.2, 0.3, 0.5), (0.3, 0.4, 0.6)]
        }
    
    def extract_lip_region(self, frame, landmarks):
        """Dudak bölgesini çıkarır"""
        if landmarks is None or len(landmarks) < 68:
            return None
        
        # Dudak bölgesini al
        min_x = min(landmarks[i][0] for i in self.lip_indices)
        min_y = min(landmarks[i][1] for i in self.lip_indices)
        max_x = max(landmarks[i][0] for i in self.lip_indices)
        max_y = max(landmarks[i][1] for i in self.lip_indices)
        
        # Bölgeyi genişlet
        padding = 10
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(frame.shape[1], max_x + padding)
        max_y = min(frame.shape[0], max_y + padding)
        
        # Dudak bölgesini kırp
        lip_region = frame[int(min_y):int(max_y), int(min_x):int(max_x)]
        
        # Dudak noktalarını bölgeye göre normalize et
        lip_points = []
        for i in self.lip_indices:
            x, y = landmarks[i]
            lip_points.append((x - min_x, y - min_y))
        
        return lip_region, (int(min_x), int(min_y), int(max_x), int(max_y)), lip_points
    
    def analyze_lip_shape(self, lip_points):
        """Dudak şeklini analiz eder"""
        if not lip_points:
            return "bilinmiyor", 0.0
        
        # Dudak açıklığını hesapla (dikey)
        top_lip_center = lip_points[51] if len(lip_points) > 51 else lip_points[3]
        bottom_lip_center = lip_points[57] if len(lip_points) > 57 else lip_points[9]
        
        # Dudak genişliğini hesapla (yatay)
        left_corner = lip_points[48] if len(lip_points) > 48 else lip_points[0]
        right_corner = lip_points[54] if len(lip_points) > 54 else lip_points[6]
        
        # Eğer tuple ise koordinatları al
        if isinstance(top_lip_center, tuple):
            top_y = top_lip_center[1]
        else:
            top_y = top_lip_center
            
        if isinstance(bottom_lip_center, tuple):
            bottom_y = bottom_lip_center[1]
        else:
            bottom_y = bottom_lip_center
            
        if isinstance(left_corner, tuple):
            left_x = left_corner[0]
        else:
            left_x = left_corner
            
        if isinstance(right_corner, tuple):
            right_x = right_corner[0]
        else:
            right_x = right_corner
        
        # Dudak açıklığı ve genişliği
        lip_height = bottom_y - top_y
        lip_width = right_x - left_x
        
        # Dudak şekli sınıflandırması
        if lip_height < 5:
            shape = "kapalı"
            confidence = 0.9
        elif lip_height < 10:
            shape = "hafif açık"
            confidence = 0.8
        elif lip_height < 15:
            shape = "açık"
            confidence = 0.7
        else:
            shape = "çok açık"
            confidence = 0.6
        
        return shape, confidence
    
    def extract_lip_features(self, lip_region, lip_points):
        """Dudak bölgesinden özellikler çıkarır"""
        # Dudak bölgesi boyutları
        if lip_region.size == 0:
            return (0, 0, 0)
        
        # Renk özellikleri
        hsv = cv2.cvtColor(lip_region, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h_mean = np.mean(h) / 180.0  # 0-1 arası normalize et
        s_mean = np.mean(s) / 255.0
        v_mean = np.mean(v) / 255.0
        
        # Şekil özellikleri
        shape, _ = self.analyze_lip_shape(lip_points)
        shape_value = {
            "kapalı": 0.0,
            "hafif açık": 0.33,
            "açık": 0.66,
            "çok açık": 1.0,
            "bilinmiyor": 0.5
        }.get(shape, 0.5)
        
        # Özellik vektörü
        features = (h_mean, s_mean, shape_value)
        
        # Geçmişe ekle
        self.lip_history.append(features)
        
        return features
    
    def predict_word(self, features):
        """Dudak özelliklerinden kelime tahmin eder"""
        if len(self.lip_history) < 3:
            return "", 0.0
        
        best_word = ""
        best_score = 0.0
        
        # Her kelime kalıbı için benzerlik hesapla
        for word, patterns in self.word_patterns.items():
            similarity = self._calculate_similarity(list(self.lip_history), patterns)
            
            if similarity > best_score and similarity > 0.6:  # Eşik değeri
                best_score = similarity
                best_word = word
        
        # Sonuçları güncelle
        if best_word:
            self.last_prediction = best_word
            self.confidence = best_score
        
        return best_word, best_score
    
    def _calculate_similarity(self, history, patterns):
        """Dudak hareketi geçmişi ile kelime kalıpları arasındaki benzerliği hesaplar"""
        if len(history) < len(patterns):
            return 0.0
        
        max_score = 0.0
        
        # Geçmiş içinde kaydırarak en iyi eşleşmeyi bul
        for i in range(len(history) - len(patterns) + 1):
            segment = history[i:i+len(patterns)]
            score = 0.0
            
            # Her özellik için benzerlik hesapla
            for j in range(len(patterns)):
                pattern = patterns[j]
                actual = segment[j]
                
                # Özellikler arasındaki benzerliği hesapla
                feature_similarity = 1.0 - min(1.0, sum(abs(a - b) for a, b in zip(pattern, actual)) / len(pattern))
                score += feature_similarity
            
            # Ortalama benzerlik
            score /= len(patterns)
            
            if score > max_score:
                max_score = score
        
        return max_score
    
    def visualize_lip_reading(self, frame, landmarks):
        """Dudak okuma sonuçlarını görselleştirir"""
        if landmarks is None or len(landmarks) < 68:
            return frame
        
        # Dudak bölgesini al
        min_x = min(landmarks[i][0] for i in self.lip_indices)
        min_y = min(landmarks[i][1] for i in self.lip_indices)
        max_x = max(landmarks[i][0] for i in self.lip_indices)
        max_y = max(landmarks[i][1] for i in self.lip_indices)
        
        # Dudak bölgesini çiz
        cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 255), 2)
        
        # Dudak noktalarını çiz
        for i in self.lip_indices:
            cv2.circle(frame, landmarks[i], 2, (0, 0, 255), -1)
        
        # Tahmin edilen kelimeyi göster
        if self.last_prediction:
            confidence_text = f"{int(self.confidence * 100)}%"
            cv2.putText(frame, f"{self.last_prediction.upper()} ({confidence_text})", 
                        (int(min_x), int(min_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame