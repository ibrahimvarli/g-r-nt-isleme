import cv2
import numpy as np
import os
from collections import deque

class LipReading:
    def __init__(self):
        # Dudak okuma için gerekli değişkenler
        self.lip_history = deque(maxlen=30)  # Son 30 kare için dudak pozisyonlarını sakla
        self.word_buffer = ""
        self.confidence = 0.0
        
        # Basit dudak şekli - kelime eşleştirmeleri (gerçek uygulamada daha karmaşık bir model kullanılır)
        self.lip_shapes = {
            "open": ["A", "E", "I", "O", "U"],
            "closed": ["M", "B", "P"],
            "wide": ["E", "I", "S"],
            "round": ["O", "U"],
            "neutral": ["T", "D", "N"]
        }
        
        # Türkçe kelime veritabanı (basit bir örnek)
        self.turkish_words = [
            "MERHABA", "NASIL", "EVET", "HAYIR", "TAMAM", 
            "TEŞEKKÜR", "LÜTFEN", "YARDIM", "ANLADIM", "GÖRÜŞÜRÜZ"
        ]
    
    def extract_lip_region(self, frame, points):
        """Dudak bölgesini çıkar"""
        if points is None or len(points) < 68:
            return None
        
        # Dudak noktaları (48-67)
        lip_points = points[48:68]
        
        # Dudak bölgesinin sınırlarını belirle
        x_min = min(p[0] for p in lip_points)
        y_min = min(p[1] for p in lip_points)
        x_max = max(p[0] for p in lip_points)
        y_max = max(p[1] for p in lip_points)
        
        # Bölgeyi biraz genişlet
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)
        
        # Dudak bölgesini kırp
        lip_region = frame[y_min:y_max, x_min:x_max]
        
        # Bölge geçerli değilse None döndür
        if lip_region.size == 0:
            return None
            
        return lip_region, (x_min, y_min, x_max, y_max)
    
    def analyze_lip_shape(self, lip_points):
        """Dudak şeklini analiz et"""
        if lip_points is None or len(lip_points) < 20:  # 48-67 arası dudak noktaları
            return "unknown", 0.0
        
        # Dış dudak noktaları (48-59)
        outer_lip = lip_points[:12]
        
        # İç dudak noktaları (60-67)
        inner_lip = lip_points[12:]
        
        # Dudak genişliği ve yüksekliği hesapla
        width = np.linalg.norm(np.array(outer_lip[0]) - np.array(outer_lip[6]))
        height = np.linalg.norm(np.array(outer_lip[3]) - np.array(outer_lip[9]))
        
        # İç dudak alanı hesapla
        inner_area = cv2.contourArea(np.array(inner_lip).astype(np.int32))
        
        # Dudak şeklini belirle
        shape = "neutral"  # Varsayılan
        confidence = 0.5   # Varsayılan güven değeri
        
        # Genişlik/yükseklik oranı
        aspect_ratio = width / height if height > 0 else 0
        
        if inner_area < 100:  # Kapalı dudaklar
            shape = "closed"
            confidence = 0.7
        elif aspect_ratio > 2.0:  # Geniş dudaklar
            shape = "wide"
            confidence = 0.8
        elif aspect_ratio < 1.2 and inner_area > 300:  # Yuvarlak dudaklar
            shape = "round"
            confidence = 0.8
        elif inner_area > 400:  # Açık dudaklar
            shape = "open"
            confidence = 0.9
        
        return shape, confidence
    
    def predict_phoneme(self, lip_shape, confidence):
        """Dudak şekline göre olası fonemleri tahmin et"""
        if lip_shape in self.lip_shapes:
            return self.lip_shapes[lip_shape], confidence
        return [], 0.0
    
    def update(self, frame, points):
        """Dudak okuma işlemini güncelle"""
        if points is None or len(points) < 68:
            return None, None, 0.0
        
        # Dudak bölgesini çıkar
        lip_result = self.extract_lip_region(frame, points)
        if lip_result is None:
            return None, None, 0.0
            
        lip_region, lip_bbox = lip_result
        
        # Dudak noktaları (48-67)
        lip_points = points[48:68]
        
        # Dudak şeklini analiz et
        lip_shape, confidence = self.analyze_lip_shape(lip_points)
        
        # Dudak şeklini geçmişe ekle
        self.lip_history.append(lip_shape)
        
        # Son birkaç karedeki dudak şekillerini analiz et
        if len(self.lip_history) >= 10:
            # En sık görülen dudak şekli
            from collections import Counter
            shape_counts = Counter(self.lip_history)
            most_common_shape = shape_counts.most_common(1)[0][0]
            
            # Olası fonemleri tahmin et
            phonemes, phoneme_confidence = self.predict_phoneme(most_common_shape, confidence)
            
            # Kelime tahmini (basit bir yaklaşım)
            if phonemes and phoneme_confidence > 0.7:
                # Rastgele bir fonem seç
                import random
                phoneme = random.choice(phonemes)
                
                # Kelime tamponuna ekle
                self.word_buffer += phoneme
                
                # Tampon çok uzunsa veya noktalama işareti varsa
                if len(self.word_buffer) > 10 or phoneme in [".", "?", "!"]:
                    # En yakın kelimeyi bul
                    predicted_word = self.find_closest_word(self.word_buffer)
                    self.word_buffer = ""
                    return lip_region, predicted_word, phoneme_confidence
        
        return lip_region, None, confidence
    
    def find_closest_word(self, buffer):
        """Tampondaki harflere en yakın kelimeyi bul"""
        if not buffer:
            return ""
        
        # Basit bir yaklaşım: Türkçe kelime veritabanında ara
        best_match = ""
        best_score = 0
        
        for word in self.turkish_words:
            # Basit bir benzerlik skoru hesapla
            score = 0
            for c in buffer:
                if c in word:
                    score += 1
            
            # Kelime uzunluğuna göre normalize et
            score = score / max(len(word), len(buffer))
            
            if score > best_score:
                best_score = score
                best_match = word
        
        # Eşik değeri
        if best_score > 0.3:
            return best_match
        return ""