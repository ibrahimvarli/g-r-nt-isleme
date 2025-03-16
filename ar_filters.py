import cv2
import numpy as np
import os
from math import sin, cos, radians

class ARFilters:
    def __init__(self):
        # AR filtreleri için gerekli kaynakları yükle
        self.filters_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources', 'filters')
        
        # Filtreleri saklamak için sözlük
        self.filters = {
            'Gözlük': None,
            'Şapka': None,
            'Maske': None,
            'Sakal': None,
            'Hayvan Kulakları': None,
            'Işık Efekti': None
        }
        
        # Filtre dosyaları varsa yükle
        self._load_filter_resources()
        
        # Aktif filtre
        self.active_filter = None
    
    def _load_filter_resources(self):
        """Filtre kaynaklarını yükler"""
        # Kaynaklar klasörü yoksa oluştur
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources', 'filters'), exist_ok=True)
        
        # Burada gerçek filtre görüntüleri yüklenebilir
        # Şimdilik basit renkli şekiller oluşturalım
        
        # Gözlük filtresi (basit mavi dikdörtgen)
        glasses = np.zeros((100, 200, 4), dtype=np.uint8)
        glasses[:, :, 0] = 0    # Blue
        glasses[:, :, 1] = 0    # Green
        glasses[:, :, 2] = 255  # Red
        glasses[:, :, 3] = 150  # Alpha (transparency)
        cv2.rectangle(glasses, (10, 20), (190, 45), (0, 0, 0, 255), 2)
        cv2.rectangle(glasses, (10, 20), (90, 80), (0, 0, 0, 255), 2)
        cv2.rectangle(glasses, (110, 20), (190, 80), (0, 0, 0, 255), 2)
        self.filters['Gözlük'] = glasses
        
        # Şapka filtresi (basit yeşil üçgen)
        hat = np.zeros((150, 250, 4), dtype=np.uint8)
        hat[:, :, 0] = 0    # Blue
        hat[:, :, 1] = 255  # Green
        hat[:, :, 2] = 0    # Red
        hat[:, :, 3] = 180  # Alpha
        pts = np.array([[125, 10], [20, 140], [230, 140]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(hat, [pts], (0, 200, 0, 200))
        self.filters['Şapka'] = hat
        
        # Maske filtresi (basit beyaz dikdörtgen)
        mask = np.zeros((100, 150, 4), dtype=np.uint8)
        mask[:, :, 0] = 255  # Blue
        mask[:, :, 1] = 255  # Green
        mask[:, :, 2] = 255  # Red
        mask[:, :, 3] = 150  # Alpha
        cv2.rectangle(mask, (10, 10), (140, 90), (200, 200, 200, 200), -1)
        self.filters['Maske'] = mask
        
        # Sakal filtresi (basit siyah yarım daire)
        beard = np.zeros((100, 150, 4), dtype=np.uint8)
        beard[:, :, 3] = 0  # Tamamen saydam başlangıç
        cv2.ellipse(beard, (75, 0), (70, 100), 0, 0, 180, (0, 0, 0, 200), -1)
        self.filters['Sakal'] = beard
        
        # Hayvan kulakları (basit üçgenler)
        animal_ears = np.zeros((150, 250, 4), dtype=np.uint8)
        animal_ears[:, :, 3] = 0  # Tamamen saydam başlangıç
        # Sol kulak
        pts1 = np.array([[50, 20], [30, 100], [70, 100]], np.int32)
        pts1 = pts1.reshape((-1, 1, 2))
        # Sağ kulak
        pts2 = np.array([[200, 20], [180, 100], [220, 100]], np.int32)
        pts2 = pts2.reshape((-1, 1, 2))
        cv2.fillPoly(animal_ears, [pts1], (150, 100, 200, 200))
        cv2.fillPoly(animal_ears, [pts2], (150, 100, 200, 200))
        self.filters['Hayvan Kulakları'] = animal_ears
        
        # Işık efekti (parlak sarı daire)
        light_effect = np.zeros((200, 200, 4), dtype=np.uint8)
        light_effect[:, :, 3] = 0  # Tamamen saydam başlangıç
        cv2.circle(light_effect, (100, 100), 80, (0, 255, 255, 150), -1)
        # Işık parlaması efekti
        for r in range(60, 100, 10):
            alpha = 150 - r
            if alpha < 0:
                alpha = 0
            cv2.circle(light_effect, (100, 100), r, (0, 255, 255, alpha), 2)
        self.filters['Işık Efekti'] = light_effect
    
    def set_active_filter(self, filter_name):
        """Aktif filtreyi ayarlar"""
        if filter_name in self.filters:
            self.active_filter = filter_name
            return True
        return False
    
    def apply_filter(self, frame, landmarks):
        """Seçili filtreyi kareye uygular"""
        if self.active_filter is None or landmarks is None:
            return frame
        
        # Filtreyi al
        filter_img = self.filters[self.active_filter]
        if filter_img is None:
            return frame
        
        # Yüz ölçülerini al
        if len(landmarks) < 68:  # En az 68 landmark noktası gerekli
            return frame
        
        # Yüz genişliği ve yüksekliği hesapla
        face_width = max(landmark[0] for landmark in landmarks) - min(landmark[0] for landmark in landmarks)
        face_height = max(landmark[1] for landmark in landmarks) - min(landmark[1] for landmark in landmarks)
        
        # Filtreyi yüz boyutuna göre ölçekle
        if self.active_filter == 'Gözlük':
            # Gözlük için göz konumlarını kullan
            left_eye = landmarks[36]  # Sol göz köşesi
            right_eye = landmarks[45]  # Sağ göz köşesi
            eye_width = right_eye[0] - left_eye[0]
            
            # Gözlük boyutunu ayarla
            filter_width = int(eye_width * 1.5)
            filter_height = int(filter_width * filter_img.shape[0] / filter_img.shape[1])
            filter_resized = cv2.resize(filter_img, (filter_width, filter_height))
            
            # Gözlük konumunu ayarla
            x_offset = int(left_eye[0] - filter_width * 0.25)
            y_offset = int(left_eye[1] - filter_height * 0.5)
            
        elif self.active_filter == 'Şapka':
            # Şapka için alın konumunu kullan
            forehead = landmarks[27]  # Burun üstü
            
            # Şapka boyutunu ayarla
            filter_width = int(face_width * 1.2)
            filter_height = int(filter_width * filter_img.shape[0] / filter_img.shape[1])
            filter_resized = cv2.resize(filter_img, (filter_width, filter_height))
            
            # Şapka konumunu ayarla
            x_offset = int(forehead[0] - filter_width / 2)
            y_offset = int(forehead[1] - filter_height)
            
        elif self.active_filter == 'Maske':
            # Maske için ağız konumunu kullan
            mouth = landmarks[48]  # Ağız köşesi
            
            # Maske boyutunu ayarla
            filter_width = int(face_width * 0.8)
            filter_height = int(face_height * 0.4)
            filter_resized = cv2.resize(filter_img, (filter_width, filter_height))
            
            # Maske konumunu ayarla
            x_offset = int(mouth[0] - filter_width / 2)
            y_offset = int(mouth[1] - filter_height / 2)
            
        elif self.active_filter == 'Sakal':
            # Sakal için çene konumunu kullan
            chin = landmarks[8]  # Çene ucu
            
            # Sakal boyutunu ayarla
            filter_width = int(face_width * 0.8)
            filter_height = int(face_height * 0.4)
            filter_resized = cv2.resize(filter_img, (filter_width, filter_height))
            
            # Sakal konumunu ayarla
            x_offset = int(chin[0] - filter_width / 2)
            y_offset = int(chin[1] - filter_height / 2)
            
        elif self.active_filter == 'Hayvan Kulakları':
            # Kulaklar için baş üstünü kullan
            top_head = landmarks[27]  # Burun üstü
            
            # Kulak boyutunu ayarla
            filter_width = int(face_width * 1.5)
            filter_height = int(filter_width * filter_img.shape[0] / filter_img.shape[1])
            filter_resized = cv2.resize(filter_img, (filter_width, filter_height))
            
            # Kulak konumunu ayarla
            x_offset = int(top_head[0] - filter_width / 2)
            y_offset = int(top_head[1] - filter_height)
            
        elif self.active_filter == 'Işık Efekti':
            # Işık efekti için yüz merkezini kullan
            face_center_x = int(sum(landmark[0] for landmark in landmarks) / len(landmarks))
            face_center_y = int(sum(landmark[1] for landmark in landmarks) / len(landmarks))
            
            # Işık efekti boyutunu ayarla
            filter_width = int(face_width * 2)
            filter_height = filter_width
            filter_resized = cv2.resize(filter_img, (filter_width, filter_height))
            
            # Işık efekti konumunu ayarla
            x_offset = int(face_center_x - filter_width / 2)
            y_offset = int(face_center_y - filter_height / 2)
        else:
            return frame
        
        # Filtreyi kareye uygula
        self._overlay_image(frame, filter_resized, x_offset, y_offset)
        
        return frame
    
    def _overlay_image(self, background, foreground, x_offset, y_offset):
        """Ön plan görüntüsünü arka plan üzerine bindirme"""
        # Ön plan görüntüsünün boyutlarını al
        h, w = foreground.shape[:2]
        
        # Arka plan görüntüsünün boyutlarını al
        bg_h, bg_w = background.shape[:2]
        
        # Ön plan görüntüsünün arka plan sınırları içinde kalmasını sağla
        if x_offset < 0:
            foreground = foreground[:, -x_offset:]
            w += x_offset
            x_offset = 0
        if y_offset < 0:
            foreground = foreground[-y_offset:, :]
            h += y_offset
            y_offset = 0
        if x_offset + w > bg_w:
            foreground = foreground[:, :bg_w-x_offset]
            w = bg_w - x_offset
        if y_offset + h > bg_h:
            foreground = foreground[:bg_h-y_offset, :]
            h = bg_h - y_offset
        
        # Ön plan görüntüsünün alfa kanalını al
        if foreground.shape[2] == 4:
            alpha = foreground[:, :, 3] / 255.0
            for c in range(3):
                background[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                    background[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha) + \
                    foreground[:, :, c] * alpha
        else:
            background[y_offset:y_offset+h, x_offset:x_offset+w] = foreground
        
        return background
    
    def get_available_filters(self):
        """Kullanılabilir filtrelerin listesini döndürür"""
        return list(self.filters.keys())