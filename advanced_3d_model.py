import cv2
import numpy as np
from math import sin, cos, radians, hypot
import time

class Advanced3DModel:
    def __init__(self):
        # 3D model parametreleri
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0
        self.scale = 1.0
        self.depth_factor = 1.0
        
        # Model kalitesi (nokta sayısı)
        self.quality = "normal"  # low, normal, high
        
        # Animasyon parametreleri
        self.animation_enabled = False
        self.animation_speed = 1.0
        self.animation_frame = 0
        self.last_animation_time = time.time()
        
        # Işık kaynağı parametreleri
        self.light_position = [0, 0, -500]  # x, y, z
        self.light_color = [255, 255, 255]  # r, g, b
        self.light_intensity = 1.0
        
        # Doku parametreleri
        self.texture_enabled = False
        self.wireframe_enabled = True
        self.shading_enabled = True
        
        # Yüz ifadesi parametreleri
        self.expression = "neutral"  # neutral, smile, sad, surprise, angry
        self.expression_intensity = 0.5
        
        # Renk paleti
        self.color_palette = {
            "wireframe": (0, 255, 0),
            "points": (0, 0, 255),
            "surface": (200, 200, 200),
            "highlight": (255, 255, 0)
        }
    
    def set_quality(self, quality):
        """Model kalitesini ayarlar"""
        if quality in ["low", "normal", "high"]:
            self.quality = quality
            return True
        return False
    
    def set_rotation(self, x=None, y=None, z=None):
        """Model rotasyonunu ayarlar"""
        if x is not None:
            self.rotation_x = x
        if y is not None:
            self.rotation_y = y
        if z is not None:
            self.rotation_z = z
    
    def set_scale(self, scale):
        """Model ölçeğini ayarlar"""
        if 0.1 <= scale <= 3.0:
            self.scale = scale
            return True
        return False
    
    def set_depth_factor(self, factor):
        """Derinlik faktörünü ayarlar"""
        if 0.1 <= factor <= 3.0:
            self.depth_factor = factor
            return True
        return False
    
    def toggle_animation(self, enabled=None):
        """Animasyonu açar/kapatır"""
        if enabled is not None:
            self.animation_enabled = enabled
        else:
            self.animation_enabled = not self.animation_enabled
        return self.animation_enabled
    
    def set_animation_speed(self, speed):
        """Animasyon hızını ayarlar"""
        if 0.1 <= speed <= 5.0:
            self.animation_speed = speed
            return True
        return False
    
    def set_light_position(self, x=None, y=None, z=None):
        """Işık kaynağı konumunu ayarlar"""
        if x is not None:
            self.light_position[0] = x
        if y is not None:
            self.light_position[1] = y
        if z is not None:
            self.light_position[2] = z
    
    def set_light_color(self, r=None, g=None, b=None):
        """Işık kaynağı rengini ayarlar"""
        if r is not None:
            self.light_color[0] = max(0, min(255, r))
        if g is not None:
            self.light_color[1] = max(0, min(255, g))
        if b is not None:
            self.light_color[2] = max(0, min(255, b))
    
    def set_light_intensity(self, intensity):
        """Işık kaynağı yoğunluğunu ayarlar"""
        if 0.0 <= intensity <= 2.0:
            self.light_intensity = intensity
            return True
        return False
    
    def toggle_texture(self, enabled=None):
        """Dokuyu açar/kapatır"""
        if enabled is not None:
            self.texture_enabled = enabled
        else:
            self.texture_enabled = not self.texture_enabled
        return self.texture_enabled
    
    def toggle_wireframe(self, enabled=None):
        """Tel kafesi açar/kapatır"""
        if enabled is not None:
            self.wireframe_enabled = enabled
        else:
            self.wireframe_enabled = not self.wireframe_enabled
        return self.wireframe_enabled
    
    def toggle_shading(self, enabled=None):
        """Gölgelendirmeyi açar/kapatır"""
        if enabled is not None:
            self.shading_enabled = enabled
        else:
            self.shading_enabled = not self.shading_enabled
        return self.shading_enabled
    
    def set_expression(self, expression, intensity=None):
        """Yüz ifadesini ayarlar"""
        valid_expressions = ["neutral", "smile", "sad", "surprise", "angry"]
        if expression in valid_expressions:
            self.expression = expression
            if intensity is not None and 0.0 <= intensity <= 1.0:
                self.expression_intensity = intensity
            return True
        return False
    
    def _rotate_point(self, point):
        """Bir noktayı 3D uzayda döndürür"""
        x, y, z = point
        
        # X ekseni etrafında döndürme
        rad_x = radians(self.rotation_x)
        y_new = y * cos(rad_x) - z * sin(rad_x)
        z_new = y * sin(rad_x) + z * cos(rad_x)
        y, z = y_new, z_new
        
        # Y ekseni etrafında döndürme
        rad_y = radians(self.rotation_y)
        x_new = x * cos(rad_y) + z * sin(rad_y)
        z_new = -x * sin(rad_y) + z * cos(rad_y)
        x, z = x_new, z_new
        
        # Z ekseni etrafında döndürme
        rad_z = radians(self.rotation_z)
        x_new = x * cos(rad_z) - y * sin(rad_z)
        y_new = x * sin(rad_z) + y * cos(rad_z)
        x, y = x_new, y_new
        
        return (x, y, z)
    
    def _apply_expression(self, landmarks):
        """Yüz ifadesini uygular"""
        if self.expression == "neutral" or len(landmarks) < 68:
            return landmarks
        
        # İfade için değiştirilecek landmark noktaları
        modified_landmarks = landmarks.copy()
        
        # İfade yoğunluğu faktörü
        factor = self.expression_intensity * 10
        
        if self.expression == "smile":
            # Ağız köşelerini yukarı kaldır
            modified_landmarks[48] = (landmarks[48][0], landmarks[48][1] - factor)
            modified_landmarks[54] = (landmarks[54][0], landmarks[54][1] - factor)
            # Ağzı genişlet
            for i in range(48, 55):
                x_diff = landmarks[i][0] - landmarks[51][0]
                modified_landmarks[i] = (landmarks[i][0] + x_diff * 0.2, modified_landmarks[i][1])
            
        elif self.expression == "sad":
            # Ağız köşelerini aşağı indir
            modified_landmarks[48] = (landmarks[48][0], landmarks[48][1] + factor)
            modified_landmarks[54] = (landmarks[54][0], landmarks[54][1] + factor)
            # Kaşları ortada yukarı kaldır
            for i in range(17, 27):
                if 19 <= i <= 23:
                    modified_landmarks[i] = (landmarks[i][0], landmarks[i][1] - factor * 0.5)
            
        elif self.expression == "surprise":
            # Ağzı aç
            for i in range(56, 68):
                y_diff = landmarks[i][1] - landmarks[51][1]
                modified_landmarks[i] = (landmarks[i][0], landmarks[i][1] + y_diff * 0.5)
            # Kaşları yukarı kaldır
            for i in range(17, 27):
                modified_landmarks[i] = (landmarks[i][0], landmarks[i][1] - factor)
            
        elif self.expression == "angry":
            # Kaşların iç kısmını aşağı indir
            modified_landmarks[21] = (landmarks[21][0], landmarks[21][1] + factor)
            modified_landmarks[22] = (landmarks[22][0], landmarks[22][1] + factor)
            # Ağzı küçült
            for i in range(48, 68):
                x_diff = landmarks[i][0] - landmarks[51][0]
                y_diff = landmarks[i][1] - landmarks[51][1]
                modified_landmarks[i] = (landmarks[i][0] - x_diff * 0.2, landmarks[i][1] - y_diff * 0.2)
        
        return modified_landmarks
    
    def _calculate_lighting(self, point, normal):
        """Bir nokta için ışık hesaplaması yapar"""
        if not self.shading_enabled:
            return 1.0
        
        # Nokta ile ışık kaynağı arasındaki vektör
        light_vec = [
            self.light_position[0] - point[0],
            self.light_position[1] - point[1],
            self.light_position[2] - point[2]
        ]
        
        # Vektör uzunluğu
        length = (light_vec[0]**2 + light_vec[1]**2 + light_vec[2]**2)**0.5
        if length == 0:
            return 0.5
        
        # Vektörü normalize et
        light_vec = [light_vec[0]/length, light_vec[1]/length, light_vec[2]/length]
        
        # Normal vektör ile ışık vektörü arasındaki açının kosinüsü
        dot_product = light_vec[0]*normal[0] + light_vec[1]*normal[1] + light_vec[2]*normal[2]
        
        # Işık faktörü (0.2 ile 1.0 arasında)
        light_factor = max(0.2, min(1.0, dot_product * self.light_intensity))
        
        return light_factor
    
    def create_model(self, landmarks, frame_width, frame_height):
        """Landmark noktalarından 3D model oluşturur"""
        if landmarks is None or len(landmarks) < 68:
            # Boş bir görüntü döndür
            return np.zeros((500, 500, 3), np.uint8)
        
        # Animasyon güncelleme
        if self.animation_enabled:
            current_time = time.time()
            elapsed = current_time - self.last_animation_time
            if elapsed > 0.05:  # 20 FPS
                self.animation_frame += 1
                self.rotation_y = (self.animation_frame * self.animation_speed) % 360
                self.last_animation_time = current_time
        
        # İfadeyi uygula
        landmarks = self._apply_expression(landmarks)
        
        # 3D model için boş görüntü oluştur
        model_img = np.zeros((500, 500, 3), np.uint8)
        
        # Landmark noktalarını 3D uzaya dönüştür
        points_3d = []
        for point in landmarks:
            # 2D koordinatları normalize et
            x = (point[0] - frame_width/2) * self.scale
            y = (point[1] - frame_height/2) * self.scale
            
            # Z koordinatını hesapla (basitleştirilmiş)
            z = 0
            
            # Burun için daha fazla derinlik
            if point in [landmarks[i] for i in range(27, 36)]:
                z = -50 * self.depth_factor
            # Gözler ve kaşlar için orta derinlik
            elif point in [landmarks[i] for i in range(17, 27)] or point in [landmarks[i] for i in range(36, 48)]:
                z = -25 * self.depth_factor
            # Çene ve dudaklar için az derinlik
            else:
                z = -10 * self.depth_factor
            
            # Noktayı döndür
            x, y, z = self._rotate_point((x, y, z))
            
            # 3D noktayı 2D ekrana projeksiyon
            # Basit perspektif projeksiyon
            scale_factor = 500 / (500 - z)  # z ne kadar küçükse (uzaksa) o kadar küçük görünür
            x_2d = int(x * scale_factor + 250)  # Merkezi 250,250 olarak ayarla
            y_2d = int(y * scale_factor + 250)
            
            points_3d.append((x_2d, y_2d, z))
        
        # Yüz bölgelerini çiz
        # Çene çizgisi
        for i in range(16):
            p1 = points_3d[i]
            p2 = points_3d[i+1]
            cv2.line(model_img, (p1[0], p1[1]), (