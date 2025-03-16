import cv2
import numpy as np
import os
from math import hypot, sin, cos, radians

class Enhanced3DFaceModel:
    def __init__(self):
        # 3D model parametreleri
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0
        self.scale = 1.0
        self.depth_factor = 1.5
        self.mesh_quality = "high"  # low, medium, high
        self.render_mode = "solid"  # wireframe, solid, textured
        
        # Yüz bölgeleri için renkler
        self.face_colors = {
            "jaw": (0, 255, 0),      # Yeşil
            "eyebrows": (255, 0, 0),  # Mavi
            "nose": (0, 0, 255),     # Kırmızı
            "eyes": (255, 255, 0),   # Turkuaz
            "lips": (255, 0, 255)     # Mor
        }
        
        # Yüz bölgeleri indeksleri
        self.face_regions = {
            "jaw": list(range(0, 17)),
            "eyebrows": list(range(17, 27)),
            "nose": list(range(27, 36)),
            "eyes": list(range(36, 48)),
            "lips": list(range(48, 68))
        }
        
        # Yüz mesh üçgenleri (basitleştirilmiş)
        self.face_triangles = self._generate_face_triangles()
        
        # Doku haritası için görüntü
        self.texture_image = None
    
    def _generate_face_triangles(self):
        """Yüz mesh üçgenlerini oluştur"""
        # Basitleştirilmiş üçgen listesi
        # Gerçek uygulamada daha kapsamlı bir üçgen listesi kullanılır
        triangles = [
            # Çene üçgenleri
            (0, 1, 17), (1, 2, 17), (2, 3, 17), (3, 4, 17), (4, 5, 17),
            (5, 6, 17), (6, 7, 17), (7, 8, 17), (8, 9, 17), (9, 10, 17),
            (10, 11, 17), (11, 12, 17), (12, 13, 17), (13, 14, 17), (14, 15, 17), (15, 16, 17),
            
            # Sol kaş üçgenleri
            (17, 18, 19), (18, 19, 20), (19, 20, 21),
            
            # Sağ kaş üçgenleri
            (22, 23, 24), (23, 24, 25), (24, 25, 26),
            
            # Burun üçgenleri
            (27, 28, 29), (28, 29, 30), (30, 31, 32), (31, 32, 33), (32, 33, 34), (33, 34, 35),
            
            # Sol göz üçgenleri
            (36, 37, 41), (37, 38, 41), (38, 39, 41), (39, 40, 41),
            
            # Sağ göz üçgenleri
            (42, 43, 47), (43, 44, 47), (44, 45, 47), (45, 46, 47),
            
            # Dış dudak üçgenleri
            (48, 49, 59), (49, 50, 59), (50, 51, 59), (51, 52, 59),
            (52, 53, 59), (53, 54, 59), (54, 55, 59), (55, 56, 59),
            (56, 57, 59), (57, 58, 59), (58, 59, 48),
            
            # İç dudak üçgenleri
            (60, 61, 67), (61, 62, 67), (62, 63, 67), (63, 64, 67),
            (64, 65, 67), (65, 66, 67), (66, 67, 60)
        ]
        
        return triangles
    
    def calculate_depth(self, points):
        """Gelişmiş derinlik hesaplama"""
        if points is None or len(points) < 68:
            return []
        
        # Gözler arası mesafeyi referans olarak hesapla
        left_eye = (sum([points[36][0], points[37][0], points[38][0], points[39][0], points[40][0], points[41][0]]) / 6,
                    sum([points[36][1], points[37][1], points[38][1], points[39][1], points[40][1], points[41][1]]) / 6)
        right_eye = (sum([points[42][0], points[43][0], points[44][0], points[45][0], points[46][0], points[47][0]]) / 6,
                     sum([points[42][1], points[43][1], points[44][1], points[45][1], points[46][1], points[47][1]]) / 6)
        
        eye_distance = hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
        
        # Yüz merkezi
        face_center_x = (points[0][0] + points[16][0]) / 2
        face_center_y = (points[0][1] + points[16][1]) / 2
        
        # Her nokta için derinlik hesapla (gelişmiş model)
        depth_points = []
        for i, point in enumerate(points):
            # Bölgeye göre derinlik değerleri
            if i in self.face_regions["nose"]:
                # Burun noktaları - burun köprüsünden uca doğru artan derinlik
                if i == 27:  # Burun köprüsü
                    depth = 10 * self.depth_factor
                elif i == 30:  # Burun ortası
                    depth = 25 * self.depth_factor
                else:  # Burun ucu ve kanatları
                    depth = 40 * self.depth_factor
            elif i in self.face_regions["eyes"]:
                # Göz çukuru etkisi
                depth = 15 * self.depth_factor
            elif i in self.face_regions["eyebrows"]:
                # Kaşlar hafif öne çıkık
                depth = 18 * self.depth_factor
            elif i in self.face_regions["jaw"]:
                # Çene hattı - yüzün kenarlarına doğru azalan derinlik
                # Merkeze olan uzaklığa göre derinliği ayarla
                dist_from_center = hypot(point[0] - face_center_x, point[1] - face_center_y)
                depth = max(5, 20 - dist_from_center / eye_distance * 15) * self.depth_factor
            elif i in self.face_regions["lips"]:
                # Dudaklar - dış dudak ve iç dudak ayrımı
                if i < 60:  # Dış dudak
                    depth = 30 * self.depth_factor
                else:  # İç dudak
                    depth = 32 * self.depth_factor
            else:
                depth = 10 * self.depth_factor
                
            depth_points.append((point[0], point[1], depth))
        
        return depth_points
    
    def create_3d_model(self, points, frame=None):
        """Gelişmiş 3D yüz modeli oluştur"""
        if points is None or len(points) < 68:
            return np.zeros((500, 500, 3), np.uint8)
        
        # 3D model görselleştirmesi için boş bir görüntü oluştur
        model_img = np.zeros((500, 500, 3), np.uint8)
        
        # Derinlik noktalarını hesapla
        depth_points = self.calculate_depth(points)
        
        # Noktaları 3D uzaya dönüştür
        points_3d = []
        for x, y, z in depth_points:
            # Görüntü merkezine göre normalize et
            x_centered = x - frame.shape[1]/2 if frame is not None else x - 250
            y_centered = y - frame.shape[0]/2 if frame is not None else y - 250
            
            # 3D dönüşüm uygula
            x_rot, y_rot, z_rot = self._apply_rotation(x_centered, y_centered, z)
            
            # Ölçekle ve merkeze geri taşı
            x_final = x_rot * self.scale + 250
            y_final = y_rot * self.scale + 250
            z_final = z_rot * self.scale
            
            points_3d.append((int(x_final), int(y_final), z_final))
        
        # Render moduna göre çizim yap
        if self.render_mode == "wireframe":
            self._draw_wireframe(model_img, points_3d)
        elif self.render_mode == "solid":
            self._draw_solid(model_img, points_3d)
        elif self.render_mode == "textured" and frame is not None:
            self._draw_textured(model_img, points_3d, points, frame)
        else:
            self._draw_wireframe(model_img, points_3d)
        
        return model_img
    
    def _apply_rotation(self, x, y, z):
        """3D rotasyon uygula"""
        # X ekseni etrafında döndür
        y_rot = y * cos(radians(self.rotation_x)) - z * sin(radians(self.rotation_x))
        z_rot = y * sin(radians(self.rotation_x)) + z * cos(radians(self.rotation_x))
        y, z = y_rot, z_rot
        
        # Y ekseni etrafında döndür
        x_rot = x * cos(radians(self.rotation_y)) + z * sin(radians(self.rotation_y))
        z_rot = -x * sin(radians(self.rotation_y)) + z * cos(radians(self.rotation_y))
        x, z = x_rot, z_rot
        
        # Z ekseni etrafında döndür
        x_rot = x * cos(radians(self.rotation_z)) - y * sin(radians(self.rotation_z))
        y_rot = x * sin(radians(self.rotation_z)) + y * cos(radians(self.rotation_z))
        x, y = x_rot, y_rot
        
        return x, y, z
    
    def _draw_wireframe(self, img, points_3d):
        """Tel kafes modeli çiz"""
        # Yüz bölgelerine göre çizim yap
        for region, indices in self.face_regions.items():
            color = self.face_colors[region]
            
            # Bölgedeki noktaları birleştir
            for i in range(len(indices) - 1):
                idx1 = indices[i]
                idx2 = indices[i + 1]
                
                # Derinliğe göre çizgi kalınlığını ayarla
                thickness = max(1, min(3, int((points_3d[idx1][2] + points_3d[idx2][2]) / 40)))
                
                cv2.line(img, (points_3d[idx1][0], points_3d[idx1][1]), 
                         (points_3d[idx2][0], points_3d[idx2][1]), color, thickness)
            
            # Kapalı bölgeler için son noktayı ilk noktaya bağla
            if region in ["eyes", "lips"]:
                if region == "eyes":
                    # Sol göz
                    cv2.line(img, (points_3d[36][0], points_3d[36][1]), 
                             (points_3d[41][0], points_3d[41][1]), color, thickness)
                    # Sağ göz
                    cv2.line(img, (points_3d[42][0], points_3d[42][1]), 
                             (points_3d[47][0], points_3d[47][1]), color, thickness)
                elif region == "lips":
                    # Dış dudak
                    cv2.line(img, (points_3d[48][0], points_3d[48][1]), 
                             (points_3d[59][0], points_3d[59][1]), color, thickness)
                    # İç dudak
                    cv2.line(img, (points_3d[60][0], points_3d[60][1]), 
                             (points_3d[67][0], points_3d[67][1]), color, thickness)
    
    def _draw_solid(self, img, points_3d):
        """Dolu yüzey modeli çiz"""
        # Z-buffer oluştur (derinlik bilgisi için)
        z_buffer = np.ones((img.shape[0], img.shape[1])) * float('inf')
        
        # Üçgenleri çiz
        for triangle in self.face_triangles:
            p1 = points_3d[triangle[0]]
            p2 = points_3d[triangle[1]]
            p3 = points_3d[triangle[2]]
            
            # Üçgenin hangi bölgeye ait olduğunu belirle
            region = None
            for r_name, indices in self.face_regions.items():
                if triangle[0] in indices:
                    region = r_name
                    break
            
            if region is None:
                continue
                
            color = self.face_colors[region]
            
            # Üçgeni doldur
            pts = np.array([[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Z-değerlerinin ortalamasını hesapla (derinlik için)
            avg_z = (p1[2] + p2[2] + p3[2]) / 3
            
            # Üçgenin tüm pikselleri için
            cv2.fillPoly(img, [pts], color)
            
            # Z-buffer güncelle (derinlik testi)
            min_x = max(0, min(p1[0], p2[0], p3[0]))
            min_y = max(0, min(p1[1], p2[1], p3[1]))
            max_x = min(img.shape[1]-1, max(p1[0], p2[0], p3[0]))
            max_y = min(img.shape[0]-1, max(p1[1], p2[1], p3[1]))
            
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                        if avg_z < z_buffer[y, x]:
                            z_buffer[y, x] = avg_z
    
    def _draw_textured(self, img, points_3d, points_2d, frame):
        """Dokulu model çiz"""
        if frame is None:
            self._draw_solid(img, points_3d)
            return
        
        # Z-buffer oluştur
        z_buffer = np.ones((img.shape[0], img.shape[1])) * float('inf')
        
        # Üçgenleri çiz
        for triangle in self.face_triangles:
            p1_3d = points_3d[triangle[0]]
            p2_3d = points_3d[triangle[1]]
            p3_3d = points_3d[triangle[2]]
            
            p1_2d = points_2d[triangle[0]]
            p2_2d = points_2d[triangle[1]]
            p3_2d = points_2d[triangle[2]]
            
            # 3D üçgen için noktalar
            pts_3d = np.array([[p1_3d[0], p1_3d[1]], [p2_3d[0], p2_3d[1]], [p3_3d[0], p3_3d[1]]], np.int32)
            pts_3d = pts_3d.reshape((-1, 1, 2))
            
            # 2D üçgen için noktalar (orijinal görüntüden doku almak için)
            pts_2d = np.array([[p1_2d[0], p1_2d[1]], [p2_2d[0], p2_2d[1]], [p3_2d[0], p3_2d[1]]], np.float32)
            
            # Z-değerlerinin ortalamasını hesapla
            avg_z = (p1_3d[2] + p2_3d[2] + p3_3d[2]) / 3
            
            # Üçgenin sınırlarını belirle
            min_x = max(0, min(p1_3d[0], p2_3d[0], p3_3d[0]))
            min_y = max(0, min(p1_3d[1], p2_3d[1], p3_3d[1]))
            max_x = min(img.shape[1]-1, max(p1_3d[0], p2_3d[0], p3_3d[0]))
            max_y = min(img.shape[0]-1, max(p1_3d[1], p2_3d[1], p3_3d[1]))
            
            # Affine dönüşüm matrisi hesapla
            dst_tri = np.array([[0, 0], [0, 1], [1, 1]], np.float32)
            warp_mat = cv2.getAffineTransform(pts_2d, dst_tri)
            
            # Her piksel için
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    if cv2.pointPolygonTest(pts_3d, (x, y), False) >= 0 and avg_z < z_buffer[y, x]:
                        # Orijinal görüntüdeki karşılık gelen noktayı bul
                        src_pt = np.dot(warp_mat, np.array([x, y, 1]))
                        src_x, src_y = int(src_pt[0]), int(src_pt[1])
                        
                        # Görüntü sınırları içinde mi kontrol et
                        if 0 <= src_x < frame.shape[1] and 0 <= src_y < frame.shape[0]:
                            img[y, x] = frame[src_y, src_x]
                            z_buffer[y, x] = avg_z