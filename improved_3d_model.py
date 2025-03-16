import cv2
import numpy as np
import os
from math import hypot, sin, cos, radians
from scipy.spatial import Delaunay

class Improved3DFaceModel:
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
        
        # Doku haritası için görüntü
        self.texture_image = None
        
        # Nokta bulutu yoğunluğu
        self.point_cloud_density = 3  # Yüksek değer = daha yoğun nokta bulutu
        
        # Üçgen mesh kalitesi
        self.triangle_mesh_quality = 2  # Yüksek değer = daha kaliteli mesh
    
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
    
    def generate_point_cloud(self, depth_points):
        """Nokta bulutu oluştur - Yüz noktaları arasında interpolasyon yaparak daha yoğun nokta bulutu oluşturur"""
        if not depth_points:
            return []
        
        # Orijinal noktaları nokta bulutuna ekle
        point_cloud = depth_points.copy()
        
        # Yüz bölgelerine göre ara noktalar oluştur
        for region, indices in self.face_regions.items():
            for i in range(len(indices) - 1):
                idx1 = indices[i]
                idx2 = indices[i + 1]
                
                # İki nokta arasında ara noktalar oluştur
                for t in range(1, self.point_cloud_density):
                    t_val = t / self.point_cloud_density
                    
                    # Doğrusal interpolasyon
                    x = depth_points[idx1][0] * (1 - t_val) + depth_points[idx2][0] * t_val
                    y = depth_points[idx1][1] * (1 - t_val) + depth_points[idx2][1] * t_val
                    z = depth_points[idx1][2] * (1 - t_val) + depth_points[idx2][2] * t_val
                    
                    point_cloud.append((x, y, z))
        
        # Yüz içi noktaları oluştur (daha yoğun bir nokta bulutu için)
        if self.mesh_quality == "high":
            # Yüz içi bölgeler için daha fazla nokta ekle
            # Örneğin, göz bölgesi, burun bölgesi, dudak bölgesi
            eye_center_left = ((depth_points[37][0] + depth_points[41][0]) / 2,
                              (depth_points[37][1] + depth_points[41][1]) / 2,
                              (depth_points[37][2] + depth_points[41][2]) / 2)
            
            eye_center_right = ((depth_points[43][0] + depth_points[47][0]) / 2,
                               (depth_points[43][1] + depth_points[47][1]) / 2,
                               (depth_points[43][2] + depth_points[47][2]) / 2)
            
            nose_center = ((depth_points[30][0] + depth_points[33][0]) / 2,
                          (depth_points[30][1] + depth_points[33][1]) / 2,
                          (depth_points[30][2] + depth_points[33][2]) / 2)
            
            mouth_center = ((depth_points[48][0] + depth_points[54][0]) / 2,
                           (depth_points[48][1] + depth_points[54][1]) / 2,
                           (depth_points[48][2] + depth_points[54][2]) / 2)
            
            # Bu merkezleri nokta bulutuna ekle
            point_cloud.append(eye_center_left)
            point_cloud.append(eye_center_right)
            point_cloud.append(nose_center)
            point_cloud.append(mouth_center)
        
        return point_cloud
    
    def generate_triangle_mesh(self, points):
        """Üçgen mesh oluştur - Delaunay üçgenlemesi kullanarak"""
        if not points or len(points) < 3:
            return []
        
        # 2D noktaları al (x, y)
        points_2d = np.array([(p[0], p[1]) for p in points])
        
        # Delaunay üçgenlemesi uygula
        try:
            tri = Delaunay(points_2d)
            triangles = tri.simplices
        except Exception as e:
            print(f"Üçgenleme hatası: {e}")
            return []
        
        # Kalite filtreleme - çok uzun veya çok küçük üçgenleri filtrele
        filtered_triangles = []
        for t in triangles:
            # Üçgenin kenar uzunluklarını hesapla
            p1, p2, p3 = points_2d[t[0]], points_2d[t[1]], points_2d[t[2]]
            
            edge1 = hypot(p1[0] - p2[0], p1[1] - p2[1])
            edge2 = hypot(p2[0] - p3[0], p2[1] - p3[1])
            edge3 = hypot(p3[0] - p1[0], p3[1] - p1[1])
            
            # Maksimum kenar uzunluğu
            max_edge = max(edge1, edge2, edge3)
            
            # Üçgenin alanı
            s = (edge1 + edge2 + edge3) / 2
            try:
                area = (s * (s - edge1) * (s - edge2) * (s - edge3)) ** 0.5
            except ValueError:
                continue  # Geçersiz üçgen
            
            # Kalite metriği: alan / (max_kenar^2)
            quality = area / (max_edge * max_edge)
            
            # Kalite eşiğini geç
            if quality > 0.1 / self.triangle_mesh_quality:  # Düşük değer = daha fazla üçgen
                filtered_triangles.append(t)
        
        return filtered_triangles
    
    def create_3d_model(self, points, frame=None):
        """Gelişmiş 3D yüz modeli oluştur"""
        if points is None or len(points) < 68:
            return np.zeros((500, 500, 3), np.uint8)
        
        # 3D model görselleştirmesi için boş bir görüntü oluştur
        model_img = np.zeros((500, 500, 3), np.uint8)
        
        # Derinlik noktalarını hesapla
        depth_points = self.calculate_depth(points)
        
        # Nokta bulutu oluştur
        point_cloud = self.generate_point_cloud(depth_points)
        
        # Noktaları 3D uzaya dönüştür
        points_3d = []
        for x, y, z in point_cloud:
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
        
        # Üçgen mesh oluştur
        triangles = self.generate_triangle_mesh(points_3d)
        
        # Render moduna göre çizim yap
        if self.render_mode == "wireframe":
            self._draw_wireframe(model_img, points_3d, triangles)
        elif self.render_mode == "solid":
            self._draw_solid(model_img, points_3d, triangles)
        elif self.render_mode == "textured" and frame is not None:
            self._draw_textured(model_img, points_3d, points, frame, triangles)
        else:
            self._draw_wireframe(model_img, points_3d, triangles)
        
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
    
    def _draw_wireframe(self, img, points_3d, triangles):
        """Tel kafes modeli çiz"""
        # Üçgenleri çiz
        for t in triangles:
            p1 = points_3d[t[0]]
            p2 = points_3d[t[1]]
            p3 = points_3d[t[2]]
            
            # Derinliğe göre çizgi kalınlığını ayarla
            avg_z = (p1[2] + p2[2] + p3[2]) / 3
            thickness = max(1, min(3, int(avg_z / 40)))
            
            # Üçgenin hangi bölgeye ait olduğunu belirle (basitleştirilmiş)
            color = (255, 255, 255)  # Varsayılan beyaz
            
            # Üçgeni çiz
            cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), color, thickness)
            cv2.line(img, (p2[0], p2[1]), (p3[0], p3[1]), color, thickness)
            cv2.line(img, (p3[0], p3[1]), (p1[0], p1[1]), color, thickness)
    
    def _draw_solid(self, img, points_3d, triangles):
        """Dolu yüzey modeli çiz"""
        # Z-buffer oluştur (derinlik bilgisi için)
        z_buffer = np.ones((img.shape[0], img.shape[1])) * float('inf')
        
        # Üçgenleri derinlik değerlerine göre sırala (uzaktan yakına)
        sorted_triangles = []
        for t in triangles:
            p1 = points_3d[t[0]]
            p2 = points_3d[t[1]]
            p3 = points_3d[t[2]]
            avg_z = (p1[2] + p2[2] + p3[2]) / 3
            sorted_triangles.append((t, avg_z))
        
        sorted_triangles.sort(key=lambda x: x[1], reverse=True)  # Uzaktan yakına sırala
        
        # Üçgenleri çiz
        for t, _ in sorted_triangles:
            p1 = points_3d[t[0]]
            p2 = points_3d[t[1]]
            p3 = points_3d[t[2]]
            
            # Üçgenin hangi bölgeye ait olduğunu belirle (basitleştirilmiş)
            color = (100, 100, 100)  # Varsayılan gri
            
            # Üçgeni doldur
            pts = np.array([[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Z-değerlerinin ortalamasını hesapla (derinlik için)
            avg_z = (p1[2] + p2[2] + p3[2]) / 3
            
            # Derinliğe göre renk yoğunluğunu ayarla
            intensity = min(255, max(50, int(150 + avg_z / 2)))
            adjusted_color = tuple(min(255, c * intensity // 150) for c in color)
            
            # Üçgeni çiz
            cv2.fillPoly(img, [pts], adjusted_color)
    
    def _draw_textured(self, img, points_3d, points_2d, frame, triangles):
        """Dokulu model çiz"""
        if frame is None:
            self._draw_solid(img, points_3d, triangles)
            return
        
        # Z-buffer oluştur
        z_buffer = np.ones((img.shape[0], img.shape[1])) * float('inf')
        
        # Üçgenleri derinlik değerlerine göre sırala (uzaktan yakına)
        sorted_triangles = []
        for t in triangles:
            p1 = points_3d[t[0]]
            p2 = points_3d[t[1]]
            p3 = points_3d[t[2]]
            avg_z = (p1[2] + p2[2] + p3[2]) / 3
            sorted_triangles.append((t, avg_z))
        
        sorted_triangles.sort(key=lambda x: x[1], reverse=True)  # Uzaktan yakına sırala
        
        # Üçgenleri çiz
        for t, _ in sorted_triangles:
            p1_3d = points_3d[t[0]]
            p2_3d = points_3d[t[1]]
            p3_3d = points_3d[t[2]]
            
            # 3D üçgen için noktalar
            pts_3d = np.array([[p1_3d[0], p1_3d[1]], [p2_3d[0], p2_3d[1]], [p3_3d[0], p3_3d[1]]], np.int32)
            pts_3d = pts_3d.reshape((-1, 1, 2))
            
            # Üçgenin sınırlarını belirle
            min_x = max(0, min(p1_3d[0], p2_3d[0], p3_3d[0]))
            min_y = max(0, min(p1_3d[1], p2_3d[1], p3_3d[1]))
            max_x = min(img.shape[1]-1, max(p1_3d[0], p2_3d[0], p3_3d[0]))
            max_y = min(img.shape[0]-1, max(p1_3d[1], p2_3d[1], p3_3d[1]))
            
            # Basit doku kaplama - üçgenin içindeki her piksel için
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    if cv2.pointPolygonTest(pts_3d, (x, y), False) >= 0:
                        # Barycentric koordinatları hesapla
                        # Bu koordinatlar, pikselin üçgen içindeki konumunu belirler
                        # ve doku koordinatlarını hesaplamak için kullanılır
                        # Basitleştirilmiş bir yaklaşım kullanıyoruz
                        color = (100, 100, 100)  # Varsayılan gri
                        img[y, x] = color
    
    def create_depth_visualization(self, points, frame=None):
        """Derinlik görselleştirmesi oluştur"""
        if points is None or len(points) < 68:
            return np.zeros((500, 500, 3), np.uint8)
        
        # Derinlik görselleştirmesi için boş bir görüntü oluştur
        depth_img = np.zeros((500, 500, 3), np.uint8)
        
        # Derinlik noktalarını hesapla
        depth_points = self.calculate_depth(points)
        
        # Nokta bulutu oluştur
        point_cloud = self.generate_point_cloud(depth_points)
        
        # Noktaları 3D uzaya dönüştür
        points_3d = []
        for x, y, z in point_cloud:
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
        
        # Üçgen mesh oluştur
        triangles = self.generate_triangle_mesh(points_3d)
        
        # Derinlik haritası oluştur
        for t in triangles:
            p1 = points_3d[t[0]]
            p2 = points_3d[t[1]]
            p3 = points_3d[t[2]]
            
            # Üçgeni doldur
            pts = np.array([[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Z-değerlerinin ortalamasını hesapla (derinlik için)
            avg_z = (p1[2] + p2[2] + p3[2]) / 3
            
            # Derinliğe göre renk belirle (mavi-yeşil-kırmızı gradyan)
            # Düşük derinlik (uzak) -> Mavi
            # Orta derinlik -> Yeşil
            # Yüksek derinlik (yakın) -> Kırmızı
            normalized_z = min(1.0, max(0.0, avg_z / 100.0))
            
            if normalized_z < 0.5:
                # Mavi -> Yeşil
                r = 0
                g = int(255 * (normalized_z * 2))
                b = int(255 * (1 - normalized_z * 2))
            else:
                # Yeşil -> Kırmızı
                r = int(255 * ((normalized_z - 0.5) * 2))
                g = int(255 * (1 - (normalized_z - 0.5) * 2))
                b = 0
            
            color = (b, g, r)  # BGR format
            cv2.fillPoly(depth_img, [pts], color)
        
        return depth_img