import tkinter as tk
from tabbed_face_app import TabbedFaceDetectionApp

def main():
    # Uygulama penceresini oluştur
    root = tk.Tk()
    # Uygulama başlığını ayarla
    root.title("Gelişmiş Yüz Tarama ve Modelleme Uygulaması")
    # Pencere boyutunu ayarla
    root.geometry("1200x700")
    
    # Uygulama nesnesini oluştur
    app = TabbedFaceDetectionApp(root)
    
    # Bilgi mesajı göster
    print("Gelişmiş Yüz Tarama ve Modelleme Uygulaması başlatılıyor...")
    print("Kullanılabilir özellikler:")
    print("- Gerçek zamanlı yüz tespiti")
    print("- 68 yüz noktası (landmark) tespiti")
    print("- Gelişmiş 3D yüz modeli oluşturma (nokta bulutu ve üçgen mesh tekniği)")
    print("- Derinlik görselleştirme")
    print("- Görüntü filtreleri (Siyah-Beyaz, Sepya, Negatif, Bulanık)")
    print("- AR filtreleri (Gözlük, Şapka, Maske, Sakal, Hayvan Kulakları, Işık Efekti)")
    print("- Duygu analizi simülasyonu")
    print("- Yaş/Cinsiyet tahmini simülasyonu")
    print("- Görüntü kaydetme")
    print("- Yüz ifadelerine dayalı avatar animasyonu")
    print("- Sesli komut kontrolü")
    print("- Geliştirilmiş dudak okuma")
    print("- Göz takibi ve yorgunluk tespiti")
    print("- Yaşlandırma/gençleştirme simülasyonu")
    print("- Sanal makyaj uygulaması")
    print("- Yüz hareketleriyle kontrol")
    print("- Göz izleme tabanlı kontrol")
    print("\nYeni Özellikler:")
    print("- Sekmeli arayüz ile tüm özellikler ekrana sığıyor")
    print("- Gelişmiş 3D yüz modeli (nokta bulutu ve üçgen mesh tekniği)")
    print("- Doku kaplama özelliği")
    print("- Daha gerçekçi yüz ifadeleri")
    print("\nUygulamadan çıkmak için pencereyi kapatın veya Ctrl+C tuşlarına basın.")
    
    # Ctrl+C (KeyboardInterrupt) için temizleme işlevi
    def cleanup_on_exit():
        if hasattr(app, 'cap') and app.cap is not None:
            print("\nUygulama kapatılıyor, kamera kaynakları serbest bırakılıyor...")
            app.cap.release()
        # Sesli komut dinlemeyi durdur
        if hasattr(app, 'voice_command_active') and app.voice_command_active:
            app.voice_commands.stop_listening()
            print("Sesli komut dinleme durduruldu.")
        print("Uygulama güvenli bir şekilde kapatıldı.")
    
    # Ctrl+C (KeyboardInterrupt) yakalamak için
    try:
        # Ana döngüyü başlat
        root.mainloop()
    except KeyboardInterrupt:
        print("\nKlavye kesintisi algılandı (Ctrl+C)")
    finally:
        cleanup_on_exit()

if __name__ == "__main__":
    main()