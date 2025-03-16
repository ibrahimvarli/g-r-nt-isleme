# Gelişmiş Yüz Tarama ve Modelleme Projesi

Bu proje, bilgisayar kamerasını kullanarak kullanıcının yüzünü tarayıp basit bir 3D yüz modeli oluşturan gelişmiş bir Python uygulamasıdır. Modern bir kullanıcı arayüzü ve çeşitli görüntü işleme özellikleri sunar.

## Özellikler

- Gerçek zamanlı yüz tespiti
- 68 yüz noktası (landmark) tespiti
- Basit 3D yüz modeli oluşturma
- Derinlik görselleştirme
- Modern kullanıcı arayüzü (Tkinter)
- Görüntü filtreleri (Siyah-Beyaz, Sepya, Negatif, Bulanık)
- Duygu analizi simülasyonu
- Yaş/Cinsiyet tahmini simülasyonu
- Görüntü kaydetme özelliği

## Gereksinimler

Bu projeyi çalıştırmak için aşağıdaki kütüphanelere ihtiyacınız vardır:

- Python 3.6 veya üstü
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- dlib (`pip install dlib`)

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
   ```
   pip install opencv-python numpy dlib
   ```

2. dlib'in yüz landmark modelini indirin:
   - [shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) dosyasını indirin
   - Dosyayı açın ve `shape_predictor_68_face_landmarks.dat` dosyasını proje klasörüne yerleştirin

## Kullanım

Programı çalıştırmak için:

```
python main.py
```

Program çalıştığında:
- Kamera açılacak ve yüzünüzü tespit edecektir
- Tespit edilen yüz üzerinde yeşil noktalar (landmarks) gösterilecektir
- Ayrı bir pencerede basit 3D yüz modeli gösterilecektir
- Başka bir pencerede derinlik görselleştirmesi gösterilecektir

Programdan çıkmak için 'q' tuşuna basın.

## Nasıl Çalışır

1. Kameradan görüntü alınır
2. dlib kütüphanesi kullanılarak yüz tespiti yapılır
3. Tespit edilen yüz üzerinde 68 landmark noktası belirlenir
4. Bu noktalar kullanılarak basit bir 3D model oluşturulur
5. Derinlik bilgisi hesaplanarak görselleştirilir

## Notlar

- Program çalışmadan önce `shape_predictor_68_face_landmarks.dat` dosyasının proje klasöründe olduğundan emin olun
- İyi aydınlatma koşullarında daha iyi sonuçlar alınır
- Kameranın doğrudan yüzünüze bakması en iyi sonuçları verir