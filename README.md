# 🎓 AI Akademik Koç (Academic Vision)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-green)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)

**AI Akademik Koç**, öğrencilerin demografik özelliklerini ve günlük alışkanlıklarını analiz ederek akademik başarı puanlarını tahmin eden ve kişiye özel, veriye dayalı gelişim tavsiyeleri sunan yapay zeka destekli bir web uygulamasıdır.

## 🚀 Özellikler

- **Başarı Tahmini:** Random Forest algoritması kullanarak öğrencinin sınav puanını (%90+ doğrulukla) tahmin eder.
- **Akıllı Kıyaslama:** Öğrencinin "değiştirilemez" özellikleri (yaş, cinsiyet, ebeveyn eğitimi vb.) baz alınarak, veri setindeki en başarılı "benzer" öğrencileri bulur.
- **Veriye Dayalı Tavsiyeler:** Sizin alışkanlıklarınız ile başarılı benzerlerinizin alışkanlıklarını (uyku, çalışma saati vb.) karşılaştırarak somut öneriler sunar.
- **Görsel Analiz:** Matplotlib entegrasyonu ile kişisel durumunuzu görselleştiren dinamik grafikler üretir.

## 🛠 Kullanılan Teknolojiler

- **Backend:** Python, Flask
- **Makine Öğrenmesi:** Scikit-Learn (Random Forest Regressor, Nearest Neighbors, Pipeline, ColumnTransformer)
- **Veri İşleme:** Pandas, NumPy
- **Model Serileştirme:** Joblib
- **Frontend:** HTML5, CSS3 (Responsive Tasarım)
- **Görselleştirme:** Matplotlib

## 🧠 Nasıl Çalışır?

Proje iki temel AI yaklaşımını birleştirir:
1.  **Regresyon Modeli (Random Forest):** Girdilere dayalı olarak 0-100 arası bir başarı puanı tahmin eder.
2.  **Öneri Motoru (KNN - K-Nearest Neighbors):** Kullanıcının değiştiremeyeceği profiline (Profil Özellikleri) en çok benzeyen ama sınav puanı 80 üzeri olan öğrencileri bulur. Bu "hedef grubun" ortalama alışkanlıklarını hesaplayarak kullanıcıya "Daha fazla uyu" veya "Sosyal medyayı azalt" gibi dinamik geri bildirimler verir.

## 📸 Ekran Görüntüleri

uygulama_gorseller klasöründen uygulamanın çalışan görüntülerine ulaşıp inceleyebilirsiniz.

## 💻 Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için adımları izleyin:

1. **Repoyu klonlayın:**
   ```bash
   git clone [https://github.com/kullaniciadi/ai-academic-coach.git](https://github.com/kullaniciadi/ai-academic-coach.git)
   cd ai-academic-coach
