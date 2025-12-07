from flask import Flask, render_template, request
import pandas as pd
import joblib
import io
import base64
import numpy as np

# --- DÜZELTME BURADA ---
# Matplotlib'in sunucu hatası vermemesi için "Agg" modunu açıyoruz.
# Bu kod, uygulamanın çökmesini engeller.
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
# -----------------------

app = Flask(__name__)

# Modeli Yükle (Hata almamak için korumalı blok)
try:
    model_data = joblib.load('student_coach_model.pkl')
    regressor = model_data['regressor']
    profile_transformer = model_data['profile_transformer']
    knn = model_data['knn']
    high_performers = model_data['high_performers']
    actionable = model_data['actionable_features']
    profile_cols = model_data['profile_features']
except:
    print("UYARI: Model dosyası bulunamadı!")

# Haritalama Sözlükleri
maps = {
    'gender': {'Kadın': 'Female', 'Erkek': 'Male', 'Diğer': 'Other'},
    'edu': {'Lise': 'High School', 'Üniversite (Lisans)': 'Bachelor', 'Yüksek Lisans': 'Master', 'Doktora': 'PhD'},
    'yes_no': {'Evet': 'Yes', 'Hayır': 'No'},
    'internet': {'Zayıf': 'Poor', 'Orta': 'Average', 'İyi': 'Good'},
    'diet': {'Dengesiz': 'Poor', 'Orta': 'Fair', 'Sağlıklı': 'Good'}
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    graph_url = None
    feedback = [] 

    if request.method == 'POST':
        # Formdan verileri al
        input_data = {
            'age': int(request.form['age']),
            'gender': maps['gender'][request.form['gender']],
            'parental_education_level': maps['edu'][request.form['education']],
            'part_time_job': maps['yes_no'][request.form['part_time']],
            'internet_quality': maps['internet'][request.form['internet']],
            'mental_health_rating': int(request.form['mental']),
            'diet_quality': maps['diet'][request.form['diet']],
            'study_hours_per_day': float(request.form['study_hrs']),
            'sleep_hours': float(request.form['sleep_hrs']),
            'social_media_hours': float(request.form['social_hrs']),
            'netflix_hours': float(request.form['netflix_hrs']),
            'exercise_frequency': int(request.form['exercise'])
        }

        # DataFrame oluştur
        df = pd.DataFrame([input_data])

        # 1. Tahmin Yap
        score = regressor.predict(df)[0]
        prediction = round(score, 1)

        # 2. Benzer Öğrencilerle Kıyaslama
        user_profile = df[profile_cols]
        user_profile_matrix = profile_transformer.transform(user_profile)
        distances, indices = knn.kneighbors(user_profile_matrix)
        similar_students = high_performers.iloc[indices[0]]
        avg_habits = similar_students[actionable].mean()

        # --- TAVSİYE MOTORU ---
        
        # Ders Çalışma Tavsiyesi
        gap_study = avg_habits['study_hours_per_day'] - input_data['study_hours_per_day']
        if gap_study > 0.5:
            feedback.append(f"📉 **Akademik Odak:** Hedeflediğin başarı grubu günde ortalama **{avg_habits['study_hours_per_day']:.1f} saat** çalışıyor. Çalışma süreni artırmalısın.")
        else:
            feedback.append("✅ **Akademik Odak:** Çalışma disiplinin harika! Başarı grubunun standartlarını yakalamışsın.")

        # Sosyal Medya Tavsiyesi
        gap_social = input_data['social_media_hours'] - avg_habits['social_media_hours']
        if gap_social > 1:
            feedback.append(f"📵 **Dijital Denge:** Sosyal medyada çok vakit harcıyorsun. Günde **{gap_social:.1f} saat** tasarruf edip bunu uykuya veya derse ayırabilirsin.")
        
        # Uyku Tavsiyesi
        gap_sleep = abs(input_data['sleep_hours'] - avg_habits['sleep_hours'])
        if gap_sleep > 1.5:
             feedback.append(f"🌙 **Uyku Düzeni:** Başarılı öğrenciler günde ortalama **{avg_habits['sleep_hours']:.1f} saat** uyuyor. Uyku düzenini gözden geçirmelisin.")

        # Egzersiz Tavsiyesi 
        if input_data['exercise_frequency'] == 0:
            feedback.append("🏃‍♂️ **Fiziksel Aktivite:** Haftada en az 1-2 gün egzersiz yapmak zihni açar ve odaklanmayı artırır.")

        # ----------------------

        # Grafik Çizimi (Burada da ufak bir temizlik yapıyoruz)
        plt.clf() # Eski grafiği hafızadan sil
        plt.figure(figsize=(10, 5))
        
        categories = ['Ders', 'Uyku', 'Sosyal Medya', 'TV/Dizi', 'Spor']
        user_vals = [input_data['study_hours_per_day'], input_data['sleep_hours'], input_data['social_media_hours'], input_data['netflix_hours'], input_data['exercise_frequency']]
        target_vals = [avg_habits['study_hours_per_day'], avg_habits['sleep_hours'], avg_habits['social_media_hours'], avg_habits['netflix_hours'], avg_habits['exercise_frequency']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        plt.bar(x - width/2, user_vals, width, label='Siz', color='#004e92')
        plt.bar(x + width/2, target_vals, width, label='Hedef Profil', color='#b0c4de')
        plt.xticks(x, categories)
        plt.legend()
        plt.title('Alışkanlık Karşılaştırması')
        
        # Çerçeveleri temizle
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Kaydet ve Çevir
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close('all') # Tüm pencereleri zorla kapat

    return render_template('index.html', prediction=prediction, graph_url=graph_url, feedback=feedback)

if __name__ == '__main__':
    app.run(debug=True)