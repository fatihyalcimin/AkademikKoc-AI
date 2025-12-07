import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
import joblib

# 1. Veriyi Yükle
print("Veri yükleniyor...")
df = pd.read_csv('student_habits_performance.csv')

# 2. Özellikleri Tanımla
# Kullanıcının değiştirebileceği özellikler (Tavsiye vereceğimiz kısımlar)
actionable_features = ['study_hours_per_day', 'social_media_hours', 'netflix_hours', 'sleep_hours', 'exercise_frequency']
# Kullanıcının değiştiremeyeceği profili (Benzer kişileri bulmak için)
profile_features = ['age', 'gender', 'part_time_job', 'diet_quality', 'parental_education_level', 'internet_quality', 'mental_health_rating']
target = 'exam_score'

X = df[actionable_features + profile_features]
y = df[target]

# 3. Veri Ön İşleme Hattı (Pipeline)
numeric_features = ['study_hours_per_day', 'social_media_hours', 'netflix_hours', 'sleep_hours', 'exercise_frequency', 'age', 'mental_health_rating']
categorical_features = ['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 'internet_quality']

# Kategorik verileri sayıya, sayısal verileri standarta çevirme işlemi
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. Puan Tahmin Modelini Eğit (Random Forest)
print("Tahmin modeli eğitiliyor...")
regressor = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
regressor.fit(X, y)

# 5. Benzerlik Modelini Hazırla (KNN)
# Sadece profil özelliklerine göre benzerlik bakacağız
print("Tavsiye motoru hazırlanıyor...")
profile_transformer = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'mental_health_rating']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 'internet_quality'])
    ])

# Başarılı öğrencilerin (Puanı 80 üstü olanlar) profil haritasını çıkar
high_performers = df[df['exam_score'] > 80].copy()
high_performers_matrix = profile_transformer.fit_transform(high_performers[profile_features])

# Benzerlik bulucu (En yakın komşu)
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(high_performers_matrix)

# 6. Modelleri Kaydet
# Bu dosyaları web sitesi kullanacak
print("Modeller kaydediliyor...")
model_data = {
    'regressor': regressor,
    'profile_transformer': profile_transformer,
    'knn': knn,
    'high_performers': high_performers,
    'actionable_features': actionable_features,
    'profile_features': profile_features
}
joblib.dump(model_data, 'student_coach_model.pkl')
print("Tamamlandı! 'student_coach_model.pkl' dosyası oluşturuldu.")