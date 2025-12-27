# 🇹🇷 Türkçe Spam Mesaj Tespiti

Klasik NLP ve makine öğrenimi yöntemleri kullanarak Türkçe SMS mesajlarını otomatik olarak spam/normal olarak sınıflandıran web uygulaması.

## 📋 Özellikler

- ✅ Türkçe'ye özel morfolojik analiz (TurkishNLP)
- ✅ TF-IDF tabanlı özellik çıkarımı
- ✅ Multinomial Naive Bayes sınıflandırıcı
- ✅ Modern Streamlit web arayüzü
- ✅ Tekli mesaj analizi
- ✅ Toplu dosya analizi (TXT/CSV)
- ✅ Gerçek zamanlı tahmin ve olasılık gösterimi

## 🛠️ Teknolojiler

- **Python 3.8+**
- **TurkishNLP**: Türkçe morfolojik analiz
- **Scikit-learn**: ML model ve TF-IDF
- **Streamlit**: Web arayüzü
- **NLTK**: Temel NLP işlemleri
- **Pandas & NumPy**: Veri manipülasyonu

## 📦 Kurulum

### 1. Repository'yi klonlayın

```bash
git clone <repo-url>
cd turkce_spam
```

### 2. Virtual environment oluşturun (önerilir)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Bağımlılıkları yükleyin

```bash
pip install -r requirements.txt
```

### 4. NLTK verilerini indirin

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 🚀 Kullanım

### Model Eğitimi

```bash
python src/train_model.py
```

Bu komut:
- Veri setini yükler
- Ön işleme yapar
- TF-IDF özellikleri çıkarır
- Naive Bayes modelini eğitir
- Model ve vectorizer'ı `models/` klasörüne kaydeder

### Web Arayüzünü Başlatma

```bash
streamlit run app.py
```

Tarayıcınızda `http://localhost:8501` adresine gidin.

## 📊 Proje Yapısı

```
turkce_spam/
├── data/
│   ├── raw/                        # Ham veri seti
│   └── processed/                  # İşlenmiş veri
├── models/
│   ├── spam_classifier.joblib      # Eğitilmiş model
│   └── tfidf_vectorizer.joblib     # TF-IDF vektörleştirici
├── src/
│   ├── preprocessing.py            # Metin ön işleme
│   ├── feature_extraction.py       # TF-IDF özellikleri
│   ├── train_model.py             # Model eğitimi
│   └── utils.py                   # Yardımcı fonksiyonlar
├── notebooks/
│   └── exploratory_analysis.ipynb  # Veri analizi
├── app.py                         # Streamlit uygulaması
├── requirements.txt               # Python bağımlılıkları
└── README.md                      # Bu dosya
```

## 📈 Model Performansı

Model performans metrikleri eğitim sonrası güncellenecek.

## 🔍 Özellikler Detayı

### Metin Ön İşleme
- Küçük harf dönüşümü
- Noktalama temizleme
- Türkçe stopwords kaldırma
- Kök bulma (stemming)

### TF-IDF Vektörleştirme
- Unigram ve bigram desteği
- Max features: 5000
- Document frequency filtreleme

### Sınıflandırma
- Multinomial Naive Bayes
- Olasılık tahminleri
- Binary sınıflandırma (Spam/Normal)

## 🎯 Kullanım Senaryoları

1. **Tekli Mesaj Kontrolü**: Web arayüzünde tek bir mesajı kontrol edin
2. **Toplu Analiz**: CSV/TXT dosyasındaki tüm mesajları analiz edin
3. **API Entegrasyonu**: Model dosyalarını kendi API'nize entegre edin

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Pull request göndermekten çekinmeyin.

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 👨‍💻 Geliştirici

Türkçe NLP Spam Tespiti Projesi

---
**Not**: Bu proje eğitim amaçlıdır ve sürekli geliştirilmektedir.
