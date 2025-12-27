"""
TF-IDF Özellik Çıkarımı Modülü
Bu modül metin mesajlarını TF-IDF vektörlerine dönüştürür.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple
import joblib
import os


class TurkishTfidfVectorizer:
    """
    Türkçe metinler için özelleştirilmiş TF-IDF vektörleştirici.
    """
    
    def __init__(
        self, 
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95
    ):
        """
        Args:
            max_features: Maksimum özellik sayısı
            ngram_range: N-gram aralığı (unigram, bigram)
            min_df: Minimum doküman frekansı
            max_df: Maksimum doküman frekansı (oran)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,  # Logaritmik TF kullan
            lowercase=True,
            strip_accents=None  # Türkçe karakterler korunsun
        )
    
    def fit(self, texts: List[str]):
        """
        TF-IDF vektörleştiriciyi eğitim verisiyle fit et.
        
        Args:
            texts: Eğitim metinleri
        """
        self.vectorizer.fit(texts)
        return self
    
    def transform(self, texts: List[str]):
        """
        Metinleri TF-IDF vektörlerine dönüştür.
        
        Args:
            texts: Dönüştürülecek metinler
            
        Returns:
            TF-IDF matris (sparse)
        """
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts: List[str]):
        """
        Fit ve transform işlemlerini birlikte yap.
        
        Args:
            texts: Eğitim metinleri
            
        Returns:
            TF-IDF matris (sparse)
        """
        return self.vectorizer.fit_transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """
        Özellik isimlerini (kelimeler/n-gramlar) döndürür.
        
        Returns:
            Özellik isimleri listesi
        """
        return self.vectorizer.get_feature_names_out()
    
    def get_vocabulary_size(self) -> int:
        """
        Toplam kelime hazinesi boyutunu döndürür.
        
        Returns:
            Vocabulary boyutu
        """
        return len(self.vectorizer.vocabulary_)
    
    def save(self, filepath: str):
        """
        Vektörleştiriciyi dosyaya kaydet.
        
        Args:
            filepath: Kayıt yolu
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.vectorizer, filepath)
        print(f"✅ TF-IDF vectorizer kaydedildi: {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Kaydedilmiş vektörleştiriciyi yükle.
        
        Args:
            filepath: Dosya yolu
            
        Returns:
            TurkishTfidfVectorizer örneği
        """
        instance = cls()
        instance.vectorizer = joblib.load(filepath)
        print(f"✅ TF-IDF vectorizer yüklendi: {filepath}")
        return instance


def create_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2)
) -> TurkishTfidfVectorizer:
    """
    TF-IDF vektörleştirici oluştur.
    
    Args:
        max_features: Maksimum özellik sayısı
        ngram_range: N-gram aralığı
        
    Returns:
        TurkishTfidfVectorizer örneği
    """
    return TurkishTfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("TF-IDF VEKTÖRLEŞTİRİCİ TESTİ")
    print("=" * 70)
    
    # Test verileri
    sample_texts = [
        "tebrikler kazandınız hemen tıklayın",
        "bugün buluşalım kahve içelim",
        "acele edin indirim şimdi arayın",
        "yarın toplantı var mısın",
        "kazandınız ödül almak için tıklayın"
    ]
    
    # Vektörleştirici oluştur
    vectorizer = create_tfidf_vectorizer(max_features=20, ngram_range=(1, 2))
    
    # Fit ve transform
    tfidf_matrix = vectorizer.fit_transform(sample_texts)
    
    print(f"\n📊 TF-IDF Matris Boyutu: {tfidf_matrix.shape}")
    print(f"📚 Vocabulary Boyutu: {vectorizer.get_vocabulary_size()}")
    print(f"\n🔤 Özellik İsimleri (ilk 10):")
    features = vectorizer.get_feature_names()
    for i, feature in enumerate(features[:10], 1):
        print(f"  {i}. {feature}")
    
    print("\n✅ TF-IDF vektörleştirici başarıyla test edildi!")
    print("=" * 70)
