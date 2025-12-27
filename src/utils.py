"""
Yardımcı Fonksiyonlar ve Genel Utilities
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def load_model(filepath: str) -> Any:
    """
    Kaydedilmiş modeli yükle.
    
    Args:
        filepath: Model dosya yolu
        
    Returns:
        Yüklenmiş model
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {filepath}")
    
    model = joblib.load(filepath)
    print(f"✅ Model yüklendi: {filepath}")
    return model


def save_model(model: Any, filepath: str):
    """
    Modeli dosyaya kaydet.
    
    Args:
        model: Kaydedilecek model
        filepath: Kayıt yolu
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"✅ Model kaydedildi: {filepath}")


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    labels: list = None,
    save_path: str = None
):
    """
    Confusion matrix görselleştir.
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        labels: Sınıf isimleri
        save_path: Kayıt yolu (opsiyonel)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels if labels else ['Normal', 'Spam'],
        yticklabels=labels if labels else ['Normal', 'Spam'],
        cbar_kws={'label': 'Sayı'}
    )
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Gerçek Etiket', fontsize=12)
    plt.xlabel('Tahmin', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix kaydedildi: {save_path}")
    
    plt.show()


def print_classification_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    target_names: list = None
):
    """
    Sınıflandırma metriklerini yazdır.
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        target_names: Sınıf isimleri
    """
    if target_names is None:
        target_names = ['Normal', 'Spam']
    
    print("\n" + "=" * 60)
    print("SINIFLANDIRMA METRİKLERİ")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=target_names))
    print("=" * 60)


def split_data(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Veriyi train/test olarak ayır.
    
    Args:
        df: DataFrame
        text_column: Metin kolonu adı
        label_column: Etiket kolonu adı
        test_size: Test oranı
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    X = df[text_column]
    y = df[label_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Sınıf dengesini koru
    )
    
    return X_train, X_test, y_train, y_test


def get_turkish_stopwords() -> set:
    """
    Türkçe stopwords setini döndür.
    
    Returns:
        Stopwords seti
    """
    return {
        've', 'veya', 'ile', 'ama', 'fakat', 'ancak', 'lakin', 
        'ki', 'de', 'da', 'mi', 'mu', 'mı', 'mü',
        'bir', 'bu', 'şu', 'o', 'ben', 'sen', 'biz', 'siz', 'onlar',
        'için', 'gibi', 'kadar', 'daha', 'çok', 'az', 'her', 'bazı',
        'hiç', 'çünkü', 'neden', 'nasıl', 'ne', 'nerede', 'kim', 'hangi',
        'ya', 'yani', 'veya', 'yahut', 'hem', 'ise', 'eğer', 'şayet',
        'var', 'yok', 'olarak', 'olan', 'olur', 'olmak', 'olan',
        'değil', 'gibi', 'göre', 'karşı', 'sonra', 'önce', 'üzere',
        'beri', 'dolayı', 'rağmen', 'artık', 'henüz', 'bile', 'dahi'
    }


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("UTILS MODÜLÜ TESTİ")
    print("=" * 70)
    
    # Stopwords testi
    stopwords = get_turkish_stopwords()
    print(f"\n📚 Türkçe Stopwords Sayısı: {len(stopwords)}")
    print(f"Örnek stopwords: {list(stopwords)[:10]}")
    
    # Test confusion matrix
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    
    print("\n📊 Test Confusion Matrix:")
    plot_confusion_matrix(y_true, y_pred)
    
    print("\n✅ Utils modülü başarıyla test edildi!")
    print("=" * 70)
