"""
Türkçe SMS Spam Sınıflandırma - Model Eğitim Scripti
Bu script veri setini yükler, işler, modeli eğitir ve kaydeder.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib
import os
import sys

# Kendi modüllerimizi import et
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_message, batch_preprocess
from feature_extraction import TurkishTfidfVectorizer
from utils import plot_confusion_matrix, print_classification_metrics


# Sabitler
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'raw', 'TurkishSMSCollection.csv')
MODEL_PATH = os.path.join(PROJECT_DIR, 'models', 'spam_classifier.joblib')
VECTORIZER_PATH = os.path.join(PROJECT_DIR, 'models', 'tfidf_vectorizer.joblib')
CONFUSION_MATRIX_PATH = os.path.join(PROJECT_DIR, 'data', 'confusion_matrix.png')
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_dataset(filepath):
    """
    Veri setini yükle ve hazırla.
    
    Args:
        filepath: CSV dosya yolu
        
    Returns:
        DataFrame
    """
    print("\n" + "=" * 70)
    print("VERİ SETİ YÜKLEME")
    print("=" * 70)
    
    # CSV'yi yükle
    df = pd.read_csv(
        filepath,
        sep=';',
        header=None,
        names=['message', 'group', 'group_text'],
        encoding='utf-8',
        on_bad_lines='skip'
    )
    
    print(f"✅ Veri seti yüklendi: {len(df)} mesaj")
    print(f"📊 Kolonlar: {list(df.columns)}")
    
    # Group kolonunu güvenli şekilde integer'a çevir
    df['group'] = pd.to_numeric(df['group'], errors='coerce')
    
    # Geçersiz satırları at
    initial_count = len(df)
    df = df.dropna(subset=['group'])
    removed_invalid = initial_count - len(df)
    if removed_invalid > 0:
        print(f"⚠️ {removed_invalid} adet geçersiz group değeri temizlendi")
    
    df['group'] = df['group'].astype(int)
    
    # Group'u binary hale getir (1 = Spam, 0 = Normal)
    # Group 1 = Spam, Group 2 = Normal
    # Explicit mapping ile label oluştur
    df['label'] = df['group'].map({1: 1, 2: 0})
    
    # Beklenmeyen değerleri temizle (1 veya 2 dışında)
    initial_count = len(df)
    df = df.dropna(subset=['label'])
    removed_unexpected = initial_count - len(df)
    if removed_unexpected > 0:
        print(f"⚠️ {removed_unexpected} adet beklenmeyen group değeri (1 veya 2 dışında) temizlendi")
    
    df['label'] = df['label'].astype(int)
    
    # Sınıf dağılımı
    spam_count = (df['label'] == 1).sum()
    normal_count = (df['label'] == 0).sum()
    
    print(f"\n📈 Sınıf Dağılımı:")
    print(f"  Spam: {spam_count} (%{spam_count/len(df)*100:.1f})")
    print(f"  Normal: {normal_count} (%{normal_count/len(df)*100:.1f})")
    
    return df


def preprocess_data(df):
    """
    Metinleri ön işlemden geçir.
    
    Args:
        df: DataFrame
        
    Returns:
        DataFrame with processed messages
    """
    print("\n" + "=" * 70)
    print("METIN ÖN İŞLEME")
    print("=" * 70)
    
    print("🔄 Metinler işleniyor...")
    
    # Batch preprocessing
    df['processed_message'] = batch_preprocess(
        df['message'].tolist(),
        remove_punct=True,
        normalize_nums=True,  # Sayıları <NUM> token'a çevir
        remove_stop=True,
        use_stemming=True,
        advanced_stem=True  # Advanced stemming kullan (varsayılan)
    )
    
    # Boş mesajları filtrele
    initial_count = len(df)
    df = df[df['processed_message'].str.len() > 0]
    removed = initial_count - len(df)
    
    if removed > 0:
        print(f"⚠️ {removed} adet boş mesaj kaldırıldı")
    
    print(f"✅ {len(df)} mesaj başarıyla işlendi")
    
    # Örnek göster
    print("\n📝 Örnek İşlenmiş Mesajlar:")
    for i in range(min(3, len(df))):
        print(f"\n{i+1}. HAM: {df.iloc[i]['message'][:80]}...")
        print(f"   İŞLENMİŞ: {df.iloc[i]['processed_message'][:80]}...")
        print(f"   ETİKET: {'SPAM' if df.iloc[i]['label'] == 1 else 'NORMAL'}")
    
    return df


def split_dataset(df, test_size=0.2, random_state=42):
    """
    Veri setini train/test olarak ayır.
    
    Args:
        df: DataFrame
        test_size: Test seti oranı
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 70)
    print("VERİ SETİ AYIRMA")
    print("=" * 70)
    
    X = df['processed_message']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Sınıf dengesini koru
    )
    
    print(f"✅ Eğitim seti: {len(X_train)} mesaj")
    print(f"✅ Test seti: {len(X_test)} mesaj")
    print(f"📊 Train Spam oranı: %{(y_train.sum()/len(y_train)*100):.1f}")
    print(f"📊 Test Spam oranı: %{(y_test.sum()/len(y_test)*100):.1f}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, X_test, y_test):
    """
    TF-IDF vektörleştir ve Multinomial Naive Bayes eğit.
    
    Args:
        X_train: Eğitim metinleri
        y_train: Eğitim etiketleri
        X_test: Test metinleri
        y_test: Test etiketleri
        
    Returns:
        model, vectorizer, metrics
    """
    print("\n" + "=" * 70)
    print("TF-IDF VEKTÖRLEŞTİRME")
    print("=" * 70)
    
    # TF-IDF Vectorizer oluştur
    vectorizer = TurkishTfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # Unigram ve bigram
        min_df=2,
        max_df=0.95
    )
    
    # Fit ve transform
    X_train_tfidf = vectorizer.fit_transform(X_train.tolist())
    X_test_tfidf = vectorizer.transform(X_test.tolist())
    
    print(f"✅ TF-IDF matris boyutu: {X_train_tfidf.shape}")
    print(f"📚 Vocabulary boyutu: {vectorizer.get_vocabulary_size()}")
    
    print("\n" + "=" * 70)
    print("MODEL EĞİTİMİ")
    print("=" * 70)
    
    # Multinomial Naive Bayes
    model = MultinomialNB(alpha=1.0)
    
    print("🔄 Model eğitiliyor...")
    model.fit(X_train_tfidf, y_train)
    print("✅ Model eğitimi tamamlandı!")
    
    print("\n" + "=" * 70)
    print("MODEL DEĞERLENDİRME")
    print("=" * 70)
    
    # Tahmin
    y_train_pred = model.predict(X_train_tfidf)
    y_test_pred = model.predict(X_test_tfidf)
    
    # Metrikler
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    print(f"\n📊 EĞITIM SETİ:")
    print(f"  Accuracy: %{train_accuracy*100:.2f}")
    
    print(f"\n📊 TEST SETİ:")
    print(f"  Accuracy:  %{test_accuracy*100:.2f}")
    print(f"  Precision: %{test_precision*100:.2f}")
    print(f"  Recall:    %{test_recall*100:.2f}")
    print(f"  F1-Score:  %{test_f1*100:.2f}")
    
    # Detaylı classification report
    print_classification_metrics(y_test, y_test_pred, target_names=['Normal', 'Spam'])
    
    # Confusion Matrix
    print("\n📊 CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Görselleştirme
    plot_confusion_matrix(
        y_test, 
        y_test_pred,
        labels=['Normal', 'Spam'],
        save_path=CONFUSION_MATRIX_PATH
    )
    
    metrics = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1
    }
    
    return model, vectorizer, metrics


def save_models(model, vectorizer, model_path, vectorizer_path):
    """
    Model ve vectorizer'ı kaydet.
    
    Args:
        model: Eğitilmiş model
        vectorizer: TF-IDF vectorizer
        model_path: Model kayıt yolu
        vectorizer_path: Vectorizer kayıt yolu
    """
    print("\n" + "=" * 70)
    print("MODEL KAYDETME")
    print("=" * 70)
    
    # Klasörü oluştur
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Modeli kaydet
    joblib.dump(model, model_path)
    print(f"✅ Model kaydedildi: {model_path}")
    
    # Vectorizer'ı kaydet
    vectorizer.save(vectorizer_path)
    
    print("\n✅ Tüm dosyalar başarıyla kaydedildi!")


def main():
    """Ana eğitim pipeline'ı."""
    print("\n" + "=" * 70)
    print("TÜRKÇE SMS SPAM TESPİTİ - MODEL EĞİTİMİ")
    print("=" * 70)
    
    # 1. Veri yükle
    df = load_dataset(DATA_PATH)
    
    # 2. Ön işleme
    df = preprocess_data(df)
    
    # 3. Train/Test ayır
    X_train, X_test, y_train, y_test = split_dataset(
        df, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    # 4. Model eğit
    model, vectorizer, metrics = train_model(X_train, y_train, X_test, y_test)
    
    # 5. Modelleri kaydet
    save_models(model, vectorizer, MODEL_PATH, VECTORIZER_PATH)
    
    # Özet
    print("\n" + "=" * 70)
    print("ÖZET RAPOR")
    print("=" * 70)
    print(f"✅ Toplam mesaj: {len(df)}")
    print(f"✅ Test Accuracy: %{metrics['test_accuracy']*100:.2f}")
    print(f"✅ F1-Score: %{metrics['f1_score']*100:.2f}")
    print(f"✅ Model: {MODEL_PATH}")
    print(f"✅ Vectorizer: {VECTORIZER_PATH}")
    print("=" * 70)
    
    print("\n🎉 MODEL EĞİTİMİ BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
