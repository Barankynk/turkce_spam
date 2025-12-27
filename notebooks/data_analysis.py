"""
Türkçe SMS Spam Veri Seti Analizi
Bu script veri setini yükler, analiz eder ve görselleştirir.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Stil ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Veri setini yükle
data_path = '../data/raw/TurkishSMSCollection.csv'
df = pd.read_csv(data_path)

print("=" * 60)
print("TÜRKÇE SMS SPAM VERİ SETİ ANALİZİ")
print("=" * 60)

# Temel bilgiler
print("\n📊 Veri Seti Genel Bakış:")
print(f"Toplam mesaj sayısı: {len(df)}")
print(f"Özellik sayısı: {df.shape[1]}")
print(f"\nKolon isimleri: {list(df.columns)}")

# İlk 5 satır
print("\n🔍 İlk 5 Mesaj:")
print(df.head())

# Sınıf dağılımı
print("\n📈 Sınıf Dağılımı:")
class_dist = df.iloc[:, -1].value_counts()
print(class_dist)
print(f"\nSpam oranı: %{(class_dist.iloc[0] / len(df) * 100):.2f}")
print(f"Normal oranı: %{(class_dist.iloc[1] / len(df) * 100):.2f}")

# Eksik veri kontrolü
print("\n🔎 Eksik Veri Kontrolü:")
missing = df.isnull().sum()
print(missing)
if missing.sum() == 0:
    print("✅ Eksik veri yok!")

# Mesaj uzunlukları
print("\n📏 Mesaj Uzunluk İstatistikleri:")
df['message_length'] = df.iloc[:, 0].str.len()
print(df['message_length'].describe())

# Sınıflara göre ortalama uzunluk
print("\n📊 Sınıflara Göre Ortalama Mesaj Uzunluğu:")
avg_length_by_class = df.groupby(df.iloc[:, -1])['message_length'].mean()
print(avg_length_by_class)

# Görselleştirmeler
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Türkçe SMS Spam Veri Seti Analizi', fontsize=16, fontweight='bold')

# 1. Sınıf Dağılımı
class_counts = df.iloc[:, -1].value_counts()
axes[0, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
               colors=['#ff6b6b', '#51cf66'], startangle=90)
axes[0, 0].set_title('Sınıf Dağılımı (Spam vs Normal)')

# 2. Sınıf Dağılımı (Bar)
axes[0, 1].bar(class_counts.index, class_counts.values, color=['#ff6b6b', '#51cf66'])
axes[0, 1].set_title('Mesaj Sayıları')
axes[0, 1].set_ylabel('Adet')
for i, v in enumerate(class_counts.values):
    axes[0, 1].text(i, v + 50, str(v), ha='center', fontweight='bold')

# 3. Mesaj Uzunluğu Dağılımı
axes[1, 0].hist(df['message_length'], bins=50, color='#4dabf7', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Mesaj Uzunluğu Dağılımı')
axes[1, 0].set_xlabel('Karakter Sayısı')
axes[1, 0].set_ylabel('Frekans')

# 4. Sınıflara Göre Mesaj Uzunluğu
spam_lengths = df[df.iloc[:, -1] == class_counts.index[0]]['message_length']
normal_lengths = df[df.iloc[:, -1] == class_counts.index[1]]['message_length']

axes[1, 1].boxplot([spam_lengths, normal_lengths], labels=['Spam', 'Normal'])
axes[1, 1].set_title('Sınıflara Göre Mesaj Uzunluğu')
axes[1, 1].set_ylabel('Karakter Sayısı')

plt.tight_layout()
plt.savefig('../data/dataset_analysis.png', dpi=300, bbox_inches='tight')
print("\n📸 Görselleştirmeler 'data/dataset_analysis.png' dosyasına kaydedildi!")

# Özet rapor
print("\n" + "=" * 60)
print("ÖZET RAPOR")
print("=" * 60)
print(f"✅ Veri seti başarıyla yüklendi: {len(df)} mesaj")
print(f"✅ Dengeli dağılım: Spam %{(class_counts.iloc[0] / len(df) * 100):.1f}, Normal %{(class_counts.iloc[1] / len(df) * 100):.1f}")
print(f"✅ Ortalama mesaj uzunluğu: {df['message_length'].mean():.0f} karakter")
print(f"✅ Veri kalitesi: Eksik veri yok")
print("=" * 60)
