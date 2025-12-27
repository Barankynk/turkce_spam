# Veri Seti Bilgileri

## Turkish SMS Collection Dataset

**Kaynak**: 
- GitHub: https://github.com/onrkrsy/TurkishSMS-Collection
- Kaggle: https://www.kaggle.com/datasets/onurkursiy/turkish-sms-collection-dataset

**İçerik**:
- **Toplam Mesaj**: 4,751 adet
- **Spam**: 2,536 mesaj
- **Normal (Ham)**: 2,215 mesaj
- **Dil**: Türkçe
- **Kaynak**: Türkiye'nin farklı bölgelerinden ve yaş gruplarından kişiler

**Sınıf Dağılımı**:
- Spam: %53.4
- Normal: %46.6
- Dengeli bir dağılım (balanced dataset)

**Format**: CSV dosyası
- Kolon 1: Metin (SMS içeriği)
- Kolon 2: Etiket (spam/ham veya 1/0)

## Kullanım Planı

1. Kaggle'dan CSV dosyasını indir
2. `data/raw/` klasörüne yerleştir
3. Veri analizi ve ön işleme
4. %80 eğitim / %20 test ayrımı
5. İşlenmiş veriyi `data/processed/` klasörüne kaydet

## Notlar

- Dataset açık kaynak ve araştırma amaçlı kullanıma uygun
- Gerçek Türkçe SMS mesajları içeriyor
- Spam/Normal dağılımı dengeli
- Ek veri toplamasına gerek yok
