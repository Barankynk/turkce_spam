"""
Türkçe Metin Ön İşleme Modülü
Bu modül Türkçe SMS mesajları için özel ön işleme fonksiyonları içerir.
"""

import re
import string
from typing import List, Optional

# TurkishNLP'yi modül seviyesinde bir kez yükle (performans için)
try:
    from turkishnlp import detector
    _TR_NLP = detector.TurkishNLP()
except Exception:
    _TR_NLP = None

# Türkçe stopwords listesi
TURKISH_STOPWORDS = {
    've', 'veya', 'ile', 'ama', 'fakat', 'ancak', 'lakin', 
    'ki', 'de', 'da', 'mi', 'mu', 'mı', 'mü',
    'bir', 'bu', 'şu', 'o', 'ben', 'sen', 'biz', 'siz', 'onlar',
    'için', 'gibi', 'kadar', 'daha', 'çok', 'az', 'her', 'bazı',
    'hiç', 'çünkü', 'neden', 'nasıl', 'ne', 'nerede', 'kim', 'hangi',
    'ya', 'yani', 'yahut', 'hem', 'ise', 'eğer', 'şayet',
    'var', 'yok', 'olarak', 'olan', 'olur', 'olmak',
    'değil', 'göre', 'karşı', 'sonra', 'önce', 'üzere',
    'beri', 'dolayı', 'rağmen', 'artık', 'henüz', 'bile', 'dahi'
}


def clean_text(text: str) -> str:
    """
    Metni temel temizleme işlemlerinden geçirir.
    
    Args:
        text: Ham metin
        
    Returns:
        Temizlenmiş metin
    """
    if not isinstance(text, str):
        return ""
    
    # Küçük harfe çevir (Türkçe karakterler korunur)
    text = text.lower()
    
    # Birden fazla boşluğu tek boşluğa indir
    text = re.sub(r'\s+', ' ', text)
    
    # Baş ve son boşlukları kaldır
    text = text.strip()
    
    return text


def remove_punctuation(text: str) -> str:
    """
    Noktalama işaretlerini kaldırır.
    
    Args:
        text: Metin
        
    Returns:
        Noktalama işareti olmayan metin
    """
    # Türkçe noktalama işaretleri dahil
    punctuation = string.punctuation + '""''–—…'
    translator = str.maketrans('', '', punctuation)
    return text.translate(translator)


def normalize_numbers(text: str) -> str:
    """
    Sayıları <NUM> token'a çevirir (silmek yerine).
    Spam tespitinde sayılar önemli sinyal (ör: "1000 TL", "50 bonus").
    
    Args:
        text: Metin
        
    Returns:
        Sayılar normalize edilmiş metin
    """
    return re.sub(r'\d+', ' <NUM> ', text)


def normalize_urls_and_phones(text: str) -> str:
    """
    URL'leri ve telefon numaralarını özel token'lara çevirir.
    Spam tespitinde URL ve telefon önemli sinyal.
    
    Args:
        text: Metin
        
    Returns:
        URL ve telefonlar normalize edilmiş metin
    """
    # URL'leri <URL> token'a çevir
    text = re.sub(r'http[s]?://\S+|www\.\S+', ' <URL> ', text)
    
    # Türk cep telefonu formatlarını <PHONE> token'a çevir
    # Formatlar: 05551234567, 0 555 123 4567, +905551234567
    text = re.sub(r'\+?90?\s*0?5\d{2}\s?\d{3}\s?\d{2}\s?\d{2}', ' <PHONE> ', text)
    text = re.sub(r'\b0?5\d{9}\b', ' <PHONE> ', text)
    
    return text


def remove_special_chars(text: str) -> str:
    """
    Özel karakterleri kaldırır (Türkçe karakterler korunur).
    
    Args:
        text: Metin
        
    Returns:
        Özel karakter içermeyen metin
    """
    # Türkçe karakterler ve normal harfler dışındakileri kaldır
    # Türkçe: ç, ğ, ı, ö, ş, ü, İ, Ç, Ğ, Ö, Ş, Ü
    pattern = r'[^a-zA-ZçğıöşüÇĞİÖŞÜ\s]'
    return re.sub(pattern, '', text)


def remove_stopwords(text: str, custom_stopwords: Optional[set] = None) -> str:
    """
    Türkçe stopwords kaldırır.
    
    Args:
        text: Metin
        custom_stopwords: Opsiyonel özel stopwords seti
        
    Returns:
        Stopwords olmayan metin
    """
    stopwords = custom_stopwords if custom_stopwords else TURKISH_STOPWORDS
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)


def turkish_stemming(text: str) -> str:
    """
    Türkçe kelimeleri köklerine indirgeme (basit kurallar).
    
    Not: Bu basit bir fallback. advanced_turkish_stemming tercih edilmeli.
    
    Args:
        text: Metin
        
    Returns:
        Köklerine indirgenmiş metin
    """
    # Yaygın Türkçe ekler (konservatif liste - tek harfliler çıkarıldı)
    suffixes = [
        'lar', 'ler', 'lik', 'lık', 'luk', 'lük',
        'da', 'de', 'ta', 'te', 'dan', 'den', 'tan', 'ten',
        'nın', 'nin', 'nun', 'nün', 'ın', 'in', 'un', 'ün',
        'na', 'ne',
        'dır', 'dir', 'dur', 'dür', 'tır', 'tir', 'tur', 'tür',
        'mış', 'miş', 'muş', 'müş',
        'yor', 'ıyor', 'iyor', 'uyor', 'üyor'
    ]
    
    words = text.split()
    stemmed_words = []
    
    for word in words:
        if len(word) > 5:  # Sadece uzun kelimelere dokun
            for suffix in sorted(suffixes, key=len, reverse=True):
                if word.endswith(suffix):
                    # Kökün en az 3 karakter olmasını sağla
                    if len(word) - len(suffix) >= 3:
                        word = word[:-len(suffix)]
                        break
        stemmed_words.append(word)
    
    return ' '.join(stemmed_words)


def advanced_turkish_stemming(text: str) -> str:
    """
    TurkishNLP kütüphanesi ile gelişmiş kök bulma.
    
    Args:
        text: Metin
        
    Returns:
        Morfolojik analiz sonucu köklerine indirgenmiş metin
    """
    if _TR_NLP is None:
        # TurkishNLP yüklü değilse basit stemming kullan
        return turkish_stemming(text)
    
    try:
        # Kelimeleri ayır ve her birini köklerine indir
        words = text.split()
        stemmed_words = []
        
        for word in words:
            try:
                # TurkishNLP ile kök bulma
                stem = _TR_NLP.get_stem(word)
                stemmed_words.append(stem if stem else word)
            except:
                stemmed_words.append(word)
        
        return ' '.join(stemmed_words)
    
    except Exception:
        # Hata durumunda basit stemming'e geri dön
        return turkish_stemming(text)


def preprocess_message(
    text: str, 
    remove_punct: bool = True,
    normalize_nums: bool = True,  # remove_nums yerine normalize
    remove_stop: bool = True,
    use_stemming: bool = True,
    advanced_stem: bool = True  # Varsayılan olarak advanced stemming
) -> str:
    """
    Tam ön işleme pipeline'ı.
    
    Args:
        text: Ham metin
        remove_punct: Noktalama kaldırılsın mı?
        normalize_nums: Sayılar normalize edilsin mi? (silmek yerine <NUM>)
        remove_stop: Stopwords kaldırılsın mı?
        use_stemming: Kök bulma yapılsın mı?
        advanced_stem: Gelişmiş stemming (TurkishNLP) kullanılsın mı?
        
    Returns:
        Tam işlenmiş metin
    """
    # 1. Temel temizleme
    text = clean_text(text)
    
    # 2. URL ve telefonları normalize et
    text = normalize_urls_and_phones(text)
    
    # 3. Noktalama işaretlerini kaldır
    if remove_punct:
        text = remove_punctuation(text)
    
    # 4. Sayıları normalize et (silmek yerine <NUM> token)
    if normalize_nums:
        text = normalize_numbers(text)
    
    # 5. Özel karakterleri kaldır (Türkçe karakterler korunur)
    text = remove_special_chars(text)
    
    # 6. Stopwords kaldır
    if remove_stop:
        text = remove_stopwords(text)
    
    # 7. Kök bulma
    if use_stemming:
        if advanced_stem:
            text = advanced_turkish_stemming(text)
        else:
            text = turkish_stemming(text)
    
    # 8. Son temizlik (boşluklar)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def batch_preprocess(texts: List[str], **kwargs) -> List[str]:
    """
    Bir liste mesajı toplu olarak işler.
    
    Args:
        texts: Metin listesi
        **kwargs: preprocess_message için parametreler
        
    Returns:
        İşlenmiş metin listesi
    """
    return [preprocess_message(text, **kwargs) for text in texts]


# Test
if __name__ == "__main__":
    # Test mesajları
    test_messages = [
        "Tebrikler! 10.000 TL kazandınız. Hemen tıklayın: www.example.com",
        "Bugün buluşalım mı? Saat 5'te kahve içelim.",
        "ACELE EDİN!!! %90 İNDİRİM!!! Şimdi arayın: 0555 123 4567"
    ]
    
    print("=" * 70)
    print("TÜRKÇE METIN ÖN İŞLEME TESTİ")
    print("=" * 70)
    
    for i, msg in enumerate(test_messages, 1):
        print(f"\n{i}. MESAJ:")
        print(f"Ham: {msg}")
        processed = preprocess_message(msg)
        print(f"İşlenmiş: {processed}")
        print("-" * 70)
