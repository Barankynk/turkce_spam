"""
Türkçe SMS Spam Tespiti - Streamlit Web Arayüzü
Modern ve kullanıcı dostu spam tespit uygulaması
"""

import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path

# Preprocessing modülünü import et
from src.preprocessing import preprocess_message

# Sayfa Konfigürasyonu
st.set_page_config(
    page_title="🇹🇷 Türkçe Spam Tespiti",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .spam-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .normal-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #2d3748;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #f7fafc;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .stTextArea textarea {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Model Yükleme (Cache ile)
@st.cache_resource
def load_models():
    """Model ve vectorizer'ı yükle"""
    try:
        model_path = Path('models/spam_classifier.joblib')
        vectorizer_path = Path('models/tfidf_vectorizer.joblib')
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Model yükleme hatası: {e}")
        st.stop()

# Tahmin Fonksiyonu
def predict_spam(text, model, vectorizer):
    """
    Mesajın spam olup olmadığını tahmin et
    
    Returns:
        tuple: (is_spam, probability, processed_text)
    """
    # Ön işleme
    processed_text = preprocess_message(
        text,
        remove_punct=True,
        normalize_nums=True,
        remove_stop=True,
        use_stemming=True,
        advanced_stem=True
    )
    
    # Boş metin kontrolü
    if not processed_text or len(processed_text.strip()) == 0:
        return None, None, processed_text
    
    # TF-IDF vektörleştirme
    text_vector = vectorizer.transform([processed_text])
    
    # Tahmin
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    
    is_spam = prediction == 1
    spam_prob = probability[1] * 100
    
    return is_spam, spam_prob, processed_text

# Ana Uygulama
def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Türkçe SMS Spam Tespiti</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Yapay zeka destekli, Türkçe mesajlar için spam tespit sistemi</p>', unsafe_allow_html=True)
    
    # Model yükle
    model, vectorizer = load_models()
    
    # Sidebar - Bilgilendirme
    with st.sidebar:
        st.header("ℹ️ Hakkında")
        st.markdown("""
        Bu uygulama **Türkçe SMS mesajlarını** analiz ederek spam olup olmadığını tespit eder.
        
        ### 🔬 Kullanılan Teknolojiler
        - **TF-IDF**: Metin vektörleştirme
        - **Naive Bayes**: Makine öğrenimi
        - **TurkishNLP**: Türkçe morfolojik analiz
        
        ### 📊 Model Performansı
        - **Accuracy**: %93.36
        - **F1-Score**: %94.14
        - **Spam Recall**: %99.80
        
        ### ⚙️ Özellikler
        ✅ Tekli mesaj analizi  
        ✅ Toplu dosya analizi  
        ✅ Gerçek zamanlı tahmin  
        ✅ Türkçe'ye özel işleme
        """)
        
        st.divider()
        st.caption("🎯 NLP Projesi 2025")
    
    # Ana İçerik - Tabs
    tab1, tab2 = st.tabs(["📝 Tekli Mesaj Analizi", "📁 Toplu Dosya Analizi"])
    
    # TAB 1: Tekli Mesaj Analizi
    with tab1:
        st.subheader("Mesajınızı Analiz Edin")
        
        # Örnek mesajlar
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📌 Örnek Spam Mesaj"):
                st.session_state.sample_text = "Tebrikler! 1000 TL kazandınız. Hemen tıklayın: www.spam.com Bedava bonus 0555 123 4567"
        with col2:
            if st.button("📌 Örnek Normal Mesaj"):
                st.session_state.sample_text = "Merhaba, bugün saat 5'te kahve içmeye ne dersin? Görüşmek isterim."
        
        # Metin girişi
        user_message = st.text_area(
            "Mesajınızı buraya yazın:",
            value=st.session_state.get('sample_text', ''),
            height=150,
            placeholder="Analiz edilecek mesajı buraya yazın...",
            key="message_input"
        )
        
        # Analiz butonu
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("🔍 Mesajı Analiz Et", type="primary", use_container_width=True)
        
        if analyze_button and user_message:
            with st.spinner("🔄 Mesaj analiz ediliyor..."):
                is_spam, spam_prob, processed_text = predict_spam(user_message, model, vectorizer)
                
                if is_spam is None:
                    st.warning("⚠️ Mesaj işlenemedi. Lütfen geçerli bir metin girin.")
                else:
                    # Sonuç gösterimi
                    st.divider()
                    
                    # Ana sonuç
                    if is_spam:
                        st.markdown(f"""
                        <div class="spam-box">
                            🚨 SPAM MESAJ TESPİT EDİLDİ!
                        </div>
                        """, unsafe_allow_html=True)
                        st.error(f"⚡ Bu mesaj **%{spam_prob:.2f}** olasılıkla SPAM!")
                    else:
                        st.markdown(f"""
                        <div class="normal-box">
                            ✅ NORMAL MESAJ
                        </div>
                        """, unsafe_allow_html=True)
                        st.success(f"✨ Bu mesaj **%{100-spam_prob:.2f}** olasılıkla GÜVENLİ!")
                    
                    # Detaylar
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Spam Olasılığı",
                            f"%{spam_prob:.1f}",
                            delta=f"{spam_prob - 50:.1f}% eşik üstü" if spam_prob > 50 else f"{50 - spam_prob:.1f}% eşik altı"
                        )
                    
                    with col2:
                        st.metric(
                            "Güven Skoru",
                            f"%{max(spam_prob, 100-spam_prob):.1f}",
                            delta="Yüksek Güven" if max(spam_prob, 100-spam_prob) > 80 else "Orta Güven"
                        )
                    
                    with col3:
                        st.metric(
                            "Sonuç",
                            "SPAM" if is_spam else "NORMAL",
                            delta="Tehlikeli" if is_spam else "Güvenli"
                        )
                    
                    # Olasılık bar
                    st.subheader("📊 Tahmin Olasılığı")
                    col1, col2 = st.columns([spam_prob, 100-spam_prob] if spam_prob > 0 else [1, 1])
                    with col1:
                        st.progress(spam_prob/100, text=f"Spam: %{spam_prob:.1f}")
                    with col2:
                        st.progress((100-spam_prob)/100, text=f"Normal: %{100-spam_prob:.1f}")
                    
                    # Ön işlenmiş metin (expandable)
                    with st.expander("🔍 Ön İşleme Detayları"):
                        st.markdown("**Orijinal Mesaj:**")
                        st.code(user_message, language="text")
                        
                        st.markdown("**İşlenmiş Metin:**")
                        st.code(processed_text, language="text")
                        
                        st.info("""
                        **Ön İşleme Adımları:**
                        1. Küçük harfe çevirme
                        2. URL ve telefon normalizasyonu
                        3. Sayı normalizasyonu (<NUM>)
                        4. Noktalama temizleme
                        5. Türkçe stopwords kaldırma
                        6. Kök bulma (stemming)
                        """)
    
    # TAB 2: Toplu Dosya Analizi
    with tab2:
        st.subheader("📁 Toplu Mesaj Analizi")
        
        st.info("""
        **Desteklenen Formatlar:** TXT, CSV  
        **TXT:** Her satırda bir mesaj  
        **CSV:** 'message' veya 'text' kolonunda mesajlar
        """)
        
        # Dosya yükleme
        uploaded_file = st.file_uploader(
            "Dosyanızı yükleyin",
            type=['txt', 'csv'],
            help="TXT veya CSV formatında mesaj listesi"
        )
        
        if uploaded_file:
            try:
                # Dosya okuma
                if uploaded_file.name.endswith('.txt'):
                    content = uploaded_file.read().decode('utf-8')
                    messages = [line.strip() for line in content.split('\n') if line.strip()]
                elif uploaded_file.name.endswith('.csv'):
                    # Önce ; delimiter'ı dene (TurkishSMSCollection için)
                    try:
                        df_upload = pd.read_csv(uploaded_file, sep=';')
                    except:
                        # Başarısız olursa , delimiter'ı dene
                        uploaded_file.seek(0)  # Dosya pointerını başa sar
                        df_upload = pd.read_csv(uploaded_file, sep=',')
                    
                    # Mesaj kolonunu bul
                    msg_col = next((col for col in df_upload.columns if col.lower() in ['message', 'text', 'mesaj']), df_upload.columns[0])
                    messages = df_upload[msg_col].dropna().astype(str).tolist()
                
                st.success(f"✅ {len(messages)} mesaj yüklendi!")
                
                # Analiz butonu
                if st.button("🚀 Tüm Mesajları Analiz Et", type="primary"):
                    results = []
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, msg in enumerate(messages):
                        is_spam, spam_prob, processed = predict_spam(msg, model, vectorizer)
                        
                        # Boş/geçersiz mesaj kontrolü
                        if is_spam is None:
                            results.append({
                                'Mesaj': msg[:100] + '...' if len(msg) > 100 else msg,
                                'Sonuç': 'HATALI',
                                'Spam Olasılığı (%)': 'N/A',
                                'İşlenmiş Metin': processed if processed else ''
                            })
                        else:
                            results.append({
                                'Mesaj': msg[:100] + '...' if len(msg) > 100 else msg,
                                'Sonuç': 'SPAM' if is_spam else 'NORMAL',
                                'Spam Olasılığı (%)': f"{spam_prob:.2f}",
                                'İşlenmiş Metin': processed
                            })
                        
                        # Progress güncelle
                        progress = (idx + 1) / len(messages)
                        progress_bar.progress(progress)
                        status_text.text(f"İşlenen: {idx + 1}/{len(messages)}")
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Sonuçlar DataFrame
                    df_results = pd.DataFrame(results)
                    
                    # İstatistikler
                    st.subheader("📊 Analiz Sonuçları")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    spam_count = (df_results['Sonuç'] == 'SPAM').sum()
                    normal_count = (df_results['Sonuç'] == 'NORMAL').sum()
                    
                    with col1:
                        st.metric("Toplam Mesaj", len(messages))
                    with col2:
                        st.metric("🚨 Spam", spam_count, delta=f"%{spam_count/len(messages)*100:.1f}")
                    with col3:
                        st.metric("✅ Normal", normal_count, delta=f"%{normal_count/len(messages)*100:.1f}")
                    with col4:
                        st.metric("Spam Oranı", f"%{spam_count/len(messages)*100:.1f}")
                    
                    # Pasta grafik
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 5))
                    colors = ['#ff6b6b', '#51cf66']
                    ax.pie(
                        [spam_count, normal_count],
                        labels=['Spam', 'Normal'],
                        autopct='%1.1f%%',
                        colors=colors,
                        startangle=90
                    )
                    ax.set_title('Spam / Normal Dağılımı', fontsize=14, fontweight='bold')
                    st.pyplot(fig)
                    
                    # Sonuç tablosu
                    st.subheader("📋 Detaylı Sonuçlar")
                    st.dataframe(
                        df_results[['Mesaj', 'Sonuç', 'Spam Olasılığı (%)']],
                        use_container_width=True,
                        height=400
                    )
                    
                    # CSV İndirme
                    csv = df_results.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Sonuçları İndir (CSV)",
                        data=csv,
                        file_name="spam_analiz_sonuclari.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"❌ Dosya işleme hatası: {e}")

# Session State Başlatma
if 'sample_text' not in st.session_state:
    st.session_state.sample_text = ''

# Uygulamayı Çalıştır
if __name__ == "__main__":
    main()
