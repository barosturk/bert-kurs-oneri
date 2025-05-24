# -*- coding: utf-8 -*-
import streamlit as st
import streamlit.components.v1 as stc
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import math
import base64# logo.png dosyasını base64'e çevirmek için

# ------------------------------
# MODEL YUKLEME
# ------------------------------
@st.cache_resource
def load_bert_model():
    return SentenceTransformer('distiluse-base-multilingual-cased-v1')

# ------------------------------
# VERI YUKLEME
# ------------------------------
@st.cache_data
def load_data():
    dosya_yolu = os.path.join(os.path.dirname(__file__), "data", "merged-udemy_courses_tr.csv")
    df = pd.read_csv(dosya_yolu, encoding="utf-8-sig", on_bad_lines='skip', sep=",")
    df.columns = df.columns.str.strip()
    df['course_title'] = df['course_title'].fillna('')  # NaN başlıkları temizle
    print(df.dtypes)
    # İnceleme sayısını sayısal tipe çevir
    df['review'] = pd.to_numeric(df['review'], errors='coerce').fillna(0).astype(int)
    return df

def show_course_card(title, score, price, review, url):
    if "youtube.com" in url:
        platform = "YouTube"
    elif "udemy.com" in url:
        platform = "Udemy"
    else:
        platform = "Diğer"

    st.markdown(
        f"""
        <div style="border-radius: 10px; padding: 15px; background-color: #f8f9fa;
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.1); margin: 10px; color: #000;">
            <div style="font-size: 16px; font-weight: bold; color: #000; margin-bottom: 8px;">{title}</div>
            <p><strong>💡 Benzerlik Skoru:</strong> {score:.2f}</p>
            <p><strong>💰 Fiyat:</strong> {price}</p>
            <p><strong>📝 İnceleme Sayısı:</strong> {review}</p>
            <p><strong>🌐 Platform:</strong> {platform}</p>
            <a href="{url}" target="_blank">
                <button style="background-color:#007bff; color:white; padding:8px 12px;
                               border:none; border-radius:5px; cursor:pointer;">
                    🔗 Kursa Git ({platform})
                </button>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------
# BERT TABANLI ONERI
# ------------------------------
@st.cache_data(show_spinner=False)
def get_recommendation_bert(query, df, selected_level="Tüm Seviyeler", num_of_rec=7, min_similarity=0.4):
    model = load_bert_model()

    course_titles = df['course_title'].fillna('').tolist()
    title_embeddings = model.encode(course_titles, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)

    cos_scores = util.pytorch_cos_sim(query_embedding, title_embeddings)[0]

    query_words = query.lower().split()

    # 1. Benzerlik eşik değerinden geçenleri al
    top_results = [i for i in range(len(cos_scores)) if cos_scores[i] >= min_similarity]

    # 2. Başlıkta arama kelimesi geçenleri al
    top_results = [i for i in top_results if any(word in str(df.iloc[i]['course_title']).lower() for word in query_words)]

    # Eğer hiç sonuç yoksa fallback
    if not top_results:
        fallback = df[df['course_title'].str.contains(query, case=False, na=False)]
        if not fallback.empty:
            fallback = fallback.head(num_of_rec).copy()
            fallback['similarity_score'] = 0.0
            return fallback[['course_title', 'similarity_score', 'url', 'price', 'review']]
        return None

    # 🔥 BURASI KRİTİK: Skor ve inceleme sayısına göre sırala
    sorted_results = sorted(
        top_results,
        key=lambda x: (round(float(cos_scores[x]), 4), df.iloc[x]['review']),
        reverse=True
    )

    # 🔥 Sonra ilk num_of_rec kadarını al
    sorted_indices = sorted_results[:num_of_rec]

    recommended = df.iloc[sorted_indices].copy()
    recommended['similarity_score'] = [float(cos_scores[i]) for i in sorted_indices]

    # Seviye filtresi uygula
    if selected_level != "Tüm Düzeyler":
        recommended = recommended[recommended['level'].str.contains(selected_level, case=False, na=False)]

    return recommended[['course_title', 'similarity_score', 'url', 'price', 'review']]

# ------------------------------

# UYGULAMA ARAYUZU
# ------------------------------
def main():
   

    st.title("BERT Tabanli Kurs Oneri Uygulamasi")
    st.caption("Kodlama Kursları İçin Türkçe Destekli Anlamsal Filtreleme")
    logo_base64 = get_base64_logo("logo.png")

    st.sidebar.markdown(
    f"""
    <div style='text-align: center; padding-top: 10px; padding-bottom: 10px;'>
        <img src="data:image/png;base64,{logo_base64}" style="max-width: 80%; height: auto;" />
        <div style='font-size: 16px; font-weight: bold; color: #333;'>©2025 Yıldırım Bilsem </div>
        <div style='font-size: 12px; color: gray; font-style: italic;'>Eğitimde Yapay Zeka Gücü</div>
   
    </div>
    """,
    unsafe_allow_html=True
)
    menu = ["Ana Sayfa", "Öneri", "Hakkında"]
    choice = st.sidebar.selectbox("Menü", menu)
    # Menü altına responsive logo ekle
    # Sidebar altına sabit logo
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"] > div:first-child {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100vh;
    }
    .sidebar-logo {
        text-align: center;
        padding: 10px 0;
    }
    .sidebar-logo img {
        max-width: 100%;
        height: auto;
    }
    </style>
    """,
        unsafe_allow_html=True
    )
    
    df = load_data()    

    if choice == "Ana Sayfa":
        st.subheader("Veri Onizleme")
        st.dataframe(df.head(10))

    elif choice == "Öneri":
        st.subheader("Kurs Onerileri")
        search_term = st.text_input("Nasıl bir kurs arıyorsunuz?")
        num_of_rec = st.sidebar.slider("Kaç kurs önerilsin?", 1, 20, 7)
        min_similarity = st.sidebar.slider("Minimum benzerlik skoru", 0.1, 1.0, 0.4, 0.05)
        level_options = ["Tüm Düzeyler", "Yeni Başlayan", "Orta", "Uzman"]
        selected_level = st.sidebar.selectbox("Seviye Seçimi", level_options)

        if st.button("Öneri Getir") and search_term:
            with st.spinner("🔄 Sonuçlar getiriliyor..."):
                results = get_recommendation_bert(search_term, df, selected_level, num_of_rec, min_similarity)


            if results is not None:
                for row in results.itertuples(index=False):
                    try:
                        price_value = float(row.price)
                        if math.isnan(price_value):
                            fiyat = "Ücretsiz"
                        else:
                            fiyat = f"₺{price_value:.2f}"
                    except:
                        fiyat = "Ücretsiz"

                    show_course_card(row.course_title, row.similarity_score, fiyat, row.review, row.url)

            else:
                st.warning("Uygun kurs bulunamadı. Daha farklı ifadelerle arama yapmayı deneyin.")

    else:
        st.subheader("Hakkında")
        st.info("Bu uygulama Yıldırım BİLSEM Bilişim ÖYG/2 Grubu tarafından Streamlit platformunda BERT modeli ile geliştirilmiştir.")
        
def get_base64_logo(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

if __name__ == '__main__':
    main()