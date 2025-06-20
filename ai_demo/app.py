import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io
import os
import pandas as pd
import base64
import uuid

# Sayfa yapılandırması
st.set_page_config(
    page_title="SE 3508 - AI Final Project",
    page_icon="🦅",
    layout="wide",
)

# Benzersiz bir ID oluştur (her sayfa yüklemesinde farklı olacak)
unique_id = str(uuid.uuid4())[:8]

# PDF gösterme fonksiyonu
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    
    # PDF'yi iframe içinde göster
    pdf_display = f"""
    <iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700" type="application/pdf"></iframe>
    """
    return pdf_display

# Başlık
st.title("Disease Detection from Chest X-Ray Images")
st.markdown("---")

# PDF görüntüleyici durumunu session_state'de saklayalım
if 'pdf_göster' not in st.session_state:
    st.session_state.pdf_göster = False

# Sidebar için PDF butonu
with st.sidebar:
    st.markdown("---")
    st.subheader("Project Presentation")
    
    # PDF butonu
    if st.button("Show PDF", key="show_pdf_btn", use_container_width=True):
        st.session_state.pdf_göster = True

# PDF görüntüleme bölümü
if st.session_state.pdf_göster:
    # Ana sayfayı gizle
    st.markdown("""
    <style>
    div.block-container {padding-top: 1rem; padding-bottom: 0rem;}
    div.stButton > button {display: inline-block; background-color: #FF4B4B; color: white;}
    </style>
    """, unsafe_allow_html=True)
    
    # Başlık ve kapat butonu
    col1, col2 = st.columns([6, 1])
    
    with col1:
        st.header("Project Presentation")
    
    with col2:
        if st.button("Close", key="close_pdf"):
            st.session_state.pdf_göster = False
            st.experimental_rerun()
    
    # PDF dosyasını doğrudan göster
    pdf_path = "proje_sunum.pdf"
    
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as file:
            base64_pdf = base64.b64encode(file.read()).decode('utf-8')
        
        # PDF'yi iframe içinde göster
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.error("PDF file not found! Please add the 'proje_sunum.pdf' file to the project folder.")
        st.session_state.pdf_göster = False
        
else:
    # Hastalık sınıflarını alfabetik olarak düzenleme
    CLASSES = sorted([
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
        "Effusion", "Emphysema", "Fibrosis", "Infiltration",
        "Mass", "No Finding", "Nodule", "Pleural_Thickening", 
        "Pneumonia", "Pneumothorax"
    ])

    # CSV dosyasını yükleme
    @st.cache_data
    def load_csv_data():
        try:
            df = pd.read_csv("Data_Entry_2017.csv")
            return df
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            return None

    # Model dosyalarının doğru yolda olup olmadığını kontrol et
    @st.cache_resource
    def check_models():
        required_models = ["densenet121_best_model.pth", "vgg19_best_multilabel.pth"]
        found_models = []
        
        for model in required_models:
            if os.path.exists(model):
                found_models.append(model)
        
        return found_models

    # DenseNet121 model tanımlaması
    @st.cache_resource
    def load_densenet121():
        model = models.densenet121(weights=None)
        # DenseNet modelinin classifier kısmını özel olarak değiştirme
        num_ftrs = model.classifier.in_features
        
        # Notebook'taki modele uygun classifier yapısı
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, len(CLASSES))
        )
        
        # Eğitilmiş model ağırlıklarını yükle
        model.load_state_dict(torch.load("densenet121_best_model.pth", map_location=torch.device('cpu')))
        model.eval()
        return model

    # VGG19 model tanımlaması
    @st.cache_resource
    def load_vgg19():
        model = models.vgg19(weights=None)
        
        # VGG19 modelinde eğitilmiş modelle aynı yapıda classifier tanımla
        num_ftrs = 512 * 7 * 7  # VGG19'un son conv katmanından sonraki boyut
        
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),  # Orta katman 4096 değil 1024 olmalı
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, len(CLASSES))  # Giriş boyutu 1024 olarak değiştirildi
        )
        
        # Eğitilmiş model ağırlıklarını yükle
        model.load_state_dict(torch.load("vgg19_best_multilabel.pth", map_location=torch.device('cpu')))
        model.eval()
        return model

    # Görüntü ön işleme
    def preprocess_image(image):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Grayscale görüntüleri 3 kanala dönüştür
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # batch boyutu ekle
        
        return input_batch

    # Tahmin fonksiyonu - @st.cache_data kaldırıldı
    def predict(_model, _img):
        with torch.no_grad():
            output = _model(_img)
            probabilities = torch.sigmoid(output)
            return probabilities.squeeze().numpy()

    # Kullanıcı arayüzü - sidebar
    st.sidebar.title("Model Selection")

    # Mevcut modelleri kontrol et
    available_models = check_models()

    if not available_models:
        st.error("Error: Model files not found!")
        st.info("Ensure that the DenseNet121 and VGG19 model files (.pth) are in the same directory as this application.")
        st.stop()

    # Model seçim kutucuğu
    model_choice = st.sidebar.selectbox(
        "Select the model you want to use:",
        ["DenseNet121", "VGG19"]
    )

    # Seçilen modele göre model yükleme
    if model_choice == "DenseNet121":
        if "densenet121_best_model.pth" in available_models:
            model = load_densenet121()
        else:
            st.sidebar.error("DenseNet121 model not found")
            st.stop()
    else:  # VGG19
        if "vgg19_best_multilabel.pth" in available_models:
            model = load_vgg19()
        else:
            st.sidebar.error("VGG19 model not found")
            st.stop()

    st.sidebar.info(f"{model_choice} model loaded")
    st.sidebar.markdown("---")

    # CSV verilerini yükle
    csv_data = load_csv_data()

    # Ana sayfa düzeni
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Chest X-Ray Image")
        
        # Dosya yükleme alanı
        uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Görüntüyü göster
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Chest X-Ray Image", use_column_width=True)
            
            # Görüntüyü işle
            processed_image = preprocess_image(image)
            
            # Tahmin yap
            predictions = predict(model, processed_image)

    with col2:
        st.subheader("Results")
        
        if uploaded_file is not None:
            # Dosya adını al
            file_name = uploaded_file.name
            
            # Sonuçları göster
            results = {CLASSES[i]: float(predictions[i]) for i in range(len(CLASSES))}
            
            # Sonuçları azalan sırada sırala
            sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
            
            # CSV'de eşleşme kontrolü
            gerçek_etiketler = None
            if csv_data is not None and file_name in csv_data["Image Index"].values:
                csv_row = csv_data[csv_data["Image Index"] == file_name].iloc[0]
                gerçek_etiketler = csv_row["Finding Labels"].split("|")
                
                st.markdown("### Real Labels")
                for etiket in gerçek_etiketler:
                    st.info(f"**{etiket}**")
                
                st.markdown("---")
            
            st.markdown("### Model Predictions")
            # Sonuçları görselleştir
            for disease, probability in sorted_results.items():
                # Eğer gerçek etiketlerde varsa vurgula
                if gerçek_etiketler is not None and disease in gerçek_etiketler:
                    st.markdown(f"**{disease}:** {probability:.2%} ✓")
                    st.progress(float(probability))
                else:
                    # 0.5'ten büyük olasılıkları vurgula
                    if probability >= 0.5:
                        st.markdown(f"**{disease}:** {probability:.2%}")
                        st.progress(float(probability))
                    else:
                        st.markdown(f"{disease}: {probability:.2%}")
                        st.progress(float(probability))
            
            # En yüksek olasılıklı hastalığı vurgula
            top_disease = list(sorted_results.keys())[0]
            top_prob = sorted_results[top_disease]
            
            st.markdown("---")
            if top_prob >= 0.5:
                st.success(f"**Most likely diagnosis: {top_disease} ({top_prob:.2%})**")
            else:
                st.info("No obvious diagnosis was detected. All probabilities are low.")
                
            # Karşılaştırma özeti
            if gerçek_etiketler is not None:
                st.markdown("### Comparison Summary")
                
                # Modelin tahmin ettiği sınıflar (olasılık > 0.5)
                model_tahminleri = [disease for disease, prob in sorted_results.items() if prob >= 0.5]
                
                # Doğru tahminler (gerçek etiketlerde olan ve modelin de tahmin ettiği)
                doğru_tahminler = [disease for disease in model_tahminleri if disease in gerçek_etiketler]
                
                # Yanlış tahminler (gerçek etiketlerde olmayıp modelin tahmin ettiği)
                yanlış_tahminler = [disease for disease in model_tahminleri if disease not in gerçek_etiketler]
                
                # Kaçırılan tahminler (gerçek etiketlerde olup modelin tahmin etmediği)
                kaçırılan_tahminler = [etiket for etiket in gerçek_etiketler if etiket not in model_tahminleri]
                
                # Metrikleri göster
                if len(doğru_tahminler) > 0:
                    st.success(f"**Correct predictions:** {', '.join(doğru_tahminler)}")
                
                if len(yanlış_tahminler) > 0:
                    st.error(f"**Wrong predictions:** {', '.join(yanlış_tahminler)}")
                
                if len(kaçırılan_tahminler) > 0:
                    st.warning(f"**Missed predictions:** {', '.join(kaçırılan_tahminler)}")
                    
                # Başarı oranı
                if len(gerçek_etiketler) > 0:
                    başarı_oranı = len(doğru_tahminler) / len(gerçek_etiketler) * 100
                    st.metric(label="Success Rate", value=f"%{başarı_oranı:.1f}")

    # Alt kısım bilgilendirmesi
    st.markdown("---")
    st.caption("This application uses DenseNet121 and VGG19 models to detect diseases from chest X-ray images.")
    st.caption("Note: This application is only for educational purposes and should not replace real medical diagnosis.")