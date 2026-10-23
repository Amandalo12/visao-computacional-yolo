import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Detector de Ativos de Marketing",
    page_icon="🖼️",
    layout="centered"
)

@st.cache_resource
def carregar_modelo():
    """Carrega o modelo YOLOv8 em cache para otimizar memória e performance."""
    return YOLO("yolov8n-oiv7.pt")

def processar_imagem(image_np, model, conf_threshold):
    """Executa a detecção e desenha as caixas na imagem."""
    # O YOLO aceita arrays numpy diretamente (formato RGB do PIL)
    results = model(image_np, conf=conf_threshold, verbose=False)[0]
    
    classes_interesse = {"Cell phone", "Car", "House", "Building"}
    deteccoes_validas = []
    
    # Converte de RGB para BGR para desenhar com OpenCV corretamente
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        
        if cls_name in classes_interesse:
            coords = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            deteccoes_validas.append({
                "classe": cls_name,
                "confianca": round(conf, 2)
            })
            
            x1, y1, x2, y2 = map(int, coords)
            label = f"{cls_name} ({conf:.2f})"
            
            # Desenha retângulo e texto
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, label, (x1, max(y1 - 10, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    # Retorna a imagem convertida de volta para RGB para exibição no Streamlit
    img_result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_result_rgb, deteccoes_validas

# Interface do Usuário
st.title("🎯 Detector de Ativos de Marketing")
st.markdown("Identifique **celulares**, **carros** e **casas** em campanhas visuais instantaneamente.")

# Barra lateral para parâmetros
st.sidebar.header("Configurações")
confianca = st.sidebar.slider("Limiar de Confiança (Confidence)", 0.1, 0.9, 0.4, 0.05)

# Upload de imagem
arquivo_enviado = st.file_uploader("Envie uma imagem (.jpg, .png)", type=["jpg", "jpeg", "png"])

if arquivo_enviado is not None:
    # Carrega a imagem enviada
    image = Image.open(arquivo_enviado)
    image_np = np.array(image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
        
    with st.spinner("Analisando imagem..."):
        model = carregar_modelo()
        imagem_processada, deteccoes = processar_imagem(image_np, model, confianca)
        
    with col2:
        st.subheader("Resultado Detectado")
        st.image(imagem_processada, use_container_width=True)
        
    # Exibe resumo analítico
    st.markdown("### 📊 Resumo das Detecções")
    if deteccoes:
        # Conta ocorrências por classe
        contagem = {}
        for d in deteccoes:
            contagem[d["classe"]] = contagem.get(d["classe"], 0) + 1
            
        cols = st.columns(len(contagem))
        for i, (cls, qtd) in enumerate(contagem.items()):
            cols[i].metric(label=cls, value=qtd)
    else:
        st.info("Nenhum objeto de interesse detectado com o limiar atual.")