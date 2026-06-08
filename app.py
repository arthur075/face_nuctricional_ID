import os
import logging
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import cv2
from deepface import DeepFace
import pandas as pd
import numpy as np

st.set_page_config(page_title="FaceID System", layout="wide", page_icon="🎯")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e8f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem !important; padding-bottom: 1rem !important; }
.titulo-sistema { font-size: 1.6rem; font-weight: 800; color: #ffffff; margin-bottom: 0; }
.subtitulo-sistema { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #4ade80; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 1rem; }
.badge-online  { display:inline-flex;align-items:center;gap:5px;background:#052e16;color:#4ade80;font-family:'DM Mono',monospace;font-size:0.65rem;padding:3px 10px;border-radius:999px;border:1px solid #166534; }
.badge-offline { display:inline-flex;align-items:center;gap:5px;background:#1a0a0a;color:#f87171;font-family:'DM Mono',monospace;font-size:0.65rem;padding:3px 10px;border-radius:999px;border:1px solid #7f1d1d; }
.dot { width:5px;height:5px;border-radius:50%;display:inline-block; }
.dot-green { background:#4ade80; }
.dot-red   { background:#f87171; }
.stButton > button { background:#13131a !important; color:#e8e8f0 !important; border:1px solid #2a2a40 !important; border-radius:8px !important; font-family:'Syne',sans-serif !important; font-weight:600 !important; padding:0.3rem 0.8rem !important; font-size:0.8rem !important; }
.stButton > button:hover { border-color:#4ade80 !important; color:#4ade80 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="titulo-sistema">FACE<span style="color:#4ade80">ID</span> SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitulo-sistema">▸ Reconhecimento Facial com Tabela Nutricional</div>', unsafe_allow_html=True)

# ─── DADOS ────────────────────────────────────────────────────────────────────
@st.cache_data
def carregar_dados():
    csv_path = os.path.join(os.path.dirname(__file__), "dados_pessoais.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=["Nome","Idade","Peso_kg","Altura_cm","Calorias","Proteinas_g","Carboidratos_g","Gorduras_g","Fibras_g","Sodio_mg","Acucar_g","Agua_ml","Restricoes"])

ROSTOS_DIR = os.path.join(os.path.dirname(__file__), "rostos_conhecidos")

# ─── FUNÇÕES ──────────────────────────────────────────────────────────────────
def identificar_pessoa(frame):
    try:
        resultados = DeepFace.find(img_path=frame, db_path=ROSTOS_DIR, enforce_detection=False, silent=True)
        if resultados and len(resultados[0]) > 0:
            melhor = resultados[0].iloc[0]["identity"]
            nome = os.path.splitext(os.path.basename(melhor))[0].replace("_"," ").split(" ")[0]
            return nome
    except Exception:
        pass
    return None

def desenhar_overlay(frame, nome):
    try:
        faces = DeepFace.extract_faces(img_path=frame, enforce_detection=False, detector_backend="opencv")
        for face_data in faces:
            r = face_data.get("facial_area", {})
            x, y, w, h = r.get("x",0), r.get("y",0), r.get("w",0), r.get("h",0)
            cor = (74,222,128) if nome else (248,113,113)
            cv2.rectangle(frame, (x,y), (x+w,y+h), cor, 2)
            label = nome if nome else "DESCONHECIDO"
            (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y-th-14), (x+tw+12, y), cor, -1)
            cv2.putText(frame, label, (x+6,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,10,20), 2)
    except Exception:
        pass
    return frame

def pct(val, maximo):
    try:
        return min(int((float(val)/maximo)*100), 100)
    except:
        return 0

def render_info(pessoa, placeholder):
    df = carregar_dados()
    with placeholder.container():
        if not pessoa:
            st.markdown("""
            <div style="background:#0e0e1a;border:1px dashed #2a2a40;border-radius:12px;padding:1.5rem;text-align:center;color:#4b5563;margin-top:0.5rem">
                <div style="font-size:2rem">👤</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.75rem;margin-top:0.3rem">Aguardando identificação...</div>
            </div>""", unsafe_allow_html=True)
            return

        resultado = df[df["Nome"].str.lower().str.contains(pessoa.lower(), na=False)]
        if resultado.empty:
            st.markdown(f"""
            <div style="background:#1a0e0e;border:1px solid #7f1d1d;border-radius:12px;padding:1rem;text-align:center">
                <div style="font-size:1.5rem">⚠️</div>
                <div style="color:#f87171;font-weight:600;font-size:0.9rem">{pessoa}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#6b7280;margin-top:0.3rem">Sem cadastro no banco de dados</div>
            </div>""", unsafe_allow_html=True)
            return

        d = resultado.iloc[0]
        nome   = d.get("Nome","—")
        idade  = d.get("Idade","—")
        peso   = d.get("Peso_kg","—")
        altura = d.get("Altura_cm","—")
        cal    = d.get("Calorias", 0)
        prot   = d.get("Proteinas_g", 0)
        carb   = d.get("Carboidratos_g", 0)
        gord   = d.get("Gorduras_g", 0)
        fibra  = d.get("Fibras_g", 0)
        sodio  = d.get("Sodio_mg", 0)
        acucar = d.get("Acucar_g", 0)
        agua   = d.get("Agua_ml", 0)
        rest   = d.get("Restricoes","Nenhuma")

        p_prot  = pct(prot, 50)
        p_carb  = pct(carb, 300)
        p_gord  = pct(gord, 65)
        p_fibra = pct(fibra, 25)
        p_sodio = pct(sodio, 2300)

        # Card do nome (compacto)
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0f1f0f,#0a1a10);border:1px solid #166534;border-radius:10px;padding:0.6rem 1rem;margin-bottom:0.5rem;display:flex;align-items:center;justify-content:space-between">
            <div>
                <div style="font-size:1.2rem;font-weight:800;color:#4ade80">{nome}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#6b7280">{idade}a · {peso}kg · {altura}cm</div>
            </div>
            <span style="background:#052e16;color:#4ade80;font-family:'DM Mono',monospace;font-size:0.6rem;padding:2px 8px;border-radius:999px;border:1px solid #166534">✓ ID</span>
        </div>""", unsafe_allow_html=True)

        # Tabela nutricional compacta
        st.markdown(f"""
        <div style="background:#13131a;border:1px solid #1e1e2e;border-radius:10px;overflow:hidden;font-size:0.78rem">

          <div style="background:linear-gradient(90deg,#1a1a2e,#16213e);padding:0.5rem 0.9rem;border-bottom:2px solid #000">
            <div style="font-weight:800;color:#fff;letter-spacing:0.05em;font-size:0.8rem">TABELA NUTRICIONAL</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#6b7280">Ingestão diária recomendada</div>
          </div>

          <div style="background:#0a0a0f;padding:0.4rem 0.9rem;display:flex;justify-content:space-between;align-items:baseline;border-bottom:6px solid #000">
            <span style="font-weight:700;color:#e8e8f0">Calorias</span>
            <span><span style="font-size:1.5rem;font-weight:800;color:#4ade80">{cal}</span><span style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#6b7280"> kcal</span></span>
          </div>

          <div style="padding:0.2rem 0.9rem 0rem">
            <span style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#4b5563">% VALORES DIÁRIOS</span>
          </div>

          <div style="display:flex;justify-content:space-between;padding:0.3rem 0.9rem;border-bottom:1px solid #1a1a2e">
            <span style="font-weight:700;color:#e8e8f0">Proteínas</span>
            <span><span style="font-family:'DM Mono',monospace;color:#c8c8d8">{prot}g</span> <span style="font-family:'DM Mono',monospace;color:#4ade80;font-size:0.7rem">{p_prot}%</span></span>
          </div>
          <div style="padding:0.15rem 0.9rem;border-bottom:1px solid #1a1a2e">
            <div style="background:#1e1e2e;border-radius:999px;height:4px"><div style="width:{p_prot}%;background:#4ade80;height:4px;border-radius:999px"></div></div>
          </div>

          <div style="display:flex;justify-content:space-between;padding:0.3rem 0.9rem;border-bottom:1px solid #1a1a2e">
            <span style="font-weight:700;color:#e8e8f0">Carboidratos</span>
            <span><span style="font-family:'DM Mono',monospace;color:#c8c8d8">{carb}g</span> <span style="font-family:'DM Mono',monospace;color:#4ade80;font-size:0.7rem">{p_carb}%</span></span>
          </div>
          <div style="padding:0.15rem 0.9rem;border-bottom:1px solid #1a1a2e">
            <div style="background:#1e1e2e;border-radius:999px;height:4px"><div style="width:{p_carb}%;background:#60a5fa;height:4px;border-radius:999px"></div></div>
          </div>
          <div style="display:flex;justify-content:space-between;padding:0.2rem 0.9rem 0.3rem 1.4rem;border-bottom:1px solid #1a1a2e">
            <span style="color:#9ca3af">· Açúcares</span>
            <span style="font-family:'DM Mono',monospace;color:#c8c8d8">{acucar}g</span>
          </div>

          <div style="display:flex;justify-content:space-between;padding:0.3rem 0.9rem;border-bottom:1px solid #1a1a2e">
            <span style="font-weight:700;color:#e8e8f0">Gorduras</span>
            <span><span style="font-family:'DM Mono',monospace;color:#c8c8d8">{gord}g</span> <span style="font-family:'DM Mono',monospace;color:#4ade80;font-size:0.7rem">{p_gord}%</span></span>
          </div>
          <div style="padding:0.15rem 0.9rem;border-bottom:1px solid #1a1a2e">
            <div style="background:#1e1e2e;border-radius:999px;height:4px"><div style="width:{p_gord}%;background:#f59e0b;height:4px;border-radius:999px"></div></div>
          </div>

          <div style="display:flex;justify-content:space-between;padding:0.3rem 0.9rem;border-bottom:1px solid #1a1a2e">
            <span style="font-weight:700;color:#e8e8f0">Fibras</span>
            <span><span style="font-family:'DM Mono',monospace;color:#c8c8d8">{fibra}g</span> <span style="font-family:'DM Mono',monospace;color:#4ade80;font-size:0.7rem">{p_fibra}%</span></span>
          </div>
          <div style="padding:0.15rem 0.9rem;border-bottom:2px solid #000">
            <div style="background:#1e1e2e;border-radius:999px;height:4px"><div style="width:{p_fibra}%;background:#a78bfa;height:4px;border-radius:999px"></div></div>
          </div>

          <div style="display:flex;justify-content:space-between;padding:0.3rem 0.9rem;border-bottom:1px solid #1a1a2e">
            <span style="font-weight:700;color:#e8e8f0">Sódio</span>
            <span><span style="font-family:'DM Mono',monospace;color:#c8c8d8">{sodio}mg</span> <span style="font-family:'DM Mono',monospace;color:#f87171;font-size:0.7rem">{p_sodio}%</span></span>
          </div>
          <div style="padding:0.15rem 0.9rem;border-bottom:1px solid #1a1a2e">
            <div style="background:#1e1e2e;border-radius:999px;height:4px"><div style="width:{p_sodio}%;background:#f87171;height:4px;border-radius:999px"></div></div>
          </div>

          <div style="display:flex;justify-content:space-between;padding:0.3rem 0.9rem;border-bottom:1px solid #1a1a2e">
            <span style="font-weight:700;color:#e8e8f0">Água</span>
            <span style="font-family:'DM Mono',monospace;color:#c8c8d8">{agua} ml</span>
          </div>

          <div style="display:flex;justify-content:space-between;padding:0.3rem 0.9rem;border-bottom:none">
            <span style="font-weight:700;color:#e8e8f0">Restrições</span>
            <span style="font-family:'DM Mono',monospace;color:#fbbf24;font-size:0.7rem">{rest}</span>
          </div>

          <div style="padding:0.4rem 0.9rem;font-family:'DM Mono',monospace;font-size:0.55rem;color:#4b5563;border-top:2px solid #000;line-height:1.5">
            * % VD com base em dieta de 2.000 kcal/dia.
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─── ESTADO ───────────────────────────────────────────────────────────────────
if "camera_ativa" not in st.session_state:
    st.session_state.camera_ativa = False
if "pessoa_identificada" not in st.session_state:
    st.session_state.pessoa_identificada = None

# ─── LAYOUT ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    if st.session_state.camera_ativa:
        st.markdown('<span class="badge-online"><span class="dot dot-green"></span> CÂMERA ATIVA</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-offline"><span class="dot dot-red"></span> CÂMERA OFFLINE</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    btn1, btn2 = st.columns(2)
    with btn1:
        if st.button("▶  Iniciar Câmera", use_container_width=True):
            st.session_state.camera_ativa = True
            st.rerun()
    with btn2:
        if st.button("⏹  Parar Câmera", use_container_width=True):
            st.session_state.camera_ativa = False
            st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)
    FRAME_WINDOW = st.empty()

with col2:
    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-weight:700;font-size:0.9rem;color:#fff;margin-bottom:0.5rem;letter-spacing:0.02em">DIETA E MACRONUTRIENTES</div>', unsafe_allow_html=True)
    info_placeholder = st.empty()

render_info(st.session_state.pessoa_identificada, info_placeholder)

# ─── LOOP DA CÂMERA ───────────────────────────────────────────────────────────
if st.session_state.camera_ativa:
    cap = cv2.VideoCapture(0)
    frame_count = 0

    while st.session_state.camera_ativa:
        ret, frame = cap.read()
        if not ret:
            st.error("Câmera não encontrada.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_count % 30 == 0:
            nome = identificar_pessoa(frame_rgb)
            if nome != st.session_state.pessoa_identificada:
                st.session_state.pessoa_identificada = nome
                render_info(nome, info_placeholder)

        frame_overlay = desenhar_overlay(frame_rgb.copy(), st.session_state.pessoa_identificada)
        FRAME_WINDOW.image(frame_overlay, use_container_width=True)
        frame_count += 1

    cap.release()