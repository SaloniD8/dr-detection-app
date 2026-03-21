import streamlit as st
import numpy as np
import cv2
from PIL import Image
from fpdf import FPDF
import tempfile
import os
import uuid
import datetime
from model import load_model, predict

# ── Page config ──
st.set_page_config(
    page_title = 'DocSight',
    page_icon  = '👁️',
    layout     = 'wide'
)

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Fraunces:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #f8fafc;
    color: #1e293b;
}

.stApp { background: #f8fafc; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1300px; }

.app-header {
    background: linear-gradient(135deg, #0f766e, #0369a1);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 4px 24px rgba(15, 118, 110, 0.2);
}

.header-icon {
    font-size: 2.8rem;
    background: rgba(255,255,255,0.15);
    width: 64px;
    height: 64px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.header-title {
    font-family: 'Fraunces', serif;
    font-size: 2rem;
    font-weight: 700;
    color: white;
    margin: 0;
    line-height: 1.1;
}

.header-sub {
    font-size: 0.88rem;
    color: rgba(255,255,255,0.75);
    margin: 4px 0 0 0;
    font-weight: 400;
    letter-spacing: 0.03em;
}

.section-title {
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #0f766e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.8rem 0 1rem 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #ccfbf1;
}

.pid-badge {
    background: linear-gradient(135deg, #ecfdf5, #f0fdfa);
    border: 1.5px solid #5eead4;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 0.9rem;
    font-weight: 600;
    color: #0f766e;
    display: inline-block;
    margin-bottom: 1rem;
    letter-spacing: 0.05em;
}

.result-dr {
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border: 1.5px solid #fda4af;
    border-left: 5px solid #e11d48;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
}

.result-nodr {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1.5px solid #86efac;
    border-left: 5px solid #16a34a;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
}

.result-borderline {
    background: linear-gradient(135deg, #fefce8, #fef9c3);
    border: 1.5px solid #fde047;
    border-left: 5px solid #ca8a04;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
}

.result-title {
    font-family: 'Fraunces', serif;
    font-size: 1.3rem;
    font-weight: 700;
    margin: 0 0 6px 0;
}

.result-sub {
    font-size: 0.88rem;
    color: #64748b;
    margin: 0;
}

.metric-row {
    display: flex;
    gap: 12px;
    margin: 1.2rem 0;
}

.metric-box {
    flex: 1;
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}

.metric-val {
    font-family: 'Fraunces', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #0f766e;
}

.metric-lbl {
    font-size: 0.72rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

.img-caption {
    font-size: 0.75rem;
    color: #94a3b8;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 6px;
    font-weight: 500;
}

.stTextInput > div > div > input {
    background: white !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #1e293b !important;
    padding: 10px 14px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stTextInput > div > div > input::placeholder {
    color: #94a3b8 !important;
    font-size: 0.9rem !important;
}

.stTextInput > label {
    display: none !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0f766e, #0369a1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(15,118,110,0.3) !important;
}

[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #0f766e, #0369a1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    width: 100% !important;
}

[data-testid="stFileUploadDropzone"] {
    background: white !important;
    border: 2px dashed #94a3b8 !important;
    border-radius: 14px !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #0f766e, #0369a1) !important;
    border-radius: 4px !important;
}

[data-testid="stSidebar"] {
    background: white !important;
    border-right: 1px solid #e2e8f0 !important;
}

.stat-card {
    background: linear-gradient(135deg, #f0fdfa, #ecfeff);
    border: 1px solid #99f6e4;
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 10px;
    text-align: center;
}

.stat-value {
    font-family: 'Fraunces', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #0f766e;
}

.stat-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

hr { border-color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model ──
@st.cache_resource
def get_model():
    with st.spinner('Loading AI model...'):
        return load_model()

model = get_model()


# ── Helper: generate patient ID ──
def generate_patient_id():
    today = datetime.datetime.now().strftime('%Y%m%d')
    uid   = str(uuid.uuid4()).upper()[:6]
    return f"PAT-{today}-{uid}"


# ── Session state init ──
if 'patient_id' not in st.session_state:
    st.session_state.patient_id = generate_patient_id()
if 'reset' not in st.session_state:
    st.session_state.reset = False


# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0 1.5rem 0;">
        <div style="font-family:'Fraunces',serif; font-size:1.2rem;
                    font-weight:700; color:#0f766e;">DocSight</div>
        <div style="font-size:0.72rem; color:#94a3b8; text-transform:uppercase;
                    letter-spacing:0.1em; margin-top:4px;">AI Diagnostic Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    for val, lbl in [("0.9986","AUC-ROC"),("98.18%","Accuracy"),("98.86%","Sensitivity"),("97.16%","Specificity")]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{val}</div>
            <div class="stat-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; color:#64748b; line-height:1.8;">
        <b style="color:#334155;">Model:</b> EfficientNet-B4<br>
        <b style="color:#334155;">Data:</b> APTOS 2019 + IDRiD<br>
        <b style="color:#334155;">Images:</b> 4,117<br>
        <b style="color:#334155;">Threshold:</b> 0.35<br>
        <b style="color:#334155;">Explainability:</b> Grad-CAM
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">How to Use</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; color:#64748b; line-height:1.9;">
        1️⃣ &nbsp; Patient ID is auto-generated<br>
        2️⃣ &nbsp; Enter patient details<br>
        3️⃣ &nbsp; Upload fundus image<br>
        4️⃣ &nbsp; View AI analysis<br>
        5️⃣ &nbsp; Download PDF report<br>
        6️⃣ &nbsp; Click Next Patient to reset
    </div>""", unsafe_allow_html=True)


# ── Header ──
st.markdown("""
<div class="app-header">
    <div class="header-icon">👁️</div>
    <div>
        <p class="header-title">DocSight</p>
        <p class="header-sub">AI Powered GUI for Diabetic Retinopathy Diagnosis</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Patient Info ──
st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="pid-badge">
    🪪 &nbsp; Patient ID: &nbsp; {st.session_state.patient_id}
</div>
""", unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
with col_a:
    patient_name = st.text_input('', placeholder='👤  Patient Name',
                                  key=f'name_{st.session_state.patient_id}')
with col_b:
    patient_age  = st.text_input('', placeholder='🎂  Age',
                                  key=f'age_{st.session_state.patient_id}')
with col_c:
    patient_gender = st.text_input('', placeholder='⚧  Gender (Male / Female)',
                                    key=f'gender_{st.session_state.patient_id}')
# ── Buttons row ──
btn1, btn2 = st.columns(2)
with btn1:
    if st.button('🔄 Generate New Patient ID'):
        st.session_state.patient_id = generate_patient_id()
        st.rerun()
with btn2:
    if st.button('👤 Test Next Patient — Clear All'):
        st.session_state.patient_id = generate_patient_id()
        st.rerun()


# ── Upload ──
st.markdown('<div class="section-title">Fundus Image Upload</div>', unsafe_allow_html=True)

if st.session_state.reset:
    st.session_state.reset = False

uploaded = st.file_uploader(
    'Upload a retinal fundus image (PNG, JPG, JPEG)',
    type=['png', 'jpg', 'jpeg'],
    key=f'upload_{st.session_state.patient_id}'
)


# ── Analysis ──
if uploaded is not None:
    pil_img   = Image.open(uploaded).convert('RGB')
    img_array = np.array(pil_img)

    st.markdown('<div class="section-title">Analysis</div>', unsafe_allow_html=True)
    progress = st.progress(0, text='Initializing...')

    import time
    progress.progress(25, text='Preprocessing image...')
    time.sleep(0.3)
    progress.progress(55, text='Running AI inference...')
    img_rgb, cam, prediction, prob = predict(model, img_array, threshold=0.35)
    progress.progress(80, text='Generating Grad-CAM...')
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.5, heatmap, 0.5, 0)
    time.sleep(0.2)
    progress.progress(100, text='Analysis complete!')
    time.sleep(0.3)
    progress.empty()

    # ── Images ──
    st.markdown('<div class="section-title">Visual Analysis</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(img_rgb, use_column_width=True)
        st.markdown('<div class="img-caption">Preprocessed Image</div>', unsafe_allow_html=True)
    with c2:
        st.image(heatmap, use_column_width=True)
        st.markdown('<div class="img-caption">Grad-CAM · Red = High Attention</div>', unsafe_allow_html=True)
    with c3:
        st.image(overlay, use_column_width=True)
        st.markdown('<div class="img-caption">Overlay</div>', unsafe_allow_html=True)

    # ── Result ──
    st.markdown('<div class="section-title">Diagnostic Result</div>', unsafe_allow_html=True)
    conf = 'Refer to Specialist' if prediction == 'DR' else 'Routine Checkup Advised'

    if prediction == 'DR':
        st.markdown(f"""
        <div class="result-dr">
            <p class="result-title" style="color:#be123c;">🔴 &nbsp; Diabetic Retinopathy Detected</p>
            <p class="result-sub">DR Probability: <b>{prob:.1%}</b> &nbsp;·&nbsp;
            Recommendation: <b>{conf}</b></p>
        </div>""", unsafe_allow_html=True)
    elif prob >= 0.2:
        st.markdown(f"""
        <div class="result-borderline">
            <p class="result-title" style="color:#92400e;">🟡 &nbsp; Borderline — Manual Review Advised</p>
            <p class="result-sub">DR Probability: <b>{prob:.1%}</b> &nbsp;·&nbsp;
            Further examination recommended</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-nodr">
            <p class="result-title" style="color:#15803d;">🟢 &nbsp; No Diabetic Retinopathy Detected</p>
            <p class="result-sub">DR Probability: <b>{prob:.1%}</b> &nbsp;·&nbsp;
            Recommendation: <b>{conf}</b></p>
        </div>""", unsafe_allow_html=True)

    # ── Metrics ──
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box">
            <div class="metric-val">{'DR' if prediction=='DR' else 'No DR'}</div>
            <div class="metric-lbl">Prediction</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{prob:.4f}</div>
            <div class="metric-lbl">DR Probability</div>
        </div>
       <div class="metric-box">
            <div class="metric-val" style="font-size:0.95rem;">{conf}</div>
            <div class="metric-lbl">Recommendation</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">0.35</div>
            <div class="metric-lbl">Threshold</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── PDF ──
    st.markdown('<div class="section-title">Download Report</div>', unsafe_allow_html=True)
    if st.button('📄 Generate PDF Report'):
        with st.spinner('Generating report...'):
            tmp_dir      = tempfile.mkdtemp()
            orig_path    = os.path.join(tmp_dir, 'original.png')
            heat_path    = os.path.join(tmp_dir, 'heatmap.png')
            overlay_path = os.path.join(tmp_dir, 'overlay.png')
            Image.fromarray(img_rgb).save(orig_path)
            Image.fromarray(heatmap).save(heat_path)
            Image.fromarray(overlay).save(overlay_path)

            pdf = FPDF()
            pdf.add_page()

            pdf.set_fill_color(15, 118, 110)
            pdf.rect(0, 0, 210, 28, 'F')
            pdf.set_font('Arial', 'B', 18)
            pdf.set_text_color(255, 255, 255)
            pdf.set_y(8)
            pdf.cell(0, 10, 'DocSight - AI Diagnostic Report', ln=True, align='C')
            pdf.set_font('Arial', '', 9)
            pdf.set_text_color(204, 251, 241)
            pdf.cell(0, 6, 'AI Powered GUI for Diabetic Retinopathy Diagnosis', ln=True, align='C')
            pdf.set_y(34)

            pdf.set_fill_color(240, 253, 250)
            pdf.rect(10, pdf.get_y(), 190, 30, 'F')
            pdf.set_y(pdf.get_y() + 4)
            pdf.set_font('Arial', 'B', 10)
            pdf.set_text_color(15, 118, 110)
            pdf.cell(0, 6, 'PATIENT INFORMATION', ln=True, align='C')
            pdf.set_font('Arial', '', 10)
            pdf.set_text_color(30, 41, 59)
            pdf.cell(47, 6, f'ID: {st.session_state.patient_id}', align='C')
            pdf.cell(47, 6, f'Name: {patient_name if patient_name else "N/A"}', align='C')
            pdf.cell(47, 6, f'Age: {patient_age if patient_age else "N/A"}', align='C')
            pdf.cell(47, 6, f'Gender: {patient_gender if patient_gender else "N/A"}', align='C', ln=True)
            pdf.ln(4)

            pdf.set_font('Arial', '', 9)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(0, 6, f'Report generated: {datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")}',
                     ln=True, align='C')
            pdf.ln(4)

            if prediction == 'DR':
                pdf.set_fill_color(255, 241, 242)
                r, g, b = 190, 18, 60
            else:
                pdf.set_fill_color(240, 253, 244)
                r, g, b = 21, 128, 61

            pdf.rect(10, pdf.get_y(), 190, 20, 'F')
            pdf.set_y(pdf.get_y() + 4)
            pdf.set_font('Arial', 'B', 13)
            pdf.set_text_color(r, g, b)
            result_text = 'DIABETIC RETINOPATHY DETECTED' if prediction == 'DR' else 'NO DIABETIC RETINOPATHY DETECTED'
            pdf.cell(0, 8, result_text, ln=True, align='C')
            pdf.set_font('Arial', '', 9)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(0, 5,
                     f'DR Probability: {prob:.4f} ({prob:.1%})   |   Confidence: {conf}   |   Threshold: 0.35',
                     ln=True, align='C')
            pdf.ln(6)

            pdf.set_font('Arial', 'B', 11)
            pdf.set_text_color(15, 118, 110)
            pdf.cell(0, 8, 'IMAGE ANALYSIS', ln=True)
            img_y = pdf.get_y()
            pdf.image(orig_path,    x=10,  y=img_y, w=58)
            pdf.image(heat_path,    x=76,  y=img_y, w=58)
            pdf.image(overlay_path, x=142, y=img_y, w=58)
            pdf.set_y(img_y + 60)
            pdf.set_font('Arial', '', 9)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(63, 5, 'Preprocessed Image', align='C')
            pdf.cell(63, 5, 'Grad-CAM Heatmap',   align='C')
            pdf.cell(63, 5, 'Overlay',             align='C')
            pdf.ln(12)

            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)
            pdf.set_font('Arial', 'I', 8)
            pdf.set_text_color(120, 120, 120)
            pdf.multi_cell(0, 5,
                'Note: This report has been generated by an AI-assisted diagnostic tool '
                'for clinical reference only. The findings presented are based on automated '
                'image analysis and must be reviewed and verified by a qualified medical '
                'professional before any clinical decision is made. AI results may not '
                'account for all clinical factors. Final diagnosis remains the sole '
                'responsibility of the treating physician.'
            )

            pdf_path = os.path.join(tmp_dir, 'report.pdf')
            pdf.output(pdf_path)
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()

        fname = f"DocSight_Report_{st.session_state.patient_id}.pdf"
        st.download_button(
            label     = '⬇️ Download PDF Report',
            data      = pdf_bytes,
            file_name = fname,
            mime      = 'application/pdf'
        )
        st.success('PDF ready!')

# ── Footer ──
st.markdown("""
<hr>
<div style="text-align:center; padding:1rem 0; font-size:0.75rem; color:#94a3b8;">
    DocSight &nbsp;·&nbsp; EfficientNet-B4 &nbsp;·&nbsp;
    APTOS 2019 + IDRiD &nbsp;·&nbsp; AUC 0.9986
</div>
""", unsafe_allow_html=True)
