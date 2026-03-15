import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
from fpdf import FPDF
import tempfile
import os
from model import load_model, predict

# ── Page config ──
st.set_page_config(
    page_title = 'DocSight',
    page_icon  = '👁️',
    layout     = 'wide'
)

# ── CSS Styling ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080c14;
    color: #e2e8f0;
}

.stApp {
    background: #080c14;
}

/* ── Hide default Streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1400px; }

/* ── Header ── */
.header-container {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 2rem 0 1rem 0;
    border-bottom: 1px solid rgba(56, 189, 248, 0.15);
    margin-bottom: 2rem;
}

.header-icon {
    width: 52px;
    height: 52px;
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    box-shadow: 0 0 24px rgba(14, 165, 233, 0.35);
}

.header-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
}

.header-sub {
    font-size: 0.85rem;
    color: #64748b;
    margin: 0;
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Stat cards in sidebar ── */
.stat-card {
    background: rgba(14, 165, 233, 0.06);
    border: 1px solid rgba(14, 165, 233, 0.15);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    text-align: center;
}

.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #38bdf8;
}

.stat-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Upload zone ── */
.upload-zone {
    background: rgba(14, 165, 233, 0.04);
    border: 2px dashed rgba(56, 189, 248, 0.25);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    margin-bottom: 1.5rem;
    transition: all 0.3s ease;
}

.upload-zone:hover {
    border-color: rgba(56, 189, 248, 0.5);
    background: rgba(14, 165, 233, 0.08);
}

/* ── Image cards ── */
.img-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(56, 189, 248, 0.1);
    border-radius: 14px;
    padding: 12px;
    text-align: center;
}

.img-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 8px;
    font-weight: 500;
}

/* ── Result banner ── */
.result-dr {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-left: 4px solid #ef4444;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
    animation: fadeIn 0.5s ease;
}

.result-nodr {
    background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(16,185,129,0.06));
    border: 1px solid rgba(34, 197, 94, 0.35);
    border-left: 4px solid #22c55e;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
    animation: fadeIn 0.5s ease;
}

.result-borderline {
    background: linear-gradient(135deg, rgba(234,179,8,0.12), rgba(245,158,11,0.06));
    border: 1px solid rgba(234, 179, 8, 0.35);
    border-left: 4px solid #eab308;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
    animation: fadeIn 0.5s ease;
}

.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0 0 6px 0;
}

.result-sub {
    font-size: 0.88rem;
    color: #94a3b8;
    margin: 0;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 1rem 0;
}

.metric-box {
    flex: 1;
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #38bdf8;
}

.metric-lbl {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* ── Section title ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(56, 189, 248, 0.1);
}

/* ── Patient input ── */
.stTextInput > div > div > input {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    padding: 10px 14px !important;
}

.stTextInput > div > div > input:focus {
    border-color: rgba(56, 189, 248, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.1) !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(14, 165, 233, 0.35) !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
    border-radius: 4px !important;
}

/* ── Divider ── */
hr { border-color: rgba(56, 189, 248, 0.1) !important; }

/* ── Animations ── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(14,165,233,0.4); }
    50%       { box-shadow: 0 0 0 8px rgba(14,165,233,0); }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a0f1a !important;
    border-right: 1px solid rgba(56, 189, 248, 0.08) !important;
}

[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem !important;
}

/* ── File uploader ── */
[data-testid="stFileUploadDropzone"] {
    background: rgba(14, 165, 233, 0.04) !important;
    border: 2px dashed rgba(56, 189, 248, 0.25) !important;
    border-radius: 14px !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ──
@st.cache_resource
def get_model():
    return load_model()

model = get_model()


# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
        <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700;
                    background:linear-gradient(135deg,#38bdf8,#818cf8);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            DocSight
        </div>
        <div style="font-size:0.72rem; color:#475569; text-transform:uppercase;
                    letter-spacing:0.1em; margin-top:4px;">
            AI Powered DR Diagnosis
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

    stats = [
        ("0.9986", "AUC-ROC Score"),
        ("98.18%", "Accuracy"),
        ("98.86%", "Sensitivity"),
        ("97.16%", "Specificity"),
    ]
    for val, lbl in stats:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{val}</div>
            <div class="stat-label">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; color:#64748b; line-height:1.7;">
        <b style="color:#94a3b8;">Model:</b> EfficientNet-B4<br>
        <b style="color:#94a3b8;">Training Data:</b> APTOS 2019 + IDRiD<br>
        <b style="color:#94a3b8;">Images Trained:</b> 4,117<br>
        <b style="color:#94a3b8;">Threshold:</b> 0.35 (screening)<br>
        <b style="color:#94a3b8;">Explainability:</b> Grad-CAM
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Instructions</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; color:#64748b; line-height:1.8;">
        1. Enter patient details<br>
        2. Upload fundus image<br>
        3. View AI analysis<br>
        4. Download PDF report
    </div>
    """, unsafe_allow_html=True)


# ── Header ──
st.markdown("""
<div class="header-container">
    <div class="header-icon">👁️</div>
    <div>
        <p class="header-title">DocSight</p>
        <p class="header-sub">AI Powered GUI for Diabetic Retinopathy Diagnosis</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Patient Info ──
st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)
col_a, col_b, col_c = st.columns(3)
with col_a:
    patient_name = st.text_input('Patient Name', placeholder='e.g. Rahul Sharma')
with col_b:
    patient_id = st.text_input('Patient ID', placeholder='e.g. PAT-00123')
with col_c:
    patient_age = st.text_input('Age', placeholder='e.g. 52')


# ── Upload ──
st.markdown('<div class="section-title">Fundus Image Upload</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    'Upload a retinal fundus image (PNG, JPG, JPEG)',
    type=['png', 'jpg', 'jpeg']
)


# ── Analysis ──
if uploaded is not None:

    # Read image via PIL → always RGB, no color mismatch
    pil_img   = Image.open(uploaded).convert('RGB')
    img_array = np.array(pil_img)

    # Progress bar
    st.markdown('<div class="section-title">Analysis</div>', unsafe_allow_html=True)
    progress = st.progress(0, text='Initializing...')

    progress.progress(25, text='Preprocessing image...')
    import time; time.sleep(0.3)

    progress.progress(55, text='Running AI inference...')
    img_rgb, cam, prediction, prob = predict(model, img_array, threshold=0.35)

    progress.progress(80, text='Generating Grad-CAM...')
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.5, heatmap, 0.5, 0)
    time.sleep(0.2)

    progress.progress(100, text='Analysis complete!')
    time.sleep(0.4)
    progress.empty()

    # ── Images ──
    st.markdown('<div class="section-title">Visual Analysis</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(img_rgb, use_column_width=True)
        st.markdown('<div class="img-label">Preprocessed Image</div>', unsafe_allow_html=True)
    with c2:
        st.image(heatmap, use_column_width=True)
        st.markdown('<div class="img-label">Grad-CAM Heatmap · Red = High Attention</div>', unsafe_allow_html=True)
    with c3:
        st.image(overlay, use_column_width=True)
        st.markdown('<div class="img-label">Overlay</div>', unsafe_allow_html=True)

    # ── Result Banner ──
    st.markdown('<div class="section-title">Diagnostic Result</div>', unsafe_allow_html=True)

    if prediction == 'DR':
        st.markdown(f"""
        <div class="result-dr">
            <p class="result-title" style="color:#f87171;">🔴 &nbsp; Diabetic Retinopathy Detected</p>
            <p class="result-sub">DR probability: <b style="color:#fca5a5;">{prob:.1%}</b> &nbsp;·&nbsp;
            Confidence: <b style="color:#fca5a5;">{'High' if abs(prob-0.5)>0.25 else 'Low'}</b> &nbsp;·&nbsp;
            Threshold: 0.35</p>
        </div>
        """, unsafe_allow_html=True)
    elif prob >= 0.2:
        st.markdown(f"""
        <div class="result-borderline">
            <p class="result-title" style="color:#fbbf24;">🟡 &nbsp; Borderline — Manual Review Advised</p>
            <p class="result-sub">DR probability: <b style="color:#fde68a;">{prob:.1%}</b> &nbsp;·&nbsp;
            Further examination recommended</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-nodr">
            <p class="result-title" style="color:#4ade80;">🟢 &nbsp; No Diabetic Retinopathy Detected</p>
            <p class="result-sub">DR probability: <b style="color:#86efac;">{prob:.1%}</b> &nbsp;·&nbsp;
            Confidence: <b style="color:#86efac;">{'High' if abs(prob-0.5)>0.25 else 'Low'}</b> &nbsp;·&nbsp;
            Threshold: 0.35</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Metrics ──
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box">
            <div class="metric-val">{'DR' if prediction == 'DR' else 'No DR'}</div>
            <div class="metric-lbl">Prediction</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{prob:.4f}</div>
            <div class="metric-lbl">DR Probability</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{'High' if abs(prob-0.5) > 0.25 else 'Low'}</div>
            <div class="metric-lbl">Confidence</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">0.35</div>
            <div class="metric-lbl">Threshold Used</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── PDF Report ──
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

            # Header bar
            pdf.set_fill_color(8, 12, 20)
            pdf.rect(0, 0, 210, 30, 'F')
            pdf.set_font('Arial', 'B', 18)
            pdf.set_text_color(56, 189, 248)
            pdf.set_y(9)
            pdf.cell(0, 10, 'DocSight - Diagnostic Report', ln=True, align='C')

            pdf.set_y(32)
            pdf.set_font('Arial', '', 9)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(0, 6, 'AI Powered GUI for Diabetic Retinopathy Diagnosis  |  EfficientNet-B4  |  AUC: 0.9986', ln=True, align='C')
            pdf.ln(4)

            # Patient info
            pdf.set_fill_color(240, 249, 255)
            pdf.rect(10, pdf.get_y(), 190, 28, 'F')
            pdf.set_y(pdf.get_y() + 4)
            pdf.set_font('Arial', 'B', 10)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 6, 'PATIENT INFORMATION', ln=True, align='C')
            pdf.set_font('Arial', '', 10)
            pdf.cell(65, 6, f'Name  : {patient_name if patient_name else "N/A"}', align='C')
            pdf.cell(65, 6, f'ID      : {patient_id if patient_id else "N/A"}', align='C')
            pdf.cell(65, 6, f'Age   : {patient_age if patient_age else "N/A"}', align='C', ln=True)
            pdf.ln(6)

            # Result
            if prediction == 'DR':
                pdf.set_fill_color(254, 226, 226)
                r, g, b = 185, 28, 28
            else:
                pdf.set_fill_color(220, 252, 231)
                r, g, b = 21, 128, 61

            pdf.set_fill_color(*(254,226,226) if prediction=='DR' else (220,252,231))
            pdf.rect(10, pdf.get_y(), 190, 22, 'F')
            pdf.set_y(pdf.get_y() + 4)
            pdf.set_font('Arial', 'B', 13)
            pdf.set_text_color(r, g, b)
            result_text = 'DIABETIC RETINOPATHY DETECTED' if prediction == 'DR' else 'NO DIABETIC RETINOPATHY DETECTED'
            pdf.cell(0, 8, result_text, ln=True, align='C')
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 6, f'DR Probability: {prob:.4f} ({prob:.1%})   |   Confidence: {"High" if abs(prob-0.5)>0.25 else "Low"}   |   Threshold: 0.35', ln=True, align='C')
            pdf.ln(6)

            # Images
            pdf.set_text_color(30, 30, 30)
            pdf.set_font('Arial', 'B', 11)
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
            pdf.ln(10)

            # Disclaimer
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

        fname = f"DR_Report_{patient_id if patient_id else 'patient'}.pdf"
        st.download_button(
            label     = '⬇️ Download PDF Report',
            data      = pdf_bytes,
            file_name = fname,
            mime      = 'application/pdf'
        )

# ── Footer ──
st.markdown("""
<div style="text-align:center; padding: 3rem 0 1rem 0;">
    <div style="font-size:0.75rem; color:#1e293b;">
        DocSight &nbsp;·&nbsp; EfficientNet-B4 &nbsp;·&nbsp;
        Trained on APTOS 2019 + IDRiD &nbsp;·&nbsp; AUC 0.9986
    </div>
</div>
""", unsafe_allow_html=True)