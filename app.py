import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
from fpdf import FPDF
import tempfile
import os
from model import load_model, predict

# ── Page config ──
st.set_page_config(
    page_title = 'DR Detection',
    page_icon  = '👁️',
    layout     = 'wide'
)

# ── Load model once (cached) ──
@st.cache_resource
def get_model():
    with st.spinner('Loading AI model... (first time only, ~30 seconds)'):
        model = load_model()
    return model

model = get_model()

# ── Title ──
st.title('👁️ Diabetic Retinopathy Detection')
st.markdown('Upload a **fundus retinal image** to detect Diabetic Retinopathy.')
st.markdown('---')

# ── Upload ──
uploaded = st.file_uploader(
    'Upload Fundus Image',
    type=['png', 'jpg', 'jpeg'],
    help='Upload a fundus camera image'
)

if uploaded is not None:
    # Read image — always read as RGB via PIL to avoid color issues
    pil_img   = Image.open(uploaded).convert('RGB')
    img_array = np.array(pil_img)

    st.markdown('### Analysis Results')

    with st.spinner('Running DR analysis...'):
        img_rgb, cam, prediction, prob = predict(model, img_array, threshold=0.35)

    # ── Build heatmap and overlay ──
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.5, heatmap, 0.5, 0)

    # ── Show images ──
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_rgb, caption='Preprocessed Image', use_column_width=True)
    with col2:
        st.image(heatmap, caption='Grad-CAM (Red = High Attention)', use_column_width=True)
    with col3:
        st.image(overlay, caption='Overlay', use_column_width=True)

    # ── Result box ──
    st.markdown('---')
    if prediction == 'DR':
        st.error(f'🔴 DR DETECTED  |  Probability: {prob:.1%}  |  High Confidence')
    elif prob >= 0.2:
        st.warning(f'🟡 BORDERLINE — MANUAL REVIEW ADVISED  |  Probability: {prob:.1%}')
    else:
        st.success(f'🟢 No DR Detected  |  Probability: {prob:.1%}  |  High Confidence')

    # ── Metrics ──
    st.markdown('### Detailed Metrics')
    m1, m2, m3 = st.columns(3)
    m1.metric('Prediction',    prediction)
    m2.metric('DR Probability', f'{prob:.4f}')
    m3.metric('Confidence',    'High' if abs(prob - 0.5) > 0.25 else 'Low')

    # ── PDF Report ──
    st.markdown('---')
    st.markdown('### Download Report')

    if st.button('Generate PDF Report'):
        with st.spinner('Generating PDF...'):

            # Save images temporarily
            tmp_dir     = tempfile.mkdtemp()
            orig_path   = os.path.join(tmp_dir, 'original.png')
            heat_path   = os.path.join(tmp_dir, 'heatmap.png')
            overlay_path = os.path.join(tmp_dir, 'overlay.png')

            Image.fromarray(img_rgb).save(orig_path)
            Image.fromarray(heatmap).save(heat_path)
            Image.fromarray(overlay).save(overlay_path)

            # Build PDF
            pdf = FPDF()
            pdf.add_page()

            # Header
            pdf.set_font('Arial', 'B', 20)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 12, 'Diabetic Retinopathy Detection Report', ln=True, align='C')
            pdf.set_font('Arial', '', 11)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 8, 'AI-Assisted Fundus Image Analysis', ln=True, align='C')
            pdf.ln(5)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)

            # Result
            pdf.set_font('Arial', 'B', 14)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 10, 'RESULT', ln=True)
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 8, f'Prediction      : {prediction}', ln=True)
            pdf.cell(0, 8, f'DR Probability  : {prob:.4f} ({prob:.1%})', ln=True)
            pdf.cell(0, 8, f'Threshold Used  : 0.35 (recommended for screening)', ln=True)
            pdf.cell(0, 8, f'Confidence      : {"High" if abs(prob-0.5) > 0.25 else "Low — Manual Review Advised"}', ln=True)
            pdf.ln(5)

            # Images
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'IMAGE ANALYSIS', ln=True)
            pdf.set_font('Arial', '', 10)

            img_y = pdf.get_y()
            pdf.image(orig_path,    x=10,  y=img_y, w=58)
            pdf.image(heat_path,    x=76,  y=img_y, w=58)
            pdf.image(overlay_path, x=142, y=img_y, w=58)

            pdf.set_y(img_y + 62)
            pdf.cell(58, 6, 'Preprocessed Image', align='C')
            pdf.cell(58, 6, 'Grad-CAM Heatmap',   align='C')
            pdf.cell(58, 6, 'Overlay',             align='C')
            pdf.ln(10)

            # Disclaimer
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)
            pdf.set_font('Arial', 'I', 9)
            pdf.set_text_color(120, 120, 120)
            pdf.multi_cell(0, 6,
                'Disclaimer: This report has been generated by an AI-assisted diagnostic tool '
                'for clinical reference only. The findings presented are based on automated '
                'image analysis and must be reviewed and verified by a qualified medical '
                'professional before any clinical decision is made. AI results may not '
                'account for all clinical factors. Final diagnosis remains the sole '
                'responsibility of the treating physician.'
            )
            # Save PDF
            pdf_path = os.path.join(tmp_dir, 'dr_report.pdf')
            pdf.output(pdf_path)

            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()

        st.download_button(
            label     = '📄 Download PDF Report',
            data      = pdf_bytes,
            file_name = f'DR_Report_{uploaded.name}.pdf',
            mime      = 'application/pdf'
        )
        st.success('PDF ready! Click above to download.')

# ── Footer ──
st.markdown('---')
st.markdown(
    '<p style="text-align:center; color:grey; font-size:12px;">'
    'Model: EfficientNet-B4 | Trained on APTOS 2019 + IDRiD | AUC: 0.9986'
    '</p>',
    unsafe_allow_html=True
)