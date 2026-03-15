\# 👁️ DocSight — AI Powered GUI for Diabetic Retinopathy Diagnosis



!\[Python](https://img.shields.io/badge/Python-3.10-blue)

!\[PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)

!\[Streamlit](https://img.shields.io/badge/Streamlit-Live-green)

!\[AUC](https://img.shields.io/badge/AUC--ROC-0.9986-brightgreen)



\## 🔗 Live Demo

\*\*\[Launch DocSight App](https://dr-detection-app-nw6x37aycsamjydhzp9xiq.streamlit.app/)\*\*



\---



\## 📌 About

DocSight is an AI-powered clinical tool for automated detection of \*\*Diabetic Retinopathy (DR)\*\* from retinal fundus images. It is designed for use by medical professionals as a screening aid.



The system analyzes fundus camera images and provides:

\- Binary classification — \*\*DR Detected / No DR\*\*

\- \*\*Grad-CAM heatmap\*\* showing which retinal regions the model focused on

\- \*\*Downloadable PDF report\*\* with patient details and findings



\---



\## 🧠 Model

| Property | Details |

|---|---|

| Architecture | EfficientNet-B4 |

| Training Data | APTOS 2019 + IDRiD (4,117 images) |

| Image Size | 300 × 300 px |

| AUC-ROC | 0.9986 |

| Accuracy | 98.18% |

| Sensitivity | 98.86% |

| Specificity | 97.16% |

| Threshold | 0.35 (optimized for screening) |



\---



\## 🔬 Preprocessing Pipeline

Every image goes through the same pipeline regardless of source camera:

1\. \*\*Black border cropping\*\* — removes dark edges

2\. \*\*CLAHE\*\* — enhances local contrast, improves lesion visibility

3\. \*\*Ben Graham Normalization\*\* — eliminates lighting/brightness bias (prevents shortcut learning)



\---



\## 📊 Results

\- Only \*\*6 missed DR cases\*\* out of 437 DR images

\- Only \*\*9 false alarms\*\* out of 387 normal images

\- Validated on \*\*Indian population dataset (IDRiD)\*\* — matches hospital demographics

\- Tested on real hospital fundus images from \*\*Desai Eye Hospital\*\*



\---



\## 🏗️ Project Structure

```

dr-detection-app/

&#x20;   ├── app.py              # Streamlit web application

&#x20;   ├── model.py            # Model definition + preprocessing + Grad-CAM

&#x20;   └── requirements.txt    # Python dependencies

```



\---



\## 🚀 Run Locally

```bash

git clone https://github.com/SaloniD8/dr-detection-app.git

cd dr-detection-app

pip install -r requirements.txt

streamlit run app.py

```



\---



\## 📦 Model Weights

Model hosted on HuggingFace:

\*\*\[Salonideshmukh/dr-detection-model](https://huggingface.co/Salonideshmukh/dr-detection-model)\*\*



Downloaded automatically when app starts — no manual setup needed.



\---



\## 🗂️ Datasets Used

| Dataset | Purpose | Link |

|---|---|---|

| APTOS 2019 | Classification training | \[Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection) |

| IDRiD | Indian population data | \[Kaggle](https://www.kaggle.com/datasets/mariaherrerot/idrid-dataset) |



\---



\## ⚙️ Tech Stack

\- \*\*Model:\*\* PyTorch + Timm (EfficientNet-B4)

\- \*\*Explainability:\*\* Grad-CAM

\- \*\*App:\*\* Streamlit

\- \*\*Model Hosting:\*\* HuggingFace Hub

\- \*\*Deployment:\*\* Streamlit Cloud



\---



\## ⚠️ Disclaimer

This tool is intended for use by qualified medical professionals as a clinical screening aid only. AI findings must be reviewed and verified by a physician before any clinical decision is made. Final diagnosis remains the sole responsibility of the treating physician.



\---



\## 👩‍💻 Author

\*\*Saloni Deshmukh\*\*

\[GitHub](https://github.com/SaloniD8) · \[HuggingFace](https://huggingface.co/Salonideshmukh)

