import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import json
import tensorflow as tf
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.platypus import KeepTogether

# --------- CONFIG ----------
IMG_SIZE = (256, 256)
MODEL_PATH = "tomato_disease_model.h5"
# --------------------------

# --------- LOAD CLASS NAMES ----------
with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

def format_class_name(raw_name: str) -> str:
    """
    Convert dataset class name to user-friendly disease name
    Example:
    Tomato___Tomato_Yellow_Leaf_Curl_Virus
    -> Tomato Yellow Leaf Curl Virus
    """
    name = raw_name.replace("Tomato___", "")
    name = name.replace("_", " ")
    name = name.replace("  ", " ")
    if name == "Spider mites Two-spotted spider mite":
        name = "Two-spotted spider mites"
    return name.strip()
# --------- EXTRA INFO ----------
DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "description": "Bacterial spot causes dark, water-soaked lesions on tomato leaves and fruits.",
        "advice": "Avoid overhead irrigation, remove infected plants, and use disease-free seeds."
    },
    "Tomato___Early_blight": {
        "description": "Early blight causes concentric brown rings on older leaves and may reduce yield.",
        "advice": "Remove infected leaves and apply appropriate fungicides if necessary."
    },
    "Tomato___Late_blight": {
        "description": "Late blight is a serious disease that spreads rapidly in cool, moist conditions.",
        "advice": "Destroy infected plants immediately and avoid planting near infected fields."
    },
    "Tomato___Leaf_Mold": {
        "description": "Leaf mold appears as yellow spots on the upper leaf surface with mold underneath.",
        "advice": "Improve air circulation and reduce humidity inside greenhouses."
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "Septoria leaf spot causes small circular spots with dark borders on leaves.",
        "advice": "Remove infected leaves and avoid splashing water onto foliage."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Spider mites cause stippling and yellowing of leaves due to sap sucking.",
        "advice": "Increase humidity and use miticides or biological control if infestation is severe."
    },
    "Tomato___Target_Spot": {
        "description": "Target spot causes brown lesions with concentric rings on leaves and stems.",
        "advice": "Remove plant debris and apply fungicides when symptoms appear."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "This virus causes yellowing, curling of leaves, and stunted plant growth.",
        "advice": "Control whiteflies and remove infected plants immediately."
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "Mosaic virus causes mottled leaf coloration and distorted growth.",
        "advice": "Disinfect tools and avoid handling plants when wet."
    },
    "Tomato___healthy": {
        "description": "The tomato leaf appears healthy with no visible disease symptoms.",
        "advice": "Continue regular monitoring and good farming practices."
    }
}

# --------- MODEL LOADING ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# --------- PREPROCESS ----------
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.resize(IMG_SIZE)
    img_array = np.array(img).astype("float32") / 255.0

    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model, img: Image.Image):
    processed = preprocess_image(img)
    preds = model.predict(processed, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx]), preds

# --------- FOOTER ----------
def draw_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Oblique", 9)

    footer_lines = [
        "© 2026 All rights reserved",
        "AI-Based Tomato Pest Removal System",
        "PG-Bot – Breakin Point"
    ]


    page_width, _ = A4
    y = 30
    for line in footer_lines:
        text_width = canvas.stringWidth(line, "Helvetica-Oblique", 9)
        x = (page_width - text_width) / 2
        canvas.drawString(x, y, line)
        y += 11

    canvas.restoreState()

# --------- PDF ----------

def create_pdf_report(
    predicted_raw_class: str,
    confidence: float,
    probabilities: dict,
    image: Image.Image
):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=60,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()
    elements = []

    # -------- TITLE --------
    elements.append(
        Paragraph("<b>PG-Bot – AI Vision Analysis Report</b>", styles["Title"])
    )
    elements.append(
        Paragraph(
            "Simulation report for the PG-Bot autonomous crop-scanning robot.",
            styles["Italic"]
        )
    )
    elements.append(Spacer(1, 8))

    # -------- TIMESTAMP --------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(
        Paragraph(f"Report generated at: {timestamp}", styles["Normal"])
    )
    elements.append(Spacer(1, 12))

    # -------- PREDICTION SUMMARY --------
    clean_name = format_class_name(predicted_raw_class)

    elements.append(
        Paragraph(f"Detected Disease: <b>{clean_name}</b>", styles["Normal"])
    )
    elements.append(
        Paragraph(f"Confidence: <b>{confidence * 100:.2f}%</b>", styles["Normal"])
    )
    elements.append(Spacer(1, 14))

    # -------- IMAGE --------
    if image is not None:
        img_buffer = BytesIO()
        image.convert("RGB").save(img_buffer, format="PNG")
        img_buffer.seek(0)

        rl_img = RLImage(img_buffer, width=220, height=220 * image.height / image.width)
        elements.append(Paragraph("Uploaded Leaf Image:", styles["Heading3"]))
        elements.append(rl_img)
        elements.append(Spacer(1, 14))

    # -------- PROBABILITY TABLE --------
    table_data = [["Disease", "Probability (%)"]]
    for cls, p in probabilities.items():
        table_data.append([cls, f"{p * 100:.2f}"])

    table = Table(table_data, colWidths=[260, 120])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
    ]))

    table_block = KeepTogether([
        Paragraph("Class Probabilities:", styles["Heading3"]),
        Spacer(1, 6),
        table,
        Spacer(1, 14),
    ])

    elements.append(table_block)

    # -------- BAR CHART --------
    max_prob = max(probabilities.values())
    bar_width = 260
    row_height = 16
    chart_height = row_height * len(probabilities) + 10

    drawing = Drawing(bar_width + 160, chart_height)
    y = chart_height - 15

    for cls, p in probabilities.items():
        bar_len = (p / max_prob) * bar_width if max_prob > 0 else 0

        drawing.add(String(0, y, cls, fontSize=8))
        drawing.add(Rect(110, y - 4, bar_len, 10,
                         fillColor=colors.green,
                         strokeColor=colors.green))
        drawing.add(String(
            110 + bar_width + 6,
            y - 1,
            f"{p * 100:.1f}%",
            fontSize=8
        ))

        y -= row_height

    probability_block = KeepTogether([
        Paragraph("Probability Distribution:", styles["Heading3"]),
        Spacer(1, 6),
        drawing,
        Spacer(1, 14),
    ])

    elements.append(probability_block)


    # -------- DISEASE INFO --------
    info = DISEASE_INFO.get(predicted_raw_class)
    elements.append(
        Paragraph("Disease Description & Suggested Action:", styles["Heading3"])
    )

    if info:
        elements.append(Paragraph(info["description"], styles["Normal"]))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(info["advice"], styles["Normal"]))
    else:
        elements.append(
            Paragraph("No additional information available.", styles["Normal"])
        )

    # -------- BUILD PDF --------
    doc.build(elements, onFirstPage=draw_footer, onLaterPages=draw_footer)
    buffer.seek(0)
    return buffer

# --------- MAIN ----------
def main():
    st.set_page_config(
        page_title="PG-Bot | AI Crop Scanning System",
        page_icon="🤖",
        layout="centered"
    )

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.title("🍅 PG-Bot")
        st.markdown(
            "**PG-Bot** is an AI-powered agricultural robot designed to operate in tomato fields."
        )

        st.markdown("### 🧠 Vision Module")
        st.markdown(
            "- Captures images from onboard camera\n"
            "- Classifies crop health & diseases\n"
            "- Sends signals to robotic arm for targeted cutting"
        )

        st.markdown("### 🧪 This Demo")
        st.markdown(
            "This application represents a **simulation of the vision system**, "
            "allowing testing and visualization of the AI model before full hardware deployment."
        )

        st.markdown("### ⚙️ Model Info")
        st.markdown(
            "- Input size: 256×256\n"
            "- Model: DenseNet (Transfer Learning)\n"
            "- Classes: Tomato diseases + Healthy"
        )

    # ---------- TABS ----------
    tab_detector, tab_about = st.tabs(["🔍 Detector", "ℹ️ About Project"])

    # ====== DETECTOR TAB ======
    with tab_detector:
        st.title("🍅 PG-Bot – AI Vision Module (Simulation)")
        st.write(
            "This interface simulates the **computer vision system of PG-Bot**, "
            "an autonomous robot designed to scan tomato crops, detect diseases and pests, "
            "and assist in their physical removal."
        )

        with st.spinner("Loading AI model..."):
            model = load_model()

        col1, col2 = st.columns(2)

        with col1:
            uploaded_file = st.file_uploader(
                "📁 Upload image",
                type=["jpg", "jpeg", "png"]
            )

        with col2:
            camera_image = st.camera_input("📷 Take a photo")

        img = None
        if uploaded_file:
            img = Image.open(uploaded_file)
        elif camera_image:
            img = Image.open(camera_image)

        if img is not None:
            st.image(img, caption="Input Image", use_container_width=True)

            if st.button("🔍 Predict"):
                with st.spinner("Analyzing the leaf..."):
                    predicted_class, confidence, all_probs = predict_image(model, img)

                st.subheader("Detected Disease")
                clean_name = format_class_name(predicted_class)
                st.write(f"**Disease:** {clean_name}")

                st.subheader("Confidence")
                st.write(f"{confidence * 100:.2f}%")
                st.progress(int(confidence * 100))

                if confidence < 0.6:
                    st.warning(
                        "Low confidence prediction. Consider using a clearer image."
                    )

                # Disease info
                st.subheader("Disease Information & Advice")
                info = DISEASE_INFO.get(predicted_class)
                if info:
                    st.markdown(f"**Description:** {info['description']}")
                    st.markdown(f"**Suggested Action:** {info['advice']}")

                # Probabilities
                st.subheader("Detection Confidence Distribution")
                prob_dict = {
                    format_class_name(cls): float(p)
                    for cls, p in zip(CLASS_NAMES, all_probs)
                }

                df = pd.DataFrame(
                    {"Class": prob_dict.keys(), "Probability": prob_dict.values()}
                ).set_index("Class")

                st.dataframe(df.style.format({"Probability": "{:.2%}"}))
                st.bar_chart(df)

                # PDF Report
                pdf_buffer = create_pdf_report(
                    predicted_class,
                    confidence,
                    prob_dict,   # clean-name probabilities
                    img
                )

                st.download_button(
                    label="📄 Download Prediction Report (PDF)",
                    data=pdf_buffer,
                    file_name="tomato_leaf_prediction_report.pdf",
                    mime="application/pdf"
                )

                st.success("You can try another image.")

        else:
            st.info("Upload an image or use the camera to get started.")
        
        st.markdown("---")
        st.caption("Breakin Point | PG-Bot",text_alignment="center")

    # ====== ABOUT TAB ======
    with tab_about:
        st.title("ℹ️ About PG-Bot")

        st.markdown("### 🤖 Project Overview")
        st.write(
            "**PG-Bot** is an AI-powered agricultural robotics project designed to assist in "
            "**automated tomato crop monitoring and pest control**. The system combines "
            "computer vision, deep learning, and robotics to detect unhealthy or infected "
            "plants and support targeted physical intervention in the field."
        )

        st.markdown("### 🌱 Problem Statement")
        st.write(
            "Tomato crops are highly vulnerable to a wide range of diseases and pests that can "
            "spread rapidly across fields, reducing yield and increasing economic losses. "
            "Traditional inspection methods rely on manual observation, which is time-consuming, "
            "labor-intensive, and often impractical for large-scale farms."
        )

        st.markdown("### 🎯 Project Objective")
        st.write(
            "PG-Bot aims to provide an **autonomous solution** that continuously scans tomato crops "
            "using an onboard camera, detects diseases or pest-related symptoms early, and enables "
            "precise, localized removal of infected plant parts. This approach helps reduce "
            "chemical usage, minimize crop damage, and improve overall farm efficiency."
        )

        st.markdown("### 🧠 AI Vision Module (This Application)")
        st.write(
            "This Streamlit application represents a **simulation of PG-Bot’s vision module**. "
            "It demonstrates how the AI model analyzes camera images captured from the field, "
            "classifies the detected condition of tomato leaves, and produces confidence scores "
            "that guide robotic decision-making."
        )

        st.markdown("### 🧪 Dataset & Disease Classes")
        st.write(
            "The AI model was trained on a labeled tomato leaf image dataset containing multiple "
            "disease categories as well as healthy samples. The classes include common tomato "
            "diseases such as bacterial, fungal, viral infections, and pest-related damage."
        )

        st.markdown("### 🧠 Model & Technical Approach")
        st.write(
            "- **Architecture:** DenseNet-based convolutional neural network (transfer learning)\n"
            "- **Input:** RGB images resized to 256×256 pixels\n"
            "- **Output:** Disease class probabilities\n"
            "- **Training Strategy:** Feature extraction with frozen base model layers\n"
            "- **Inference Role:** Provide reliable perception data for robotic control logic"
        )

        st.markdown("### ⚙️ System Integration Concept")
        st.write(
            "In the complete PG-Bot system, the AI vision module operates as part of a larger "
            "robotic pipeline:\n\n"
            "1. Camera captures crop images in real time.\n"
            "2. AI model classifies plant condition.\n"
            "3. Decision logic determines required action (monitor, cut, or ignore).\n"
            "4. Robotic mechanism removes infected plant parts and collects waste for disposal."
        )

        st.markdown("### ⚠️ Limitations")
        st.write(
            "- Model performance depends on image quality and lighting conditions.\n"
            "- Early-stage or visually subtle infections may be harder to detect.\n"
            "- This system is intended to **assist**, not replace, expert agricultural judgment."
        )

        st.markdown("### 🚀 Future Work")
        st.write(
            "- Integration with real-time video streams from the robot camera.\n"
            "- Addition of severity estimation and action confidence.\n"
            "- Expansion to support other crops and disease types.\n"
            "- Full hardware deployment and field testing."
        )

        st.markdown("---")
        st.markdown("## 👥 Our Team")

        st.write(
            "PG-Bot is developed by a multidisciplinary team combining skills in "
            "artificial intelligence, software engineering, and robotics, working together "
            "to build an autonomous agricultural solution."
        )

        # -------- TEAM DATA (EDIT ONLY THIS) --------
        team_members = [
            {"name": "Alaa Hassan", "role": "Team Leader", "img": "Team/Alaa.jpg"},
            {"name": "Yousef Gad", "role": "Technical Leader", "img": "Team/Yousef.jpg"},
            {"name": "Seif Elboghdady", "role": "Finance Leader", "img": "Team/Seif.jpg"},
            {"name": "Aya Seraj", "role": "R&D Leader", "img": "Team/Aya.jpg"},
            {"name": "Mohamed Amr", "role": "Mechanical Design Leader", "img": "Team/Mohamed.jpg"},
            {"name": "Asmaa Elfagal", "role": "Marketing Leader", "img": "Team/Asmaa.jpg"},
            {"name": "Mahmoud Mira", "role": "AI Leader", "img": "Team/Mahmoud.jpg"},
        ]
        # --------------------------------------------

        # -------- DISPLAY GRID --------
        rows = [team_members[:4], team_members[4:]]

        for i, row in enumerate(rows):
            cols = st.columns(len(row))
            for col, member in zip(cols, row):
                with col:
                    st.image(member["img"], use_container_width=True)
                    st.markdown(
                        f"""
                        <div style="text-align:center; margin-top:8px;">
                            <strong>{member['name']}</strong><br/>
                            <span style="color: gray; font-size: 0.9em;">
                                {member['role']}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # 👉 add space AFTER the first row only
            if i == 0:
                st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

            # Space between the two rows
            if i == 0:
                st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)


        st.markdown("---")
        st.caption("Breakin Point | PG-Bot",text_alignment="center")

if __name__ == "__main__":
    main()
