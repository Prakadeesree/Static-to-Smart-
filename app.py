import streamlit as st
import pytesseract
from PIL import Image
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
import difflib
import time


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


csv_path = "data/equipment_data.csv"
if os.path.exists(csv_path):
    df_csv = pd.read_csv(csv_path)
else:
    df_csv = pd.DataFrame(columns=["Tag", "Description", "Type"])


st.markdown("""
    <style>
        .main {
            background-color: #0d1117;
            color: white;
        }
        .stMetric {
            background: #1e1e1e;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .stAlert {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)



st.title("Static To Smart Digital P&Id")
st.markdown("### 🔍 Live Sensor Simulation")

uploaded_files = st.file_uploader(
    "📂 Upload P&ID Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

with st.sidebar:
    st.header("⚡ Settings")
    update_interval = st.slider("⏱ Update Interval (seconds)", 1, 10, 3)
    warning_threshold = st.slider("⚠ Warning Threshold", 0, 100, 60)
    critical_threshold = st.slider("❌ Critical Threshold", 0, 100, 80)

all_matches = pd.DataFrame()


if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        st.subheader(f"🖼 Processing: {uploaded_file.name}")

        img = Image.open(uploaded_file)
        st.image(img, caption=uploaded_file.name, use_column_width=True)

        
        img_cv = np.array(img)
        if len(img_cv.shape) == 2:
            gray = img_cv
        elif img_cv.shape[2] == 3:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        elif img_cv.shape[2] == 4:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2GRAY)
        else:
            st.error("Unsupported image format!")
            continue

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        
        ocr_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DATAFRAME)
        ocr_data = ocr_data[ocr_data.text.notna() & (ocr_data.text.str.strip() != "")]

        detected_tags = []
        tags_list = df_csv["Tag"].tolist()

        for w in ocr_data["text"]:
            matches = difflib.get_close_matches(w, tags_list, n=1, cutoff=0.8)
            if matches:
                detected_tags.append(matches[0])

        if detected_tags:
            tag_counts = {tag: detected_tags.count(tag) for tag in detected_tags}
            df_matches = df_csv[df_csv["Tag"].isin(detected_tags)].copy()
            df_matches["Connections"] = df_matches["Tag"].apply(lambda t: tag_counts[t])
            all_matches = pd.concat([all_matches, df_matches], ignore_index=True)
        else:
            st.info(f"ℹ No tags found in {uploaded_file.name}")


if not all_matches.empty:
    csv_data = all_matches.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download OCR + Sensor Data",
        data=csv_data,
        file_name="pid_ocr_data.csv",
        mime="text/csv",
    )

    st.markdown("### 📡 Live Monitoring Dashboard")


    placeholder_table = st.empty()
    placeholder_alert = st.empty()
    placeholder_graph = st.empty()

    run_live = st.toggle("▶ Start Live Sensor Updates")

    if run_live:
        for _ in range(1000):  
            all_matches["SensorValue"] = all_matches["Tag"].apply(lambda t: round(random.uniform(20, 100), 2))
            all_matches["Status"] = all_matches["SensorValue"].apply(
                lambda v: "Normal" if v < warning_threshold else ("Warning" if v < critical_threshold else "Critical")
            )

            
            col1, col2, col3 = st.columns(3)
            col1.metric("✅ Normal", str((all_matches["Status"]=="Normal").sum()))
            col2.metric("⚠ Warnings", str((all_matches["Status"]=="Warning").sum()))
            col3.metric("❌ Critical", str((all_matches["Status"]=="Critical").sum()))

            
            alerts = []
            for _, row in all_matches.iterrows():
                if row["Status"] == "Warning":
                    alerts.append(f"⚠ {row['Tag']}: {row['SensorValue']}")
                elif row["Status"] == "Critical":
                    alerts.append(f"❌ {row['Tag']}: {row['SensorValue']}")

            if alerts:
                placeholder_alert.warning("\n".join(alerts))
            else:
                placeholder_alert.success("All systems normal ✅")

            
            placeholder_table.dataframe(all_matches, use_container_width=True)

            
            G = nx.Graph()
            for _, row in all_matches.iterrows():
                node_label = f"{row['Tag']}\n{row['SensorValue']}"
                node_color = {"Normal": "green", "Warning": "orange", "Critical": "red"}[row["Status"]]
                G.add_node(node_label, color=node_color, connections=row["Connections"])

            labels_list = list(G.nodes)
            for i, node1 in enumerate(labels_list):
                for j, node2 in enumerate(labels_list):
                    if i < j:
                        G.add_edge(node1, node2)

            plt.figure(figsize=(6, 6))
            pos = nx.spring_layout(G)
            colors = [G.nodes[n]["color"] for n in G.nodes()]
            nx.draw(G, pos, with_labels=True, node_color=colors, node_size=1800, font_size=9, font_weight='bold')
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels={(u, v): G.nodes[u]["connections"] + G.nodes[v]["connections"] for u, v in G.edges()},
            )
            placeholder_graph.pyplot(plt)
            plt.clf()

            time.sleep(update_interval)
