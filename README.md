# Static-to-Smart-
# ⚙ Industrial P&ID Digital Twin Dashboard  

A **real-time digital twin dashboard** that extracts equipment tags from P&ID diagrams using **OCR + YOLOv8 + OpenCV**, simulates/streams sensor values, and visualizes live system status in an interactive Streamlit app.  

---

## 🚀 Features
- 📂 Upload **P&ID images** for processing  
- 🔍 **YOLOv8 + OpenCV + Tesseract OCR** for symbol & text detection  
- 🗂 Match OCR results with equipment metadata (CSV)  
- 📡 **IoT Sensor Simulation** (flow, pressure, temperature) or real IoT data (Azure IoT Hub / MQTT)  
- 🤖 **Anomaly Detection**: threshold rules or ML-based classification  
- 📊 **Live Monitoring Dashboard**: metrics, alerts, tables, and network graph  
- 📥 Export OCR + sensor data as CSV  

---

## 📂 Project Structure
│── 📂 data/ # Sample data & equipment metadata
│ ├── equipment_data.csv
│ └── sample_pid.png
│
│── 📂 models/ # Pretrained / trained ML models
│ └── yolov8_pid.pt
│
│── 📂 src/ # Core source code
│ ├── app.py # Main Streamlit dashboard
│ ├── ocr_utils.py # OpenCV + Tesseract OCR functions
│ ├── sensor_simulation.py# IoT data simulation / streaming
│ ├── anomaly_detection.py# Threshold & ML anomaly detection
│ └── graph_builder.py # Build & visualize equipment graph
│
│── 📂 notebooks/ # Jupyter notebooks
│ 
│
│── requirements.txt # Python dependencies
│── README.md # Project documentation
└── .gitignore # Ignore cache, venv, logs

🔧 Tech Stack

Frontend/UI → Streamlit, Matplotlib, NetworkX

OCR & CV → OpenCV, Tesseract OCR, YOLOv8 (Ultralytics)

Data → Pandas, difflib

Simulation → Random, Time (or IoT Hub/MQTT)

ML (optional) → scikit-learn, PyTorch

🌐 Future Enhancements

✅ Real IoT integration (Azure IoT Hub / MQTT)

✅ Advanced anomaly detection (LSTM / Autoencoder)

✅ Edge-based graph connections from actual P&ID lines

✅ Cloud deployment (AWS/GCP/Azure)
