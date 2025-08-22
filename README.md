# Static-to-Smart-
# ⚙ Industrial P&ID Digital Twin Dashboard  

A "real-time digital twin dashboard" that extracts equipment tags from P&ID diagrams using **OCR + YOLOv8 + OpenCV**, simulates/streams sensor values, and visualizes live system status in an interactive Streamlit app.  

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

│── 📜 README.md             # Project documentation  
│── 📜 requirements.txt      # Python dependencies  

│── 📂 src/                  # Core application code  
│   ├── app.py               # Main Streamlit dashboard  
│   ├── detection.py         # Detection logic (YOLOv8 + OpenCV + OCR)  
│   └── pid_ocr.py           # OCR utilities & preprocessing  

│── 📂 models/               # Pretrained / trained ML models  
│   └── yolov8_pid.pt        # (example YOLOv8 model file)  

│── 📂 data/                 # Input data & samples  
│   ├── sample_data.csv      # Equipment metadata  
│   └── sample_pid.png       # Sample P&ID image  

│── 📂 results/              # Output / processed results  
│   └── output.csv           # OCR + sensor results  

---
## 🔧 Tech Stack

Frontend/UI → Streamlit, Matplotlib, NetworkX

OCR & CV → OpenCV, Tesseract OCR, YOLOv8 (Ultralytics)

Data → Pandas, difflib

Simulation → Random, Time 

---
## 🌐 Future Enhancements

✅ Real IoT integration with database (Azure IoT Hub / MQTT)

✅ Advanced anomaly detection (LSTM / Autoencoder)

✅ Edge-based graph connections from actual P&ID lines


