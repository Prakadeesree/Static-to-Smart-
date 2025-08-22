from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


base = Path("pandid_demo")
(base / "labels" / "train").mkdir(parents=True, exist_ok=True)
(base / "images" / "train").mkdir(parents=True, exist_ok=True)


dexpi_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Dexpi version="0.1">
  <!-- Simplified demo for P&ID: Tank -> Valve -> Pump -> Reactor -->
  <Plant id="PLANT-001" name="Demo Plant">
    <EquipmentList>
      <Equipment id="T-101" type="Tank"><Tag>T-101</Tag></Equipment>
      <Equipment id="V-101" type="ControlValve"><Tag>LV-101</Tag></Equipment>
      <Equipment id="P-201" type="CentrifugalPump"><Tag>P-201</Tag></Equipment>
      <Equipment id="R-301" type="Reactor"><Tag>R-301</Tag></Equipment>

      <Instrument id="LT-101" type="LevelTransmitter"><Tag>LT-101</Tag></Instrument>
      <Instrument id="PT-201" type="PressureTransmitter"><Tag>PT-201</Tag></Instrument>
    </EquipmentList>

    <Connectivity>
      <Pipe id="L-1001">
        <From equipment="T-101"/>
        <To equipment="V-101"/>
      </Pipe>
      <Pipe id="L-1002">
        <From equipment="V-101"/>
        <To equipment="P-201"/>
      </Pipe>
      <Pipe id="L-1003">
        <From equipment="P-201"/>
        <To equipment="R-301"/>
      </Pipe>

      <!-- Signal links -->
      <Signal id="S-1">
        <From instrument="LT-101"/>
        <To equipment="V-101"/>
      </Signal>
      <Signal id="S-2">
        <From instrument="PT-201"/>
        <To equipment="P-201"/>
      </Signal>
    </Connectivity>
  </Plant>
</Dexpi>
"""
(base / "dexpi_demo.xml").write_text(dexpi_xml)


data_yaml = """# P&ID YOLO dataset config
path: .
train: images/train
val: images/train

names:
  0: tank
  1: pump
  2: control_valve
  3: instrument
  4: pipe_connector
  5: text_tag
"""
(base / "pandid.yaml").write_text(data_yaml)

classes_txt = "\n".join([
    "tank",
    "pump",
    "control_valve",
    "instrument",
    "pipe_connector",
    "text_tag"
])
(base / "classes.txt").write_text(classes_txt)


sample_label = """0 0.20 0.30 0.10 0.12   # tank T-101
2 0.35 0.30 0.07 0.08   # control valve V-101
1 0.50 0.30 0.10 0.10   # pump P-201
3 0.20 0.18 0.05 0.05   # instrument LT-101
3 0.50 0.18 0.05 0.05   # instrument PT-201
5 0.20 0.42 0.12 0.04   # text tag 'T-101'
"""
(base / "labels" / "train" / "sample_image.txt").write_text(sample_label)


(base / "images" / "train" / "sample_image.jpg").touch()


readme = """P&ID Demo: DEXPI + YOLO

Files:
- dexpi_demo.xml        -> Simplified DEXPI-style XML (machine-readable P&ID)
- pandid.yaml           -> YOLO dataset config with class names
- classes.txt           -> Class list (one per line)
- labels/train/sample_image.txt -> Example YOLO labels for sample_image.jpg

YOLO label format:
<class_id> <x_center> <y_center> <width> <height>
(all values normalized 0..1 relative to image size)

Classes:
0 tank
1 pump
2 control_valve
3 instrument
4 pipe_connector
5 text_tag

Training example (Ultralytics YOLOv8):
pip install ultralytics
yolo detect train data=pandid.yaml model=yolov8n.pt imgsz=1024 epochs=50

Replace sample_image.jpg with real P&ID images
and update/create .txt files in labels/train accordingly.
"""


print(f"Dataset + XML created at: {base.resolve()}")



X, y = make_classification(
    n_samples=500,    
    n_features=5,     
    n_classes=2,      
    random_state=42
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("✅ Accuracy:", acc)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))