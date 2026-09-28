```markdown
# ⛏️ MineSafe Titan — SIH 2026 Final Prototype
> **Official Submission for Smart India Hackathon (SIH) 2026 — Problem Statement 26039**  
> *Real-Time AI & IoT Underground Mine Safety Platform — Multi-Hazard Detection, Worker Biometrics Tracking, YOLOv8 Vision & Autonomous Teleoperation.*

---

## 📌 1. Project Overview & Core Concept
**MineSafe Titan** is an integrated IoT hardware and AI/ML decision-support system engineered to prevent and mitigate underground coal mining disasters. 

Underground mines present severe environmental hazards including explosive methane ($\text{CH}_4$) ingress[cite: 16, 18], oxygen ($\text{O}_2$) deficiency[cite: 16, 18], toxic carbon monoxide ($\text{CO}$) poisoning[cite: 16, 18], roof collapses[cite: 16, 18], and water sump flooding[cite: 16, 18]. **MineSafe Titan** bridges the gap between field sensors and emergency control centers by providing:

1. **Multi-Sensor Telemetry Validation:** Real-time ingestion of atmospheric, thermal, and seismic data from custom ESP32 hardware nodes over HTTP/REST APIs[cite: 17].
2. **Explainable Hazard Fusion Engine:** Mathematical calculation of a single, transparent $0\text{--}100$ Mine Hazard Score combining atmospheric, geotechnical, and thermal risk factors[cite: 18].
3. **Worker Triage Risk Index (TRPI):** Dynamic $0\text{--}100$ health scoring for miners fusing biometric wearable telemetry (Heart Rate, $\text{SpO}_2$, Body Temp) with toxic gas exposure and fall detection[cite: 17, 18].
4. **Multimodal Computer Vision (YOLOv8 + ByteTrack):** Object detection and persistent tracking for PPE compliance (`no_helmet`, `no_vest`), worker fall evidence (`fallen_person`), structural hazards, and subterranean smoke/fire[cite: 22].
5. **Statutory Compliance Audit:** Built-in verification against Directorate General of Mines Safety (DGMS) Coal Mines Regulations (CMR 2017) with one-click forensic JSON audit logging[cite: 19, 23].

---

## 🏗️ 2. System Architecture

```text
       ┌────────────────────────┐      ┌─────────────────────────┐
       │   ESP32 Rover Gateway   │      │   Worker Wearable Nodes │
       │  (Atmospheric Sensors) │      │  (HR, SpO₂, Fall Sensor)│
       └───────────┬────────────┘      └────────────┬────────────┘
                   │ GET /data                      │ GET /workers
                   └──────────────────┬─────────────┘
                                      │
                                      ▼
       ┌─────────────────────────────────────────────────────────┐
       │              Multimodal AI & Hazard Fusion               │
       │  ┌───────────────────────┬───────────────────────────┐  │
       │  │ Hazard Engine (0-100) │ TinyMLP Sensor Anomaly AI │  │
       │  ├───────────────────────┼───────────────────────────┤  │
       │  │ TRPI Triage Index     │ YOLOv8 Computer Vision    │  │
       │  └───────────────────────┴───────────────────────────┘  │
       └───────────────────────────┬─────────────────────────────┘
                                   │
                                   ▼
       ┌─────────────────────────────────────────────────────────┐
       │                  Command Dashboard (UI)                 │
       │  ┌───────────────────────┬───────────────────────────┐  │
       │  │ Real-time Telemetry   │ Coal Mine Gallery Map     │  │
       │  ├───────────────────────┼───────────────────────────┤  │
       │  │ MJPEG Camera Stream   │ Rover Teleoperation       │  │
       │  ├───────────────────────┼───────────────────────────┤  │
       │  │ Emergency Protocols   │ Statutory Standards & Audit│ │
       │  └───────────────────────┴───────────────────────────┘  │
       └─────────────────────────────────────────────────────────┘

```

---

## 🛠️ 3. Hardware & Software Tech Stack

* **Hardware Nodes:** ESP32-S3, ESP32-S3-CAM, DHT11 (Temp/Humidity), MQ-2 (Combustible Gas/Methane), MQ-9 (Carbon Monoxide), MQ-135 (Air Quality/$\text{CO}_2$), SW-420 (Vibration Sensor), Water Level Sensor.


* **Frontend / Command Center:** Python 3.12+, Streamlit, Plotly, HTML5/CSS3.


* **Machine Learning & Vision:** PyTorch, Ultralytics YOLOv8, ByteTrack, Scikit-Learn, NumPy, Pandas, PIL.


* **Protocols & API:** HTTP RESTful APIs, JSON, MJPEG Video Streaming.



---

## 📂 4. Repository File Structure

The project is structured modularly under the `minesafe/` package to ensure clean separation between UI components, core algorithms, hardware adapters, and AI models.

```text
MineSafe_Titan_26039/
│
├── minesafe/                          <-- Main Application Package
│   ├── __init__.py
│   ├── app.py                         <-- Streamlit command dashboard UI
│   │
│   ├── core/                          <-- Core Business Logic & Adapters
│   │   ├── __init__.py
│   │   ├── config.py                  <-- Thresholds, limits & graph coordinates
│   │   ├── gateway.py                 <-- ESP32 REST API client & sensor scaling
│   │   ├── hazard.py                  <-- Multi-hazard engine & TRPI triage score
│   │   ├── methods.py                 <-- CMR 2017 references & O₂ buffer proxy
│   │   ├── route.py                   <-- Dijkstra graph search & path ranking
│   │   ├── telemetry.py               <-- Telemetry data model & sanitization
│   │   └── workers.py                 <-- Wearable biometrics data model
│   │
│   └── models/                        <-- Artificial Intelligence & Computer Vision
│       ├── __init__.py
│       ├── sensor_ai.py               <-- TinyMLP neural network for sensor anomalies
│       └── vision.py                  <-- YOLOv8 object detector & ByteTrack pipeline
│
├── sample_data/
│   └── sample_sensor_history.csv     <-- Historical sensor CSV for training
├── dataset/
│   ├── data.yaml.example             <-- YOLO dataset path configuration
│   └── README.md                      <-- Guide for mine vision classes
├── evaluate_sensor_model.py           <-- Evaluates TinyMLP precision/recall
├── train_yolov8.py                    <-- Script to train custom YOLO weights
├── validate_yolo_dataset.py           <-- Validates YOLO label formatting
├── smoke_test.py                      <-- Automated core system health test
├── start.bat                          <-- Windows 1-click launcher
├── start.sh                           <-- Linux/macOS 1-click launcher
├── requirements.txt                   <-- Python package dependencies
└── .gitignore                         <-- Git exclusion rules

```

---

## 🔌 5. Live ESP32 Hardware API Contract

When connected to an active ESP32 rover node (`LIVE ESP GATEWAY` mode), the system communicates over these REST endpoints:

* **`GET /data`**: Fetches real-time sensor JSON (`temperature`, `humidity`, `mq2`, `mq9`, `mq135`, `vibration`, `waterValue`).


* **`GET /capture`**: Serves a JPEG camera frame snapshot from the ESP32-S3-CAM.


* **`POST /api/command`**: Dispatches motor actions (`{"command": "forward" | "reverse" | "left" | "right" | "start" | "emergency_stop"}`).



---

## ⚡ 6. Quickstart & Installation

### Prerequisites

* Python 3.10+ installed on your computer.


* ESP32-S3 board connected to the same Wi-Fi / Hotspot network as your laptop.



### 1. Clone Repository

```bash
git clone https://github.com/techforge26039/MineSafe_Titan_26039.git

```

### 2. Launch Dashboard (1-Click)

* **Windows:** Double-click `start.bat` or run in PowerShell:



```powershell
.\start.bat

```

* **Linux / macOS:** Run in terminal:



```bash
chmod +x start.sh
./start.sh

```

---

## 📜 7. Statutory References & Disclaimer

* **Regulatory Baseline:** Threshold references used in this project are derived from the **Directorate General of Mines Safety (DGMS), Coal Mines Regulations, 2017** (Regulation 53).


* **Disclaimer:** MineSafe Titan is an internal-round hackathon research prototype designed for decision-support. It is not certified mine-control equipment and should not replace statutory emergency procedures or certified mine instrumentation.



```

```
