# 🐅 Project BAGH - SentinelSound Poaching Detection System

An end-to-end acoustic monitoring system for wildlife poaching detection using TinyML and IoT technology. The system uses an Arduino Uno with a MAX4466 microphone to detect gunshots and animal distress calls, classifies them using a machine learning model, and displays real-time alerts on a web dashboard.

![System Status](https://img.shields.io/badge/Status-Operational-green)
![ML Model](https://img.shields.io/badge/ML-LogisticRegression-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-96%25+-brightgreen)

---

## 📋 Table of Contents

- [System Architecture](#-system-architecture)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Detailed Setup](#-detailed-setup)
- [Running the System](#-running-the-system)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Troubleshooting](#-troubleshooting)

---

## 🏗️ System Architecture

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   Arduino Uno       │     │   Python Script     │     │   Supabase Cloud    │
│   + MAX4466 Mic     │────▶│   (Laptop)          │────▶│   (Database)        │
│                     │     │                     │     │                     │
│   Serial @ 1Mbps    │     │   - Audio Capture   │     │   - poaching_events │
│   Sample: 20kHz     │     │   - ML Inference    │     │   - sensor_nodes    │
│                     │     │   - MFCC Features   │     │   - audio_evidence  │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
                                                                  │
                                                                  ▼
                                                        ┌─────────────────────┐
                                                        │   Next.js Dashboard │
                                                        │   (Real-time)       │
                                                        │                     │
                                                        │   - Live Map        │
                                                        │   - Alerts Log      │
                                                        │   - Analytics       │
                                                        │   - AI Assistant    │
                                                        └─────────────────────┘
```

---

## 📦 Prerequisites

### Hardware
- Arduino Uno (or compatible board)
- MAX4466 Electret Microphone Amplifier
- USB cable for Arduino connection
- Laptop/PC with Windows

### Software
- Python 3.10+ 
- Node.js 18+ and npm
- Arduino IDE (for uploading firmware)
- Git

### Accounts
- Supabase account (free tier works)

---

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Project-BAGH-Poaching-Detection.git
cd Project-BAGH-Poaching-Detection

# Create Python virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the environment template
copy .env.example .env

# Edit .env with your Supabase credentials
# SUPABASE_URL=https://your-project.supabase.co
# SUPABASE_KEY=your-anon-key
```

### 3. Setup Website

```bash
cd Website

# Install Node.js dependencies
npm install

# Copy environment template
copy .env.example .env.local

# Edit .env.local with:
# NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
# NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
# GEMINI_API_KEY=your-gemini-key (for AI assistant)
```

### 4. Upload Arduino Firmware

1. Open `Arduino_Code/Arduino_Code.ino` in Arduino IDE
2. Select your board (Arduino Uno) and port (e.g., COM7)
3. Upload the sketch

### 5. Run the System

**Terminal 1 - Website:**
```bash
cd Website
npm run dev
```

**Terminal 2 - Detection Script:**
```bash
cd ..
.venv\Scripts\activate
python record_poaching.py
```

### 6. Test

1. Open http://localhost:3000/dashboard/alerts in your browser
2. Play a gunshot sound near the Arduino microphone
3. Watch the event appear on the dashboard! 🎯

---

## 🔧 Detailed Setup

### Python Dependencies

```bash
# Core
pip install pyserial numpy

# ML & Audio Processing
pip install scikit-learn joblib librosa

# Supabase & Utils
pip install supabase python-dotenv requests
```

### Database Schema

The system uses the following Supabase tables:

#### `poaching_events`
| Column | Type | Description |
|--------|------|-------------|
| id | bigint | Auto-increment primary key |
| node_id | text | Sensor node identifier |
| timestamp | timestamptz | Event detection time |
| event_type | text | `gunshot` or `animal_distress` |
| confidence | numeric | ML model confidence (0-1) |
| audio_url | text | Supabase Storage URL |
| severity | text | `low`, `medium`, `high`, `critical` |
| verification_status | text | `pending`, `verified_poaching`, `false_positive` |

#### `sensor_nodes`
| Column | Type | Description |
|--------|------|-------------|
| id | text | Node identifier (e.g., NODE-01) |
| name | text | Human-readable name |
| gps_lat | numeric | Latitude |
| gps_lon | numeric | Longitude |
| status | text | `online`, `offline`, `maintenance` |
| zone | text | Deployment zone |

### Storage Bucket

Create a bucket named `audio_evidence` with public access for storing .wav files.

---

## 🏃 Running the System

### Full System (Production)

```bash
# Terminal 1: Start the dashboard
cd Website
npm run dev

# Terminal 2: Run detection script
cd Project-BAGH-Poaching-Detection
.venv\Scripts\activate
python record_poaching.py
```

### Test Pipeline (No Arduino)

```bash
# Run pipeline test without hardware
python test_pipeline.py
```

### Configuration Options

Edit `record_poaching.py` to adjust:

```python
SERIAL_PORT = 'COM7'       # Your Arduino port
BAUD_RATE = 1000000        # Must match Arduino
SAMPLE_RATE = 20000        # Audio sample rate
RECORD_SECONDS = 4         # Recording duration
TRIGGER_THRESHOLD = 50     # Sensitivity (higher = less sensitive)
NODE_ID = "NODE-01"        # Sensor identifier
```

---

## 📁 Project Structure

```
Project-BAGH-Poaching-Detection/
├── Arduino_Code/
│   └── Arduino_Code.ino      # Arduino firmware
├── ML_training/
│   ├── gunshot_animal_model.pkl  # Trained ML model
│   └── data/                 # Training data (animals/gunshots)
├── Website/
│   ├── app/
│   │   ├── dashboard/        # Dashboard pages
│   │   └── api/              # API routes (chat, transcribe)
│   ├── components/           # React components
│   └── lib/                  # Utilities (Supabase client)
├── recordings/               # Saved audio files
├── record_poaching.py        # Main detection script
├── test_pipeline.py          # Pipeline test script
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
└── README.md                 # This file
```

---

## 🔌 API Reference

### Detection Pipeline Flow

1. **Audio Capture**: Arduino samples at 20kHz, sends via Serial
2. **Trigger Detection**: Python monitors for loud sounds
3. **Recording**: 4-second audio clip saved as .wav
4. **Feature Extraction**: 26 MFCC features extracted using librosa
5. **ML Inference**: LogisticRegression classifies as gunshot/animal
6. **Upload**: Audio uploaded to Supabase Storage
7. **Database**: Event record inserted to `poaching_events`
8. **Real-time**: Dashboard updates via Supabase subscriptions

### ML Model Details

- **Type**: Logistic Regression (scikit-learn)
- **Classes**: `0` = animal_distress, `1` = gunshot
- **Features**: 26 (13 MFCC means + 13 MFCC stds)
- **Training Data**: Gunshots (AK-47, M16, etc.) + Animals (elephant, wolf, etc.)

---

## 🔍 Troubleshooting

### Arduino Not Connecting

```bash
# Check available ports
python -c "import serial.tools.list_ports; print([p.device for p in serial.tools.list_ports.comports()])"

# Update SERIAL_PORT in record_poaching.py
SERIAL_PORT = 'COM3'  # Your actual port
```

### ML Model Version Warning

The warning about sklearn version mismatch is safe to ignore. To suppress:

```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
```

### Database RLS Errors

Run this SQL in Supabase:

```sql
-- Allow inserts to poaching_events
CREATE POLICY "Allow insert" ON poaching_events FOR INSERT WITH CHECK (true);

-- Allow read access
CREATE POLICY "Allow read" ON poaching_events FOR SELECT USING (true);
```

### Website Not Updating

1. Check browser console for errors
2. Verify Supabase credentials in `.env.local`
3. Ensure real-time is enabled in Supabase dashboard

---

## 📊 Dashboard Features

| Feature | Path | Description |
|---------|------|-------------|
| Overview | `/dashboard` | System stats and recent events |
| Live Map | `/dashboard/map` | Sensor locations and threat markers |
| Events Log | `/dashboard/alerts` | All detection events with audio playback |
| Analytics | `/dashboard/analytics` | Charts and trend analysis |
| AI Assistant | `/dashboard/ai-assistant` | Voice-controlled queries |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is developed for educational purposes as part of wildlife conservation research.

---

## 🙏 Acknowledgments

- Bandipur National Park for inspiration
- Training data from various open-source audio datasets
- Supabase for real-time database infrastructure

---

**Built with ❤️ for Wildlife Conservation**