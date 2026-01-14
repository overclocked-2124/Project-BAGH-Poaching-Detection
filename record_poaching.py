import serial
import wave
import time
import numpy as np
import struct
import os
import joblib
import librosa
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

# --- SUPABASE CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables. Check your .env file.")

# Ensure URL has trailing slash (required by newer supabase-py versions)
if not SUPABASE_URL.endswith('/'):
    SUPABASE_URL = SUPABASE_URL + '/'

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- HARDWARE CONFIGURATION ---
SERIAL_PORT = 'COM7'       # CHECK YOUR PORT
BAUD_RATE = 1000000        # Must match Arduino Code
SAMPLE_RATE = 20000        # Approx rate for Arduino Uno
MIN_RECORD_SECONDS = 2     # Minimum recording duration
MAX_RECORD_SECONDS = 15    # Maximum recording duration (safety limit)
SILENCE_DURATION = 1.0     # Seconds of silence before stopping recording
OUTPUT_DIR = "recordings"   # Directory to store recordings
NODE_ID = "NODE-01"         # Static node ID for this laptop

# SENSITIVITY: How much louder than background noise to trigger?
# Higher number = Harder to trigger (Less false alarms)
# Lower number = Easier to trigger (More sensitive)
TRIGGER_THRESHOLD = 50     # Trigger when sound exceeds baseline + this
TRIGGER_SAMPLES = 10       # Need this many consecutive loud samples to trigger (noise filter)
SILENCE_MARGIN = 0.5       # Consider "quiet" when loudness drops below trigger level * this factor   

# --- ML MODEL CONFIGURATION ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ML_training", "gunshot_animal_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "ML_training", "scaler.pkl")
ML_MODEL = None  # Loaded at startup
ML_SCALER = None  # Feature scaler
CLASS_MAPPING = {0: "animal_distress", 1: "gunshot"}  # Model classes: 0=animals, 1=gunshots


def load_ml_model():
    """Load the trained ML model and scaler at startup."""
    global ML_MODEL, ML_SCALER
    try:
        ML_MODEL = joblib.load(MODEL_PATH)
        print(f"[ML] Model loaded successfully from: {MODEL_PATH}")
        print(f"[ML] Model type: {type(ML_MODEL).__name__}")
        
        # Load scaler if exists
        if os.path.exists(SCALER_PATH):
            ML_SCALER = joblib.load(SCALER_PATH)
            print(f"[ML] Scaler loaded from: {SCALER_PATH}")
        else:
            print(f"[ML] No scaler found, using raw features")
            
    except Exception as e:
        print(f"[ML] WARNING: Could not load model: {e}")
        print("[ML] Falling back to mock classification")
        ML_MODEL = None


def get_host_location() -> tuple:
    """
    Get the approximate GPS location of the host laptop using IP geolocation.
    Falls back to default coordinates if unavailable.
    
    Returns:
        Tuple of (latitude, longitude)
    """
    DEFAULT_LAT, DEFAULT_LON = 11.6664, 76.6292  # Bandipur National Park
    
    try:
        # Use free IP geolocation API
        response = requests.get("http://ip-api.com/json/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                lat = data.get("lat", DEFAULT_LAT)
                lon = data.get("lon", DEFAULT_LON)
                print(f"[LOCATION] Host location: {lat}, {lon} ({data.get('city', 'Unknown')}, {data.get('country', 'Unknown')})")
                return (lat, lon)
    except Exception as e:
        print(f"[LOCATION] Could not get host location: {e}")
    
    print(f"[LOCATION] Using default location: {DEFAULT_LAT}, {DEFAULT_LON}")
    return (DEFAULT_LAT, DEFAULT_LON)


def extract_audio_features(file_path: str, target_sr: int = 22050) -> np.ndarray:
    """
    Extract comprehensive audio features from file for ML inference.
    Must match the feature extraction used during training.
    
    Args:
        file_path: Path to the .wav file
        target_sr: Target sample rate for librosa
        
    Returns:
        Feature vector (55 features total)
    """
    # Load audio file
    y, sr = librosa.load(file_path, sr=target_sr, duration=4.0)
    
    features = []
    
    # 1. MFCCs (13 coefficients) - mean and std = 26 features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))
    
    # 2. Spectral Centroid - 2 features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(spectral_centroid))
    features.append(np.std(spectral_centroid))
    
    # 3. Spectral Bandwidth - 2 features
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(spectral_bandwidth))
    features.append(np.std(spectral_bandwidth))
    
    # 4. Spectral Rolloff - 2 features
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(spectral_rolloff))
    features.append(np.std(spectral_rolloff))
    
    # 5. Zero Crossing Rate - 2 features
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))
    
    # 6. RMS Energy - 2 features
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.std(rms))
    
    # 7. Spectral Contrast - 7 features
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(spectral_contrast, axis=1))
    
    # 8. Chroma Features - 12 features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    
    return np.array(features).reshape(1, -1)


def analyze_audio(file_path: str) -> dict:
    """
    Run ML inference on audio file to classify the sound.
    
    Uses a trained LogisticRegression model to classify audio as either:
    - 'animal_distress' (class 0): Animal sounds
    - 'gunshot' (class 1): Gunshot sounds
    
    Args:
        file_path: Path to the .wav file to analyze
        
    Returns:
        dict with 'event_type' and 'confidence' keys
    """
    if ML_MODEL is None:
        # Fallback to mock if model not loaded
        import random
        event_type = random.choice(["gunshot", "animal_distress"])
        confidence = round(random.uniform(0.6, 0.99), 2)
        print(f"[ML] (Mock) Detected: {event_type} ({confidence:.0%})")
        return {"event_type": event_type, "confidence": confidence}
    
    try:
        # Extract features
        print(f"[ML] Extracting features from: {file_path}")
        features = extract_audio_features(file_path)
        
        # Scale features if scaler is available
        if ML_SCALER is not None:
            features = ML_SCALER.transform(features)
        
        # Run inference
        prediction = ML_MODEL.predict(features)[0]
        probabilities = ML_MODEL.predict_proba(features)[0]
        confidence = float(max(probabilities))
        
        # Map to event type
        event_type = CLASS_MAPPING.get(prediction, "gunshot")
        
        print(f"[ML] Prediction: {prediction} -> {event_type}")
        print(f"[ML] Confidence: {confidence:.1%}")
        print(f"[ML] Probabilities: animal={probabilities[0]:.2%}, gunshot={probabilities[1]:.2%}")
        
        return {
            "event_type": event_type,
            "confidence": round(confidence, 2)
        }
        
    except Exception as e:
        print(f"[ML] Error during inference: {e}")
        # Fallback on error
        return {"event_type": "gunshot", "confidence": 0.5}


def upload_to_supabase_storage(file_path: str) -> str:
    """
    Upload a .wav file to Supabase Storage and return the public URL.
    
    Args:
        file_path: Local path to the .wav file
        
    Returns:
        Public URL of the uploaded file
    """
    # Generate unique filename with timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{NODE_ID}_{timestamp}.wav"
    storage_path = f"alerts/{filename}"
    
    print(f"[UPLOAD] Uploading {file_path} to Supabase Storage...")
    
    with open(file_path, "rb") as f:
        file_data = f.read()
    
    # Upload to the audio_evidence bucket
    response = supabase.storage.from_("audio_evidence").upload(
        path=storage_path,
        file=file_data,
        file_options={"content-type": "audio/wav", "upsert": "true"}
    )
    
    # Get the public URL
    public_url = supabase.storage.from_("audio_evidence").get_public_url(storage_path)
    
    print(f"[UPLOAD] Success! URL: {public_url}")
    return public_url


def push_event_to_database(event_type: str, confidence: float, audio_url: str, audio_duration: float = 4.0) -> dict:
    """
    Insert a new poaching event record into the Supabase database.
    
    Args:
        event_type: Type of detected event (gunshot, chainsaw, etc.)
        confidence: ML model confidence score (0.0 - 1.0)
        audio_url: Public URL of the uploaded audio file
        audio_duration: Duration of the audio recording in seconds
        
    Returns:
        The inserted database record
    """
    # Determine severity based on confidence
    if confidence > 0.8:
        severity = "high"
    elif confidence > 0.6:
        severity = "medium"
    else:
        severity = "low"
    
    # Prepare the event record
    event_record = {
        "node_id": NODE_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "confidence": confidence,
        "audio_url": audio_url,
        "verification_status": "pending",
        "severity": severity,
        "audio_duration_seconds": int(round(audio_duration))  # Must be integer for DB
    }
    
    print(f"[DATABASE] Inserting event: {event_type} ({severity} severity)")
    
    # Insert into poaching_events table
    response = supabase.table("poaching_events").insert(event_record).execute()
    
    print(f"[DATABASE] Event recorded with ID: {response.data[0]['id']}")
    return response.data[0]


def process_detection(file_path: str):
    """
    Post-detection pipeline: Analyze, Upload, and Record the event.
    
    Args:
        file_path: Path to the saved .wav file
    """
    print("\n" + "="*50)
    print("POST-DETECTION PIPELINE STARTED")
    print("="*50)
    
    try:
        # Get actual audio duration from file
        with wave.open(file_path, 'r') as wf:
            audio_duration = wf.getnframes() / wf.getframerate()
        
        # Step 1: Run ML inference
        ml_result = analyze_audio(file_path)
        
        # Step 2: Upload audio to Supabase Storage
        audio_url = upload_to_supabase_storage(file_path)
        
        # Step 3: Push event to database
        event = push_event_to_database(
            event_type=ml_result["event_type"],
            confidence=ml_result["confidence"],
            audio_url=audio_url,
            audio_duration=audio_duration
        )
        
        print("="*50)
        print("PIPELINE COMPLETE - Event sent to Dashboard!")
        print(f"View at: {SUPABASE_URL.replace('.supabase.co', '.supabase.co')}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        print("Event NOT recorded. Check your Supabase credentials and network connection.")


def get_output_filename() -> str:
    """Generate a unique filename for each recording."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"poaching_alert_{timestamp}.wav")


def update_sensor_node_location():
    """Update the sensor node's GPS coordinates based on host location."""
    try:
        lat, lon = get_host_location()
        
        # Update the sensor node in the database
        response = supabase.table("sensor_nodes").upsert({
            "id": NODE_ID,
            "name": "Laptop Recording Station",
            "gps_lat": lat,
            "gps_lon": lon,
            "status": "online",
            "zone": "Field Station",
            "last_seen": datetime.now(timezone.utc).isoformat()
        }).execute()
        
        print(f"[DATABASE] Sensor node {NODE_ID} location updated: ({lat}, {lon})")
        return (lat, lon)
    except Exception as e:
        print(f"[DATABASE] Failed to update sensor location: {e}")
        return None


def main():
    # --- INITIALIZATION ---
    print("=" * 50)
    print("SENTINELSOUND POACHING DETECTION SYSTEM")
    print("=" * 50)
    
    # Load ML model
    load_ml_model()
    
    # Update sensor location from host system
    update_sensor_node_location()
    
    print("=" * 50)
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        print(f"Connected to {SERIAL_PORT}.")
    except Exception as e:
        print(f"Error: {e}")
        return

    # --- PHASE 1: CALIBRATION ---
    print("\n--- CALIBRATING... PLEASE BE QUIET (2 Seconds) ---")
    calibration_data = []
    start_time = time.time()
    
    # Read for 2 seconds to find the "Baseline"
    while time.time() - start_time < 2:
        if ser.in_waiting >= 2:
            data = ser.read(2)
            val = (data[0] << 8) | data[1]
            calibration_data.append(val)
            
    if len(calibration_data) == 0:
        print("Error: No data received from Arduino. Check wiring!")
        return

    # Calculate statistics
    avg_baseline = np.mean(calibration_data)
    max_noise = np.max(np.abs(np.array(calibration_data) - avg_baseline))
    
    # Set the trigger level just above the natural noise
    dynamic_threshold = max_noise + TRIGGER_THRESHOLD
    
    print(f"Baseline (Silence): {avg_baseline:.2f}")
    print(f"Background Noise Level: {max_noise:.2f}")
    print(f"Trigger Set At: +/- {dynamic_threshold:.2f} from baseline")
    print("--------------------------------------------------")
    print("SYSTEM ARMED. LISTENING FOR POACHERS...")

    # --- PHASE 2: LISTENING ---
    audio_buffer = []
    recording = False
    min_frames = int(MIN_RECORD_SECONDS * SAMPLE_RATE)
    max_frames = int(MAX_RECORD_SECONDS * SAMPLE_RATE)
    silence_frames_needed = int(SILENCE_DURATION * SAMPLE_RATE)
    recorded_frames = 0
    silence_counter = 0
    loud_counter = 0  # Track consecutive loud samples to filter noise spikes
    
    # Silence threshold = halfway between baseline noise and trigger threshold
    silence_threshold = dynamic_threshold * SILENCE_MARGIN
    
    # Clear buffer to start fresh
    ser.reset_input_buffer()
    
    print(f"Silence detection threshold: {silence_threshold:.2f}")
    print(f"Trigger requires {TRIGGER_SAMPLES} consecutive loud samples")

    while True:
        if ser.in_waiting >= 2:
            data = ser.read(2)
            val = (data[0] << 8) | data[1]
            
            # Skip obviously bad readings (electrical noise - values near 0 or max)
            if val < 10 or val > 1000:
                continue  # Ignore spike, likely noise
            
            # --- TRIGGER LOGIC ---
            # Calculate how far this sound is from the baseline
            loudness = abs(val - avg_baseline)
            
            if not recording:
                # MONITOR MODE - Keep a small pre-buffer so we don't cut off the start
                audio_buffer.append(val)
                if len(audio_buffer) > 10000:  # ~0.5 sec pre-buffer
                    audio_buffer.pop(0)
                
                # CHECK IF LOUD ENOUGH - require consecutive loud samples to avoid noise triggers
                if loudness > dynamic_threshold:
                    loud_counter += 1
                    if loud_counter >= TRIGGER_SAMPLES:
                        print(f"\n>>> SOUND DETECTED! (Level: {loudness:.0f} > {dynamic_threshold:.0f}) Starting recording...")
                        recording = True
                        recorded_frames = 0
                        silence_counter = 0
                        loud_counter = 0
                else:
                    loud_counter = 0  # Reset if not loud
            
            else:
                # RECORDING MODE
                audio_buffer.append(val)
                recorded_frames += 1
                
                # Check if sound has returned to quiet (below silence threshold)
                if loudness < silence_threshold:
                    silence_counter += 1
                else:
                    silence_counter = 0  # Reset if still loud
                
                # Stop conditions:
                # 1. Sound returned to silence for SILENCE_DURATION (after minimum recording)
                # 2. Hit maximum recording time (safety limit)
                should_stop = False
                stop_reason = ""
                
                if recorded_frames >= min_frames and silence_counter >= silence_frames_needed:
                    should_stop = True
                    stop_reason = f"Sound ended (quiet for {SILENCE_DURATION}s)"
                elif recorded_frames >= max_frames:
                    should_stop = True
                    stop_reason = f"Max duration reached ({MAX_RECORD_SECONDS}s)"
                
                if should_stop:
                    duration = recorded_frames / SAMPLE_RATE
                    print(f"<<< Recording Complete. {stop_reason}")
                    print(f"    Duration: {duration:.1f} seconds ({recorded_frames} frames)")
                    output_file = get_output_filename()
                    save_wav(audio_buffer, avg_baseline, output_file)
                    
                    # --- POST-DETECTION: Supabase Integration ---
                    process_detection(output_file)
                    
                    # Reset to MONITOR MODE
                    audio_buffer = []
                    recording = False
                    silence_counter = 0
                    loud_counter = 0
                    ser.reset_input_buffer()  # Clear any buffered data
                    print("\n[MONITOR MODE] Listening for sounds...")

def save_wav(data, baseline, output_file):
    """Save audio data to a WAV file."""
    # Center the audio by removing the calculated baseline
    np_data = np.array(data, dtype=np.float32)
    np_data = np_data - baseline
    
    # Normalize volume (Make it audible)
    max_val = np.max(np.abs(np_data))
    if max_val > 0:
        np_data = np_data * (30000 / max_val) # Scale to ~90% max volume
    
    # Convert to 16-bit integer for WAV
    np_data = np_data.astype(np.int16)

    with wave.open(output_file, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2) 
        f.setframerate(SAMPLE_RATE)
        f.writeframes(np_data.tobytes())
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    main()