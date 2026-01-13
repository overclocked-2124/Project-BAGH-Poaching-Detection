import serial
import wave
import time
import numpy as np
import struct

# --- CONFIGURATION ---
SERIAL_PORT = 'COM7'      # CHECK YOUR ARDUINO PORT!
BAUD_RATE = 1000000       # Must match Arduino Code
SAMPLE_RATE = 20000       # Approx rate Arduino Uno can push
THRESHOLD = 600           # 0-1023. Silence is usually ~512. Trigger > 600.
RECORD_SECONDS = 4
OUTPUT_FILENAME = "poaching_event.wav"

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        print(f"Connected to {SERIAL_PORT}. Listening for noise...")
    except:
        print(f"Error: Could not open {SERIAL_PORT}. Is Arduino IDE Monitor open?")
        return

    # Buffer to hold slight history (pre-trigger)
    audio_buffer = []
    recording = False
    frames_to_record = RECORD_SECONDS * SAMPLE_RATE
    recorded_frames = 0
    
    print("System Armed. Make some noise!")

    while True:
        if ser.in_waiting >= 2:
            # Read 2 bytes (16-bit integer)
            data = ser.read(2)
            # Combine bytes into integer
            val = (data[0] << 8) | data[1]
            
            # Save to buffer
            audio_buffer.append(val)
            
            # Keep buffer minimal size if not recording
            if not recording and len(audio_buffer) > 1000:
                audio_buffer.pop(0)

            # --- TRIGGER LOGIC ---
            # Center of 0-1023 is ~512.
            # We look for deviations (Loudness)
            loudness = abs(val - 512)
            
            if not recording and loudness > (THRESHOLD - 512):
                print("LOUD NOISE DETECTED! Recording...")
                recording = True
                recorded_frames = 0

            # --- RECORDING LOGIC ---
            if recording:
                recorded_frames += 1
                if recorded_frames >= frames_to_record:
                    print("Recording Complete. Saving file...")
                    save_wav(audio_buffer)
                    
                    # Reset
                    audio_buffer = []
                    recording = False
                    print("Armed and listening again...")

def save_wav(data):
    # Normalize 0-1023 to 16-bit audio range (-32768 to 32767)
    # 0 -> -32768, 512 -> 0, 1023 -> 32767
    np_data = np.array(data, dtype=np.int16)
    np_data = (np_data - 512) * 64 

    with wave.open(OUTPUT_FILENAME, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2) # 2 bytes (16-bit)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(np_data.tobytes())
    print(f"Saved: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    main()