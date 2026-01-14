"""
Test script to verify ML model integration and Supabase connection.
Run this without Arduino connected to test the pipeline components.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from record_poaching import (
    load_ml_model, 
    get_host_location, 
    analyze_audio, 
    ML_MODEL,
    supabase,
    NODE_ID
)

def test_pipeline():
    print("=" * 60)
    print("SENTINELSOUND PIPELINE TEST")
    print("=" * 60)
    
    # Test 1: ML Model Loading
    print("\n[TEST 1] Loading ML Model...")
    load_ml_model()
    
    # Re-import to get updated value
    from record_poaching import ML_MODEL as loaded_model
    
    if loaded_model is not None:
        print("✅ ML Model loaded successfully!")
        print(f"   Model: {type(loaded_model).__name__}")
        print(f"   Classes: {loaded_model.classes_}")
    else:
        print("❌ ML Model failed to load (will use mock)")
    
    # Test 2: Location Service
    print("\n[TEST 2] Getting Host Location...")
    lat, lon = get_host_location()
    print(f"✅ Location: ({lat}, {lon})")
    
    # Test 3: Supabase Connection
    print("\n[TEST 3] Testing Supabase Connection...")
    try:
        result = supabase.table("sensor_nodes").select("id, name, status").limit(3).execute()
        print("✅ Supabase connected!")
        print(f"   Found {len(result.data)} sensor nodes:")
        for node in result.data:
            print(f"   - {node['id']}: {node['name']} ({node['status']})")
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
    
    # Test 4: Audio Analysis (with sample file if exists)
    print("\n[TEST 4] Testing Audio Analysis...")
    test_files = [
        "recordings/test.wav",
        "poaching_alert.wav",
    ]
    
    test_file = None
    for f in test_files:
        if os.path.exists(f):
            test_file = f
            break
    
    if test_file:
        result = analyze_audio(test_file)
        print(f"✅ Audio analysis result: {result}")
    else:
        print("⚠️ No test audio file found. Create a test recording to verify ML inference.")
        print("   (The ML model expects 26 MFCC features from audio)")
    
    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETE")
    print("=" * 60)
    print("\nTo run full system with Arduino:")
    print("  python record_poaching.py")

if __name__ == "__main__":
    test_pipeline()
