"""
upload_alerts.py
Standalone script to upload simulated poaching events to Supabase.
"""

import os
import datetime
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client


def get_supabase_client() -> Client:
    """Initialize and return Supabase client using environment variables."""
    load_dotenv()
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
    
    return create_client(supabase_url, supabase_key)


def upload_event(audio_file_path: str) -> bool:
    """
    Upload a poaching event to Supabase.
    
    Args:
        audio_file_path: Path to the audio file to upload as evidence.
        
    Returns:
        True if upload was successful, False otherwise.
    """
    try:
        # Initialize Supabase client
        supabase = get_supabase_client()
        
        # Generate timestamp for unique filename
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Read audio file binary data
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        
        # Upload to Supabase Storage bucket 'audio_evidence'
        storage_filename = f"evidence_{timestamp_str}.wav"
        
        storage_response = supabase.storage.from_("audio_evidence").upload(
            path=storage_filename,
            file=audio_data,
            file_options={"content-type": "audio/wav"}
        )
        
        # Get public URL of uploaded file
        public_url = supabase.storage.from_("audio_evidence").get_public_url(storage_filename)
        
        print(f"Audio uploaded to: {public_url}")
        
        # Insert event into poaching_events table
        event_data = {
            "node_id": "NODE-01",
            "timestamp": timestamp.isoformat(),
            "event_type": "gunshot",
            "confidence": 0.95,
            "audio_url": public_url,
            "verification_status": "pending",
            "severity": "critical"
        }
        
        db_response = supabase.table("poaching_events").insert(event_data).execute()
        
        print("SUCCESS: Event uploaded")
        print(f"Event ID: {db_response.data[0].get('id', 'N/A')}")
        return True
        
    except FileNotFoundError as e:
        print(f"FAILED: {e}")
        return False
    except ValueError as e:
        print(f"FAILED: {e}")
        return False
    except Exception as e:
        print(f"FAILED: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python upload_alerts.py <audio_file_path>")
        print("Example: python upload_alerts.py ./recordings/sample.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    success = upload_event(audio_file)
    sys.exit(0 if success else 1)
