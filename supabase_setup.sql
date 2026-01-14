-- ============================================
-- SUPABASE SETUP FOR POACHING DETECTION SYSTEM
-- ============================================
-- Run this in the Supabase SQL Editor if not already done via MCP

-- 1. Create the audio_evidence storage bucket
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'audio_evidence', 
  'audio_evidence', 
  true,  -- Public bucket for easy access
  52428800,  -- 50MB max file size
  ARRAY['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/x-wav']
)
ON CONFLICT (id) DO NOTHING;

-- 2. Storage bucket RLS policies (allow public read/write for sensor nodes)
-- Note: In production, consider more restrictive policies with service role key

CREATE POLICY "Public Audio Read" ON storage.objects 
  FOR SELECT USING (bucket_id = 'audio_evidence');

CREATE POLICY "Public Audio Insert" ON storage.objects 
  FOR INSERT WITH CHECK (bucket_id = 'audio_evidence');

CREATE POLICY "Public Audio Update" ON storage.objects 
  FOR UPDATE USING (bucket_id = 'audio_evidence');

-- 3. Create the sensor node for your laptop (if not exists)
INSERT INTO sensor_nodes (id, name, gps_lat, gps_lon, zone, status, battery_level, firmware_version)
VALUES (
  'NODE-01', 
  'Laptop Recording Station', 
  0.0,    -- Update with actual GPS coordinates
  0.0,    -- Update with actual GPS coordinates
  'Base Camp', 
  'online', 
  100, 
  '1.0.0'
)
ON CONFLICT (id) DO UPDATE SET
  status = 'online',
  last_seen = now();

-- 4. Verify the poaching_events table exists with correct schema
-- (This should already exist from your dashboard setup)
-- The table should have these columns:
--   - id (bigint, auto-increment)
--   - node_id (text, FK to sensor_nodes)
--   - timestamp (timestamptz)
--   - event_type (text: gunshot, chainsaw, vehicle, animal_distress, human_voice, explosion, trap_sound)
--   - confidence (numeric, 0-1)
--   - audio_url (text)
--   - verification_status (text: pending, verified_poaching, false_positive, under_review)
--   - severity (text: low, medium, high, critical)
--   - audio_duration_seconds (integer)

-- 5. Verify RLS policies allow inserts from anon key
-- Run this to check existing policies:
-- SELECT * FROM pg_policies WHERE tablename = 'poaching_events';
