import whisperx
import gc
import time
import os
import torch
from datetime import datetime

print("hello auto-device")
print("=" * 50)
print("WHISPERX AUTO-DEVICE PROCESSING PIPELINE")
print("=" * 50)

# Auto-detect device
if torch.cuda.is_available():
    device = "cuda"
    compute_type = "float16"  # better for GPU
else:
    device = "cpu"
    compute_type = "int8"  # better for CPU

audio_file = "audio/tukevasti-ilmassa-3min.mp3"
batch_size = 16  # reduce if low on GPU mem

print(f"\n[1/10] CONFIGURATION:")
print(f"  Device: {device} ({'CUDA detected' if device == 'cuda' else 'CUDA not available'})")
print(f"  Audio file: {audio_file}")
print(f"  Batch size: {batch_size}")
print(f"  Compute type: {compute_type}")

# Start timing
processing_start_time = time.time()

# save model to local path (optional)
#model_dir = os.path.expanduser("~/dev/models")
#print(f"\n[2/10] LOADING WHISPER MODEL...")
#print(f"  Model directory: {model_dir}")

print(f"  Loading model to {device} with {compute_type}...")
load_start = time.time()

try:
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    load_time = time.time() - load_start
    print(f"  ✓ Whisper model loaded successfully in {load_time:.2f}s")
except Exception as e:
    print(f"  ✗ Error loading model: {e}")
    raise

print(f"\n[3/10] LOADING AUDIO...")
audio = whisperx.load_audio(audio_file)
# Get audio duration
audio_duration = len(audio) / 16000  # whisperx uses 16kHz sample rate
print(f"  ✓ Audio loaded: {audio_duration:.2f} seconds duration")

print(f"\n[4/10] TRANSCRIBING AUDIO...")
transcribe_start = time.time()
result = model.transcribe(audio, batch_size=batch_size, language="fi")
transcribe_time = time.time() - transcribe_start
print(f"  ✓ Transcription completed in {transcribe_time:.2f}s")
print(f"  ✓ Found {len(result['segments'])} segments")
print(f"\n  SEGMENTS BEFORE ALIGNMENT:")
for i, segment in enumerate(result["segments"][:3]):  # Show first 3 segments
    print(f"    [{i + 1}] {segment['start']:.2f}s-{segment['end']:.2f}s: {segment['text'][:50]}...")
if len(result["segments"]) > 3:
    print(f"    ... and {len(result['segments']) - 3} more segments")

print(f"\n[5/10] LOADING ALIGNMENT MODEL...")
# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code="fi", device=device)
print(f"  ✓ Alignment model loaded for Finnish")

print(f"\n[6/10] ALIGNING TRANSCRIPT...")
align_start = time.time()
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
align_time = time.time() - align_start
print(f"  ✓ Alignment completed in {align_time:.2f}s")
print(f"\n  SEGMENTS AFTER ALIGNMENT:")
for i, segment in enumerate(result["segments"][:3]):  # Show first 3 segments
    print(f"    [{i + 1}] {segment['start']:.2f}s-{segment['end']:.2f}s: {segment['text'][:50]}...")

print(f"\n[7/10] LOADING DIARIZATION MODEL...")
hftoken = os.getenv("HF_TOKEN")
print("hftoken:", hftoken)
diarize_model = whisperx.diarize.DiarizationPipeline(token=hftoken, device=device)
#diarize_model = whisperx.diarize.DiarizationPipeline(token=os.getenv("HF_TOKEN"), device=device)
print(f"  ✓ Diarization model loaded")

print(f"\n[8/10] PERFORMING SPEAKER DIARIZATION...")
diarize_start = time.time()
# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
#diarize_segments = diarize_model(audio, min_speakers=0, max_speakers=0)
diarize_time = time.time() - diarize_start
print(f"  ✓ Diarization completed in {diarize_time:.2f}s")

print(f"\n[9/10] ASSIGNING SPEAKERS TO WORDS...")
assign_start = time.time()
result = whisperx.assign_word_speakers(diarize_segments, result)
assign_time = time.time() - assign_start
print(f"  ✓ Speaker assignment completed in {assign_time:.2f}s")

print(f"\n[10/10] SAVING RESULTS...")
# Create output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Generate timestamp filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f"{timestamp}.txt")

# Write diarized transcript to file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(f"Diarized Transcript - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Audio file: {audio_file}\n")
    f.write(f"Device: {device}\n")
    f.write(f"Duration: {audio_duration:.2f} seconds\n")
    f.write("=" * 50 + "\n\n")

    for i, segment in enumerate(result["segments"]):
        speaker = segment.get('speaker', 'UNKNOWN')
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        f.write(f"[{i + 1:03d}] {speaker} ({start_time:.2f}s-{end_time:.2f}s): {text}\n")

print(f"  ✓ Results saved to {output_file}")

# Calculate and display timing results
end_time = time.time()
processing_time = time.time() - processing_start_time
speed_ratio = audio_duration / processing_time

print(f"\n" + "=" * 50)
print(f"DETAILED TIMING BREAKDOWN:")
print(f"=" * 50)
print(f"Transcription: {transcribe_time:.2f}s ({transcribe_time / processing_time * 100:.1f}%)")
print(f"Alignment: {align_time:.2f}s ({align_time / processing_time * 100:.1f}%)")
print(f"Diarization: {diarize_time:.2f}s ({diarize_time / processing_time * 100:.1f}%)")
print(f"Speaker assignment: {assign_time:.2f}s ({assign_time / processing_time * 100:.1f}%)")
print(f"Other operations: {processing_time - transcribe_time - align_time - diarize_time - assign_time:.2f}s")

print(f"\n=== FINAL TIMING RESULTS ===")
print(f"Audio duration: {audio_duration:.2f} seconds")
print(f"Processing time: {processing_time:.2f} seconds")
print(f"Speed ratio: {speed_ratio:.2f}x (processing time vs audio length)")
if speed_ratio > 1:
    print(f"Processing was {speed_ratio:.2f}x faster than real-time")
else:
    print(f"Processing was {1 / speed_ratio:.2f}x slower than real-time")

print(f"\nTranscript saved to: {output_file}")