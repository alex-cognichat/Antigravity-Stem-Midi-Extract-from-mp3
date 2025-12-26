import os
import sys
import shutil
import subprocess
import torch
import platform
import logging
from pathlib import Path
from tqdm import tqdm
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Hardware Detection & Optimization ---

def get_apple_chip_name():
    """Get the specific Apple Silicon chip name (M1, M2, M3, etc.)"""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except:
        return "Apple Silicon"

def get_optimal_device():
    """
    Detect and return the best available compute device.
    Priority: CUDA > MPS > CPU
    """
    device_info = {
        "device": "cpu",
        "name": "CPU",
        "memory_gb": None,
        "is_accelerated": False
    }
    
    # Check for NVIDIA CUDA
    if torch.cuda.is_available():
        device_info["device"] = "cuda"
        device_info["name"] = torch.cuda.get_device_name(0)
        try:
            device_info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            device_info["memory_gb"] = 8.0 # Default fallback
        device_info["is_accelerated"] = True
        logger.info(f"✓ CUDA GPU detected: {device_info['name']} ({device_info['memory_gb']:.1f} GB VRAM)")
        
    # Check for Apple Silicon MPS
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device_info["device"] = "mps"
        device_info["name"] = get_apple_chip_name()
        device_info["is_accelerated"] = True
        
        # Optimize Environment for MPS
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        
        logger.info(f"✓ Apple Silicon detected: {device_info['name']}")
        logger.info(f"  Using Metal Performance Shaders (MPS) for GPU acceleration")
        
    else:
        logger.warning("⚠ WARNING: No GPU acceleration available!")
        logger.warning("  Processing will be VERY slow. Consider using a machine with CUDA or Apple Silicon.")
    
    return device_info

def get_demucs_settings(device_info):
    """
    Get optimal Demucs settings based on detected hardware.
    """
    device = device_info["device"]
    settings = {
        "model": config.DEFAULT_MODEL,
        "segment": None,
        "shifts": 2,
        "overlap": 0.25,
        "float32": False,
        "device_flag": f"-d {device}"
    }

    if device == "cuda":
        vram = device_info.get("memory_gb", 8.0)
        if vram >= 16:
            settings.update({
                "model": "htdemucs_ft",
                "segment": None, # No split
                "shifts": 5, # Max quality
                "float32": True,
                "overlap": 0.5
            })
        elif vram >= 8:
            settings.update({
                "model": "htdemucs_ft",
                "segment": 10,
                "shifts": 3,
                "float32": True
            })
        else:
            settings.update({
                "model": "htdemucs", # Faster model
                "segment": 5, # Save VRAM
                "shifts": 2,
                "float32": False # Use half precision
            })
            
    elif device == "mps":
        # Apple Silicon Optimization - Use 6-stem model for more granular extraction
        settings.update({
            "model": "htdemucs_6s",  # 6 stems: vocals, drums, bass, guitar, piano, other
            "segment": 7,  # Good balance for unified memory
            "shifts": 2,   # Higher quality separation
            "float32": True, # MPS doesn't like float16 mixed precision sometimes
            "overlap": 0.5  # More overlap for smoother transitions
        })
        
    else: # CPU
        settings.update({
            "model": "htdemucs",
            "segment": 5,
            "shifts": 1,
            "float32": False,
            "overlap": 0.1
        })
        
    return settings

def get_next_versioned_path(base_path: Path, name: str) -> Path:
    """
    Returns a Path that doesn't exist yet, appending _1, _2, etc.
    e.g. Stems/Song, Stems/Song_1, Stems/Song_2
    """
    candidate = base_path / name
    if not candidate.exists():
        return candidate
    
    i = 1
    while True:
        candidate = base_path / f"{name}_{i}"
        if not candidate.exists():
            return candidate
        i += 1

# --- Processing Functions ---

def separate_stems(file_path, settings):
    """Run Demucs separation."""
    song_name = file_path.stem
    output_dir = get_next_versioned_path(config.STEMS_DIR, song_name)
    
    logger.info(f"Separating stems for: {song_name} -> {output_dir.name}")
    
    logger.info(f"Separating stems for: {song_name}...")
    
    # Construct Command
    # Use python -m demucs to ensure we find the installed package even if not in PATH
    cmd = [
        sys.executable, "-m", "demucs",
    ]
    
    # Add device flag (split cleanly)
    if settings["device_flag"]:
        cmd.extend(settings["device_flag"].split())

    cmd.extend([
        "-n", settings["model"],
        "--out", str(config.STEMS_DIR),
        "--shifts", str(settings["shifts"]),
        "--overlap", str(settings["overlap"]),
    ])
    
    if settings["segment"]:
        cmd.extend(["--segment", str(settings["segment"])])
        
    if settings["float32"]:
        cmd.append("--float32")
        
    cmd.append(str(file_path))
    
    # Run Demucs
    try:
        # We assume demucs is in path. If running from venv, it should be.
        subprocess.run(cmd, check=True, capture_output=False) # Let it print progress to stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Demucs failed for {song_name}: {e}")
        return None
        
    # Organize Output
    # Demucs output structure: Stems/model_name/song_name
    # We want: Stems/song_name
    
    raw_output_path = config.STEMS_DIR / settings["model"] / song_name
    
    if raw_output_path.exists():
        # Even if output_dir exists (unlikely given get_next_versioned_path), we overwrite or move
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(raw_output_path), str(output_dir))
        
        # Cleanup
        try:
            (config.STEMS_DIR / settings["model"]).rmdir()
        except:
            pass
            
        return output_dir
    else:
        logger.error(f"Could not find expected output at {raw_output_path}")
        return None

def stem_has_content(wav_path: Path, silence_threshold_db: float = -40.0) -> bool:
    """
    Analyze a stem to determine if it has meaningful audio content.
    Returns False if the stem is mostly silence (e.g., guitar stem with no guitar).
    
    Args:
        wav_path: Path to the WAV file
        silence_threshold_db: dB threshold below which audio is considered silence
    """
    try:
        import librosa
        import numpy as np
        
        # Load audio
        y, sr = librosa.load(str(wav_path), sr=None, mono=True)
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Calculate percentage of frames above threshold
        active_frames = np.sum(rms_db > silence_threshold_db)
        total_frames = len(rms_db)
        
        if total_frames == 0:
            return False
            
        active_ratio = active_frames / total_frames
        
        # If less than 5% of frames have content, consider it empty
        has_content = active_ratio > 0.05
        
        if not has_content:
            logger.info(f"  ⏭ Skipping {wav_path.stem} (no significant content detected)")
        
        return has_content
        
    except Exception as e:
        # If analysis fails, assume there's content to be safe
        logger.warning(f"Could not analyze {wav_path.name}: {e}")
        return True

def convert_melodic_to_midi(stems_dir, song_name):
    """Convert melodic stems (vocals, bass, etc.) to MIDI using Basic Pitch.
    Only converts stems that have meaningful audio content."""
    from basic_pitch.inference import predict_and_save
    from basic_pitch import ICASSP_2022_MODEL_PATH
    
    midi_song_dir = config.MIDI_DIR / song_name
    midi_song_dir.mkdir(parents=True, exist_ok=True)
    
    converted_count = 0
    
    for stem in config.MELODIC_STEMS:
        wav_path = stems_dir / f"{stem}.wav"
        if not wav_path.exists():
            continue
        
        # Check if stem has actual content (skip empty stems like guitar in synth tracks)
        if not stem_has_content(wav_path):
            continue
            
        # Check if MIDI already exists
        # Basic pitch outputs: {stem}_basic_pitch.mid
        expected_midi = midi_song_dir / f"{stem}_basic_pitch.mid"
        if expected_midi.exists():
             continue
             
        logger.info(f"Converting {stem} to MIDI...")
        
        try:
            predict_and_save(
                audio_path_list=[str(wav_path)],
                output_directory=str(midi_song_dir),
                save_midi=True,
                sonify_midi=False,
                save_model_outputs=False,
                save_notes=False,
                model_or_model_path=ICASSP_2022_MODEL_PATH,
            )
            logger.info(f"  ✓ Created {stem}_basic_pitch.mid")
            converted_count += 1
        except Exception as e:
            logger.error(f"Failed to convert {stem} to midi: {e}")
    
    if converted_count == 0:
        logger.info("  No melodic stems with content found for MIDI conversion")

def convert_drums_to_midi(stems_dir, song_name):
    """Convert drums to MIDI using BeatNet (if available/installed)."""
    try:
        from BeatNet.BeatNet import BeatNet
    except ImportError:
        # BeatNet optional
        return

    wav_path = stems_dir / "drums.wav"
    if not wav_path.exists():
        return
        
    midi_path = config.MIDI_DIR / song_name / "drums.mid"
    if midi_path.exists():
        return
        
    logger.info(f"Converting drums to MIDI (BeatNet)...")
    
    try:
        estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
        # Process returns beats array, we would need to convert this to MIDI file
        # This implementation requires mido or similar to write the file
        # Placeholder for actual BeatNet implementation details which can be complex
        # For now, we skip full implementation to verify basic connectivity
        pass 
    except Exception as e:
        logger.warning(f"BeatNet processing failed: {e}")

def main():
    print("\n🎸 Antigravity Stem & MIDI Extractor 🎹")
    print("=========================================")
    
    # 1. Setup
    config.SOURCE_DIR.mkdir(exist_ok=True)
    config.STEMS_DIR.mkdir(exist_ok=True)
    config.MIDI_DIR.mkdir(exist_ok=True)
    
    # 2. Hardware Detect
    device_info = get_optimal_device()
    settings = get_demucs_settings(device_info)
    
    print(f"\n⚙️  Processing Settings:")
    print(f"   Model: {settings['model']}")
    print(f"   Device: {settings['device_flag']}")
    print(f"   Shifts: {settings['shifts']} | Float32: {settings['float32']}")
    print("=========================================\n")
    
    # 3. Find Files
    # 3. Find Files
    files = []
    if config.SOURCE_DIR.exists():
        files = [
            f for f in config.SOURCE_DIR.iterdir() 
            if f.suffix.lower() in config.AUDIO_EXTENSIONS
        ]
    
    if not files:
        print("\n" + "="*50)
        print("🚦 READY TO START!")
        print("="*50)
        print(f"1. Open the '{config.SOURCE_DIR.name}' folder: {config.SOURCE_DIR}")
        print("2. Drop your audio files there (.mp3, .wav, .flac)")
        print("3. Run this script again")
        print("="*50 + "\n")
        return
        
    print(f"Found {len(files)} songs to process.\n")
    
    # 4. Processing Loop
    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Processing: {file_path.name}")
        
        # A. Separate
        stems_dir = separate_stems(file_path, settings)
        
        # B. MIDI Conversion
        # B. MIDI Conversion
        if stems_dir:
            # Pass the folder name of stems_dir to match Midi folder versioning
            convert_melodic_to_midi(stems_dir, stems_dir.name)
            convert_drums_to_midi(stems_dir, stems_dir.name)
            
    print("\n✨ All Done! Check Stems/ and Midi/ directories.")

if __name__ == "__main__":
    main()
