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

# --- Processing Functions ---

def separate_stems(file_path, settings):
    """Run Demucs separation."""
    song_name = file_path.stem
    output_dir = config.STEMS_DIR / song_name
    
    # Skip if done
    if output_dir.exists() and any(output_dir.iterdir()):
        logger.info(f"Skipping stem separation for {song_name} (output exists)")
        return output_dir
    
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

def convert_melodic_to_midi(stems_dir, song_name):
    """Convert melodic stems (vocals, bass, etc.) to MIDI using Basic Pitch."""
    from basic_pitch.inference import predict_and_save
    from basic_pitch import ICASSP_2022_MODEL_PATH
    
    midi_song_dir = config.MIDI_DIR / song_name
    midi_song_dir.mkdir(parents=True, exist_ok=True)
    
    for stem in config.MELODIC_STEMS:
        wav_path = stems_dir / f"{stem}.wav"
        if not wav_path.exists():
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
        except Exception as e:
            logger.error(f"Failed to convert {stem} to midi: {e}")

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
    files = [
        f for f in config.SOURCE_DIR.iterdir() 
        if f.suffix.lower() in config.AUDIO_EXTENSIONS
    ]
    
    if not files:
        print(f"❌ No audio files found in {config.SOURCE_DIR}")
        print("   Please add .mp3, .wav, or .flac files to the Source folder.")
        return
        
    print(f"Found {len(files)} songs to process.\n")
    
    # 4. Processing Loop
    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Processing: {file_path.name}")
        
        # A. Separate
        stems_dir = separate_stems(file_path, settings)
        
        # B. MIDI Conversion
        if stems_dir:
            convert_melodic_to_midi(stems_dir, file_path.stem)
            convert_drums_to_midi(stems_dir, file_path.stem)
            
    print("\n✨ All Done! Check Stems/ and Midi/ directories.")

if __name__ == "__main__":
    main()
