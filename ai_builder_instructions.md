# AI Builder Instructions: Stem Extractor & MIDI Converter

> **Last Updated**: December 2024  
> **Optimized For**: CUDA (NVIDIA) + Apple Silicon (M1/M2/M3) with MPS  
> **State-of-the-Art Tools**: Demucs v4 (Meta AI), Basic Pitch (Spotify), BeatNet

---

## 1. Project Overview

Build a local Python application for **music remixing workflows** that:

1. Ingests audio files from `Source/`
2. Separates into stems (vocals, drums, bass, guitar, piano, other) using **Demucs v4**
3. Converts melodic stems to **MIDI** using **Basic Pitch** (optionally drums via **BeatNet**)
4. **Automatically detects and uses the fastest available hardware** (CUDA > MPS > CPU)
5. Outputs clean, artifact-free stems and accurate MIDI for DAW import

---

## 2. Technology Stack (2024-2025 Best-in-Class)

### Stem Separation: Demucs v4 (Meta AI)
Demucs is the current industry leader for open-source stem separation. Key advantages:
- Preserves **full 22kHz frequency range** (CD quality), unlike Spleeter which cuts off at 11-16kHz
- Superior handling of phase information and transients
- Multiple model options for quality vs. speed trade-off

**Recommended Models:**
| Model | Stems | Quality | Speed | Use Case |
|-------|-------|---------|-------|----------|
| `htdemucs_ft` | 4 (vocals, drums, bass, other) | **Best** | Slower | Production quality |
| `htdemucs` | 4 | Very Good | Fast | Balanced default |
| `htdemucs_6s` | 6 (+ guitar, piano) | Good | Medium | When separate guitar/piano needed |

### Audio-to-MIDI Conversion

**Primary: Basic Pitch (Spotify)**
- Best for **melodic content**: vocals, bass, guitar, piano, other instruments
- Lightweight, polyphonic, and instrument-agnostic
- Outputs pitch bends for expressive performances
- Available as Python library and npm package

**Secondary: NeuralNote (Optional)**
- VST3/AU plugin wrapping Basic Pitch algorithm
- Real-time conversion capability
- DAW-friendly for live workflow integration

**For Drums: BeatNet (Optional)**
- Specialized drum transcription model
- Outputs MIDI drum events with accurate timing/velocity
- Complements Basic Pitch which doesn't handle drums well

### Core Stack
- **Language**: Python 3.10+ (required for latest Demucs)
- **Audio Processing**: `librosa`, `soundfile`, `pydub`
- **Acceleration**: CUDA (NVIDIA) or MPS (Apple Silicon) — MANDATORY for reasonable performance

---

## 3. Hardware Acceleration Configuration

### Device Priority (Fastest to Slowest)
1. **CUDA** (NVIDIA GPU) — 5-10x faster than CPU
2. **MPS** (Apple Silicon GPU) — 3-5x faster than CPU
3. **CPU** — Fallback only, NOT recommended

### Complete Device Detection Code

```python
import torch
import platform
import subprocess
import os

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
        device_info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        device_info["is_accelerated"] = True
        print(f"✓ CUDA GPU detected: {device_info['name']}")
        print(f"  VRAM: {device_info['memory_gb']:.1f} GB")
        
    # Check for Apple Silicon MPS
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device_info["device"] = "mps"
        device_info["name"] = get_apple_chip_name()
        device_info["is_accelerated"] = True
        print(f"✓ Apple Silicon detected: {device_info['name']}")
        print(f"  Using Metal Performance Shaders (MPS) for GPU acceleration")
        
    else:
        print("⚠ WARNING: No GPU acceleration available!")
        print("  Processing will be VERY slow. Consider using a machine with CUDA or Apple Silicon.")
    
    return device_info

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

def check_mps_availability():
    """
    Comprehensive MPS availability check for Apple Silicon.
    Returns dict with detailed status.
    """
    status = {
        "available": False,
        "built": False,
        "functional": False,
        "reason": None
    }
    
    # Check if MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            status["reason"] = "PyTorch not built with MPS support. Reinstall PyTorch."
        else:
            status["reason"] = "MPS not available. Requires macOS 12.3+ and M1/M2/M3 chip."
        return status
    
    status["available"] = True
    status["built"] = torch.backends.mps.is_built()
    
    # Test MPS functionality
    try:
        test_tensor = torch.zeros(1, device="mps")
        del test_tensor
        status["functional"] = True
    except Exception as e:
        status["reason"] = f"MPS test failed: {e}"
    
    return status
```

---

## 4. Platform-Specific Optimizations

### 🍎 **Apple Silicon (M1/M2/M3) — MPS Configuration**

```python
# CRITICAL: Environment variables for optimal MPS performance
import os

# Enable MPS fallback for unsupported operations (prevents crashes)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Optimize memory usage on unified memory architecture
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable memory caching for max available

def get_mps_optimal_settings():
    """
    Get optimal Demucs settings for Apple Silicon.
    M1/M2/M3 have unified memory but limited compared to dedicated GPUs.
    """
    return {
        # Use smaller segments to fit in unified memory
        "segment": 7,
        
        # float32 is more stable on MPS (float16 has issues)
        "float32": True,
        
        # Fewer shifts for speed (MPS is slower than CUDA)
        "shifts": 1,
        
        # Standard overlap
        "overlap": 0.25,
        
        # Explicit device flag for Demucs CLI
        "device_flag": "-d mps"
    }

# Apple Silicon specific Demucs command
DEMUCS_CMD_MPS = """
demucs -d mps -n htdemucs --segment 7 --shifts 1 --overlap 0.25 --out Stems "{input_file}"
"""
```

**Apple Silicon Best Practices:**
| Setting | Value | Reason |
|---------|-------|--------|
| `--device` / `-d` | `mps` | Forces MPS GPU usage |
| `--segment` | 5-7 | Smaller segments for unified memory |
| `--shifts` | 1-2 | Fewer shifts (MPS is slower) |
| `--float32` | Yes | float16 unstable on MPS |
| `--overlap` | 0.25 | Standard, no change needed |

**Known MPS Limitations:**
- Some complex number operations fall back to CPU automatically
- Slightly slower than equivalent NVIDIA GPU
- Use `PYTORCH_ENABLE_MPS_FALLBACK=1` to prevent crashes

---

### 🖥️ **NVIDIA CUDA — Maximum Performance Configuration**

```python
def get_cuda_optimal_settings():
    """
    Get optimal Demucs settings for NVIDIA GPUs.
    Adjust based on available VRAM.
    """
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if vram_gb >= 16:
        # High-end GPU (RTX 3090, 4090, A100, etc.)
        return {
            "segment": None,  # No limit
            "float32": True,  # Best quality
            "shifts": 5,      # Maximum quality
            "overlap": 0.5,
            "device_flag": "-d cuda"
        }
    elif vram_gb >= 8:
        # Mid-range GPU (RTX 3060, 3070, 4060, etc.)
        return {
            "segment": 10,
            "float32": True,
            "shifts": 3,
            "overlap": 0.25,
            "device_flag": "-d cuda"
        }
    else:
        # Low VRAM GPU (GTX 1060, RTX 3050, etc.)
        return {
            "segment": 5,
            "float32": False,  # Use float16 to save VRAM
            "shifts": 2,
            "overlap": 0.25,
            "device_flag": "-d cuda"
        }

# High VRAM CUDA command (16GB+)
DEMUCS_CMD_CUDA_HIGH = """
demucs -d cuda -n htdemucs_ft --shifts 5 --overlap 0.5 --float32 --out Stems "{input_file}"
"""

# Low VRAM CUDA command (<8GB)
DEMUCS_CMD_CUDA_LOW = """
demucs -d cuda -n htdemucs --segment 5 --shifts 2 --overlap 0.25 --out Stems "{input_file}"
"""
```

**CUDA Performance Tips:**
| VRAM | Model | Segment | Shifts | Quality |
|------|-------|---------|--------|---------|
| 16GB+ | `htdemucs_ft` | None | 5 | Maximum |
| 8-16GB | `htdemucs_ft` | 10 | 3 | High |
| 4-8GB | `htdemucs` | 7 | 2 | Balanced |
| <4GB | `htdemucs` | 5 | 1 | Fast |

---

### 💻 **CPU Fallback (Not Recommended)**

```python
def get_cpu_settings():
    """
    CPU settings - use only if no GPU available.
    Optimized for speed at the cost of quality.
    """
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    
    return {
        "segment": 5,
        "float32": False,
        "shifts": 1,      # Minimum for speed
        "overlap": 0.1,   # Less overlap for speed
        "jobs": min(4, num_cores),  # Parallel jobs
        "device_flag": "-d cpu"
    }

# CPU command (slowest, avoid if possible)
DEMUCS_CMD_CPU = """
demucs -d cpu -n htdemucs --segment 5 --shifts 1 --overlap 0.1 -j 4 --out Stems "{input_file}"
"""
```

---

## 5. Unified Processing Configuration

### Auto-Detect and Configure

```python
def get_processing_config():
    """
    Automatically detect hardware and return optimal configuration.
    """
    device_info = get_optimal_device()
    
    if device_info["device"] == "cuda":
        settings = get_cuda_optimal_settings()
        speed_multiplier = 8.0  # Approximate vs CPU
    elif device_info["device"] == "mps":
        settings = get_mps_optimal_settings()
        speed_multiplier = 4.0
    else:
        settings = get_cpu_settings()
        speed_multiplier = 1.0
    
    return {
        "device": device_info,
        "settings": settings,
        "estimated_speed": speed_multiplier
    }

def build_demucs_command(input_file: str, config: dict) -> str:
    """Build the optimal Demucs command based on detected hardware."""
    settings = config["settings"]
    
    cmd_parts = [
        "demucs",
        settings["device_flag"],
        "-n", "htdemucs" if settings.get("shifts", 2) <= 2 else "htdemucs_ft",
    ]
    
    if settings.get("segment"):
        cmd_parts.extend(["--segment", str(settings["segment"])])
    
    cmd_parts.extend([
        "--shifts", str(settings.get("shifts", 2)),
        "--overlap", str(settings.get("overlap", 0.25)),
    ])
    
    if settings.get("float32"):
        cmd_parts.append("--float32")
    
    cmd_parts.extend(["--out", "Stems", f'"{input_file}"'])
    
    return " ".join(cmd_parts)
```

---

## 6. Speed Comparison Reference

| Platform | Approx. Speed (3-min song) | Relative |
|----------|---------------------------|----------|
| RTX 4090 (CUDA) | ~15 seconds | 20x |
| RTX 3080 (CUDA) | ~25 seconds | 12x |
| RTX 3060 (CUDA) | ~45 seconds | 7x |
| M3 Max (MPS) | ~50 seconds | 6x |
| M2 Pro (MPS) | ~70 seconds | 4.5x |
| M1 (MPS) | ~90 seconds | 3.5x |
| Intel i9 (CPU) | ~5 minutes | 1x |

> **Note**: Times are approximate and vary based on model, settings, and song complexity.

---

## 7. Directory Structure

```
/ProjectRoot
  ├── Source/           (Input: mp3, wav, flac, m4a, ogg, aiff)
  ├── Stems/            (Output: Stems/SongName/vocals.wav, drums.wav, etc.)
  ├── Midi/             (Output: Midi/SongName/vocals.mid, bass.mid, etc.)
  ├── main.py           (Entry point)
  ├── config.py         (Configuration options)
  └── requirements.txt
```

---

## 8. MIDI Conversion (Basic Pitch)

```python
from basic_pitch.inference import predict_and_save, predict
import os

# Target melodic stems (NOT drums - Basic Pitch doesn't work well on drums)
MELODIC_STEMS = ['vocals', 'bass', 'guitar', 'piano', 'other']

def convert_stems_to_midi(song_name: str, stems_dir: str, midi_dir: str):
    """Convert all melodic stems to MIDI files."""
    for stem_name in MELODIC_STEMS:
        stem_path = os.path.join(stems_dir, song_name, f"{stem_name}.wav")
        if not os.path.exists(stem_path):
            continue
        
        output_dir = os.path.join(midi_dir, song_name)
        os.makedirs(output_dir, exist_ok=True)
        
        midi_path = os.path.join(output_dir, f"{stem_name}.mid")
        if os.path.exists(midi_path):
            print(f"  Skipping {stem_name}.mid (already exists)")
            continue
        
        print(f"  Converting {stem_name} to MIDI...")
        predict_and_save(
            audio_path_list=[stem_path],
            output_directory=output_dir,
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False
        )
```

---

## 9. Requirements & Installation

### requirements.txt
```
demucs>=4.0.0
basic-pitch>=0.3.0
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
librosa>=0.10.0
soundfile>=0.12.0
pydub>=0.25.0
tqdm>=4.65.0
```

### Installation Commands

**For Apple Silicon (M1/M2/M3):**
```bash
# Use native arm64 Python (NOT Rosetta)
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with MPS support (comes by default on arm64)
pip install torch torchaudio

# Verify MPS is available
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Install other dependencies
pip install -r requirements.txt
```

**For NVIDIA CUDA:**
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install PyTorch with CUDA (adjust cu121 to your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Install other dependencies
pip install -r requirements.txt
```

---

## 10. Implementation Checklist

1. ☐ Set up Python 3.10+ virtual environment (native arm64 on Mac)
2. ☐ Install PyTorch with CUDA or MPS support
3. ☐ Verify hardware acceleration: `python -c "import torch; print(torch.cuda.is_available(), torch.backends.mps.is_available())"`
4. ☐ Install all dependencies from requirements.txt
5. ☐ Create directory structure (Source, Stems, Midi)
6. ☐ Implement auto-device detection (CUDA > MPS > CPU)
7. ☐ Implement Demucs stem separation with device-specific settings
8. ☐ Implement Basic Pitch MIDI conversion for melodic stems
9. ☐ Add skip logic for already-processed files
10. ☐ Add comprehensive error handling
11. ☐ Add progress logging with tqdm
12. ☐ (Optional) Add BeatNet drum-to-MIDI conversion
