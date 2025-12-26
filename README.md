# Antigravity Stem & MIDI Extractor

A high-performance Python application for separating audio tracks into stems and converting them to MIDI files. Optimized for **Apple Silicon (M1/M2/M3)** and **NVIDIA CUDA** GPUs.

## Features

- **6-Stem Separation**: Extracts vocals, drums, bass, guitar, piano, and other instruments using Meta's Demucs v4
- **Audio-to-MIDI Conversion**: Converts melodic stems to MIDI files using Spotify's Basic Pitch
- **Hardware Acceleration**: Auto-detects and uses CUDA (NVIDIA) or MPS (Apple Silicon) for fast processing
- **Smart Caching**: Skips already-processed files on re-runs

## Output

### Stems (WAV files)
- `vocals.wav` — Isolated vocals
- `drums.wav` — Drums & percussion
- `bass.wav` — Bass instruments
- `guitar.wav` — Guitar separation
- `piano.wav` — Piano/keys separation
- `other.wav` — Everything else

### MIDI files
- `vocals_basic_pitch.mid`
- `bass_basic_pitch.mid`
- `guitar_basic_pitch.mid`
- `piano_basic_pitch.mid`
- `other_basic_pitch.mid`

## Installation

```bash
# Clone the repository
git clone https://github.com/alex-cognichat/Antigravity-Stem-Midi-Extract-from-mp3.git
cd Antigravity-Stem-Midi-Extract-from-mp3

# Install dependencies
pip install -r requirements.txt
```

### For Apple Silicon (M1/M2/M3)
PyTorch with MPS support is included by default.

### For NVIDIA CUDA
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

1. Place your audio files (`.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`, `.aiff`) in the `Source/` folder
2. Run the extractor:

```bash
python3 main.py
```

3. Find your outputs in:
   - `Stems/{SongName}/` — Separated audio stems
   - `Midi/{SongName}/` — MIDI transcriptions

## Performance

| Platform | Approx. Time (3-min song) |
|----------|---------------------------|
| RTX 4090 (CUDA) | ~15 seconds |
| RTX 3080 (CUDA) | ~25 seconds |
| M3 Max (MPS) | ~50 seconds |
| M1 (MPS) | ~90 seconds |
| CPU only | ~5 minutes |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- macOS 12.3+ (for Apple Silicon) or CUDA-capable GPU

## License

MIT
