from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.absolute()
SOURCE_DIR = BASE_DIR / "Source"
STEMS_DIR = BASE_DIR / "Stems"
MIDI_DIR = BASE_DIR / "Midi"

# Supported Input Formats
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aiff', '.wma'}

# Demucs Defaults - Using 6-stem model for granular extraction
DEFAULT_MODEL = "htdemucs_6s"  # 6 stems: vocals, drums, bass, guitar, piano, other

# Stems to convert to MIDI (Basic Pitch)
MELODIC_STEMS = ['vocals', 'bass', 'guitar', 'piano', 'other']

# Basic Pitch Parameters
BASIC_PITCH_PARAMS = {
    "onset_threshold": 0.5, 
    "frame_threshold": 0.3,
    "minimum_note_length": 58,
    "minimum_frequency": 32,
    "maximum_frequency": 2000
}
