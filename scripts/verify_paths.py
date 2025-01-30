import os
from dotenv import load_dotenv

# ✅ Use absolute path for .env
dotenv_path = r"D:\Desktop\dermproject\dermdetect\.env"

# Check if .env exists
if not os.path.exists(dotenv_path):
    raise FileNotFoundError(f"❌ ERROR: .env file not found at {dotenv_path}")

# Load environment variables
load_dotenv(dotenv_path)

# Debugging: Confirm .env is loaded
print(f"✅ Loaded .env from: {dotenv_path}")

# Get paths from .env
MLFLOW_URI = os.getenv("APP_URI_MLFLOW")
ALL_IMAGES_PATH = os.getenv("ALL_IMAGES_PATH")
ALL_IMAGES_CSV_PATH = os.getenv("ALL_IMAGES_CSV_PATH")
SAMPLE_20_PATH = os.getenv("SAMPLE_20_PATH")
SAMPLE_20CSV_PATH = os.getenv("SAMPLE_20CSV_PATH")

# Check if variables are missing
missing_vars = [var for var in ["ALL_IMAGES_PATH", "ALL_IMAGES_CSV_PATH", "SAMPLE_20_PATH", "SAMPLE_20CSV_PATH"]
                if os.getenv(var) is None]

if missing_vars:
    raise ValueError(f"❌ ERROR: Missing environment variables: {', '.join(missing_vars)}. Check your .env file.")

# ✅ Convert relative paths to absolute paths based on dermproject directory
PROJECT_ROOT = r"D:\Desktop\dermproject"  # Absolute path to project root

ALL_IMAGES_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, ALL_IMAGES_PATH))
ALL_IMAGES_CSV_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, ALL_IMAGES_CSV_PATH))
SAMPLE_20_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, SAMPLE_20_PATH))
SAMPLE_20CSV_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, SAMPLE_20CSV_PATH))

# ✅ Print resolved paths and check existence
print("✅ Resolved Paths:")
print(f"ALL_IMAGES_PATH: {ALL_IMAGES_PATH} (Exists: {os.path.exists(ALL_IMAGES_PATH)})")
print(f"ALL_IMAGES_CSV_PATH: {ALL_IMAGES_CSV_PATH} (Exists: {os.path.exists(ALL_IMAGES_CSV_PATH)})")
print(f"SAMPLE_20_PATH: {SAMPLE_20_PATH} (Exists: {os.path.exists(SAMPLE_20_PATH)})")
print(f"SAMPLE_20CSV_PATH: {SAMPLE_20CSV_PATH} (Exists: {os.path.exists(SAMPLE_20CSV_PATH)})")
