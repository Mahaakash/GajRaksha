import os
import sys
import time
import shutil
import subprocess
import urllib.request
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
PYTHON_EXE = BASE_DIR / "backend" / "venv" / "Scripts" / "python.exe"
CHROME_PATH = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", str(BASE_DIR)))

SCREENSHOT_MODAL = BASE_DIR / "modal_screenshot.png"
SCREENSHOT_TRACKS = BASE_DIR / "tracks_screenshot.png"

def is_server_ready(url, timeout=2.0):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HealthCheck/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False

print("=" * 70)
print("GajRaksha - Capturing Data Transparency Modal & Elephant Tracks Screenshots")
print("=" * 70)

# 1. Start FastAPI Backend if not running
if not is_server_ready("http://127.0.0.1:8000/model-info"):
    print("[1/4] Starting FastAPI Backend on http://127.0.0.1:8000...")
    backend_cmd = [
        str(PYTHON_EXE),
        "-m", "uvicorn",
        "backend.main:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    ]
    subprocess.Popen(
        backend_cmd,
        cwd=str(BASE_DIR),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    )
else:
    print("[1/4] FastAPI Backend is already running on http://127.0.0.1:8000.")

# 2. Start Vite Frontend if not running
if not is_server_ready("http://127.0.0.1:5173"):
    print("[2/4] Starting React + Vite Frontend on http://127.0.0.1:5173...")
    frontend_cmd = ["cmd.exe", "/c", "npm", "run", "dev"]
    subprocess.Popen(
        frontend_cmd,
        cwd=str(FRONTEND_DIR),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    )
else:
    print("[2/4] Vite Frontend is already running on http://127.0.0.1:5173.")

# 3. Wait for both servers to be fully responsive
print("Waiting for both servers to be healthy...")
backend_online = False
frontend_online = False

for attempt in range(25):
    if not backend_online and is_server_ready("http://127.0.0.1:8000/model-info"):
        backend_online = True
        print(f"  ✓ Backend responsive after {attempt + 1}s")

    if not frontend_online and is_server_ready("http://127.0.0.1:5173"):
        frontend_online = True
        print(f"  ✓ Frontend responsive after {attempt + 1}s")

    if backend_online and frontend_online:
        break
    time.sleep(1.0)

if not (backend_online and frontend_online):
    print(f"ERROR: Servers failed to start. Backend: {backend_online}, Frontend: {frontend_online}")
    sys.exit(1)

if not CHROME_PATH.exists():
    print(f"ERROR: Chrome not found at {CHROME_PATH}")
    sys.exit(1)

# 4. Capture Screenshot 1: "About this data" Modal
print("\n[3/4] Capturing 'About this data' Modal Screenshot...")
cmd_modal = [
    str(CHROME_PATH),
    "--headless=new",
    "--disable-gpu",
    "--window-size=1440,900",
    "--virtual-time-budget=6000",
    f"--screenshot={str(SCREENSHOT_MODAL)}",
    "http://127.0.0.1:5173/?modal=true"
]
res_modal = subprocess.run(cmd_modal, capture_output=True, text=True, timeout=30)
if res_modal.returncode == 0 and SCREENSHOT_MODAL.exists() and SCREENSHOT_MODAL.stat().st_size > 0:
    print(f"  ✓ Modal Screenshot saved: {SCREENSHOT_MODAL} ({SCREENSHOT_MODAL.stat().st_size:,} bytes)")
    if ARTIFACT_DIR.exists():
        shutil.copy2(SCREENSHOT_MODAL, ARTIFACT_DIR / "modal_screenshot.png")
        print("  ✓ Copied Modal screenshot to artifact directory.")
else:
    print(f"  ✗ Error capturing modal screenshot: {res_modal.stderr}")

# 5. Capture Screenshot 2: Real Elephant Tracks Layer (Kruger)
print("\n[4/4] Capturing Real Elephant Tracks Layer (Kruger) Screenshot...")
cmd_tracks = [
    str(CHROME_PATH),
    "--headless=new",
    "--disable-gpu",
    "--window-size=1440,900",
    "--virtual-time-budget=7000",
    f"--screenshot={str(SCREENSHOT_TRACKS)}",
    "http://127.0.0.1:5173/?tracks=true"
]
res_tracks = subprocess.run(cmd_tracks, capture_output=True, text=True, timeout=30)
if res_tracks.returncode == 0 and SCREENSHOT_TRACKS.exists() and SCREENSHOT_TRACKS.stat().st_size > 0:
    print(f"  ✓ Tracks Layer Screenshot saved: {SCREENSHOT_TRACKS} ({SCREENSHOT_TRACKS.stat().st_size:,} bytes)")
    if ARTIFACT_DIR.exists():
        shutil.copy2(SCREENSHOT_TRACKS, ARTIFACT_DIR / "tracks_screenshot.png")
        print("  ✓ Copied Tracks Layer screenshot to artifact directory.")
else:
    print(f"  ✗ Error capturing tracks screenshot: {res_tracks.stderr}")

# Open browser to interactive app
print("\nOpening interactive GajRaksha app in default browser...")
webbrowser.open("http://127.0.0.1:5173")

print("=" * 70)
print("Transparency modal & tracks verification complete!")
print("=" * 70)
