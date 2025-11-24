# HandPose Drone Control Demo

This demo is a desktop Python app built with `pywebview` that performs simple hand-pose estimation (via MediaPipe) and sends high-level commands to a frontend drone simulation.

What it includes
- `app.py` — Python backend that captures webcam frames, runs MediaPipe Hands, classifies simple poses and sends commands to the frontend.
- `web/index.html` — Frontend simulation (canvas) that receives commands and moves a simulated drone.
- `requirements.txt` — Python dependencies.

Quick start (Windows PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
python app.py
```

Notes
- If you don't have a webcam or MediaPipe installation fails, install dependencies manually and you can still inspect the frontend in a browser by opening `web/index.html`.
- The pose classifier is a simple heuristic for demo purposes — you may want to replace it with a trained model for production use.

Next steps (suggested)
- Improve pose classification or use a small ML model.
- Add a connection to a real drone SDK (e.g., DJI, MAVLink) replacing the frontend call.
- Add logging and a UI display of the camera feed.
