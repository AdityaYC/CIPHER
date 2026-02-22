# Run Drone (main app) with Drone2 features. Backend on 8000, frontend on 5173.
# From repo root. Opens backend in new window, then frontend here. Open http://localhost:5173
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# Ensure Node is on PATH (so npm works)
if (Test-Path "${env:ProgramFiles}\nodejs") { $env:Path = "${env:ProgramFiles}\nodejs;$env:Path" }

# Free port 8000
$conn = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($conn) { $conn | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Start-Sleep -Seconds 1 }

# Find Python: prefer the C:\env venv (has ultralytics), then real installs
$pythonExe = $null
foreach ($p in @(
    "C:\env\Scripts\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
    "$env:ProgramFiles\Python312\python.exe",
    "$env:ProgramFiles\Python311\python.exe"
)) {
    if (Test-Path $p) { $pythonExe = $p; break }
}
if (-not $pythonExe) {
    try {
        $pyOut = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($pyOut) { $pythonExe = $pyOut.Trim() }
    } catch { }
}
if (-not $pythonExe) {
    try {
        $ver = & python --version 2>&1
        if ($ver -notmatch "Microsoft Store|not found") { $pythonExe = "python" }
    } catch { }
}
if (-not $pythonExe) {
    Write-Host "Python not found. Install Python 3.12 from python.org (not the Store)." -ForegroundColor Red
    exit 1
}
# If we use a full path, add Python and Scripts to PATH for this session so backend and tools work
if ($pythonExe -match "\\") {
    $pyDir = Split-Path $pythonExe -Parent
    $scriptsDir = Join-Path $pyDir "Scripts"
    $env:Path = "$pyDir;$scriptsDir;$env:Path"
}

Write-Host "  Python: $pythonExe" -ForegroundColor Cyan

# Pre-flight: check YOLO ONNX model
$yoloModel = Join-Path $root "models\yolov8_det.onnx"
if (-not (Test-Path $yoloModel)) {
    Write-Host "  YOLO model missing — downloading..." -ForegroundColor Yellow
    $dlScript = Join-Path $root "scripts\download_yolo_onnx.py"
    if (Test-Path $dlScript) {
        & $pythonExe $dlScript 2>&1 | Out-Null
    } else {
        Write-Host "    (download script not found, backend will fall back to Ultralytics)" -ForegroundColor DarkYellow
    }
}
if (Test-Path $yoloModel) {
    Write-Host "  YOLO ONNX: $yoloModel" -ForegroundColor Cyan
} else {
    Write-Host "  YOLO ONNX: not found (will use Ultralytics CPU fallback)" -ForegroundColor DarkYellow
}

# Pre-flight: detect NPU availability
$npuInfo = & $pythonExe -c @"
try:
    import onnxruntime as ort
    provs = ort.get_available_providers()
    from pathlib import Path
    qnn = Path(ort.__file__).parent / 'capi' / 'QnnHtp.dll'
    if 'QNNExecutionProvider' in provs and qnn.exists():
        print('QNN NPU (Hexagon) available')
    else:
        print('CPU only (providers: ' + ', '.join(provs) + ')')
except ImportError:
    print('onnxruntime not installed')
except Exception as e:
    print(f'check failed: {e}')
"@ 2>&1
Write-Host "  NPU: $npuInfo" -ForegroundColor Cyan

# Drone backend (includes Drone2: laptop webcam, YOLO, advisory, /api/status, Agent tactical query, etc.)
# PYTHONPATH must be repo root so "backend" (vector_db, query_agent) can be imported for Agent tab queries
$env:PYTHONPATH = $root
$env:PHANTOM_HTTP_ONLY = "1"
Write-Host ""
Write-Host "Drone backend (with Drone2 features) starting on http://localhost:8000 ..."
Start-Process -FilePath $pythonExe -ArgumentList "-m", "uvicorn", "Drone.local_backend.app:app", "--host", "0.0.0.0", "--port", "8000" -WorkingDirectory $root

Start-Sleep -Seconds 4

Write-Host ""
Write-Host "  >>> Open in browser:  http://localhost:5173  <<<" -ForegroundColor Green
Write-Host "  (Cipher: Manual = webcam + YOLO; Agent, Replay.)" -ForegroundColor Gray
Write-Host "  Ctrl+C here stops the frontend; close the backend window to stop the server." -ForegroundColor Gray
Write-Host ""
Set-Location "$root\Drone\frontend"
npm run dev
