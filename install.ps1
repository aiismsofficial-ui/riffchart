# RiffChart installer for Windows 10 / 11. Double-click install.bat (it runs this script).
#
# Everything goes into this folder: uv (the Python tool), Python 3.12, the app's packages (.venv) and the
# models. Nothing is installed system-wide and nothing needs admin rights. To uninstall, delete this folder and
# the RiffChart shortcuts (Desktop and Start menu).
#
#   install.ps1 [-Cpu | -Cuda] [-NoShortcut] [-NoStart] [-Yes]
#     -Cpu / -Cuda  skip the question (PyTorch for any PC, or for NVIDIA graphics cards)
#     -Yes          accept the suggested answers (for unattended installs)

param([switch]$Cpu, [switch]$Cuda, [switch]$NoShortcut, [switch]$NoStart, [switch]$Yes)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"            # Invoke-WebRequest is very slow with its progress bar
$Root = $PSScriptRoot
$UvVersion = "0.13.6"
$UvDir = Join-Path $Root "tools\uv"
$Uv = Join-Path $UvDir "uv.exe"
$MinNvidiaDriver = 580                              # CUDA 13 builds of PyTorch need driver 580 or newer

function Say($msg, $color = "Gray") { Write-Host $msg -ForegroundColor $color }
function Step($n, $msg) { Write-Host ""; Write-Host "[$n/5] $msg" -ForegroundColor Cyan }
function Fail($msg) {
    Write-Host ""; Write-Host "Installation stopped: $msg" -ForegroundColor Red
    Write-Host "Fix that and run install.bat again: finished steps are skipped." -ForegroundColor Yellow
    exit 1
}
function Ask($question, $default) {
    if ($Yes) { return $default }
    $a = Read-Host "$question"
    if ([string]::IsNullOrWhiteSpace($a)) { return $default }
    return $a.Trim()
}

Write-Host ""
Write-Host "  RiffChart installer" -ForegroundColor Yellow
Say "  Installs into: $Root"
if ($Root -match "[^\x00-\x7F]") {
    Say "  Note: this folder's path has non-English letters; if something fails, move the folder to e.g. C:\RiffChart." "Yellow"
}

# ---------------------------------------------------------------- 1. uv
Step 1 "uv (installs Python and the app's packages)"
if (Test-Path $Uv) {
    Say "  already here"
} else {
    $zip = Join-Path $env:TEMP "uv-$UvVersion.zip"
    $url = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
    Say "  downloading uv $UvVersion"
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    } catch { Fail "couldn't download uv ($($_.Exception.Message)). Check the internet connection." }
    New-Item -ItemType Directory -Force -Path $UvDir | Out-Null
    Expand-Archive -Path $zip -DestinationPath $UvDir -Force
    Remove-Item $zip -ErrorAction SilentlyContinue
    if (-not (Test-Path $Uv)) { Fail "uv.exe wasn't in the download" }
}
# keep uv's Python inside this folder; uv's download cache stays in the user's AppData (shared, reusable)
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root "tools\python"
$env:UV_PYTHON_PREFERENCE = "only-managed"
$env:UV_NO_PROGRESS = "1"

# ---------------------------------------------------------------- 2. CPU or NVIDIA
Step 2 "Graphics card"
$gpus = @()
try { $gpus = @(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }) } catch { }
$nvidia = $gpus | Where-Object { $_ -match "NVIDIA" } | Select-Object -First 1
$driverOk = $false
if ($nvidia) {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        try {
            $ver = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1).Trim()
            $driverOk = ([double]($ver.Split(".")[0])) -ge $MinNvidiaDriver
            Say "  found $nvidia (driver $ver)"
        } catch { Say "  found $nvidia (driver version unknown)" }
    } else { Say "  found $nvidia (nvidia-smi not found)" }
} else {
    Say "  no NVIDIA card found ($((@($gpus) -join ', ')))"
}
if ($Cpu) { $flavour = "cpu" }
elseif ($Cuda) { $flavour = "cuda" }
elseif ($nvidia -and $driverOk) {
    $a = Ask "  Use the NVIDIA card (about twice as fast, ~3 GB download)? [Y/n]" "y"
    $flavour = if ($a -match "^[nN]") { "cpu" } else { "cuda" }
} elseif ($nvidia) {
    Say "  The NVIDIA driver is older than $MinNvidiaDriver, which RiffChart's GPU build needs." "Yellow"
    Say "  Installing for the CPU now. Update the driver (nvidia.com/drivers) and run install.bat -Cuda to switch." "Yellow"
    $flavour = "cpu"
} else {
    $flavour = "cpu"
}
Say "  installing for: $(if ($flavour -eq 'cuda') { 'NVIDIA graphics card' } else { 'CPU (works on any PC)' })" "Green"

# ---------------------------------------------------------------- 3. Python + packages
Step 3 "Python 3.12 and the app's packages (several minutes the first time)"
Push-Location $Root
try {
    & $Uv sync --frozen --python 3.12 --extra $flavour
    if ($LASTEXITCODE -ne 0) { Fail "installing the packages failed (see the messages above)" }
} finally { Pop-Location }
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { Fail "the app's Python wasn't created" }
Set-Content -Path (Join-Path $Root "tools\flavour.txt") -Value $flavour -Encoding ascii

# ---------------------------------------------------------------- 4. models
Step 4 "Models"
& $Py (Join-Path $Root "scripts\get_models.py")
$modelsOk = ($LASTEXITCODE -eq 0)

# ---------------------------------------------------------------- 5. shortcuts
Step 5 "Shortcuts"
$launcher = Join-Path $Root "RiffChart.bat"
$icon = Join-Path $Root "assets\riffchart.ico"
if ($NoShortcut) {
    Say "  skipped (start RiffChart with RiffChart.bat)"
} else {
    $shell = New-Object -ComObject WScript.Shell
    $places = @([Environment]::GetFolderPath("Desktop"), (Join-Path ([Environment]::GetFolderPath("StartMenu")) "Programs"))
    foreach ($dir in $places) {
        $lnk = $shell.CreateShortcut((Join-Path $dir "RiffChart.lnk"))
        $lnk.TargetPath = $launcher
        $lnk.WorkingDirectory = $Root
        $lnk.Description = "RiffChart: chart songs for Clone Hero and YARG"
        if (Test-Path $icon) { $lnk.IconLocation = $icon }
        $lnk.WindowStyle = 7                        # start minimised: the window only shows the app's log
        $lnk.Save()
    }
    Say "  RiffChart on the Desktop and in the Start menu"
}

Write-Host ""
if ($modelsOk) {
    Write-Host "RiffChart is installed." -ForegroundColor Green
} else {
    Write-Host "RiffChart is installed, but some model files are missing (see step 4)." -ForegroundColor Yellow
    Write-Host "The app opens, but charting songs needs them." -ForegroundColor Yellow
}
if ($NoStart) { exit 0 }
$a = Ask "Start RiffChart now? [Y/n]" "y"
if ($a -notmatch "^[nN]") { Start-Process -FilePath $launcher -WorkingDirectory $Root -WindowStyle Minimized }
