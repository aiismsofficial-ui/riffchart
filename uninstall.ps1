# RiffChart uninstaller. Double-click uninstall.bat (it runs this script).
#
# Removes what install.bat added: the shortcuts, the app's Python and packages (.venv), uv and Python
# (tools), the model files and caches. Your charted songs are kept unless you say otherwise, and the copies
# in your Clone Hero songs folder are never touched. Clone Hero's own settings are left as they are.
#
#   uninstall.ps1 [-Yes]      -Yes: no questions (keeps your songs, keeps uv's download cache)

param([switch]$Yes)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

function Say($msg, $color = "Gray") { Write-Host $msg -ForegroundColor $color }
function Ask($question) {
    if ($Yes) { return $false }
    $a = Read-Host "$question [y/N]"
    return ($a -match "^[yY]")
}
function Remove-Dir($rel) {
    $p = Join-Path $Root $rel
    if (Test-Path $p) {
        Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction SilentlyContinue
        if (Test-Path $p) { Say "  couldn't remove all of $rel (a file may be in use)" "Yellow" } else { Say "  removed $rel" }
    }
}

Write-Host ""
Write-Host "  RiffChart uninstaller" -ForegroundColor Yellow
Say "  Folder: $Root"
if (-not ((Test-Path (Join-Path $Root "install.ps1")) -and (Test-Path (Join-Path $Root "src\riffchart")))) {
    Write-Host "This doesn't look like a RiffChart folder; nothing was changed." -ForegroundColor Red
    exit 1
}
if (Test-Path (Join-Path $Root ".git")) {
    Write-Host "This is a development copy (it has .git); the uninstaller only runs on installed copies." -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------- 1. stop RiffChart if it's running from here
$venv = Join-Path $Root ".venv"
$running = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($venv, [StringComparison]::OrdinalIgnoreCase) })
if ($running.Count) {
    Say "  RiffChart is running; closing it." "Yellow"
    $running | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

# ---------------------------------------------------------------- 2. shortcuts (only ours: they point into this folder)
$shell = New-Object -ComObject WScript.Shell
$places = @([Environment]::GetFolderPath("Desktop"), (Join-Path ([Environment]::GetFolderPath("StartMenu")) "Programs"))
foreach ($dir in $places) {
    $lnk = Join-Path $dir "RiffChart.lnk"
    if (Test-Path $lnk) {
        $target = $shell.CreateShortcut($lnk).TargetPath
        if ($target -and $target.StartsWith($Root, [StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $lnk -Force
            Say "  removed the shortcut in $dir"
        }
    }
}

# ---------------------------------------------------------------- 3. uv's download cache (shared with other uv users)
$uv = Join-Path $Root "tools\uv\uv.exe"
if ((Test-Path $uv) -and (Ask "  Also clear uv's download cache in AppData (frees ~2-4 GB; only matters if you use uv for anything else)?")) {
    & $uv cache clean
}

# ---------------------------------------------------------------- 4. the app itself
Remove-Dir ".venv"
Remove-Dir "tools"
Remove-Dir "data\checkpoints"
Remove-Dir "data\features"
Remove-Dir "data\ui\cache"
Remove-Dir "data\ui\uploads"
Remove-Dir "logs"
Get-ChildItem -Path $Root -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# ---------------------------------------------------------------- 5. your songs
$songs = Join-Path $Root "data\ui"
$n = 0
if (Test-Path (Join-Path $songs "songs")) { $n = @(Get-ChildItem -Path (Join-Path $songs "songs") -Directory).Count }
if (Test-Path $songs) {
    if ($n -gt 0) {
        Say ""
        $what = if ($n -eq 1) { "1 charted song" } else { "$n charted songs" }
        Say "  Your $what (with your edits and highway designs) are in data\ui." "Cyan"
        Say "  The copies you exported to Clone Hero's songs folder are separate and stay either way." "Cyan"
    }
    if (Ask "  Delete the charted songs in data\ui too?") { Remove-Dir "data\ui" } else { Say "  kept data\ui" }
}

Write-Host ""
Write-Host "RiffChart is uninstalled." -ForegroundColor Green
if (Test-Path $songs) {
    Say "What's left in this folder is the source code and your songs (data\ui). Delete the folder whenever you like."
} else {
    Say "What's left in this folder is the source code. Delete the folder to finish."
}
Say "Clone Hero still lists your RiffChart songs folder; remove it in Clone Hero's settings if you want."
