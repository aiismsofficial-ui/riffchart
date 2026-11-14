; RiffChart Setup (Inno Setup 6). Built by scripts/build_setup.py, which exports the public copy of the app and
; passes /DSrc=<that folder> /DAppVersion=<version> /DOut=<output folder>.
;
; What Setup does: copies the app (a few MB of code) into the user's own Programs folder (no admin rights),
; then runs install.bat in a window of its own, which downloads Python, the app's packages and the AI models
; into that folder (several minutes, needs the internet; an NVIDIA card is used when its driver is new enough).
; Start menu and desktop shortcuts start RiffChart.bat; Windows' Installed apps lists RiffChart with an
; uninstaller, which runs uninstall.ps1 -Yes (removes Python, packages, models and caches) and keeps the
; charted songs in data\ (and never touches the copies in Clone Hero's songs folder).
;
; Not code-signed: Windows SmartScreen shows "Windows protected your PC" until the file has a download
; reputation (More info > Run anyway). See the README.

#ifndef Src
  #error Pass /DSrc=<exported app folder> (scripts/build_setup.py does)
#endif
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef Out
  #define Out "."
#endif

[Setup]
AppId={{6B0E5C7A-2F4D-4C2B-9D7E-1A3F5B8C9D21}
AppName=RiffChart
AppVersion={#AppVersion}
AppVerName=RiffChart {#AppVersion}
AppPublisher=RiffChart
AppPublisherURL=https://github.com/aiismsofficial-ui/riffchart
AppSupportURL=https://github.com/aiismsofficial-ui/riffchart/issues
AppUpdatesURL=https://github.com/aiismsofficial-ui/riffchart/releases
DefaultDirName={autopf}\RiffChart
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
; 64-bit mode: [Run] then starts the 64-bit PowerShell (the 32-bit one can't see nvidia-smi in System32, and uv
; fails to link its Python folder from it)
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Inno Setup 6.5+ turns on Windows' RedirectionGuard for itself and the programs it runs, which blocks the
; folder junction uv makes for Python (os error 448). Setup runs without admin rights in the user's own
; folder, where that protection guards nothing.
RedirectionGuard=no
OutputDir={#Out}
OutputBaseFilename=RiffChart-Setup-{#AppVersion}
SetupIconFile={#Src}\assets\riffchart.ico
UninstallDisplayIcon={app}\assets\riffchart.ico
UninstallDisplayName=RiffChart
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; the install folder holds the app's own Python and models: a path with only English letters is safest
AllowUNCPath=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Put RiffChart on the desktop"; GroupDescription: "Shortcuts:"

[Files]
Source: "{#Src}\*"; Excludes: ".git\*,.github\*,dist\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion createallsubdirs

[Icons]
Name: "{autoprograms}\RiffChart"; Filename: "{app}\RiffChart.bat"; WorkingDir: "{app}"; IconFilename: "{app}\assets\riffchart.ico"; Comment: "Chart songs for Clone Hero and YARG"; Flags: runminimized
Name: "{autodesktop}\RiffChart"; Filename: "{app}\RiffChart.bat"; WorkingDir: "{app}"; IconFilename: "{app}\assets\riffchart.ico"; Comment: "Chart songs for Clone Hero and YARG"; Flags: runminimized; Tasks: desktopicon

[Run]
; the download step: a console window shows its progress (and stays open if something fails, so it can be read)
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install.ps1"" -Yes -NoShortcut -NoStart -PauseOnError"; WorkingDir: "{app}"; StatusMsg: "Downloading Python, the app's packages and the AI models (several minutes; the black window shows the progress)..."; Flags: waituntilterminated
Filename: "{app}\RiffChart.bat"; WorkingDir: "{app}"; Description: "Start RiffChart"; Flags: postinstall nowait skipifsilent runminimized

[UninstallRun]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\uninstall.ps1"" -Yes"; WorkingDir: "{app}"; RunOnceId: "RiffChartCleanup"; Flags: runhidden waituntilterminated

[Messages]
; RiffChart 1.0 came as a ZIP unzipped anywhere: updating it means installing into that same folder
SelectDirBrowseLabel=To continue, click Next. Updating RiffChart 1.0 that you unzipped yourself? Click Browse and choose that RiffChart folder: your songs and settings are kept, and the download is much shorter.
FinishedLabel=RiffChart is installed. It opens in your web browser; your charted songs are kept in its data folder.
