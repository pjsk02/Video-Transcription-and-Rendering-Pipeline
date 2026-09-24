# One-shot local setup for Video Transcription and Rendering Pipeline on Windows (no package manager required).
# Idempotent: safe to re-run. Downloads static ffmpeg + prebuilt whisper.cpp,
# a whisper model, and creates the Python venv.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#
# Mirrors scripts/setup.sh (macOS). Everything lands under vendor\ (gitignored).
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # makes Invoke-WebRequest fast for big files

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Model = if ($env:CM_WHISPER_MODEL) { $env:CM_WHISPER_MODEL } else { "base.en" }

$Bin       = Join-Path $Root "vendor\bin"
$WhisperBin = Join-Path $Root "vendor\whisper.cpp\build\bin"
$Models    = Join-Path $Root "vendor\whisper.cpp\models"
New-Item -ItemType Directory -Force -Path $Bin, $WhisperBin, $Models | Out-Null
$Tmp = Join-Path $env:TEMP "cm-setup"
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null

function Get-File($url, $dest) {
  if (Test-Path $dest) { Write-Host "  cached: $(Split-Path -Leaf $dest)"; return }
  Write-Host "  downloading $(Split-Path -Leaf $dest) ..."
  Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
}

Write-Host "==> 1/4 ffmpeg + ffprobe (NVENC-capable, pinned to 7.1)"
# Pinned to BtbN's ffmpeg 7.1 build, NOT bleeding-edge master. Master's NVENC
# (h264_nvenc) requires NVIDIA driver >=610; common current drivers (e.g. 591.74)
# only expose the older NVENC API, so master fails to init NVENC. ffmpeg 7.1's
# NVENC works on those drivers (verified on an RTX 5060 / driver 591.74). The app
# probes NVENC at runtime and falls back to CPU x264 if it's unusable, so this is
# purely additive — the pin just lets the GPU encode path actually engage.
$FFmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-win64-gpl-7.1.zip"
function Install-FFmpeg {
  $zip = Join-Path $Tmp "ffmpeg.zip"
  Get-File $FFmpegUrl $zip
  $ex = Join-Path $Tmp "ffmpeg"
  if (Test-Path $ex) { Remove-Item -Recurse -Force $ex }
  Expand-Archive -Path $zip -DestinationPath $ex -Force
  Get-ChildItem -Path $ex -Recurse -Include ffmpeg.exe, ffprobe.exe |
    ForEach-Object { Copy-Item $_.FullName -Destination $Bin -Force }
}
$ffmpegExe = Join-Path $Bin "ffmpeg.exe"
$need = $true
if (Test-Path $ffmpegExe) {
  $ver = (& $ffmpegExe -hide_banner -version 2>$null | Select-Object -First 1)
  if ($ver -match "n7\.1") { $need = $false; Write-Host "  already present (ffmpeg 7.1)" }
  else { Write-Host "  replacing '$ver' with pinned 7.1 (NVENC driver compat)" }
}
if ($need) { Install-FFmpeg; Write-Host "  ffmpeg 7.1 -> vendor\bin" }

Write-Host "==> 2/4 whisper.cpp (prebuilt x64 BLAS binaries)"
if (-not (Test-Path (Join-Path $WhisperBin "whisper-cli.exe"))) {
  $zip = Join-Path $Tmp "whisper.zip"
  Get-File "https://github.com/ggml-org/whisper.cpp/releases/download/v1.9.1/whisper-blas-bin-x64.zip" $zip
  $ex = Join-Path $Tmp "whisper"
  if (Test-Path $ex) { Remove-Item -Recurse -Force $ex }
  Expand-Archive -Path $zip -DestinationPath $ex -Force
  # zip layout: Release\*.exe + *.dll  — flatten into build\bin
  Get-ChildItem -Path $ex -Recurse -Include *.exe, *.dll |
    ForEach-Object { Copy-Item $_.FullName -Destination $WhisperBin -Force }
  Write-Host "  whisper binaries -> vendor\whisper.cpp\build\bin"
} else { Write-Host "  already present" }

Write-Host "==> 3/4 whisper model ($Model)"
$modelFile = Join-Path $Models "ggml-$Model.bin"
Get-File "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$Model.bin" $modelFile

Write-Host "==> 4/4 Python venv + deps"
if (-not (Test-Path (Join-Path $Root ".venv"))) { python -m venv .venv }
$py = Join-Path $Root ".venv\Scripts\python.exe"
& $py -m pip install --quiet --upgrade pip
& $py -m pip install --quiet -e ".[dev,vision]"
if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed." }
& $py scripts/setup_vision.py
if ($LASTEXITCODE -ne 0) { throw "Face detector model setup failed." }

Write-Host ""
Write-Host "Setup complete. Vendored tools:"
& (Join-Path $Bin "ffmpeg.exe") -version 2>&1 | Select-Object -First 1
Write-Host "Run the UI with:  .venv\Scripts\python.exe -m content_machine.cli serve"
