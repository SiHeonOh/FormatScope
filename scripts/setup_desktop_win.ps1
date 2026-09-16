<#
  FormatScope: Windows host bootstrap for the desktop box (build-plan.md S3.2).

    Right-click PowerShell -> "Run as administrator", then:
      powershell -ExecutionPolicy Bypass -File scripts\setup_desktop_win.ps1

  Installs WSL2 + Ubuntu 24.04 and checks the NVIDIA driver. Everything after
  this runs inside Ubuntu via scripts/setup_desktop_wsl.sh. Safe to re-run.

  Do NOT install an NVIDIA driver inside WSL: the Windows driver already
  exposes the GPU to Ubuntu through /dev/dxg, and a second one breaks it.
#>

$ErrorActionPreference = 'Stop'
function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "warning: $m" -ForegroundColor Yellow }

$admin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
  ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw "Run this from an elevated PowerShell (Run as administrator)." }

Step "NVIDIA driver"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
} else {
  Warn "nvidia-smi not found. Install the current GeForce driver from https://www.nvidia.com/download/index.aspx before the CUDA steps. WSL setup can still continue."
}

Step "WSL2 + Ubuntu 24.04"
$distros = (wsl --list --quiet) -replace "`0", ""   # wsl.exe emits UTF-16
if ($distros -match 'Ubuntu-24\.04') {
  Write-Host "Ubuntu-24.04 already installed."
  wsl --set-default-version 2 | Out-Null
  wsl --set-default Ubuntu-24.04 | Out-Null
} else {
  Write-Host "Installing (this pulls ~500 MB and needs a reboot afterwards)..."
  wsl --install -d Ubuntu-24.04
  Write-Host "`nREBOOT NOW, then launch 'Ubuntu 24.04' from the Start menu once to create your Linux username and password." -ForegroundColor Green
  Write-Host "After that, re-run this script to continue." -ForegroundColor Green
  exit 0
}

Step "WSL kernel update"
wsl --update

Step "GPU visible inside WSL"
$gpu = wsl -d Ubuntu-24.04 -- bash -lc "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true"
if ($gpu) {
  Write-Host "Ubuntu sees: $gpu"
} else {
  Warn "Ubuntu cannot see the GPU yet. Update the Windows NVIDIA driver, then run 'wsl --shutdown' and try again."
}

Step "Next"
@"
Open Ubuntu (Start menu -> 'Ubuntu 24.04') and run:

    git clone https://github.com/SiHeonOh/FormatScope.git ~/FormatScope
    bash ~/FormatScope/scripts/setup_desktop_wsl.sh

Clone into the Linux home directory as shown, NOT /mnt/c -- the Windows
filesystem is roughly 10x slower under WSL and the EDA flow feels it.
"@ | Write-Host
