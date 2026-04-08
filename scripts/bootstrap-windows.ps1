$ErrorActionPreference = "Stop"
$PyVersion = "3.13"

try {
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force -ErrorAction Stop
} catch {
  Write-Host "WARNING: Could not set execution policy (possibly managed by IT)."
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "ERROR: uv is not installed."
  Write-Host "Install uv first:"
  Write-Host '  powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"'
  exit 1
}

# Check for Visual C++ Redistributable (required by PyTorch on Windows)
$vcInstalled = Test-Path "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
if (-not $vcInstalled) {
  Write-Host ""
  Write-Host "WARNING: Microsoft Visual C++ Redistributable not detected."
  Write-Host "PyTorch requires it to run on Windows."
  Write-Host "Download and install from: https://aka.ms/vs/17/release/vc_redist.x64.exe"
  Write-Host ""
  $response = Read-Host "Continue anyway? (y/N)"
  if ($response -ne "y") { exit 1 }
}

Write-Host "Installing Python $PyVersion..."
uv python install $PyVersion

Write-Host "Creating virtual environment and installing dependencies..."
uv sync

Write-Host "`u{2713} Setup complete!"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Open in VS Code: code ."
Write-Host "  2. Update README.md with your project description"
Write-Host "  3. Start coding in src/"
