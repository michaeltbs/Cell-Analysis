<#
.SYNOPSIS
  Startet das Projekt mit Docker Compose unter Windows (PowerShell).

<#
.SYNOPSIS
  Startet das Projekt per Docker Compose unter Windows (PowerShell) mit optionaler GPU-Variante.

.PARAMETER Gpu
  Wenn gesetzt startet das Skript die GPU-Variante (nutzt docker-compose.gpu.yml).

.PARAMETER HostData
  Optionales Host-Datenverzeichnis, das in den Container nach /data gemountet wird.

.EXAMPLE
  # GPU-Variante starten
  ./start_remote.ps1 -Gpu -HostData C:\srv\cell_data

.EXAMPLE
  # CPU-Variante starten
  ./start_remote.ps1 -HostData C:\path\to\data

#>

.DESCRIPTION
    [switch]$Gpu,
    [string]$HostData = "",
    [switch]$NoBuild
)]

  Dieses Skript ist das PowerShell-Pendant zu start_remote.sh.
  Es setzt Umgebungsvariablen (HOST_DATA, HOST_UID, HOST_GID) und führt
  `docker compose` aus. Optional kann die GPU-Compose-Datei verwendet werden
  (--gpu).

.EXAMPLES
  # Standard (keine GPU):
  .\start_remote.ps1 -HostData C:\data\cell_data

  # Mit GPU-Compose (wenn docker-compose.gpu.yml vorhanden ist):
  .\start_remote.ps1 -HostData C:\data\cell_data -Gpu

#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$false)]
    [string]$HostData = "",

    [Parameter(Mandatory=$false)]
    [switch]$Gpu,

    [Parameter(Mandatory=$false)]
    [int]$HostUid = 1000,

    [Parameter(Mandatory=$false)]
    [int]$HostGid = 1000
)

function Confirm-Path([string]$p) {
    if ([string]::IsNullOrWhiteSpace($p)) { return $false }
    return Test-Path $p
}

if (-not $HostData) {
    $HostData = Read-Host "HOST_DATA nicht gesetzt. Pfad zu Host-Datenverzeichnis eingeben (z.B. C:\data\cell_data)"
    if (-not $HostData) { $HostData = "." }
}

# Setze Umgebungsvariablen für den compose-Aufruf
$env:HOST_DATA = $HostData
$env:HOST_UID = $HostUid.ToString()
$env:HOST_GID = $HostGid.ToString()

Write-Host "Using HOST_DATA=$env:HOST_DATA"
Write-Host "Using HOST_UID=$env:HOST_UID HOST_GID=$env:HOST_GID"

$composeFiles = "-f docker-compose.yml"
if ($Gpu.IsPresent) {
    if (-not (Test-Path "docker-compose.gpu.yml")) {
        Write-Warning "docker-compose.gpu.yml nicht gefunden. Stelle sicher, dass die Datei im Repo vorhanden ist. Fortfahren ohne GPU-Compose..."
    } else {
        $composeFiles += " -f docker-compose.gpu.yml"
    }
}

Write-Host "Building docker image (dies kann einige Minuten dauern)..."
# Build mit übergebenen Build-Args
docker compose $composeFiles build --build-arg USER_ID=$env:HOST_UID --build-arg GROUP_ID=$env:HOST_GID

Write-Host "Starting container(s) with HOST_DATA mounted to /data..."
docker compose $composeFiles up -d

Write-Host "Containers started. Access the web UI at http://localhost:5000"
Write-Host "To follow logs: docker compose $composeFiles logs -f"

Write-Host "Done."
