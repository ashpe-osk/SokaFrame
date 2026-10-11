param(
    # Retained for compatibility; the worker now starts on every normal launch.
    [switch]$WithWorker,
    [ValidateSet("api", "frontend", "worker")]
    [string]$Service,
    [string]$ReadyFile
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$OmniBoardRoot = Join-Path $ProjectRoot "omniBoard"
$BackendRoot = Join-Path $OmniBoardRoot "backend"
$EnvFile = Join-Path $OmniBoardRoot ".env"
$FrontendRoot = Join-Path $ProjectRoot "OmniBoard-Frontend~\frontend"
$Uv = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue

function Get-EnvValue {
    param(
        [string]$Name,
        [string]$Path = $EnvFile
    )

    $line = Get-Content -LiteralPath $Path |
        Where-Object { $_ -match "^\s*$([regex]::Escape($Name))\s*=" } |
        Select-Object -Last 1
    if ($null -eq $line) {
        return $null
    }

    $value = ($line -split "=", 2)[1].Trim()
    if ($value.Length -ge 2 -and
        (($value.StartsWith('"') -and $value.EndsWith('"')) -or
         ($value.StartsWith("'") -and $value.EndsWith("'")))) {
        $value = $value.Substring(1, $value.Length - 2)
    }
    return $value
}

function Start-ServiceWindow {
    param(
        [string]$Name,
        [string]$ReadyFile
    )

    $quotedScript = '"' + $PSCommandPath + '"'
    $arguments = @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-File", $quotedScript,
        "-Service", $Name
    )
    if ($ReadyFile) {
        $arguments += @("-ReadyFile", ('"' + $ReadyFile + '"'))
    }
    return Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -PassThru
}

function Test-PortAvailable {
    param([int]$Port)

    $listener = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Loopback,
        $Port
    )
    try {
        $listener.Start()
        return $true
    }
    catch [System.Net.Sockets.SocketException] {
        return $false
    }
    finally {
        $listener.Stop()
    }
}

if ($Service) {
    if (-not $Uv) {
        throw "uv was not found on PATH. Install uv, reopen PowerShell, and try again."
    }

    $env:PYTHONUTF8 = "1"
    switch ($Service) {
        "api" {
            Set-Location -LiteralPath $BackendRoot
            & $Uv.Source run --no-sync python -m fastapi run app/main.py --host 127.0.0.1 --port 8000
        }
        "frontend" {
            Set-Location -LiteralPath $OmniBoardRoot
            & $Uv.Source run --no-project --python 3.14 python -m http.server 5174 --bind 127.0.0.1 --directory $FrontendRoot
        }
        "worker" {
            Set-Location -LiteralPath $BackendRoot
            $workerStarted = $false
            if ($ReadyFile) {
                $env:OMNIBOARD_WORKER_READY_FILE = $ReadyFile
            }
            while ($true) {
                try {
                    $workerProcess = Start-Process `
                        -FilePath $Uv.Source `
                        -ArgumentList @("run", "--no-sync", "python", "-m", "app.worker") `
                        -WorkingDirectory $BackendRoot `
                        -NoNewWindow `
                        -PassThru
                }
                catch {
                    if (-not $workerStarted -and $ReadyFile) {
                        Set-Content -LiteralPath $ReadyFile -Value "failed" -Encoding Ascii
                    }
                    throw "Could not launch analysis worker: $($_.Exception.Message)"
                }

                while (-not $workerProcess.HasExited) {
                    if (-not $workerStarted -and $ReadyFile -and
                        (Test-Path -LiteralPath $ReadyFile -PathType Leaf) -and
                        (Get-Content -LiteralPath $ReadyFile -Raw).Trim() -eq "ready") {
                        $workerStarted = $true
                    }
                    Start-Sleep -Milliseconds 500
                    $workerProcess.Refresh()
                }
                $serviceExitCode = $workerProcess.ExitCode

                if (-not $workerStarted) {
                    if ($ReadyFile) {
                        Set-Content -LiteralPath $ReadyFile -Value "failed" -Encoding Ascii
                    }
                    break
                }

                Write-Warning "Worker stopped with exit code $serviceExitCode; restarting in 5 seconds. Press Ctrl+C to stop the worker supervisor."
                Start-Sleep -Seconds 5
            }
        }
    }
    return
}

if (-not $Uv) {
    throw "uv was not found on PATH. Install uv, reopen PowerShell, and try again."
}
if (-not (Test-Path -LiteralPath $EnvFile -PathType Leaf)) {
    throw "Missing $EnvFile. Copy omniBoard\.env.example to omniBoard\.env and set the required secrets first."
}
if (-not (Test-Path -LiteralPath (Join-Path $FrontendRoot "index.html") -PathType Leaf)) {
    throw "The standalone frontend was not found at $FrontendRoot."
}

foreach ($setting in @("SECRET_KEY", "FIRST_SUPERUSER", "FIRST_SUPERUSER_PASSWORD")) {
    $value = Get-EnvValue -Name $setting
    if ([string]::IsNullOrWhiteSpace($value) -or
        $value -match "replace-with|changethis" -or
        $value -eq "admin@example.com") {
        throw "$setting is missing or still has its example value in omniBoard\.env."
    }
}

foreach ($port in @(8000, 5174)) {
    if (-not (Test-PortAvailable -Port $port)) {
        throw "Port $port is already in use. Stop the existing service before starting OmniBoard."
    }
}

if ($WithWorker) {
    Write-Warning "-WithWorker is no longer needed; the analysis worker starts automatically."
}

$cvRoot = Get-EnvValue -Name "CV_REPO_ROOT"
$cvPython = Get-EnvValue -Name "CV_PYTHON_EXECUTABLE"
if ([string]::IsNullOrWhiteSpace($cvRoot) -or -not (Test-Path -LiteralPath $cvRoot -PathType Container)) {
    throw "Set CV_REPO_ROOT in omniBoard\.env to this computer's project folder before startup."
}
if ([string]::IsNullOrWhiteSpace($cvPython) -or -not (Test-Path -LiteralPath $cvPython -PathType Leaf)) {
    throw "Set CV_PYTHON_EXECUTABLE in omniBoard\.env to the CV Python 3.10 executable before startup."
}

$cvVersion = & $cvPython --version 2>&1
if ($LASTEXITCODE -ne 0 -or $cvVersion -notmatch "Python 3\.10(?:\.|$)") {
    throw "CV_PYTHON_EXECUTABLE must run Python 3.10; the configured interpreter reported: $cvVersion"
}

$model = Get-EnvValue -Name "CV_DEFAULT_MODEL"
if ([string]::IsNullOrWhiteSpace($model)) {
    $model = "models/best.pt"
}
if (-not [IO.Path]::IsPathRooted($model)) {
    $model = Join-Path $cvRoot $model
}
if (-not (Test-Path -LiteralPath $model -PathType Leaf)) {
    throw "The configured CV model was not found: $model"
}

$cvEntrypoint = Get-EnvValue -Name "CV_ENTRYPOINT"
if ([string]::IsNullOrWhiteSpace($cvEntrypoint)) {
    $cvEntrypoint = "main.py"
}
if (-not [IO.Path]::IsPathRooted($cvEntrypoint)) {
    $cvEntrypoint = Join-Path $cvRoot $cvEntrypoint
}
if (-not (Test-Path -LiteralPath $cvEntrypoint -PathType Leaf)) {
    throw "The configured CV entry point was not found: $cvEntrypoint"
}

foreach ($executable in @("ffmpeg", "ffprobe")) {
    if (-not (Get-Command $executable -CommandType Application -ErrorAction SilentlyContinue)) {
        throw "$executable was not found on PATH. Install FFmpeg and reopen PowerShell before starting OmniBoard."
    }
}

$env:PYTHONUTF8 = "1"
Write-Host "Applying database migrations..."
Push-Location -LiteralPath $BackendRoot
try {
    & $Uv.Source run --no-sync alembic upgrade head
    if ($LASTEXITCODE -ne 0) {
        throw "Database migration failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

Write-Host "Starting API in a new PowerShell window..."
$null = Start-ServiceWindow -Name "api"

$healthUrl = "http://127.0.0.1:8000/api/v1/utils/health-check/"
$deadline = [DateTime]::UtcNow.AddSeconds(60)
$healthy = $false
while ([DateTime]::UtcNow -lt $deadline) {
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        if ($health -eq $true) {
            $healthy = $true
            break
        }
    }
    catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $healthy) {
    throw "API did not become healthy within 60 seconds. Check the API PowerShell window for errors."
}

$workerReadyFile = Join-Path ([IO.Path]::GetTempPath()) (
    "omniboard-worker-ready-{0}.txt" -f [guid]::NewGuid()
)
Write-Host "Starting analysis worker in a new PowerShell window..."
$null = Start-ServiceWindow -Name "worker" -ReadyFile $workerReadyFile
$workerDeadline = [DateTime]::UtcNow.AddSeconds(60)
$workerReady = $false
try {
    while ([DateTime]::UtcNow -lt $workerDeadline) {
        if (Test-Path -LiteralPath $workerReadyFile -PathType Leaf) {
            $workerStatus = (Get-Content -LiteralPath $workerReadyFile -Raw).Trim()
            if ($workerStatus -eq "ready") {
                $workerReady = $true
                break
            }
            if ($workerStatus -eq "failed") {
                throw "Analysis worker exited before becoming ready. Check its PowerShell window for errors."
            }
        }
        Start-Sleep -Seconds 1
    }
    if (-not $workerReady) {
        throw "Analysis worker did not report ready within 60 seconds. Check its PowerShell window for errors."
    }
}
catch {
    if (Test-Path -LiteralPath $workerReadyFile -PathType Leaf) {
        Remove-Item -LiteralPath $workerReadyFile -Force
    }
    throw
}

Write-Host "Starting standalone frontend in a new PowerShell window..."
$null = Start-ServiceWindow -Name "frontend"
$frontendUrl = "http://127.0.0.1:5174/"
$frontendDeadline = [DateTime]::UtcNow.AddSeconds(20)
$frontendReady = $false
while ([DateTime]::UtcNow -lt $frontendDeadline) {
    try {
        $response = Invoke-WebRequest -Uri $frontendUrl -TimeoutSec 2 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            $frontendReady = $true
            break
        }
    }
    catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $frontendReady) {
    throw "Frontend did not become available within 20 seconds. Check its PowerShell window for errors."
}

Write-Host ""
Write-Host "OmniBoard is ready:"
Write-Host "  Web app: $frontendUrl"
Write-Host "  API docs: http://127.0.0.1:8000/docs"
Write-Host "  Analysis worker: running"
Write-Host "Use Ctrl+C in each API, worker, or frontend window to stop it."
Start-Process $frontendUrl
