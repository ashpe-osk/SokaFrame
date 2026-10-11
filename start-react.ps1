param(
    [ValidateSet("api", "frontend", "worker")]
    [string]$Service,
    [string]$ReadyFile
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$OmniBoardRoot = Join-Path $ProjectRoot "omniBoard"
$BackendRoot = Join-Path $OmniBoardRoot "backend"
$EnvFile = Join-Path $OmniBoardRoot ".env"
$Uv = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue

function Get-EnvValue {
    param([string]$Name)

    $line = Get-Content -LiteralPath $EnvFile |
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

function Start-ServiceWindow {
    param(
        [string]$Name,
        [string]$ReadyFile
    )

    $arguments = @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-File", ('"{0}"' -f $PSCommandPath),
        "-Service", $Name
    )
    if ($ReadyFile) {
        $arguments += @("-ReadyFile", ('"{0}"' -f $ReadyFile))
    }
    return Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -PassThru
}

if ($Service) {
    if (-not $Uv) {
        throw "uv was not found on PATH. Install uv, reopen PowerShell, and try again."
    }

    $env:PYTHONUTF8 = "1"
    switch ($Service) {
        "api" {
            Set-Location -LiteralPath $BackendRoot
            $env:DATABASE_URL = "sqlite:///./sql_app.db"
            $env:LOCAL_DATA_DIR = "local-data"
            & $Uv.Source run --no-sync python -m fastapi run app/main.py --host 127.0.0.1 --port 8000
        }
        "worker" {
            Set-Location -LiteralPath $BackendRoot
            $env:DATABASE_URL = "sqlite:///./sql_app.db"
            $env:LOCAL_DATA_DIR = "local-data"
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

                if (-not $workerStarted) {
                    if ($ReadyFile) {
                        Set-Content -LiteralPath $ReadyFile -Value "failed" -Encoding Ascii
                    }
                    break
                }

                Write-Warning "Worker stopped with exit code $($workerProcess.ExitCode); restarting in 5 seconds. Press Ctrl+C to stop the worker supervisor."
                Start-Sleep -Seconds 5
            }
        }
        "frontend" {
            Set-Location -LiteralPath $OmniBoardRoot
            $env:VITE_API_URL = "http://127.0.0.1:8000"
            & npm.cmd --workspace=frontend run dev
        }
    }
    exit $LASTEXITCODE
}

if (-not $Uv) {
    throw "uv was not found on PATH. Install uv, reopen PowerShell, and try again."
}
if (-not (Get-Command npm.cmd -CommandType Application -ErrorAction SilentlyContinue)) {
    throw "npm was not found on PATH. Install Node.js, reopen PowerShell, and try again."
}
if (-not (Test-Path -LiteralPath (Join-Path $OmniBoardRoot "node_modules\.bin\vite.cmd") -PathType Leaf)) {
    throw "Frontend dependencies are missing. Run 'npm ci' from the omniBoard directory first."
}
if (-not (Test-Path -LiteralPath $EnvFile -PathType Leaf)) {
    throw "Missing omniBoard\.env. Complete the backend setup in react_frontend.md first."
}

$cvRoot = Get-EnvValue -Name "CV_REPO_ROOT"
$cvPython = Get-EnvValue -Name "CV_PYTHON_EXECUTABLE"
if ([string]::IsNullOrWhiteSpace($cvRoot) -or -not (Test-Path -LiteralPath $cvRoot -PathType Container)) {
    throw "Set CV_REPO_ROOT in omniBoard\.env to this computer's project folder before starting the worker."
}
if ([string]::IsNullOrWhiteSpace($cvPython) -or -not (Test-Path -LiteralPath $cvPython -PathType Leaf)) {
    throw "Set CV_PYTHON_EXECUTABLE in omniBoard\.env to the CV Python 3.10 executable before starting the worker."
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

$entrypoint = Get-EnvValue -Name "CV_ENTRYPOINT"
if ([string]::IsNullOrWhiteSpace($entrypoint)) {
    $entrypoint = "main.py"
}
if (-not [IO.Path]::IsPathRooted($entrypoint)) {
    $entrypoint = Join-Path $cvRoot $entrypoint
}
if (-not (Test-Path -LiteralPath $entrypoint -PathType Leaf)) {
    throw "The configured CV entry point was not found: $entrypoint"
}

foreach ($tool in @(
    @{ Name = "FFMPEG_EXECUTABLE"; Default = "ffmpeg" },
    @{ Name = "FFPROBE_EXECUTABLE"; Default = "ffprobe" }
)) {
    $executable = Get-EnvValue -Name $tool.Name
    if ([string]::IsNullOrWhiteSpace($executable)) {
        $executable = $tool.Default
    }
    if ([IO.Path]::IsPathRooted($executable)) {
        if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
            throw "$($tool.Name) was not found at $executable."
        }
    }
    elseif (-not (Get-Command $executable -CommandType Application -ErrorAction SilentlyContinue)) {
        throw "$executable was not found on PATH. Install FFmpeg and reopen PowerShell before starting the worker."
    }
}

foreach ($port in @(8000, 5173)) {
    if (-not (Test-PortAvailable -Port $port)) {
        throw "Port $port is already in use. Stop the existing service before starting the React app."
    }
}

Write-Host "Starting FastAPI in a new PowerShell window..."
$null = Start-ServiceWindow -Name "api"

$apiUrl = "http://127.0.0.1:8000/api/v1/utils/health-check/"
$apiDeadline = [DateTime]::UtcNow.AddSeconds(60)
$apiReady = $false
while ([DateTime]::UtcNow -lt $apiDeadline) {
    try {
        if ((Invoke-RestMethod -Uri $apiUrl -TimeoutSec 2) -eq $true) {
            $apiReady = $true
            break
        }
    }
    catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $apiReady) {
    throw "API did not become healthy within 60 seconds. Check its PowerShell window for errors."
}

Write-Host "Starting analysis worker in a new PowerShell window..."
$workerReadyFile = Join-Path ([IO.Path]::GetTempPath()) (
    "omniboard-react-worker-ready-{0}.txt" -f [guid]::NewGuid()
)
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

Write-Host "Starting React/Vite in a new PowerShell window..."
$null = Start-ServiceWindow -Name "frontend"

$frontendUrl = "http://localhost:5173/"
$frontendDeadline = [DateTime]::UtcNow.AddSeconds(60)
$frontendReady = $false
while ([DateTime]::UtcNow -lt $frontendDeadline) {
    try {
        $response = Invoke-WebRequest -Uri "$($frontendUrl)login" -TimeoutSec 2 -UseBasicParsing
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
    throw "React frontend did not become available within 60 seconds. Check its PowerShell window for errors."
}

Write-Host ""
Write-Host "React OmniBoard is ready:"
Write-Host "  Web app: $frontendUrl"
Write-Host "  API docs: http://127.0.0.1:8000/docs"
Write-Host "  Analysis worker: running"
Write-Host "Use Ctrl+C in each API, worker, and frontend window to stop it."
Start-Process $frontendUrl
