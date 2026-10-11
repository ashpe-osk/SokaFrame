# Start OmniBoard-Sports on Windows

This guide covers the **standalone static frontend** in
`OmniBoard-Frontend~/frontend`, the FastAPI backend, the SQLite database, and
the analysis worker.

To run the React/Vite alternative instead, complete its setup in
[react_frontend.md](./react_frontend.md), then run `.\start-react.ps1` from the
project root. That launcher starts the API, analysis worker, and React dev
server, and waits for all three to be ready.

After the one-time setup, the recommended startup is one command:

```powershell
.\start-local.ps1
```

The launcher applies database migrations, starts the API, waits for it to pass
its health check, starts the worker and waits for its ready signal, starts the
standalone frontend, then opens the frontend in your browser. The worker starts
automatically; there is no need for `-WithWorker`.

## 1. Check the extracted project

Extract the complete project bundle to a writable folder and keep its
directories together. The project root should contain `start-local.ps1`,
`main.py`, `models\best.pt`, `requirements.txt`, `omniBoard\`, and
`OmniBoard-Frontend~\frontend\index.html`.

The bundle does not include machine-specific Python environments, the private
`omniBoard\.env` file, a local database, or uploaded media. Create these on the
Windows PC that will run the application. Do not copy virtual environments or
a private `.env` from another machine.

In the commands below, replace the example Downloads location if the project
was extracted elsewhere:

```powershell
Set-Location -LiteralPath "$HOME\Downloads\omniBoard-Sportz-client"
```

## 2. Install required tools

Install [uv](https://docs.astral.sh/uv/), then close and reopen PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
```

FFmpeg and ffprobe are required to process uploaded videos. Install them and
confirm they are available in a newly opened PowerShell window:

```powershell
winget install --id Gyan.FFmpeg.Shared --exact
ffmpeg -version
ffprobe -version
```

The standalone static frontend does not require Node.js, npm, or a frontend
build. The launcher uses Python's built-in HTTP server to serve it.

## 3. Set up the CV environment and model

The worker uses a separate Python 3.10 environment from the backend. From the
project root:

```powershell
uv python install 3.10
if (-not (Test-Path -LiteralPath ".venv-1\Scripts\python.exe")) {
    uv venv --python 3.10 .venv-1
}
.\.venv-1\Scripts\Activate.ps1
python --version
uv pip install -r requirements.txt
```

Check that `python --version` reports Python 3.10. Install a compatible
PyTorch/torchvision build for the PC's hardware using the selector at
<https://pytorch.org/get-started/locally/>. Run the selected install command
with `uv pip install` while `.venv-1` is active. The model file
`models\best.pt` and CV entry point `main.py` must exist.

FFmpeg must include H.264/libx264 support for producing browser-playable
processed videos. CPU inference is available but may be slow. CUDA is usable
only with compatible NVIDIA hardware and a compatible PyTorch build.

## 4. Install the backend and configure private settings

Open PowerShell at the `omniBoard` directory:

```powershell
Set-Location -LiteralPath "$HOME\Downloads\omniBoard-Sportz-client\omniBoard"
uv sync --frozen --package app
uv run --package app python --version
```

The backend requires Python 3.14 or newer. `uv sync` creates the backend
environment in `omniBoard\.venv`; do not activate the root CV environment for
backend commands.

Create the private settings file only if it does not already exist, then edit
it:

```powershell
if (-not (Test-Path -LiteralPath .env)) {
    Copy-Item .env.example .env
}
notepad .env
```

Set these values in `omniBoard\.env`:

```dotenv
FASTAPI_ENV=development
SECRET_KEY=<a new long random secret>
FIRST_SUPERUSER=<your administrator email>
FIRST_SUPERUSER_PASSWORD=<a unique strong password>
DATABASE_URL=sqlite:///./sql_app.db
LOCAL_DATA_DIR=local-data
CV_REPO_ROOT=<absolute path to the extracted project root>
CV_PYTHON_EXECUTABLE=<absolute path to the root .venv-1 Python executable>
CV_ENTRYPOINT=main.py
CV_DEFAULT_MODEL=models/best.pt
CV_DEFAULT_BACKEND=auto
```

Generate a secret from the `omniBoard` directory:

```powershell
uv run --package app python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Copy the generated value into `SECRET_KEY`. For the paths above, use the
actual extracted project location, for example:

```dotenv
CV_REPO_ROOT=C:\Users\<your-user>\Downloads\omniBoard-Sportz-client
CV_PYTHON_EXECUTABLE=C:\Users\<your-user>\Downloads\omniBoard-Sportz-client\.venv-1\Scripts\python.exe
```

Do not use the example secrets, share `.env`, or commit it. Keep the default
SQLite URL and `LOCAL_DATA_DIR` unless you have intentionally configured the
API and worker to use another shared database and data directory.

## 5. Initialize or update the SQLite database

Run migrations from `omniBoard\backend`. The relative SQLite URL is interpreted
from this directory; the database file is
`omniBoard\backend\sql_app.db`.

```powershell
Set-Location -LiteralPath "$HOME\Downloads\omniBoard-Sportz-client\omniBoard\backend"
$env:DATABASE_URL = "sqlite:///./sql_app.db"
$env:PYTHONUTF8 = "1"
uv run alembic upgrade head
```

This is a migration, not a database reset. Never delete or replace a populated
`sql_app.db` as a troubleshooting shortcut. Back up the database before
applying migrations to a database containing important data.

The one-command launcher also runs `alembic upgrade head` on every startup to
apply any pending migrations. This is normally a no-op after the database is
up to date.

## 6. Start all services with one command (recommended)

Close the setup terminal, open a fresh PowerShell window, and run:

```powershell
Set-Location -LiteralPath "$HOME\Downloads\omniBoard-Sportz-client"
.\start-local.ps1
```

Before launching, the script checks the environment file, required settings,
ports 8000 and 5174, CV Python 3.10, the model and entry point, and FFmpeg and
ffprobe. It then migrates the database and starts:

1. FastAPI at `http://127.0.0.1:8000`
2. The analysis worker, which signals readiness before startup continues
3. The standalone frontend at `http://127.0.0.1:5174/`

API documentation is at `http://127.0.0.1:8000/docs`. The CV analysis page in
the standalone frontend is at `http://127.0.0.1:5174/cv-analysis.html`.
Sign in using the `FIRST_SUPERUSER` email and password configured in `.env`.

Keep the API, worker, and frontend PowerShell windows open while using the
application. The worker supervisor retries after an unexpected worker exit.
If the worker cannot start initially, startup reports an error; read the
worker's PowerShell window rather than assuming uploads will be processed.

## 7. Manual startup (alternative)

Use this if you need each service in its own terminal or need to diagnose a
startup problem. Complete sections 1-5 first. Start the services in this
order, using three separate PowerShell windows. Keep the API and worker pointed
at the same SQLite database and local-data directory.

### Terminal 1 — API

```powershell
Set-Location -LiteralPath "$HOME\Downloads\omniBoard-Sportz-client\omniBoard\backend"
$env:DATABASE_URL = "sqlite:///./sql_app.db"
$env:LOCAL_DATA_DIR = "local-data"
$env:PYTHONUTF8 = "1"
uv run python -m fastapi run app/main.py --host 127.0.0.1 --port 8000
```

In another PowerShell window, check that the API is healthy:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/utils/health-check/"
```

The response should be `true`.

### Terminal 2 — analysis worker

Only start one worker for this local SQLite database. Confirm no worker is
already running, then use:

```powershell
Set-Location -LiteralPath "$HOME\Downloads\omniBoard-Sportz-client\omniBoard\backend"
$env:DATABASE_URL = "sqlite:///./sql_app.db"
$env:LOCAL_DATA_DIR = "local-data"
$env:PYTHONUTF8 = "1"
uv run python -m app.worker
```

The worker needs the CV paths in `omniBoard\.env`, a working Python 3.10
environment and model, and FFmpeg/ffprobe available on `PATH`. Leave its
terminal open. Without a running worker, uploaded analyses can stay queued.

### Terminal 3 — standalone frontend

```powershell
Set-Location -LiteralPath "$HOME\Downloads\omniBoard-Sportz-client"
uv run --no-project --python 3.14 python -m http.server 5174 --bind 127.0.0.1 --directory "OmniBoard-Frontend~\frontend"
```

Open `http://127.0.0.1:5174/` in a browser. This serves the existing static
frontend; do not start Vite from `omniBoard\frontend` when following this
guide.

## 8. Stop the application

Press **Ctrl+C** in each service's PowerShell window:

- Frontend window
- Worker window (stops the worker supervisor)
- API window

The one-command launcher opens separate windows for these services. Stop each
one there. Do not close a terminal window as a substitute for a clean Ctrl+C
shutdown if a worker job is running.

## Troubleshooting

- **`uv` is not recognized:** install uv, close and reopen PowerShell, then
  check `uv --version`.
- **Missing or placeholder settings:** verify the private
  `omniBoard\.env` file and replace the example secret, email, and password.
- **CV Python version/path error:** verify `CV_PYTHON_EXECUTABLE` points to
  this PC's `.venv-1\Scripts\python.exe` and reports Python 3.10.
- **Missing model or `main.py`:** check `CV_DEFAULT_MODEL`,
  `CV_ENTRYPOINT`, and `CV_REPO_ROOT` against the extracted project.
- **FFmpeg/ffprobe not found:** install them, confirm both commands work in a
  newly opened PowerShell, and retry.
- **Port 8000 or 5174 is already in use:** stop the existing service first;
  do not start a duplicate API, worker, or frontend.
- **API does not become healthy:** inspect the API terminal, confirm the
  database migration completed, and verify the `.env` settings.
- **Worker fails readiness:** inspect its terminal for the actual error.
  Verify CV dependencies, model path, database settings, and FFmpeg.
- **Uploads remain queued:** confirm the worker is running against the same
  database as the API.
- **Wrong interface appears:** this guide starts the standalone frontend on
  port 5174. The separate React/Vite application is not started by
  `start-local.ps1`.
