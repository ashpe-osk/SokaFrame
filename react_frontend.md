# Use the React/Vite Frontend on Windows

This guide explains how to run the React/TypeScript app in
`omniBoard\frontend` as an alternative to the standalone static UI in
`OmniBoard-Frontend~\frontend`.

The existing `start-local.ps1` launcher starts the standalone frontend at
`http://127.0.0.1:5174/`; it does **not** start the React/Vite development
server. For React development, run Vite separately at
`http://127.0.0.1:5173/` and the FastAPI backend at
`http://127.0.0.1:8000/`.

After completing one-time setup and applying the database migrations, start
both the API and React development server with one command from the project
root:

```powershell
.\start-react.ps1
```

The script opens a separate PowerShell window for each service, waits for the
API health check, analysis worker readiness signal, and React page to respond,
then opens the React app in your browser. It does not run database migrations.
The worker stays online and automatically restarts after an unexpected exit.
Press **Ctrl+C** in each service window to stop it.

## Requirements

- Windows PowerShell
- Node.js and npm
- `uv`
- Python 3.14 or newer for the backend
- A separate Python 3.10 CV environment
- The local settings file `omniBoard\.env`
- SQLite migrations applied to the intended database
- CV model weights, FFmpeg, and ffprobe

Because the one-command launcher starts the worker, these CV prerequisites
must be configured even when you only intend to browse the React interface.
The worker processes uploads automatically so they do not stay queued because
the worker was omitted.

## 1. Install frontend dependencies once

Open PowerShell at the `omniBoard` directory (adjust the path if the project
was extracted somewhere else):

```powershell
Set-Location -LiteralPath "$HOME\Downloads\omniBoard-Sportz-client\omniBoard"
npm ci
```

Use the existing root `package-lock.json`; do not run `npm install` to refresh
or replace the lockfile. The React frontend is an npm workspace named
`frontend`, so install from `omniBoard`, not from the frontend subfolder.

## 2. Configure the backend and database

If the backend has not been set up on this PC, follow the one-time backend
environment and `.env` instructions in [start.md](./start.md), sections 4 and
5. In summary:

1. From `omniBoard`, install the locked backend environment:

   ```powershell
   uv sync --frozen --package app
   ```

2. Create `omniBoard\.env` from `.env.example` only if `.env` does not already
   exist. Set a private `SECRET_KEY`, `FIRST_SUPERUSER`, and
   `FIRST_SUPERUSER_PASSWORD`. Keep the local SQLite settings:

   ```dotenv
   FASTAPI_ENV=development
   DATABASE_URL=sqlite:///./sql_app.db
   LOCAL_DATA_DIR=local-data
   ```

   Do not overwrite an existing `.env` or share/commit it.

3. Initialize or update the database from `omniBoard\backend`:

   ```powershell
   Set-Location -LiteralPath "$HOME\Downloads\omniBoard-Sportz-client\omniBoard\backend"
   $env:DATABASE_URL = "sqlite:///./sql_app.db"
   $env:PYTHONUTF8 = "1"
   uv run alembic upgrade head
   ```

   The local database file is `omniBoard\backend\sql_app.db`. Back it up before
   applying future migrations if it contains important data. Do not delete or
   reset it to resolve startup problems.

If the API has already been configured and migrated, do not repeat setup
unnecessarily.

## 3. Start the API

Open a PowerShell terminal for the backend and leave it running:

```powershell
Set-Location -LiteralPath "$HOME\Downloads\omniBoard-Sportz-client\omniBoard\backend"
$env:DATABASE_URL = "sqlite:///./sql_app.db"
$env:LOCAL_DATA_DIR = "local-data"
$env:PYTHONUTF8 = "1"
uv run python -m fastapi run app/main.py --host 127.0.0.1 --port 8000
```

In another PowerShell window, verify health:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/utils/health-check/"
```

The response should be `true`. API documentation is at
`http://127.0.0.1:8000/docs`.

Keep the API's `DATABASE_URL` and `LOCAL_DATA_DIR` consistent with the worker
configuration. The one-command launcher sets both services to the local
SQLite database and `local-data` directory.

## 4. Start the React/Vite development server

Use the one-command launcher above for normal local use. To start Vite
manually, open a second PowerShell window and run the workspace script:

```powershell
Set-Location -LiteralPath "$HOME\Downloads\omniBoard-Sportz-client\omniBoard"
npm --workspace=frontend run dev
```

Vite uses its default local host and port 5173. Wait for it to print its
`Local` URL, then open that exact URL in a browser. The default is normally
`http://localhost:5173/`. In development, Vite proxies `/api` requests to
FastAPI at `http://127.0.0.1:8000`; keep the API server running as well. This
also lets the frontend work on a different Vite port without changing the
backend's CORS allowlist.

Do not append `--host 127.0.0.1 --port 5173` to the workspace command shown
above: npm can consume those options as its own and pass their values to Vite
as positional arguments. If you need to customize Vite's host or port, run
Vite directly from the frontend directory instead:

```powershell
Set-Location -LiteralPath "$HOME\Downloads\omniBoard-Sportz-client\omniBoard\frontend"
..\node_modules\.bin\vite.cmd --host 127.0.0.1 --port 5173 --strictPort
```

The backend's development CORS configuration allows `localhost` and
`127.0.0.1` on port 5173. Keep both terminals running. If port 5173 is in use,
stop the other process or choose another port and add that exact origin to the
backend's development CORS allowlist before using it.

To stop the dev server and API, press **Ctrl+C** in their respective
PowerShell windows.

## 5. Worker configuration

The React project page supports video uploads and creates analysis jobs. The
`start-react.ps1` launcher starts one worker automatically and verifies it is
ready before it announces success. Configure the root CV environment and model
in `omniBoard\.env`:

```dotenv
CV_REPO_ROOT=<absolute path to the extracted project root>
CV_PYTHON_EXECUTABLE=<absolute path to .venv-1\Scripts\python.exe>
CV_ENTRYPOINT=main.py
CV_DEFAULT_MODEL=models/best.pt
CV_DEFAULT_BACKEND=auto
```

Set the paths to the actual location on this PC. Install FFmpeg and ffprobe
and make both available on `PATH` (or configure their executable paths in
`.env`). The worker needs FFmpeg with H.264/libx264 support to create
browser-playable results. If initial worker startup fails, the React launcher
reports that failure and the worker terminal shows the cause. If the worker
later exits, its supervisor retries after five seconds.

For manual operation or troubleshooting, run one worker only, from the backend
directory and against the same database and local-data settings as the API:

```powershell
Set-Location -LiteralPath "$HOME\Downloads\omniBoard-Sportz-client\omniBoard\backend"
$env:DATABASE_URL = "sqlite:///./sql_app.db"
$env:LOCAL_DATA_DIR = "local-data"
$env:PYTHONUTF8 = "1"
uv run python -m app.worker
```

Do not run this manual worker at the same time as the worker started by
`start-react.ps1`. More CV setup details, including creation of the Python
3.10 environment, are in [start.md](./start.md).

## React authentication and routes

React uses the `/api/v1` API and a bearer access token in browser
`localStorage`; this is different from the standalone frontend's cookie/CSRF
authentication. Register at `/signup`, then explicitly sign in at `/login`.
Successful login opens the Dashboard. The React frontend does not share the
standalone UI's browser session. The app clears the saved token for an HTTP 401
or the backend's explicit invalid-credentials 403 response; other 403 responses
do not force a logout.
Password recovery requires SMTP email delivery to be configured; otherwise the
API returns an explicit service-unavailable error rather than appearing to send
a reset link.

The React app includes Dashboard, Projects, Settings, Items, and an Admin page
restricted to superusers. Projects is organized into **Computer Vision**,
**Tactical Board**, and **Telestrator** categories. Computer Vision projects
retain the existing video upload and analysis workflow. Tactical Board projects
provide a separate workspace for formations, player positions, and movement
plans; board projects are kept separate from the Computer Vision project list.
Telestrator projects have their own video-analysis workspace.
The board editor supports player positioning, movement arrows, freehand
strokes, circles, rectangles, text annotations, erasing, undoing the latest
drawing, and clearing drawings.

Telestrator projects provide a match-video annotation workspace. Upload one or
more MP4, MOV, or AVI clips, use playback, volume, keyboard, timeline, and
frame-step controls, then freeze a frame to create a saved moment. Freeze
moments are marked on the timeline, can be moved, and automatically pause
playback while their drawings are displayed for the configured 1–30 seconds.
Draw freehand marks, lines, configurable solid/curved/lob/marker/bounce arrows,
styled rings, dashed/hatched/square areas, funnels, or label/plain text. Select
and drag drawings to move them, adjust shape endpoints, double-click text to
edit it, and undo or delete annotations and moments. Use **Save annotations**
to persist drawings and freeze moments to the project. Video uploads and
playback are authenticated and project-scoped.
Tactical Board and Telestrator workspaces use the legacy
`/api/projects/{board|media}` client-project API.

## Optional production build

To compile React assets for the FastAPI app instead of using Vite's dev server,
stop Vite and run from `omniBoard`:

```powershell
Set-Location -LiteralPath "$HOME\Downloads\omniBoard-Sportz-client\omniBoard"
$env:VITE_API_URL = ""
npm run build --workspace frontend
```

Vite's build output goes to `omniBoard\backend\app\frontend`, which FastAPI
serves at `/`. Then start FastAPI as described above and open
`http://127.0.0.1:8000/`. A blank `VITE_API_URL` makes the built client use the
same origin for API requests. Running the build replaces that generated
frontend output directory; it does not run migrations or modify the SQLite
database.

To return to local development, run the Vite dev-server instructions in
section 4. Building assets is not required for Vite development.

## Playwright end-to-end tests

The frontend Playwright setup uses the `FIRST_SUPERUSER` credentials from
`omniBoard\.env` and writes browser storage state under
`omniBoard\frontend\playwright\.auth`. The frontend test scripts currently
invoke `bunx`, and `playwright.config.ts` starts the dev server with Bun when
it manages the server itself. Bun is therefore required by the existing test
scripts; the test suite is not part of normal startup.
