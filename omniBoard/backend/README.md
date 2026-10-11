# FastAPI Project - Backend

## Requirements

* [uv](https://docs.astral.sh/uv/) for Python package and environment management.
* Python 3.14 or newer, as required by `pyproject.toml`.
* Node.js and npm for the frontend (Bun is also supported).

## Local Development

The standard local workflow uses SQLite and does not require Docker or
PostgreSQL. For the complete Windows startup order (API, frontend, and
optional analysis worker), follow the [root README quick start](../README.md#local-startup-windows).
The API and worker must use the same database and local-data directory.

### Windows PowerShell

```powershell
cd C:\omniBoard-Sportz\omniBoard\backend
$env:DATABASE_URL = "sqlite:///./sql_app.db"
$env:PYTHONUTF8 = "1"
uv run fastapi dev app/main.py
```

Run `uv sync` once to install backend dependencies and
`uv run alembic upgrade head` after initial setup or when migrations are added.
Required settings such as `SECRET_KEY` and superuser settings are loaded from
the private root `.env` file. Set `DATABASE_URL` in every backend terminal so
the API and worker use the local SQLite database.

### macOS / Linux

```sh
cd /path/to/omniBoard/backend
export DATABASE_URL="sqlite:///./sql_app.db"
uv run fastapi dev app/main.py
```

The API is available at `http://localhost:8000`, with automatic interactive docs at `http://localhost:8000/docs`.
The SQLite file is created at `backend/sql_app.db` and is excluded from version
control. SQLite migrations create the existing template `User` and `Item`
tables plus the football domain metadata (`Project`, `Video`, `Analysis`, and
`Artifact`). The `0006_project_kind` migration adds a durable `Project.kind`,
defaulting existing and new projects to `tactical`. Uploaded media defaults to
`backend/local-data/uploads` and is
also excluded from version control. This is separate from the historical
PostgreSQL migration chain.

### Local analysis API

Register or sign in through the existing `/api/v1/users/signup` and
`/api/v1/login/access-token` endpoints. Create a project with
`POST /api/v1/projects/`, then submit a multipart request to
`POST /api/v1/analyses/` with `project_id` and `video_file`. Use
`GET /api/v1/analyses/` to list the signed-in user's analyses and
`GET /api/v1/analyses/{analysis_id}` to read one. Project and analysis reads
are scoped to the authenticated owner.

Uploads accept `.mp4`, `.mov`, and `.avi` files only when their container
header matches the extension (ISO-BMFF `ftyp` for MP4/MOV, RIFF/AVI for AVI).
This is a lightweight signature and outer-container-size check, not a codec
probe or full corruption scan. The default maximum upload size is 4 GiB and
can be changed with `MAX_VIDEO_UPLOAD_BYTES`; this default permits typical
10+ minute compressed match footage while bounding local disk consumption.
Files are read and hashed in 1 MiB chunks, staged under `local-data/uploads/.tmp`,
and atomically moved to a server-generated path before the SQLite transaction
commits. On metadata-commit failure, the stored file is removed. A hard process
crash between file promotion and database commit can leave an unreferenced
file; no automatic orphan cleanup is part of this phase.

After the upload is stored, analysis creation returns with status `queued`. It
does not start video inference, a worker, or any CV engine process.

### Local analysis worker

Run the worker in a separate terminal from `backend/`, using the same database
and local-data settings as the API. Configure the separate CV checkout and
Python environment before starting it:

```powershell
cd C:\omniBoard-Sportz\omniBoard\backend
$env:DATABASE_URL = "sqlite:///./sql_app.db"
$env:LOCAL_DATA_DIR = "local-data"
$env:CV_REPO_ROOT = "C:\omniBoard-Sportz"
$env:CV_PYTHON_EXECUTABLE = "C:\omniBoard-Sportz\.venv-1\Scripts\python.exe"
$env:CV_DEFAULT_MODEL = "C:\omniBoard-Sportz\models\best.pt"
$env:CV_DEFAULT_BACKEND = "auto"
uv run python -m app.worker
```

This assumes FFmpeg and ffprobe are on `PATH`. If they are not, set
`FFMPEG_EXECUTABLE` and `FFPROBE_EXECUTABLE` to their full executable paths.

```sh
export DATABASE_URL="sqlite:///./sql_app.db"
export LOCAL_DATA_DIR="local-data"
export CV_REPO_ROOT="/path/to/omniBoard-Sportz"
export CV_PYTHON_EXECUTABLE="/path/to/cv/.venv/bin/python"
export CV_DEFAULT_MODEL="/path/to/omniBoard-Sportz/models/best.pt"
export CV_DEFAULT_BACKEND="auto"
export FFMPEG_EXECUTABLE="/path/to/ffmpeg/bin/ffmpeg"
export FFPROBE_EXECUTABLE="/path/to/ffmpeg/bin/ffprobe"
uv run python -m app.worker
```

`CV_ENTRYPOINT` defaults to `main.py` (relative to `CV_REPO_ROOT`), and
`CV_TERMINATION_GRACE_SECONDS` defaults to 10 seconds. The Python executable
and repository root are required configuration; there are no hard-coded CV
checkout paths. Keep the OmniBoard Python 3.14+ environment separate from the
CV engine's Python 3.10 environment.
The worker defaults to `CV_CANONICAL_IDS=online`, `CV_BALL_GATE=true`, and
`CV_TEAM_SMOOTHING=true`. Ball gating and team smoothing require online
canonical IDs. `CV_DEFAULT_BACKEND=auto` follows the CV CLI's automatic
CUDA/OpenVINO/CPU selection; the worker does not pass a device override.
The worker also requires FFmpeg and ffprobe (with the `libx264` encoder) to be
available on `PATH`, or configured with `FFMPEG_EXECUTABLE` and
`FFPROBE_EXECUTABLE`. After CV processing it creates and validates an H.264/AVC
`browser.mp4` artifact with `yuv420p` and fast-start MP4 layout. The original
`tracked.mp4` remains unchanged and registered separately.

The worker claims the oldest queued analysis one at a time, polls while idle,
and uses a local process lock to reject a second worker. Ctrl+C requests a
cooperative shutdown. The CV pipeline atomically checkpoints its full tracking
and identity state every 100 frames, writes resumable video segments, and
assembles them after processing. On restart, interrupted analyses with a saved
checkpoint are requeued and resume from that frame; interruptions before the
first checkpoint are requeued from the beginning. Explicitly cancelled analyses
are not retried. Cancellation can be requested for an owned analysis with
`POST /api/v1/analyses/{analysis_id}/cancel`.

The worker launches the existing CV CLI as a child process with an explicit
argument list, executable, and working directory; it does not import the CV
engine or load YOLO in the API process. The CV checkout remains independently
runnable. The adapter passes the video, model, output directory, progress file,
checkpoint path, FFmpeg executable, backend, canonical-ID mode, ball gate, and
team smoothing; it does not forward arbitrary analysis options or enable
tracking-stub flags.

Each job writes under
`local-data/analyses/<analysis-id>/cv-output/` and stores separate
`logs/cv-stdout.log` and `logs/cv-stderr.log` files. Log output is streamed to
disk and each log is capped at 10 MiB. The adapter validates that its own new
CV run directory contains non-empty `tracked.mp4` and `observations.csv`
before reporting success. CPU processing can be slow, especially for full
matches. Cancellation and worker shutdown signal the CV process group, wait
for the configured grace period, and then force termination (including
descendants where the operating system supports process groups/job objects).
Backend tests use fake CV subprocesses and do not require the real CV
environment or launch YOLO; real CV smoke testing is not part of normal pytest.

To start the Vite frontend in a second terminal, from the project root run
`bun install` once and then `bun run dev`. Open the Vite URL printed by the
command (normally `http://localhost:5173`).

## General Workflow

Run backend commands from `./backend/` with `uv run`. The API environment uses
Python 3.14+; keep it separate from the existing CV engine's Python 3.10
environment.

Modify or add SQLModel models for data and SQL tables in `./backend/app/models.py`, API endpoints in `./backend/app/api/`, CRUD (Create, Read, Update, Delete) utils in `./backend/app/crud.py`.

## VS Code

There are already configurations in place to run the backend through the VS Code debugger, so that you can use breakpoints, pause and explore variables, etc.

The setup is also already configured so you can run the tests through the VS Code Python tests tab.

## Full Stack with Docker Compose

To run the backend and built frontend in Docker Compose:

```console
$ docker compose run --rm backend bash scripts/prestart.sh
$ docker compose watch
```

The application is available at `http://localhost:8000`.

### Docker Compose Override

The `compose.override.yml` file contains local settings for published ports, source synchronization, automatic image rebuilds, and backend reloads. Docker Compose applies it automatically when you run `docker compose` without an explicit file list.

To open a shell in the backend container:

```console
$ docker compose exec backend bash
```

## Backend Tests

To test the backend from the `backend` directory, run:

```console
uv run pytest
```

The test configuration always creates a fresh temporary SQLite database,
applies the default SQLite Alembic baseline, and removes the database after
the test session. It refuses to run if the configured engine is not SQLite or
the database path is outside that temporary directory. It does not use or
delete the developer database.

The tests run with Pytest. Modify existing tests or add new ones in
`./backend/tests/`.

If you use GitHub Actions, the tests will run automatically.

### Test a Running Stack

If your stack is already up and you just want to run the tests, you can use:

```bash
docker compose exec backend bash scripts/tests-start.sh
```

The `/app/backend/scripts/tests-start.sh` script calls `pytest`. If you need to pass extra arguments to `pytest`, you can pass them to that command and they will be forwarded.

For example, to stop on first error:

```bash
docker compose exec backend bash scripts/tests-start.sh -x
```

### Test Coverage

When the tests run, they generate `htmlcov/index.html`. Open it in your browser to inspect the test coverage.

## Migrations

Make sure you create a revision of your models and upgrade the database with
that revision every time you change them. From the `backend` directory, use
`uv` to run Alembic:

* The default `alembic.ini` selects the new SQLite baseline and imports
  SQLModel metadata from `./backend/app/models.py`.
* The historical PostgreSQL migration chain under
  `./backend/app/alembic/versions/` is preserved and can be selected with
  `alembic.postgresql.ini`. Use that configuration only with a PostgreSQL
  database. It is not part of the local SQLite workflow.

* After changing a model (for example, adding a column), create a revision:

```console
uv run alembic revision --autogenerate -m "Add column last_name to User model"
```

* Commit to the git repository the files generated in the alembic directory.

* After creating the revision, run the migration in the database (this is what will actually change the database):

```console
uv run alembic upgrade head
```

SQLite schema changes that require table alterations should use Alembic batch
operations; online SQLite migrations enable `render_as_batch`. PostgreSQL
migrations must be created and upgraded using the legacy PostgreSQL config and
a PostgreSQL URL.

## Email Templates

The email templates are written with [React Email](https://react.email) in `./packages/react-email/`. The `emails` directory holds one component per email and the `ui` directory holds the shared components (layout, heading, button, link, callout).

The rendered HTML in `./backend/app/email-templates/` is generated from those components. It is what the application sends and should not be edited by hand.

To preview the emails while editing them, start the dev server from the root of the project:

```console
$ bun run email:dev
```

Values coming from the backend are declared as Jinja placeholders in the component props, for example `username = "{{ username }}"`. The context for each email is built in `generate_*_email()` in `./backend/app/utils.py`, so a new placeholder needs to be added there too.

Once you are done, regenerate the templates used by the application:

```console
$ bun run email:export
```
