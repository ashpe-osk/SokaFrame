# Full Stack FastAPI Template

[![Test Docker Compose](../../actions/workflows/test-docker-compose.yml/badge.svg)](../../actions/workflows/test-docker-compose.yml)
[![Test Backend](../../actions/workflows/test-backend.yml/badge.svg)](../../actions/workflows/test-backend.yml)

## Technology Stack and Features

- ⚡ [**FastAPI**](https://fastapi.tiangolo.com) for the Python backend API.
  - 🧰 [SQLModel](https://sqlmodel.tiangolo.com) for the Python SQL database interactions (ORM).
  - 🔍 [Pydantic](https://docs.pydantic.dev), used by FastAPI, for the data validation and settings management.
  - 💾 [PostgreSQL](https://www.postgresql.org) as the SQL database.
- 🚀 [React](https://react.dev) for the frontend.
  - 🧩 Built into the backend application and served by FastAPI on the same domain as the API.
  - 💃 Using TypeScript, hooks, [Vite](https://vitejs.dev), and other parts of a modern frontend stack.
  - 🎨 [Tailwind CSS](https://tailwindcss.com) and [shadcn/ui](https://ui.shadcn.com) for the frontend components.
  - 🤖 An automatically generated frontend client.
  - 🧪 [Playwright](https://playwright.dev) for end-to-end testing.
  - 🦇 Dark mode support.
- ☁️ [FastAPI Cloud](https://fastapicloud.com) for deployment.
- 🐋 [Docker Compose](https://www.docker.com) for local services and self-hosted deployment.
  - 📞 [Traefik](https://traefik.io) as a reverse proxy with automatic HTTPS.
- 🔒 Secure password hashing by default.
- 🔑 JWT (JSON Web Token) authentication.
- 📫 Email-based password recovery.
- ✉️ [React Email](https://react.email) for email templates.
- 📬 [Mailpit](https://mailpit.axllent.org) for local email testing during development.
- ✅ Tests with [Pytest](https://pytest.org).
- 🏭 CI (continuous integration) and CD (continuous deployment) based on GitHub Actions.

### Dashboard Login

![Dashboard login screenshot](img/login.png)

### Dashboard - Admin

![Admin dashboard screenshot](img/dashboard.png)

### Dashboard - Items

![Items dashboard screenshot](img/dashboard-items.png)

### Dashboard - Dark Mode

![Dark mode dashboard screenshot](img/dashboard-dark.png)

### React Email Templates

![Email templates screenshot](img/react-email.png)

### Mailpit - Local Email Testing

![Mailpit screenshot](img/mailpit.png)

### Interactive API Documentation

![API docs](img/docs.png)

## How to Use It

Click the **Use this template** button at the top of this page to create a new repository.

## Local Startup (Windows)

For the normal local workflow, start the backend API and frontend in separate
PowerShell terminals. Start the analysis worker in a third terminal when you
want uploaded videos to be processed. Keep the terminals open while using the
app.

### One-time setup

Install [uv](https://docs.astral.sh/uv/), Python 3.14+, Node.js (which
includes npm), then install dependencies and apply the database migrations:

```powershell
cd C:\omniBoard-Sportz\omniBoard
npm.cmd install
Copy-Item .env.example .env

cd C:\omniBoard-Sportz\omniBoard\backend
uv sync
$env:DATABASE_URL = "sqlite:///./sql_app.db"
uv run alembic upgrade head
```

The backend reads required settings from the private
`C:\omniBoard-Sportz\omniBoard\.env` file. Create it from `.env.example`, then
replace the secret placeholders and set the CV paths for the machine. Keep
`.env` local; it is not tracked by Git.

### Terminal 1 — Backend API

If this terminal is using the CV virtual environment (the prompt shows
`(.venv-1)`), run `deactivate` first. Do not use `uv run --active`; uv must use
OmniBoard's backend environment.

```powershell
cd C:\omniBoard-Sportz\omniBoard\backend
$env:DATABASE_URL = "sqlite:///./sql_app.db"
$env:PYTHONUTF8 = "1"
uv run fastapi run app/main.py --host 127.0.0.1 --port 8000
```

This runs without the development file watcher, avoiding the Windows
`WinError 10013` watcher socket error. Leave this terminal running. The API is
available at `http://127.0.0.1:8000`.

### Terminal 2 — Frontend

```powershell
cd C:\omniBoard-Sportz\omniBoard
cd frontend
npm.cmd run dev
```

Leave this terminal running, then open `http://localhost:5173`. If the browser
shows `ERR_CONNECTION_REFUSED` for this address, start or restart this
frontend command.

### Terminal 3 — Analysis worker (video processing)

The worker is required to move uploaded analyses from `queued` into processing.
It uses the CV checkout at `C:\omniBoard-Sportz`:

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

Keep this terminal open while analyses are processing. The separate CV
environment requires Python 3.10; `ffmpeg` and `ffprobe`, including the H.264
`libx264` encoder, must also be available on `PATH`.

## Backend Development

Backend docs: [backend/README.md](./backend/README.md).

## Frontend Development

Frontend docs: [frontend/README.md](./frontend/README.md).

## Deployment

FastAPI Cloud deployment: [deployment.md](./deployment.md).

Self-hosted deployment with Docker Compose: [deployment-docker-compose.md](./deployment-docker-compose.md).

## Development

General development docs: [development.md](./development.md).

This includes the local FastAPI and Vite workflow, Docker Compose services, `.env` configuration, and more.

## Release Notes

Check the file [release-notes.md](./release-notes.md).

## License

The Full Stack FastAPI Template is licensed under the terms of the MIT license.
