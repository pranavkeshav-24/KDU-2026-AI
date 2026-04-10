# Production-Ready FastAPI Template

A reusable, production-grade FastAPI boilerplate prioritizing **async-first design**, **feature-based modularity**, **security by default**, and **developer ergonomics**.

## Features
- **Async PostgreSQL**: Uses `asyncpg` with async SQLAlchemy 2.0.
- **Authentication**: JWT-based login with separate short-lived access and long-lived refresh tokens. Secure password hashing with `bcrypt`.
- **RBAC**: Built-in dependency decorators to enforce User and Admin level access routing.
- **Structured JSON Logging**: `structlog` integration tailored for humans in development and systems (ELK) in production with correlation IDs on incoming requests.
- **Robust Error Handling**: Global exception capture ensuring clients always receive a consistent JSON payload for errors.
- **Security Validations**: Out-of-the-box rigid data validation using Pydantic V2 ensuring email standards and deep password-strength rules.
- **Rate limiting**: `slowapi` implementation securing the authentication routes.
- **Developer Ready**: Comes with 100% async-pytest workflows and an optimized Multi-Stage Dockerfile ready for CD systems.

## Tech Stack
| Component | Technology |
|---|---|
| **Framework** | FastAPI 0.115+ |
| **Server** | Uvicorn + Gunicorn |
| **ORM** | SQLAlchemy 2.0 (Async) |
| **Migrations** | Alembic |
| **Validation** | Pydantic V2 |
| **Security** | passlib (bcrypt) + python-jose |
| **Testing** | pytest + pytest-asyncio + httpx |
| **Containers** | Docker + docker-compose |

---

## 🏗️ Quick Start (Docker)

The fastest way to run this template is with Docker Compose, which automatically provisions the database.

1. **Clone the repository.**
2. **Setup environment variables:**
   ```bash
   cp .env.example .env
   # Make sure to update your JWT_SECRET_KEY in production!
   ```
3. **Start the application:**
   ```bash
   docker-compose up --build
   ```
   *Migrations are automatically run before the web service starts.*

4. **Verify it is running:**
   Visit: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)

---

## 💻 Manual Setup (Local Virtual Environment)

If you prefer to run things natively without docker, follow these instructions:

1. **Setup PostgreSQL** locally and ensure it is running. Update your branch's `.env` `DATABASE_URL` pointing to the live instance.
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   # (On Windows: venv\Scripts\activate)
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run migrations:**
   ```bash
   alembic upgrade head
   ```
5. **Start Uvicorn for local development:**
   ```bash
   uvicorn app.main:app --reload
   ```

---

## 📁 Project Structure (Feature-Based Layout)
```text
fastapi-template/
├── app/
│   ├── core/                      # Cross-cutting infrastructure
│   │   ├── config.py              # Pydantic BaseSettings
│   │   ├── database.py            # Async engine and ORM Base
│   │   ├── dependencies.py        # DI mapping for router validations
│   │   ├── exceptions.py          # Defined error hierarchies
│   │   ├── exception_handlers.py  # Standardizes payload drops
│   │   ├── logging.py             # Structlog 
│   │   └── security.py            # Verification context
│   │
│   ├── users/                     # "Users" feature module domain
│   │   ├── models.py              # SQLAlchemy Tables
│   │   ├── schemas.py             # Pydantic payloads 
│   │   ├── repository.py          # CRUD wrapper 
│   │   ├── service.py             # High-level domain logic
│   │   └── router.py              # Web router layer
│   │
│   ├── health/                    
│   │   └── router.py              # Minimal health monitors
│   └── main.py                    # Root app factory and lifespan
├── alembic/                       # Core migrations orchestrator
├── tests/                         # Root tests layout
```
*Note: Feature-Based structures scale infinitely safer than standard Layer-Based structures. Always create a new directory for a new feature (e.g. `app/products/`).*

---

## 🛠️ Environment Variables Configuration
Reference `.env.example` mapping properties to `app/core/config.py`:
| Variable | Description |
|---|---|
| `DATABASE_URL` | Application root DB URI containing all access variables *(Required fail-fast)* |
| `JWT_SECRET_KEY` | At least 32 bytes strong string to validate internal JWT scopes *(Required fail-fast)* |
| `ENVIRONMENT` | Choose from: `development`, `staging`, `production` |
| `DB_POOL_SIZE` | Standard connection scaling pools. Auto pings standard |
| `LOG_JSON_FORMAT` | Convert output logs from structural JSON (syslogs) to user-friendly shell output. |

---

## 📡 Core API Reference Layer
Find fully typed reference sets at `/docs`. Standard endpoints built in include:
- `POST /api/v1/auth/register` — User setup
- `POST /api/v1/auth/login` — Sign user state with JWT tokens
- `POST /api/v1/auth/refresh` — Issue an updated scope of tokens from previously generated refresh tokens.
- `GET /api/v1/users/me` *(Protected)* — Profile viewer.
- `GET /api/v1/users` *(Admin)* — Base endpoint restricting viewer content via the `require_admin` dependency mapping.
- `GET /health` and `GET /health/db` — Minimal monitors for external ping services or load balancers.

---

## 🧪 Testing Suite Setup
With a database active in Docker, all test protocols can be verified natively:
1. Verify the container instance for DB exists. The test suite operates via `postgresql+asyncpg://fastapi:secret@localhost:5432/fastapi_test`.
2. Generate independent test databases directly inside Postgres:
   ```bash
   docker exec -it <db_container_name> createdb -U fastapi fastapi_test
   ```
3. Run `pytest` natively (Tests operate under auto-rollback transaction protocols ensuring zero state decay).
   ```bash
   pytest --cov=app --cov-report=term-missing --cov-fail-under=70 -v
   ```