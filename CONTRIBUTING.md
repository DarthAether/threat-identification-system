# Contributing to Threat Identification System

Thank you for your interest in contributing. This document provides guidelines and instructions for contributing to this project.

## Development Setup

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- PostgreSQL 16 (or use Docker)
- Redis 7 (or use Docker)
- Git

### Getting Started

1. Fork and clone the repository:

```bash
git clone https://github.com/<your-username>/threat-identification-system.git
cd threat-identification-system
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

3. Install development dependencies:

```bash
make install-dev
```

4. Copy the environment file and configure it:

```bash
cp .env.example .env
# Edit .env with your local settings
```

5. Start infrastructure services:

```bash
make docker-dev
```

6. Run database migrations:

```bash
make migrate
```

7. Verify the setup:

```bash
make test
```

## Code Quality

This project enforces strict code quality standards. All checks must pass before a PR can be merged.

### Linting and Formatting

We use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting:

```bash
make lint      # Check for linting issues
make format    # Auto-format code
```

### Type Checking

We use [mypy](https://mypy-lang.org/) for static type analysis:

```bash
make typecheck
```

All public functions must have type annotations.

### Testing

We use [pytest](https://pytest.org/) for testing:

```bash
make test           # Run all tests
make test-unit      # Run unit tests only
make test-integration  # Run integration tests
make test-cov       # Run tests with coverage report
```

Guidelines for tests:
- All new features must include tests
- Aim for at least 80% code coverage on new code
- Place unit tests in `tests/unit/` and integration tests in `tests/integration/`
- Use fixtures from `tests/conftest.py` for shared setup

### Pre-Commit Hooks

Pre-commit hooks run automatically on each commit:

```bash
pre-commit install     # Install hooks (done by make install-dev)
pre-commit run --all-files  # Run all hooks manually
```

## Branching Strategy

- `main` - Production-ready code. Protected branch.
- `develop` - Integration branch for features.
- `feature/<name>` - New features. Branch from `develop`.
- `bugfix/<name>` - Bug fixes. Branch from `develop`.
- `hotfix/<name>` - Urgent production fixes. Branch from `main`.
- `release/<version>` - Release preparation. Branch from `develop`.

### Branch Naming

Use descriptive, kebab-case names:
- `feature/add-face-recognition-api`
- `bugfix/fix-camera-reconnection`
- `hotfix/patch-auth-bypass`

## Pull Request Process

1. Create your feature branch from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/my-feature
   ```

2. Make your changes and commit with clear messages:
   ```bash
   git commit -m "feat: add face recognition endpoint"
   ```

3. Ensure all checks pass:
   ```bash
   make lint
   make typecheck
   make test
   ```

4. Push your branch and open a PR against `develop`:
   ```bash
   git push origin feature/my-feature
   ```

5. Fill out the PR template completely.

6. Address review feedback promptly.

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, no logic change)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks
- `perf:` - Performance improvements
- `ci:` - CI/CD changes

## Project Structure

```
threat-identification-system/
  src/threat_id/         # Application source code
    api/                 # FastAPI routes and middleware
    camera/              # Camera capture and management
    core/                # Configuration and shared utilities
    db/                  # Database models and migrations
    detection/           # Object/threat detection engine
    recognition/         # Face recognition module
    alerting/            # Alert dispatch (email, SMS, webhook)
    observability/       # Metrics, logging, tracing
    pipeline/            # Detection pipeline orchestration
  tests/                 # Test suite
    unit/
    integration/
    load/
  docker/                # Docker and orchestration files
  scripts/               # Utility scripts
```

## Security

If you discover a security vulnerability, please **do not** open a public issue. Instead, email the maintainer directly.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
