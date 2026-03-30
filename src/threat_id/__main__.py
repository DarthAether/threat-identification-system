"""Entry point for ``python -m threat_id``.

Loads configuration, builds the FastAPI application via the factory, and
starts the uvicorn ASGI server.
"""

from __future__ import annotations

import sys

import uvicorn

from threat_id.api.app import create_app
from threat_id.core.config import Settings


def main() -> None:
    """Parse settings and run the application server."""
    settings = Settings()
    app = create_app(settings)

    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.api.log_level,
        access_log=settings.api.debug,
    )


if __name__ == "__main__":
    main()
