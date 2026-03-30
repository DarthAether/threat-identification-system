#!/usr/bin/env python3
"""Seed the database with an initial admin user and optional face records.

This script is intended for development and initial deployment setup.
It creates an admin user and optionally imports face images from a directory.

Usage:
    python scripts/seed_faces.py
    python scripts/seed_faces.py --faces-dir data/faces --admin-email admin@example.com
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Supported image extensions for face records
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed the database with an admin user and optional face records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--admin-email",
        type=str,
        default="admin@threat-id.local",
        help="Email address for the admin user.",
    )
    parser.add_argument(
        "--admin-name",
        type=str,
        default="System Administrator",
        help="Display name for the admin user.",
    )
    parser.add_argument(
        "--admin-password",
        type=str,
        default=None,
        help="Password for the admin user. If not provided, reads from ADMIN_PASSWORD env var.",
    )
    parser.add_argument(
        "--faces-dir",
        type=str,
        default=None,
        help=(
            "Directory containing face images to seed. "
            "Each subdirectory name is used as the person's identity label. "
            "Structure: faces_dir/<person_name>/<image_files>"
        ),
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Database URL. Defaults to DATABASE_URL env var.",
    )
    return parser.parse_args()


async def create_admin_user(
    session: object,
    email: str,
    name: str,
    password: str,
) -> None:
    """Create the initial admin user if they don't already exist."""
    try:
        from threat_id.db.models import User
        from sqlalchemy import select
        from sqlalchemy.ext.asyncio import AsyncSession

        assert isinstance(session, AsyncSession)

        result = await session.execute(select(User).where(User.email == email))
        existing = result.scalar_one_or_none()

        if existing:
            logger.info("Admin user already exists: %s", email)
            return

        # Hash the password
        try:
            from threat_id.core.security import hash_password

            hashed = hash_password(password)
        except ImportError:
            import hashlib

            hashed = hashlib.sha256(password.encode()).hexdigest()
            logger.warning("Using basic SHA-256 hash; threat_id.core.security not available.")

        admin = User(
            email=email,
            name=name,
            hashed_password=hashed,
            is_admin=True,
            is_active=True,
        )
        session.add(admin)
        await session.commit()
        logger.info("Created admin user: %s <%s>", name, email)

    except ImportError as e:
        logger.error("Required module not available: %s", e)
        logger.error("Make sure the threat_id package is installed.")
        sys.exit(1)


async def seed_faces(session: object, faces_dir: Path) -> None:
    """Import face records from a directory structure.

    Expected structure:
        faces_dir/
            John_Doe/
                photo1.jpg
                photo2.png
            Jane_Smith/
                photo1.jpg
    """
    if not faces_dir.exists():
        logger.warning("Faces directory does not exist: %s", faces_dir)
        return

    if not faces_dir.is_dir():
        logger.error("Faces path is not a directory: %s", faces_dir)
        return

    try:
        from threat_id.db.models import FaceRecord
        from sqlalchemy import select
        from sqlalchemy.ext.asyncio import AsyncSession

        assert isinstance(session, AsyncSession)
    except ImportError as e:
        logger.error("Required module not available: %s", e)
        return

    total_imported = 0

    for person_dir in sorted(faces_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name.replace("_", " ")
        image_files = [
            f
            for f in person_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not image_files:
            logger.warning("No images found for: %s", person_name)
            continue

        # Check if this person already has records
        result = await session.execute(
            select(FaceRecord).where(FaceRecord.label == person_name)
        )
        existing = result.scalars().all()

        if existing:
            logger.info(
                "Skipping %s: already has %d face record(s)",
                person_name,
                len(existing),
            )
            continue

        for image_file in image_files:
            image_data = image_file.read_bytes()

            record = FaceRecord(
                label=person_name,
                image_path=str(image_file),
                image_data=image_data,
            )
            session.add(record)
            total_imported += 1

        logger.info(
            "Imported %d image(s) for: %s",
            len(image_files),
            person_name,
        )

    await session.commit()
    logger.info("Total face records imported: %d", total_imported)


async def main() -> None:
    args = parse_args()

    # Resolve database URL
    database_url = args.database_url or os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error(
            "No database URL provided. Use --database-url or set DATABASE_URL env var."
        )
        sys.exit(1)

    # Resolve admin password
    password = args.admin_password or os.environ.get("ADMIN_PASSWORD")
    if not password:
        logger.error(
            "No admin password provided. Use --admin-password or set ADMIN_PASSWORD env var."
        )
        sys.exit(1)

    try:
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
        from sqlalchemy.orm import sessionmaker
    except ImportError:
        logger.error("SQLAlchemy async not available. Install with: pip install sqlalchemy[asyncio]")
        sys.exit(1)

    engine = create_async_engine(database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Create admin user
        await create_admin_user(session, args.admin_email, args.admin_name, password)

        # Seed face records if directory provided
        if args.faces_dir:
            await seed_faces(session, Path(args.faces_dir))

    await engine.dispose()
    logger.info("Database seeding complete.")


if __name__ == "__main__":
    asyncio.run(main())
