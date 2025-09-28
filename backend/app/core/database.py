"""
Database connection and session management.

This module provides database connectivity, session management, and
pgvector setup for our RAG system.
"""

from typing import AsyncGenerator
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
import logging

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Database URL for async operations (asyncpg driver)
ASYNC_DATABASE_URL = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")

# Create database engines
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=300,    # Recycle connections every 5 minutes
    echo=settings.debug  # Log SQL queries in debug mode
)

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.debug
)

# Session makers
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False
)

# Base class for all database models
Base = declarative_base()


def get_db() -> Session:
    """
    Get a database session for synchronous operations.

    Usage:
        with get_db() as db:
            user = db.query(User).first()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session for asynchronous operations.

    Usage:
        async with get_async_db() as db:
            result = await db.execute(select(User))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def create_database():
    """Create the database if it doesn't exist."""
    try:
        # Test connection to database
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Connected to database: {settings.database_name}")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise


def setup_pgvector():
    """
    Install and enable the pgvector extension.

    This function:
    1. Connects to PostgreSQL
    2. Creates the pgvector extension if not exists
    3. Verifies the extension is working
    """
    try:
        with engine.connect() as conn:
            # Enable pgvector extension
            logger.info("Setting up pgvector extension...")

            # Create extension (requires superuser privileges)
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

            # Verify extension is installed
            result = conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            ).fetchone()

            if result:
                logger.info("✅ pgvector extension successfully installed")

                # Test vector operations
                conn.execute(text("SELECT '[1,2,3]'::vector"))
                logger.info("✅ Vector operations working correctly")

            else:
                raise Exception("pgvector extension not found after installation")

    except Exception as e:
        logger.error(f"Failed to setup pgvector: {e}")
        logger.warning("Make sure PostgreSQL user has superuser privileges")
        logger.warning("Or run: CREATE EXTENSION vector; as a superuser")
        raise


def create_tables():
    """Create all database tables defined in models."""
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise


def init_database():
    """Initialize the entire database setup."""
    logger.info("Initializing database...")

    # Step 1: Verify database connection
    create_database()

    # Step 2: Setup pgvector extension
    setup_pgvector()

    # Step 3: Create tables
    create_tables()

    logger.info("🎉 Database initialization complete!")


# Connection health check
def check_database_health() -> dict:
    """Check database connectivity and extension status."""
    try:
        with engine.connect() as conn:
            # Test basic connectivity
            conn.execute(text("SELECT 1"))

            # Check pgvector extension
            vector_result = conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            ).fetchone()

            # Test vector operations
            try:
                conn.execute(text("SELECT '[1,2,3]'::vector"))
                vector_working = True
            except:
                vector_working = False

            return {
                "database_connected": True,
                "pgvector_installed": bool(vector_result),
                "vector_operations": vector_working,
                "database_url": settings.database_url.split('@')[1] if '@' in settings.database_url else "hidden"
            }
    except Exception as e:
        return {
            "database_connected": False,
            "error": str(e),
            "database_url": settings.database_url.split('@')[1] if '@' in settings.database_url else "hidden"
        }