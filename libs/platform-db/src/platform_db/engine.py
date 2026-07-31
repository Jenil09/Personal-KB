"""The async engine, the session scope, and the liveness probe.

One `Database` per service, created in the composition root and disposed in the
lifespan handler. Nothing else constructs an engine — a second engine means a
second pool, and the pool size is the thing keeping connection count under the
box's budget.

`Database` deliberately does not import from `platform-fastapi`. `ping()`
returns a bool and the service adapts it into a `HealthCheck`, so the database
layer stays usable from the CLI and from migrations.
"""

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.sql import text

from platform_core import get_logger
from platform_db.settings import DatabaseSettings

__all__ = ["Database", "SessionSource"]

_logger = get_logger("platform.db")

_PING = text("SELECT 1")


class SessionSource(Protocol):
    """All the audit tiers need from a database: a transactional scope.

    `Database` satisfies it. Narrowing the dependency this far is what lets the
    tier-1 spill decision — the one behaviour AD-013 hinges on — be exercised
    against a session that fails on demand, rather than only against a real
    Postgres that has to be killed first.
    """

    def session(self) -> AbstractAsyncContextManager[AsyncSession]: ...


class Database:
    """Owns the engine and hands out sessions.

    Constructing this does not connect; `create_async_engine` is lazy and the
    first connection is made on first use. Call `ping()` at startup if you want
    a misconfigured DSN to surface there rather than on the first request.
    """

    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings = settings
        self._engine = _create_engine(settings)
        self._sessionmaker = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
            autoflush=False,
        )

    @property
    def engine(self) -> AsyncEngine:
        """For Alembic and for the rare statement that needs no ORM session."""
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """A session that commits on a clean exit and rolls back on any error.

        The block owns the transaction boundary. A caller wanting to keep
        writes out of the request's transaction — the tier-1 audit writer is
        the one that does — opens its own scope rather than reusing this one.
        """
        session = self._sessionmaker()
        try:
            yield session
        except BaseException:
            await session.rollback()
            raise
        else:
            await session.commit()
        finally:
            await session.close()

    async def session_dependency(self) -> AsyncGenerator[AsyncSession]:
        """`Depends`-compatible view of `session()`.

        A bare async generator, so binding it costs `platform-db` no FastAPI
        import. FastAPI throws the handler's exception back in at the yield,
        which is what gives the rollback-on-error semantics here.
        """
        async with self.session() as session:
            yield session

    async def ping(self) -> bool:
        """`SELECT 1`. False rather than raising — this backs a health check."""
        try:
            async with self._engine.connect() as connection:
                await connection.execute(_PING)
        except (SQLAlchemyError, OSError) as exc:
            _logger.warning("database_ping_failed", exc_info=exc)
            return False
        return True

    async def dispose(self) -> None:
        """Close every pooled connection. Idempotent; safe to call at shutdown."""
        await self._engine.dispose()


def _create_engine(settings: DatabaseSettings) -> AsyncEngine:
    # `statement_timeout` is a server setting rather than asyncpg's
    # `command_timeout` because the server-side one also stops the query
    # itself, instead of only abandoning the client's wait for it.
    connect_args: dict[str, Any] = {
        "timeout": settings.connect_timeout_seconds,
        "server_settings": {
            "statement_timeout": str(int(settings.statement_timeout_seconds * 1000)),
        },
    }
    return create_async_engine(
        settings.dsn.get_secret_value(),
        echo=settings.echo,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout=settings.pool_timeout_seconds,
        pool_recycle=settings.pool_recycle_seconds,
        # Costs a round trip per checkout and buys immunity to the connection
        # a restarted Postgres left in the pool. At this traffic, free.
        pool_pre_ping=True,
        connect_args=connect_args,
    )
