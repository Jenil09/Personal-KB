"""Helpers for asserting on query plans.

Index coverage is a claim that gets made in a comment and quietly stops being
true — a predicate changes, a cast creeps in, and the query still returns the
right rows a thousand times slower. The only way to hold it is to ask Postgres
what it actually did.

`EXPLAIN` is a compiled construct rather than string concatenation on purpose.
Rendering a statement to SQL and prefixing it would mean either `literal_binds`
— which cannot render a JSONB parameter at all, so the one query most worth
checking is the one it fails on — or re-typing the SQL in the test, which then
proves an index covers a query nothing runs. Compiling through the normal path
keeps the binds and their types exactly as the repository sends them.
"""

from typing import Any

from sqlalchemy import ClauseElement, Executable
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.compiler import SQLCompiler

__all__ = ["Explain", "explain", "plan_for", "uses_index"]


class Explain(Executable, ClauseElement):
    """`EXPLAIN` around a statement.

    Typed as `ClauseElement` rather than `Executable` because the compiler
    visits it rather than executing it — only the `Explain` wrapper is ever
    handed to `session.execute`.
    """

    inherit_cache = False

    def __init__(self, statement: ClauseElement, *, analyze: bool = False) -> None:
        self.statement = statement
        self.analyze = analyze


@compiles(Explain, "postgresql")
def _compile_explain(element: Explain, compiler: SQLCompiler, **kw: Any) -> str:
    # `Any` in the kwargs: these are the compiler's own options, passed straight
    # through to the inner statement.
    options = "ANALYZE, BUFFERS, FORMAT TEXT" if element.analyze else "FORMAT TEXT"
    return f"EXPLAIN ({options}) {compiler.process(element.statement, **kw)}"


async def plan_for(
    session: AsyncSession, statement: ClauseElement, *, analyze: bool = False
) -> str:
    """The query plan as one string, ready to assert against.

    `analyze` runs the statement for real. Off by default: the plan is the
    claim, and executing a query to find out how it would be executed is a
    surprising side effect for a helper called from a `where` clause test.
    """
    result = await session.execute(Explain(statement, analyze=analyze))
    return "\n".join(str(line) for line in result.scalars())


async def explain(session: AsyncSession, statement: ClauseElement) -> str:
    """Alias for `plan_for` without the keyword, for readability at call sites."""
    return await plan_for(session, statement)


def uses_index(plan: str, index_name: str) -> bool:
    """Whether `plan` reaches its rows through the named index.

    Checks for the index by name rather than for the absence of `Seq Scan`: a
    plan can legitimately contain both, and a query that scans sequentially
    *and* happens to mention the index elsewhere is exactly the case a
    negative check would wave through.
    """
    return any(
        access in plan
        for access in (
            f"Index Scan using {index_name}",
            f"Index Only Scan using {index_name}",
            f"Bitmap Index Scan on {index_name}",
        )
    )
