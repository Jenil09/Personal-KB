"""The `kb` command.

Two front ends over one client. The subcommands here are the scriptable half —
they print, they exit with a status, they compose with `xargs` — and `kb` with
no arguments opens the Textual browser, which is the half for the times the
question is "what is even in here".

Every command follows the same shape: resolve settings, open one client, call
it, hand the result to `render`. Failures are not caught per command; they
propagate to `__main__.main`, which is the single place that turns a
`PlatformError` into a printed message and an exit code — the same argument as
the service having exactly one exception handler.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

import typer

from kb_cli import render
from kb_cli.client import KbClient, open_client
from kb_cli.config import KbCliSettings, config_path, load_config_file, load_settings, redact
from kb_cli.config import save_config_file as _save_config_file
from kb_cli.editor import edit_text
from kb_cli.ingest import DEFAULT_EXCLUDES, DEFAULT_GLOBS, discover, read_documents
from kb_cli.models import DocumentSummary
from platform_core import ConfigurationError, NotFoundError, PlatformError, ValidationError

__all__ = ["app"]

app = typer.Typer(
    name="kb",
    help="Operator client for the personal knowledge base.",
    no_args_is_help=False,
    add_completion=True,
    rich_markup_mode="rich",
)

config_app = typer.Typer(name="config", help="Read and write the stored configuration.")
app.add_typer(config_app)


def _run[T](coroutine: Awaitable[T]) -> T:
    return asyncio.run(_await(coroutine))


async def _await[T](coroutine: Awaitable[T]) -> T:
    return await coroutine


def _with_client[T](work: Callable[[KbClient], Awaitable[T]]) -> T:
    """Open one client, run `work`, close it — the body of nearly every command."""

    async def runner() -> T:
        async with open_client(load_settings()) as client:
            return await work(client)

    return _run(runner())


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    """Open the interactive browser when no subcommand is given.

    `no_args_is_help=False` above is what makes this reachable: Typer's default
    is to print help for a bare invocation, and the whole point of the browser
    is that it is what you get by typing `kb`.
    """
    if ctx.invoked_subcommand is not None:
        return
    _launch_tui()


@app.command("menu")
def menu() -> None:
    """Open the interactive document browser."""
    _launch_tui()


def _launch_tui() -> None:
    # Imported here rather than at module scope so that `kb search …` in a
    # script does not pay Textual's import cost, and so a headless machine with
    # a broken terminal can still run the subcommands.
    from kb_cli.tui.app import KbApp

    KbApp().run()


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------


@app.command("list")
def list_documents(
    document_type: Annotated[
        str | None, typer.Option("--type", "-t", help="Only this document type.")
    ] = None,
    tag: Annotated[
        list[str] | None, typer.Option("--tag", help="Repeatable. Matches any tag given.")
    ] = None,
    match_all_tags: Annotated[
        bool, typer.Option("--all-tags", help="Require every --tag rather than any.")
    ] = False,
    source: Annotated[str | None, typer.Option("--source", help="Exact source path.")] = None,
    collection: Annotated[str | None, typer.Option("--collection")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, max=100)] = 50,
    offset: Annotated[int, typer.Option("--offset", min=0)] = 0,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """List documents, newest first."""

    async def work(client: KbClient) -> None:
        page = await client.list_documents(
            limit=limit,
            offset=offset,
            document_type=document_type,
            source=source,
            tags=tag or (),
            match_all_tags=match_all_tags,
            collection=collection,
        )
        if as_json:
            render.console.print_json(page.model_dump_json())
            return
        render.console.print(render.documents_table(page))

    _with_client(work)


@app.command("show")
def show_document(
    document: Annotated[str, typer.Argument(help="Document id, or a unique prefix of one.")],
    metadata_only: Annotated[
        bool, typer.Option("--metadata", "-m", help="Omit the content.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show one document, content included."""

    async def work(client: KbClient) -> None:
        detail = await client.get_document(await _resolve(client, document))
        if as_json:
            render.console.print_json(detail.model_dump_json())
            return
        render.console.print(render.document_panel(detail, content=not metadata_only))

    _with_client(work)


@app.command("delete")
def delete_document(
    documents: Annotated[list[str], typer.Argument(help="Document ids, or unique prefixes.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
) -> None:
    """Delete documents from both stores.

    Prints what each id resolves to before asking, because a prefix is exactly
    the kind of argument that is easy to get wrong and impossible to undo — the
    vectors are gone and re-ingesting costs another embed.
    """

    async def work(client: KbClient) -> None:
        resolved: list[tuple[UUID, str]] = []
        for reference in documents:
            document_id = await _resolve(client, reference)
            detail = await client.get_document(document_id)
            resolved.append((document_id, detail.title))
        if not yes:
            for document_id, title in resolved:
                render.console.print(f"  {render.short_id(document_id)}  {title}")
            noun = "document" if len(resolved) == 1 else "documents"
            if not typer.confirm(f"Delete {len(resolved)} {noun}?"):
                render.console.print("Nothing deleted.", style="dim")
                raise typer.Exit(code=1)
        for document_id, title in resolved:
            deleted = await client.delete_document(document_id)
            style, word = ("green", "deleted") if deleted else ("dim", "not found")
            render.console.print(f"  [{style}]{word}[/]  {render.short_id(document_id)}  {title}")

    _with_client(work)


# ---------------------------------------------------------------------------
# ingestion
# ---------------------------------------------------------------------------


@app.command("ingest")
def ingest(
    path: Annotated[Path, typer.Argument(exists=True, readable=True, help="A file to ingest.")],
    title: Annotated[
        str | None, typer.Option("--title", help="Overrides the inferred title.")
    ] = None,
    document_type: Annotated[str | None, typer.Option("--type", "-t")] = None,
    source: Annotated[
        str | None, typer.Option("--source", help="Overrides the inferred source (AD-020).")
    ] = None,
    tag: Annotated[list[str] | None, typer.Option("--tag", help="Repeatable.")] = None,
    provider: Annotated[str | None, typer.Option("--provider", help="AD-006 collection.")] = None,
) -> None:
    """Ingest a single file."""
    documents = read_documents([path], path.parent, document_type=document_type, tags=tag or ())
    for skipped in documents.skipped:
        render.console.print(f"  [yellow]skipped[/]    {skipped.path}  ({skipped.reason})")
    if not documents.documents:
        raise typer.Exit(code=1)
    local = documents.documents[0]

    async def work(client: KbClient) -> None:
        result = await client.ingest(
            title=title or local.title,
            content=local.content,
            document_type=local.type,
            source=source or local.source,
            tags=local.tags,
            provider=provider,
        )
        render.console.print(render.ingest_result_line(source or local.source, result))
        render.console.print(f"  {result.document_id}", style="dim")

    _with_client(work)


@app.command("ingest-dir")
def ingest_dir(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    glob: Annotated[
        list[str] | None,
        typer.Option("--glob", "-g", help=f"Repeatable. Defaults to {' '.join(DEFAULT_GLOBS)}."),
    ] = None,
    exclude: Annotated[
        list[str] | None, typer.Option("--exclude", help="Directory names to skip.")
    ] = None,
    document_type: Annotated[
        str | None,
        typer.Option("--type", "-t", help="Overrides the per-file type inferred from the folder."),
    ] = None,
    tag: Annotated[
        list[str] | None, typer.Option("--tag", help="Applied to every document.")
    ] = None,
    provider: Annotated[str | None, typer.Option("--provider")] = None,
    no_recursive: Annotated[bool, typer.Option("--no-recursive")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="List what would be ingested and stop.")
    ] = False,
) -> None:
    """Walk a directory and ingest everything matching.

    Re-runnable by design. Unchanged files embed nothing (AD-008) and are
    reported as `unchanged`, so the honest way to use this on a corpus that
    partly failed is to run it again.
    """
    paths = discover(
        directory,
        globs=glob or DEFAULT_GLOBS,
        excludes=exclude or DEFAULT_EXCLUDES,
        recursive=not no_recursive,
    )
    scanned = read_documents(paths, directory, document_type=document_type, tags=tag or ())
    for skipped in scanned.skipped:
        render.console.print(f"  [yellow]skipped[/]    {skipped.path}  ({skipped.reason})")
    if dry_run:
        for local in scanned.documents:
            render.console.print(f"  [dim]would ingest[/] {local.source}  [dim]({local.type})[/]")
        render.console.print(f"\n{len(scanned.documents)} files, nothing sent.", style="dim")
        return
    if not scanned.documents:
        render.console.print("Nothing to ingest.", style="dim")
        return

    async def work(client: KbClient) -> None:
        ingested = unchanged = failed = 0
        tokens = 0
        for index, local in enumerate(scanned.documents, start=1):
            prefix = f"[dim]{index}/{len(scanned.documents)}[/]"
            try:
                result = await client.ingest(
                    title=local.title,
                    content=local.content,
                    document_type=local.type,
                    source=local.source,
                    tags=local.tags,
                    provider=provider,
                )
            except PlatformError as exc:
                # One bad file must not abandon the other thirty-nine. The run
                # is retryable in full (AD-008), so the useful behaviour is to
                # finish and report, not to stop at the first 502.
                failed += 1
                render.console.print(f"{prefix}   [red]failed[/]     {local.source}")
                render.print_error(exc)
                continue
            tokens += result.total_tokens
            if result.unchanged:
                unchanged += 1
            else:
                ingested += 1
            render.console.print(f"{prefix} {render.ingest_result_line(local.source, result)}")
        render.console.print(
            f"\n{ingested} ingested · {unchanged} unchanged · {failed} failed · {tokens:,} tokens",
            style="bold" if failed else "dim",
        )
        if failed:
            raise typer.Exit(code=1)

    _with_client(work)


@app.command("new")
def new_document(
    document_type: Annotated[str, typer.Option("--type", "-t")] = "note",
    title: Annotated[str | None, typer.Option("--title")] = None,
    tag: Annotated[list[str] | None, typer.Option("--tag")] = None,
    provider: Annotated[str | None, typer.Option("--provider")] = None,
) -> None:
    """Compose a document in $EDITOR and ingest it."""
    content = edit_text("# \n\n", editor=load_settings().editor)
    if content is None:
        render.console.print("Nothing ingested.", style="dim")
        raise typer.Exit(code=1)
    resolved_title = title or _title_from(content)
    if not resolved_title:
        raise ValidationError("The document needs a title: give --title or start it with '# '.")

    async def work(client: KbClient) -> None:
        result = await client.ingest(
            title=resolved_title,
            content=content,
            document_type=document_type,
            # No `source`: this document did not come from a file, and inventing
            # one would make the next ingest of a real file with that name
            # supersede it (AD-020).
            source=None,
            tags=tag or (),
            provider=provider,
        )
        render.console.print(render.ingest_result_line(resolved_title, result))
        render.console.print(f"  {result.document_id}", style="dim")

    _with_client(work)


def _title_from(content: str) -> str | None:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or None
        if stripped:
            return None
    return None


# ---------------------------------------------------------------------------
# search and status
# ---------------------------------------------------------------------------


@app.command("search")
def search(
    query: Annotated[list[str], typer.Argument(help="The query. Quoting is optional.")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", min=1, max=50)] = 5,
    document_type: Annotated[str | None, typer.Option("--type", "-t")] = None,
    tag: Annotated[list[str] | None, typer.Option("--tag")] = None,
    match_all_tags: Annotated[bool, typer.Option("--all-tags")] = False,
    source: Annotated[str | None, typer.Option("--source")] = None,
    provider: Annotated[str | None, typer.Option("--provider")] = None,
    full: Annotated[bool, typer.Option("--full", help="Print whole chunks, not excerpts.")] = False,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Semantic search."""
    # Joined rather than requiring quotes: `kb search bare metal kubernetes` is
    # what someone types, and making that an arity error teaches nothing.
    text = " ".join(query).strip()
    if not text:
        raise ValidationError("The query is empty.")

    async def work(client: KbClient) -> None:
        response = await client.search(
            text,
            top_k=top_k,
            document_type=document_type,
            source=source,
            tags=tag or (),
            match_all_tags=match_all_tags,
            provider=provider,
        )
        if as_json:
            render.console.print_json(response.model_dump_json())
            return
        render.console.print(render.search_results(response, full_text=full))

    _with_client(work)


@app.command("status")
def status(
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Corpus size, collections, embedding spend, and observability health."""

    async def work(client: KbClient) -> None:
        stats = await client.stats()
        if as_json:
            render.console.print_json(stats.model_dump_json())
            return
        render.console.print(render.stats_view(stats))

    _with_client(work)


@app.command("health")
def health() -> None:
    """The service's own health report, which needs no API key."""

    async def work(client: KbClient) -> None:
        render.console.print_json(json.dumps(await client.health()))

    _with_client(work)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

_SETTABLE = (
    "base_url",
    "api_key",
    "provider",
    "timeout_seconds",
    "ingest_timeout_seconds",
    "editor",
)


@config_app.command("path")
def config_path_command() -> None:
    """Print where the configuration file lives on this machine."""
    render.console.print(str(config_path()))


@config_app.command("show")
def config_show() -> None:
    """The effective configuration, with the API key masked.

    Both halves are printed — what is stored in the file, and what the process
    actually resolved after the environment had its say — because the gap
    between them is the entire explanation for "I set it and it still talks to
    localhost".
    """
    stored = load_config_file()
    render.console.print(
        render.config_table(redact(stored) or {"(empty)": None}, source=f"stored · {config_path()}")
    )
    try:
        settings = load_settings()
    except ConfigurationError as exc:
        render.print_error(exc)
        raise typer.Exit(code=1) from exc
    effective = {
        "base_url": settings.base_url,
        "api_key": "********" if settings.api_key.get_secret_value() else None,
        "provider": settings.provider,
        "timeout_seconds": settings.timeout_seconds,
        "ingest_timeout_seconds": settings.ingest_timeout_seconds,
        "editor": settings.editor,
    }
    render.console.print(render.config_table(effective, source="effective · env over stored"))


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help=f"One of: {', '.join(_SETTABLE)}")],
    value: Annotated[str, typer.Argument()],
) -> None:
    """Set one value and save it."""
    stored = load_config_file()
    stored[_check_key(key)] = _coerce(key, value)
    path = _validated_save(stored)
    render.console.print(f"{key} saved to {path}", style="green")


@config_app.command("unset")
def config_unset(
    key: Annotated[str, typer.Argument(help=f"One of: {', '.join(_SETTABLE)}")],
) -> None:
    """Remove one value, falling back to the default."""
    stored = load_config_file()
    if stored.pop(_check_key(key), None) is None:
        render.console.print(f"{key} was not set.", style="dim")
        return
    path = _validated_save(stored)
    render.console.print(f"{key} removed from {path}", style="green")


@config_app.command("init")
def config_init(
    force: Annotated[bool, typer.Option("--force", help="Overwrite without asking.")] = False,
) -> None:
    """Create the configuration file interactively.

    The first thing to run after `uv tool install`. Existing values are offered
    as defaults so this doubles as "reconfigure", and the key is prompted
    hidden so it does not end up in shell history or over someone's shoulder.
    """
    stored = load_config_file()
    target = config_path()
    if stored and not force and not typer.confirm(f"{target} exists. Update it?", default=True):
        raise typer.Exit(code=1)

    base_url = typer.prompt(
        "Service URL", default=str(stored.get("base_url") or "http://localhost:8000")
    )
    api_key = typer.prompt(
        "API key", hide_input=True, default=stored.get("api_key") or "", show_default=False
    )
    provider = typer.prompt(
        "Embedding provider (blank for the service default)",
        default=str(stored.get("provider") or ""),
        show_default=False,
    )

    stored["base_url"] = base_url.strip()
    if api_key.strip():
        stored["api_key"] = api_key.strip()
    if provider.strip():
        stored["provider"] = provider.strip()
    else:
        stored.pop("provider", None)

    path = _validated_save(stored)
    render.console.print(f"Saved to {path}", style="green")
    if not stored.get("api_key"):
        render.console.print(
            "No API key stored — set KB_CLI__API_KEY or run `kb config set api_key …`.",
            style="yellow",
        )


def _check_key(key: str) -> str:
    if key not in _SETTABLE:
        raise ValidationError(f"{key!r} is not a setting. Choose one of: {', '.join(_SETTABLE)}")
    return key


def _coerce(key: str, value: str) -> Any:
    # `Any` because the settings fields are of mixed type and this is the
    # string-to-field boundary; the model validates the result immediately.
    if key.endswith("_seconds"):
        try:
            return float(value)
        except ValueError as exc:
            raise ValidationError(f"{key} must be a number, not {value!r}.") from exc
    return value


def _validated_save(values: dict[str, Any]) -> Path:
    """Validate before writing, so a bad value never reaches the file.

    Saving first and validating on the next run would leave the tool
    unstartable, and the command that fixes it — `kb config set` — reads the
    same broken file to do its work.

    A missing `api_key` is filled with a placeholder for the duration of this
    check and never written. Requiring one here would mean `kb config set
    base_url …` failed on a fresh install until the key had been set first —
    an ordering the operator has no way to know about — and would refuse to
    save at all on a machine that supplies the key through the environment.
    Everything else is validated exactly as it will be on the next run.
    """
    # `Any` for the same reason as `_coerce`.
    candidate = {key: value for key, value in values.items() if value is not None}
    candidate.setdefault("api_key", "unset")
    KbCliSettings(**candidate)
    return _save_config_file(values)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _resolve(client: KbClient, reference: str) -> UUID:
    """A full id, or the unique document whose id starts with `reference`.

    Prefix resolution costs a listing, which is why it is only attempted when
    the reference is not already a UUID. It refuses an ambiguous prefix rather
    than picking the first match — this feeds `delete`.
    """
    try:
        return UUID(reference)
    except ValueError:
        pass
    if len(reference) < 4:
        raise ValidationError(f"{reference!r} is too short to identify a document.")
    matches: list[DocumentSummary] = []
    async for page in client.iter_documents():
        matches += [
            document for document in page.documents if str(document.id).startswith(reference)
        ]
    return _one(matches, reference)


def _one(matches: Sequence[DocumentSummary], reference: str) -> UUID:
    if not matches:
        raise NotFoundError(f"No document has an id starting with {reference!r}.")
    if len(matches) > 1:
        listed = ", ".join(
            f"{render.short_id(document.id)} ({document.title})" for document in matches[:5]
        )
        raise ValidationError(f"{reference!r} matches {len(matches)} documents: {listed}")
    return matches[0].id
