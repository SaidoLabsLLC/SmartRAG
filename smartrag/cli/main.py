"""SmartRAG CLI entry point."""

from __future__ import annotations

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="smartrag",
        description="SmartRAG — The retrieval engine that makes vector databases unnecessary.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a file or directory")
    ingest_parser.add_argument("path", help="File or directory to ingest")
    ingest_parser.add_argument(
        "--store", default="./knowledge", help="Knowledge store directory"
    )

    # ingest-url
    ingest_url_parser = subparsers.add_parser(
        "ingest-url", help="Ingest content from a URL"
    )
    ingest_url_parser.add_argument("url", help="URL to fetch and ingest")
    ingest_url_parser.add_argument(
        "--store", default="./knowledge", help="Knowledge store directory"
    )

    # query
    query_parser = subparsers.add_parser("query", help="Query the knowledge store")
    query_parser.add_argument("question", help="The question to ask")
    query_parser.add_argument(
        "--store", default="./knowledge", help="Knowledge store directory"
    )
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of results")

    # search
    search_parser = subparsers.add_parser("search", help="Search the knowledge store")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--store", default="./knowledge", help="Knowledge store directory"
    )
    search_parser.add_argument("--top-k", type=int, default=10, help="Number of results")

    # stats
    stats_parser = subparsers.add_parser("stats", help="Show knowledge store statistics")
    stats_parser.add_argument(
        "--store", default="./knowledge", help="Knowledge store directory"
    )

    # reindex
    reindex_parser = subparsers.add_parser("reindex", help="Rebuild all indexes")
    reindex_parser.add_argument(
        "--store", default="./knowledge", help="Knowledge store directory"
    )

    # export
    export_parser = subparsers.add_parser(
        "export", help="Export knowledge store as a portable .smartrag bundle"
    )
    export_parser.add_argument("output_path", help="Output path for the .smartrag bundle")
    export_parser.add_argument(
        "--store", default="./knowledge", help="Knowledge store directory"
    )
    export_parser.add_argument(
        "--with-embeddings",
        action="store_true",
        default=False,
        help="Include embedding vectors in the bundle",
    )

    # import
    import_parser = subparsers.add_parser(
        "import", help="Import a .smartrag bundle into a knowledge store"
    )
    import_parser.add_argument("bundle_path", help="Path to the .smartrag bundle file")
    import_parser.add_argument(
        "--store", default="./knowledge", help="Target knowledge store directory"
    )

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the REST API server")
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Port to listen on"
    )
    serve_parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to"
    )
    serve_parser.add_argument(
        "--store", default="./knowledge", help="Knowledge store directory"
    )

    # api-key
    apikey_parser = subparsers.add_parser("api-key", help="Manage API keys")
    apikey_sub = apikey_parser.add_subparsers(dest="apikey_action", help="API key actions")

    apikey_create = apikey_sub.add_parser("create", help="Create a new API key")
    apikey_create.add_argument("name", help="Name for the API key")
    apikey_create.add_argument(
        "--store", default="./knowledge", help="Knowledge store directory"
    )

    apikey_list = apikey_sub.add_parser("list", help="List API keys")
    apikey_list.add_argument(
        "--store", default="./knowledge", help="Knowledge store directory"
    )

    apikey_revoke = apikey_sub.add_parser("revoke", help="Revoke an API key")
    apikey_revoke.add_argument("name", help="Name of the key to revoke")
    apikey_revoke.add_argument(
        "--store", default="./knowledge", help="Knowledge store directory"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # --- Commands that do NOT need a SmartRAG instance ---

    if args.command == "serve":
        _cmd_serve(args)
        return

    if args.command == "api-key":
        _cmd_api_key(args, apikey_parser)
        return

    if args.command == "import":
        _cmd_import(args)
        return

    # --- Commands that DO need a SmartRAG instance ---

    from smartrag.core import SmartRAG

    rag = SmartRAG(args.store)

    if args.command == "ingest":
        results = rag.ingest(args.path)
        if isinstance(results, list):
            for r in results:
                status_icon = {"created": "+", "split": "~", "duplicate": "=", "failed": "!"}
                print(f"  [{status_icon.get(r.status, '?')}] {r.title} ({r.status})")
                if r.children:
                    for child in r.children:
                        print(f"      -> {child}")
            print(f"\nIngested {len(results)} files.")
        else:
            print(f"  {results.title} ({results.status})")

    elif args.command == "ingest-url":
        result = rag.ingest_url(args.url)
        status_icon = {"created": "+", "split": "~", "duplicate": "=", "failed": "!"}
        print(f"  [{status_icon.get(result.status, '?')}] {result.title} ({result.status})")
        if result.children:
            for child in result.children:
                print(f"      -> {child}")
        if result.error:
            print(f"  Error: {result.error}")

    elif args.command == "query":
        result = rag.query(args.question, top_k=args.top_k)
        if not result.results:
            print("No results found.")
        else:
            for i, r in enumerate(result.results, 1):
                print(f"\n{i}. [{r.title}] (Tier {r.tier_resolved}, score: {r.score:.3f})")
                print(f"   Slug: {r.slug}")
                if r.categories:
                    print(f"   Categories: {', '.join(r.categories)}")
                print(f"   {r.snippet[:200]}")
            print(f"\n--- {len(result.results)} results in {result.total_ms:.1f}ms, {result.total_bytes_read} bytes read ---")

    elif args.command == "search":
        results = rag.search(args.query, top_k=args.top_k)
        if not results:
            print("No results found.")
        else:
            for i, r in enumerate(results, 1):
                print(f"{i}. [{r.title}] (score: {r.score:.3f})")
                print(f"   {r.summary[:200]}")
            print(f"\n--- {len(results)} results ---")

    elif args.command == "stats":
        stats = rag.stats
        print(f"Documents:  {stats['document_count']}")
        print(f"Index size: {stats['index_size_bytes']} bytes")
        if stats.get("categories"):
            print(f"Categories: {', '.join(sorted(stats['categories']))}")

    elif args.command == "reindex":
        count = rag.reindex()
        print(f"Reindexed {count} documents.")

    elif args.command == "export":
        _cmd_export(args, rag)


def _cmd_export(args, rag):
    """Export knowledge store as a portable bundle."""
    from smartrag.export import KnowledgeExporter

    exporter = KnowledgeExporter(rag)
    bundle_path = exporter.export_bundle(
        args.output_path, include_embeddings=args.with_embeddings
    )

    # Read manifest for summary
    import json
    import zipfile

    with zipfile.ZipFile(bundle_path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))

    size_bytes = os.path.getsize(bundle_path)
    if size_bytes < 1024:
        size_str = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.1f} KB"
    else:
        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"

    print(f"Bundle created: {bundle_path}")
    print(f"Documents:      {manifest['document_count']}")
    print(f"Total size:     {size_str}")


def _cmd_import(args):
    """Import a .smartrag bundle into a knowledge store."""
    from smartrag.export import KnowledgeExporter

    rag = KnowledgeExporter.import_bundle(args.bundle_path, args.store)
    stats = rag.stats
    print(f"Imported {stats['document_count']} documents into {args.store}")


def _cmd_serve(args):
    """Start the SmartRAG REST API server."""
    try:
        import uvicorn
    except ImportError:
        print(
            "Error: uvicorn is not installed.\n"
            "Install the cloud extras: pip install smartrag[cloud]",
            file=sys.stderr,
        )
        sys.exit(1)

    # Set env vars so create_app() picks them up
    os.environ["SMARTRAG_KNOWLEDGE_DIR"] = args.store
    os.environ["SMARTRAG_PORT"] = str(args.port)
    os.environ["SMARTRAG_HOST"] = args.host

    from smartrag.api.server import create_app

    app = create_app(knowledge_dir=args.store)
    print(f"Starting SmartRAG API on {args.host}:{args.port} (store: {args.store})")
    uvicorn.run(app, host=args.host, port=args.port)


def _cmd_api_key(args, parser):
    """Manage API keys (create / list / revoke)."""
    if not args.apikey_action:
        parser.print_help()
        sys.exit(1)

    from smartrag.api.auth import create_api_key, list_api_keys, revoke_api_key

    if args.apikey_action == "create":
        try:
            raw_key = create_api_key(args.store, args.name)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"API key created for '{args.name}'.")
        print(f"Key: {raw_key}")
        print("Save this key now -- it cannot be retrieved again.")

    elif args.apikey_action == "list":
        keys = list_api_keys(args.store)
        if not keys:
            print("No API keys configured (open access mode).")
        else:
            print(f"{'Name':<30} {'Created'}")
            print("-" * 60)
            for k in keys:
                print(f"{k['name']:<30} {k['created']}")

    elif args.apikey_action == "revoke":
        success = revoke_api_key(args.store, args.name)
        if success:
            print(f"API key '{args.name}' revoked.")
        else:
            print(f"Error: No API key found with name '{args.name}'.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
