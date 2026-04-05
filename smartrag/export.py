"""Knowledge store export for mobile deployment."""

import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from smartrag import SmartRAG


class KnowledgeExporter:
    """Exports a SmartRAG knowledge store as a portable bundle."""

    def __init__(self, rag: SmartRAG):
        self._rag = rag
        self._path = rag._path

    def export_bundle(self, output_path: str, include_embeddings: bool = False) -> str:
        """Package knowledge store into a .smartrag bundle (zip with manifest).

        Args:
            output_path: Path for the output .smartrag file
            include_embeddings: Include embedding vectors in wiki.db

        Returns:
            Path to the created bundle file
        """
        output = Path(output_path)
        if not output.suffix:
            output = output.with_suffix(".smartrag")

        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = os.path.join(tmp, "bundle")
            os.makedirs(bundle_dir)

            # Copy markdown documents
            docs_src = os.path.join(self._path, "documents")
            docs_dst = os.path.join(bundle_dir, "documents")
            if os.path.isdir(docs_src):
                shutil.copytree(docs_src, docs_dst)

            # Copy _index.md
            index_src = os.path.join(self._path, "_index.md")
            if os.path.isfile(index_src):
                shutil.copy2(index_src, os.path.join(bundle_dir, "_index.md"))

            # Copy backlinks.json
            bl_src = os.path.join(self._path, "backlinks.json")
            if os.path.isfile(bl_src):
                shutil.copy2(bl_src, os.path.join(bundle_dir, "backlinks.json"))

            # Copy wiki.db (FTS5 index)
            db_src = os.path.join(self._path, ".smartrag", "wiki.db")
            if os.path.isfile(db_src):
                db_dst = os.path.join(bundle_dir, "wiki.db")
                shutil.copy2(db_src, db_dst)

                if not include_embeddings:
                    # Strip embedding table from the copy
                    import sqlite3

                    conn = sqlite3.connect(db_dst)
                    try:
                        conn.execute("DROP TABLE IF EXISTS wiki_embeddings")
                        conn.execute("VACUUM")
                    except Exception:
                        pass
                    finally:
                        conn.close()

            # Create manifest
            stats = self._rag.stats
            manifest = {
                "smartrag_version": "0.1.0",
                "format_version": "1.0",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "document_count": stats["document_count"],
                "includes_embeddings": include_embeddings,
                "categories": stats.get("categories", []),
            }

            # Calculate total size
            total_size = 0
            for root, dirs, files in os.walk(bundle_dir):
                for f in files:
                    total_size += os.path.getsize(os.path.join(root, f))
            manifest["total_size_bytes"] = total_size

            with open(os.path.join(bundle_dir, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)

            # Create zip
            with zipfile.ZipFile(str(output), "w", zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(bundle_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, bundle_dir)
                        zf.write(file_path, arcname)

        return str(output)

    @staticmethod
    def import_bundle(bundle_path: str, target_dir: str) -> SmartRAG:
        """Import a .smartrag bundle into a new knowledge store.

        Args:
            bundle_path: Path to the .smartrag bundle file
            target_dir: Directory to create the knowledge store in

        Returns:
            SmartRAG instance pointing to the imported store
        """
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(bundle_path, "r") as zf:
            zf.extractall(str(target))

        # Move wiki.db into .smartrag/
        db_extracted = target / "wiki.db"
        smartrag_dir = target / ".smartrag"
        smartrag_dir.mkdir(exist_ok=True)
        if db_extracted.exists():
            shutil.move(str(db_extracted), str(smartrag_dir / "wiki.db"))

        # Create SmartRAG instance and reindex to ensure consistency
        rag = SmartRAG(str(target))
        rag.reindex(incremental=False)
        return rag
