"""
Verify SmartRAG remains a headless Python library with no frontend dependencies.
This test prevents accidental re-introduction of web dashboard or JS/Node dependencies.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


class TestHeadlessBuild:
    """SmartRAG must have zero frontend/JS/Node dependencies."""

    # --- Forbidden files ---

    FORBIDDEN_FILES = [
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "vite.config.ts",
        "vite.config.js",
        "tailwind.config.js",
        "tailwind.config.ts",
        "postcss.config.js",
        "postcss.config.ts",
        "tsconfig.json",
        ".nvmrc",
        ".node-version",
    ]

    FORBIDDEN_DIRS = [
        "node_modules",
        "frontend",
        "dashboard",
        "dist",  # Vite/webpack output
    ]

    def test_no_forbidden_files(self):
        """No JS/Node config files should exist in project root."""
        for fname in self.FORBIDDEN_FILES:
            path = PROJECT_ROOT / fname
            assert not path.exists(), (
                f"Forbidden file found: {fname}. "
                f"SmartRAG is a headless Python library — no JS/Node config files allowed."
            )

    def test_no_forbidden_directories(self):
        """No frontend directories should exist."""
        for dirname in self.FORBIDDEN_DIRS:
            path = PROJECT_ROOT / dirname
            assert not path.exists(), (
                f"Forbidden directory found: {dirname}/. "
                f"SmartRAG is headless — no frontend directories allowed."
            )

    def test_no_html_files_in_source(self):
        """No HTML template files in the smartrag/ source package."""
        smartrag_dir = PROJECT_ROOT / "smartrag"
        if not smartrag_dir.exists():
            pytest.skip("smartrag/ source directory not found")
        html_files = list(smartrag_dir.rglob("*.html"))
        assert len(html_files) == 0, (
            f"HTML files found in smartrag/: {[str(f) for f in html_files]}. "
            f"SmartRAG must not serve HTML — API returns JSON only."
        )

    def test_no_js_ts_files_in_source(self):
        """No JavaScript/TypeScript files in the smartrag/ source package."""
        smartrag_dir = PROJECT_ROOT / "smartrag"
        if not smartrag_dir.exists():
            pytest.skip("smartrag/ source directory not found")
        js_files = list(smartrag_dir.rglob("*.js")) + list(smartrag_dir.rglob("*.ts"))
        js_files = [f for f in js_files if ".min.js" not in str(f)]  # allow vendored minified libs if any
        assert len(js_files) == 0, (
            f"JS/TS files found in smartrag/: {[str(f) for f in js_files]}. "
            f"SmartRAG is Python-only."
        )

    # --- Forbidden Python imports ---

    def test_no_static_files_mount(self):
        """FastAPI app must not mount StaticFiles (no frontend serving)."""
        api_dir = PROJECT_ROOT / "smartrag" / "api"
        if not api_dir.exists():
            pytest.skip("smartrag/api/ not found — no API server yet")
        for py_file in api_dir.rglob("*.py"):
            content = py_file.read_text()
            assert "StaticFiles" not in content, (
                f"StaticFiles mount found in {py_file}. "
                f"SmartRAG API must return JSON only — no static file serving."
            )
            assert "HTMLResponse" not in content, (
                f"HTMLResponse found in {py_file}. "
                f"SmartRAG API must return JSON only."
            )
            assert "Jinja2Templates" not in content, (
                f"Jinja2Templates found in {py_file}. "
                f"SmartRAG API must not render HTML templates."
            )

    # --- Forbidden system dependencies ---

    def test_no_node_in_dockerfile(self):
        """Dockerfile must not install Node.js or run npm."""
        dockerfile = PROJECT_ROOT / "Dockerfile"
        if not dockerfile.exists():
            pytest.skip("No Dockerfile found")
        content = dockerfile.read_text().lower()
        for forbidden in ["npm install", "npm run", "npx ", "node ", "setup-node", "nvm ", "volta "]:
            assert forbidden not in content, (
                f"Node.js reference '{forbidden}' found in Dockerfile. "
                f"SmartRAG containers must be Python-only."
            )

    # --- Python dependency check ---

    def test_no_frontend_python_deps(self):
        """pyproject.toml must not list frontend-serving packages as core deps."""
        pyproject = PROJECT_ROOT / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip("No pyproject.toml found")
        content = pyproject.read_text()
        # These are only concerning if they're core dependencies (not dev/test)
        # Check the [project.dependencies] section specifically
        in_deps = False
        for line in content.splitlines():
            if line.strip().startswith("dependencies"):
                in_deps = True
            elif in_deps and line.strip().startswith("["):
                in_deps = False
            if in_deps:
                for pkg in ["react", "tailwind", "vite"]:
                    assert pkg not in line.lower(), (
                        f"Frontend package reference '{pkg}' found in pyproject.toml dependencies."
                    )

    # --- Build verification ---

    def test_pip_install_no_node_required(self):
        """pip install -e . must succeed without Node.js on PATH."""
        env = os.environ.copy()
        # Remove node from PATH to verify pure-Python install
        paths = env.get("PATH", "").split(os.pathsep)
        filtered = [p for p in paths if "node" not in p.lower() and "nvm" not in p.lower()]
        env["PATH"] = os.pathsep.join(filtered)

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".", "--dry-run", "--no-deps"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"pip install --dry-run failed without Node.js on PATH: {result.stderr}. "
            f"SmartRAG must install with Python alone."
        )

    def test_api_returns_json_not_html(self):
        """If the API server module exists, verify it doesn't import frontend deps."""
        server_file = PROJECT_ROOT / "smartrag" / "api" / "server.py"
        if not server_file.exists():
            pytest.skip("API server not built yet")
        content = server_file.read_text()
        assert "text/html" not in content, (
            f"text/html content type found in API server. "
            f"SmartRAG API must serve JSON only."
        )
