import os
import shutil
import tempfile

import pytest


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="smartrag_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def knowledge_dir(tmp_dir):
    return os.path.join(tmp_dir, "knowledge")


@pytest.fixture
def sample_md(tmp_dir):
    p = os.path.join(tmp_dir, "sample.md")
    with open(p, "w") as f:
        f.write(
            "---\ntitle: Sample Doc\ncategories: [api, testing]\n---\n\n"
            "# Sample\n\nThis is a sample document about API testing and authentication.\n"
        )
    return p


@pytest.fixture
def sample_txt(tmp_dir):
    p = os.path.join(tmp_dir, "sample.txt")
    with open(p, "w") as f:
        f.write("# My Text File\n\nThis is plain text content about database migrations.\n")
    return p


@pytest.fixture
def sample_py(tmp_dir):
    p = os.path.join(tmp_dir, "sample.py")
    with open(p, "w") as f:
        f.write(
            '"""Auth module for JWT authentication."""\n\n'
            "def authenticate(token: str) -> bool:\n"
            '    """Validate a JWT token."""\n'
            "    return True\n"
        )
    return p


@pytest.fixture
def large_md(tmp_dir):
    """A markdown file >2000 words to trigger section splitting."""
    sections = []
    for i in range(5):
        words = " ".join([f"word{j}" for j in range(500)])
        sections.append(f"## Section {i + 1}\n\n{words}\n")
    content = (
        "---\ntitle: Large Document\n---\n\n# Overview\n\nThis is the intro.\n\n"
        + "\n".join(sections)
    )
    p = os.path.join(tmp_dir, "large.md")
    with open(p, "w") as f:
        f.write(content)
    return p
