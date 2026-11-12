"""requirements.txt (read by Streamlit Cloud, the devcontainer and the scheduled workflows) lists what pyproject.toml
lists for the package, the app and the tests, so the two cannot drift apart."""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _array(body: str, name: str) -> set:
    m = re.search(rf"^{name}\s*=\s*\[(.*?)\]", body, re.S | re.M)
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def test_requirements_match_pyproject():
    toml = (ROOT / "pyproject.toml").read_text()
    reqs = {ln.strip() for ln in (ROOT / "requirements.txt").read_text().splitlines()
            if ln.strip() and not ln.startswith("#")}
    package, app, dev = _array(toml, "dependencies"), _array(toml, "app"), _array(toml, "dev")
    assert package and app and package | app <= reqs                      # everything the app needs
    extra = reqs - package - app
    assert extra and extra <= dev, f"in requirements.txt only: {sorted(extra - dev)}"   # ... plus the test runner


def test_one_version_number():
    import lighthouse
    toml = (ROOT / "pyproject.toml").read_text()
    assert re.search(r'^version\s*=\s*"([^"]+)"', toml, re.M).group(1) == lighthouse.__version__
