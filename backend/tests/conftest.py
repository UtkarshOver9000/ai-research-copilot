"""
Ensure tests don't depend on whatever OPENAI_API_KEY happens to be set on
the machine running them (a real, non-hypothetical footgun -- this repo
was developed on a machine with a literal placeholder value permanently
set from following another project's README example).
"""

import pytest


@pytest.fixture(autouse=True)
def clear_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
