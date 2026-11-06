import pytest


@pytest.fixture(autouse=True)
def _own_highway_designs(tmp_path_factory, monkeypatch):
    """Highway presets and the library's default highway come from a fresh folder in every test, never from
    data/ui/highways on this PC."""
    from riffchart.product import highways
    monkeypatch.setattr(highways, "DESIGNS_DIR", tmp_path_factory.mktemp("highways"))
