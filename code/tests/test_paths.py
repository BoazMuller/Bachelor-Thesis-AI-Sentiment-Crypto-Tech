from thesis.paths import PROJECT_ROOT


def test_project_root_exists():
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "data").exists()

