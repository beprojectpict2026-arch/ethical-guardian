import eguard


def test_package_imports_and_has_version():
    assert eguard.__version__ == "0.1.0"
