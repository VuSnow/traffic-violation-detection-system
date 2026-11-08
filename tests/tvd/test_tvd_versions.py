from tvd.versions import PACKAGES, code_version, package_versions


def test_the_versions_of_the_libraries_that_shape_results():
    versions = package_versions()
    assert list(versions) == list(PACKAGES) and versions["traffic-violation-detection"] == "0.1.0" and versions["numpy"] == "2.4.6"


def test_a_package_that_is_not_installed_has_no_version():
    assert package_versions(("no-such-package",)) == {"no-such-package": None}


def test_the_code_version_comes_from_the_build(monkeypatch):
    monkeypatch.setenv("TVD_COMMIT", "6b1a1ba")
    assert code_version() == "6b1a1ba"
    monkeypatch.delenv("TVD_COMMIT")
    assert code_version() is None
