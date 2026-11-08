from tvd.config import ComponentConfig
from tvd.detection import DETECTORS
from tvd.versions import PACKAGES, code_version, models_used, package_versions


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


def test_the_models_used_name_each_model_and_the_software(monkeypatch):
    monkeypatch.setenv("TVD_COMMIT", "6b1a1ba")
    used = models_used(DETECTORS.create("test-moving-car"), ComponentConfig(type="ocsort", options={"use_byte": True}), None, "the plate models did not start: no GPU")
    assert used["detector"] == {"type": "MovingCarDetector"} and used["tracker"] == {"type": "ocsort", "use_byte": True}
    assert used["plates"] == {"unavailable": "the plate models did not start: no GPU"} and used["code"] == "6b1a1ba" and list(used["software"]) == list(PACKAGES)
