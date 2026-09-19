from src.tactical_model.inference import load_artifact


def test_demo_artifact_contract():
    artifact = load_artifact()

    assert artifact["demo_artifact"] is True
    assert artifact["n_components"] == 6
    assert len(artifact["feature_columns"]) == 29
    assert "nmf" in artifact
    assert "preprocessing" in artifact
    assert "training_reference" in artifact
