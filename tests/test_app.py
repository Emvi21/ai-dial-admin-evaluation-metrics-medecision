import pytest
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def test_health(app_client: TestClient):
    response = app_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_metrics_list(app_client: TestClient):
    response = app_client.get("/metrics")
    assert response.status_code == 200

    data = response.json()
    assert "metrics" in data
    assert any(metric["name"] == "exact_match" for metric in data["metrics"])
    assert all("display_name" in metric for metric in data["metrics"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "actual, ground_truth, expected_score",
    [
        ("test answer", "test answer", 1.0),
        ("wrong answer", "test answer", 0.0),
    ],
)
async def test_evaluate_exact_match(
    app_client: TestClient,
    actual: str,
    ground_truth: str,
    expected_score: float,
):
    eval_request = {
        "metric_name": "exact_match",
        "config": {},
        "input": {"actual": actual, "ground_truth": ground_truth},
    }

    response = app_client.post(
        "/evaluate",
        json=eval_request,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["metric_name"] == "exact_match"
    assert "output" in data
    assert "exact_match" in data["output"]
    assert "value" in data["output"]["exact_match"]
    assert data["output"]["exact_match"]["value"] == expected_score


@pytest.mark.asyncio
async def test_evaluate_validation_error_in_config(app_client: TestClient):
    """Test that validation errors in config are properly contextualized."""
    eval_request = {
        "metric_name": "regex_match",
        "config": {"pattern": 123},  # Invalid type, should be string
        "input": {"actual": "test"},
    }

    response = app_client.post("/evaluate", json=eval_request)
    assert response.status_code == 422

    result = response.json()
    assert "detail" in result
    assert isinstance(result["detail"], list)

    # Verify error location includes 'config' and 'pattern'
    first_error = result["detail"][0]
    assert "loc" in first_error
    assert "config" in first_error["loc"]
    assert "pattern" in first_error["loc"]


@pytest.mark.asyncio
async def test_evaluate_validation_error_in_input(app_client: TestClient):
    """Test that validation errors in input are properly contextualized."""
    eval_request = {
        "metric_name": "exact_match",
        "config": {},
        "input": {},  # Missing required fields
    }

    response = app_client.post("/evaluate", json=eval_request)
    assert response.status_code == 422

    result = response.json()
    assert "detail" in result
    assert isinstance(result["detail"], list)
    assert len(result["detail"]) >= 2  # At least 2 missing fields

    # Verify all errors have 'input' in their location
    for error in result["detail"]:
        assert "loc" in error
        assert "input" in error["loc"]
        assert "msg" in error
        assert "type" in error
