"""Tests for the deepdiff metric via the /evaluate HTTP endpoint."""

import pytest
from fastapi.testclient import TestClient


def _evaluate(app_client: TestClient, actual, ground_truth, config=None):
    return app_client.post(
        "/evaluate",
        json={
            "metric_name": "deepdiff",
            "config": config or {},
            "input": {"actual": actual, "ground_truth": ground_truth},
        },
    )


@pytest.mark.asyncio
async def test_deepdiff_identical_strings(app_client: TestClient):
    response = _evaluate(app_client, "foo", "foo")
    assert response.status_code == 200
    data = response.json()
    assert data["output"]["match"]["value"] == 1.0


@pytest.mark.asyncio
async def test_deepdiff_different_strings(app_client: TestClient):
    response = _evaluate(app_client, "foo", "bar")
    assert response.status_code == 200
    data = response.json()
    assert data["output"]["match"]["value"] == 0.0
    assert data["output"]["match"]["details"]["diff"] == {
        "values_changed": {"root": {"new_value": "foo", "old_value": "bar"}}
    }
    assert (
        data["output"]["match"]["details"]["pretty"]
        == 'Value of root changed from "bar" to "foo".'
    )


@pytest.mark.asyncio
async def test_deepdiff_identical_bools(app_client: TestClient):
    response = _evaluate(app_client, True, True)
    assert response.status_code == 200
    data = response.json()
    assert data["output"]["match"]["value"] == 1.0


@pytest.mark.asyncio
async def test_deepdiff_different_bools(app_client: TestClient):
    response = _evaluate(app_client, True, False)
    assert response.status_code == 200
    data = response.json()
    assert data["output"]["match"]["value"] == 0.0
    assert data["output"]["match"]["details"]["diff"] == {
        "values_changed": {"root": {"new_value": True, "old_value": False}}
    }
    assert (
        data["output"]["match"]["details"]["pretty"]
        == "Value of root changed from False to True."
    )


@pytest.mark.asyncio
async def test_deepdiff_same_list_ordered(app_client: TestClient):
    response = _evaluate(
        app_client, [1, 2, 3], [1, 2, 3], {"ignore_order": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["output"]["match"]["value"] == 1.0


@pytest.mark.asyncio
async def test_deepdiff_reordered_list_order_sensitive(app_client: TestClient):
    response = _evaluate(
        app_client, [2, 1, 3], [1, 2, 3], {"ignore_order": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["output"]["match"]["value"] == 0.0
    assert data["output"]["match"]["details"]["diff"] == {
        "values_changed": {
            "root[0]": {"new_value": 2, "old_value": 1},
            "root[1]": {"new_value": 1, "old_value": 2},
        }
    }
    assert data["output"]["match"]["details"]["pretty"] == (
        "Value of root[0] changed from 1 to 2.\n"
        "Value of root[1] changed from 2 to 1."
    )


@pytest.mark.asyncio
async def test_deepdiff_reordered_list_ignore_order(app_client: TestClient):
    response = _evaluate(
        app_client, [2, 1, 3], [1, 2, 3], {"ignore_order": True}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["output"]["match"]["value"] == 1.0


@pytest.mark.asyncio
async def test_deepdiff_identical_complex_object(app_client: TestClient):
    """Match: object with all JSON types - string, bool, int, list, nested list, nested dict."""
    obj = {
        "name": "test",
        "active": True,
        "score": 42,
        "tags": ["a", "b", "c"],
        "matrix": [[1, 2], [3, 4]],
        "nested": {
            "x": 1,
            "items": [{"id": 1, "val": "foo"}, {"id": 2, "val": "bar"}],
        },
    }
    response = _evaluate(app_client, obj, obj)
    assert response.status_code == 200
    data = response.json()
    assert data["output"]["match"]["value"] == 1.0


@pytest.mark.asyncio
async def test_deepdiff_complex_object_changed_value(app_client: TestClient):
    """Mismatch: one leaf value differs; verify details['diff'] is populated."""
    actual = {
        "name": "test",
        "active": True,
        "score": 99,
        "tags": ["a", "b", "c"],
        "matrix": [[1, 2], [3, 4]],
        "nested": {
            "x": 1,
            "items": [{"id": 1, "val": "foo"}, {"id": 2, "val": "bar"}],
        },
    }
    ground_truth = {
        "name": "test",
        "active": True,
        "score": 42,
        "tags": ["a", "b", "c"],
        "matrix": [[1, 2], [3, 4]],
        "nested": {
            "x": 1,
            "items": [{"id": 1, "val": "foo"}, {"id": 2, "val": "bar"}],
        },
    }
    response = _evaluate(app_client, actual, ground_truth)
    assert response.status_code == 200
    data = response.json()
    assert data["output"]["match"]["value"] == 0.0
    assert data["output"]["match"]["details"]["diff"]


@pytest.mark.asyncio
async def test_deepdiff_bboxes_epsilon_ordered_match(app_client: TestClient):
    """Match: ordered bbox comparison with math_epsilon=20; max diff is 3px."""
    actual = [{"bbox": [1436, 1071, 2288, 1715], "category": "Image"}]
    ground_truth = [{"bbox": [1435, 1071, 2291, 1714], "category": "Image"}]
    response = _evaluate(app_client, actual, ground_truth, {"math_epsilon": 20})
    assert response.status_code == 200
    data = response.json()
    assert data["output"]["match"]["value"] == 1.0


@pytest.mark.asyncio
async def test_deepdiff_bboxes_epsilon_ordered_mismatch(app_client: TestClient):
    """Mismatch: ordered bbox comparison with math_epsilon=20; first coord differs by 65px."""
    actual = [{"bbox": [1500, 1071, 2288, 1715], "category": "Image"}]
    ground_truth = [{"bbox": [1435, 1071, 2291, 1714], "category": "Image"}]
    response = _evaluate(app_client, actual, ground_truth, {"math_epsilon": 20})
    assert response.status_code == 200
    data = response.json()
    assert data["output"]["match"]["value"] == 0.0
    assert data["output"]["match"]["details"]["diff"]


@pytest.mark.asyncio
async def test_deepdiff_bboxes_significant_digits_unordered_match(
    app_client: TestClient,
):
    """Match: unordered bbox comparison with significant_digits=2 in scientific notation.

    1436 and 1435 both round to 1.44e+03; 2291 and 2288 both round to 2.29e+03;
    1073 and 1071 both round to 1.07e+03; 1712 and 1714 both round to 1.71e+03.
    """
    actual = [{"bbox": [1436, 1073, 2288, 1712], "category": "Image"}]
    ground_truth = [{"bbox": [1435, 1071, 2291, 1714], "category": "Image"}]
    response = _evaluate(
        app_client,
        actual,
        ground_truth,
        {
            "ignore_order": True,
            "significant_digits": 2,
            "number_format_notation": "e",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["output"]["match"]["value"] == 1.0


@pytest.mark.asyncio
async def test_deepdiff_bboxes_significant_digits_unordered_mismatch(
    app_client: TestClient,
):
    """Mismatch: unordered bbox comparison with significant_digits=2 in scientific notation.

    2400 rounds to 2.4e+03 while 2291 rounds to 2.3e+03 — these differ at 2 sig digits.
    """
    actual = [{"bbox": [1436, 1071, 2400, 1715], "category": "Image"}]
    ground_truth = [{"bbox": [1435, 1071, 2291, 1714], "category": "Image"}]
    response = _evaluate(
        app_client,
        actual,
        ground_truth,
        {
            "ignore_order": True,
            "significant_digits": 2,
            "number_format_notation": "e",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["output"]["match"]["value"] == 0.0
    assert data["output"]["match"]["details"]["diff"]


@pytest.mark.asyncio
async def test_deepdiff_epsilon_with_ignore_order_is_invalid(
    app_client: TestClient,
):
    """Config validation: math_epsilon cannot be combined with ignore_order=True."""
    response = _evaluate(
        app_client,
        [1, 2, 3],
        [1, 2, 3],
        {"math_epsilon": 20, "ignore_order": True},
    )
    assert response.status_code == 422
