from unittest.mock import patch


def test_rag_eval_llm_error_via_api(app_client):
    with patch(
        "aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.rag_eval.create_rag_eval_metrics_report"
    ) as mock_calc_metrics:
        mock_calc_metrics.side_effect = RuntimeError("LLM API error")

        request = {
            "metric_name": "aidial_rag_eval.rag_eval",
            "config": {"matcher": "citation", "model": "gemini-2.5-flash-lite"},
            "input": {
                "question": "What is the diameter of Earth?",
                "ground_truth_answer": "The diameter of Earth is about 12,742 km.",
                "ground_truth_facts": ["Earth fact"],
                "context": ["Earth facts"],
                "answer": "Earth is 12,742 km wide.",
            },
        }

        response = app_client.post("/evaluate", json=request)
        assert response.status_code == 200

        result = response.json()
        assert result["metric_name"] == "aidial_rag_eval.rag_eval"
        assert "recall" in result["output"]
        assert "precision" in result["output"]
        assert "f1" in result["output"]
        assert "mrr" in result["output"]
        assert "ground_truth_to_answer" in result["output"]
        assert "answer_to_ground_truth" in result["output"]
        assert "context_to_answer" in result["output"]

        # All fields should contain error messages
        def assert_error_field(field):
            assert "message" in result["output"][field]
            assert "LLM API error" in result["output"][field]["message"]
            assert result["output"][field]["type"] == "error"

        assert_error_field("recall")
        assert_error_field("precision")
        assert_error_field("f1")
        assert_error_field("mrr")
        assert_error_field("ground_truth_to_answer")
        assert_error_field("answer_to_ground_truth")
        assert_error_field("context_to_answer")


def test_rag_eval_metric_in_registry(app_client):
    response = app_client.get("/metrics")
    assert response.status_code == 200

    metrics = response.json()["metrics"]
    metric_names = [m["name"] for m in metrics]

    assert "aidial_rag_eval.rag_eval" in metric_names

    metric = next(m for m in metrics if m["name"] == "aidial_rag_eval.rag_eval")
    assert "description" in metric
    assert "config_schema" in metric
    assert "input_schema" in metric
    assert "output_schema" in metric

    # Verify config schema has required fields
    config_props = metric["config_schema"]["properties"]
    assert "model" in config_props

    # Verify input schema has required fields
    input_props = metric["input_schema"]["properties"]
    assert "question" in input_props
    assert "ground_truth_answer" in input_props
    assert "ground_truth_facts" in input_props
    assert "context" in input_props
    assert "answer" in input_props
    assert "documents" in input_props

    # Verify output schema has required inference fields
    output_props = metric["output_schema"]["properties"]
    assert "recall" in output_props
    assert "precision" in output_props
    assert "f1" in output_props
    assert "mrr" in output_props
    assert "ground_truth_to_answer" in output_props
    assert "answer_to_ground_truth" in output_props
    assert "context_to_answer" in output_props

    # Verify that description is non-empty for each field in schemas
    for _, props in config_props.items():
        assert "description" in props
        assert props["description"] != ""
    for _, props in input_props.items():
        assert "description" in props
        assert props["description"] != ""
    for _, props in output_props.items():
        assert "description" in props
        assert props["description"] != ""


def test_rag_eval_invalid_model_via_api(app_client):
    """Test RagEvalMetric error handling for invalid model name via API."""
    request = {
        "metric_name": "aidial_rag_eval.rag_eval",
        "config": {"model": "invalid-model-name"},
        "input": {
            "question": "What is Python?",
            "ground_truth_answer": "Python is a programming language.",
            "ground_truth_facts": ["Python is a programming language."],
            "context": ["Python is a high-level programming language."],
            "answer": "Python is used for software development.",
        },
    }

    response = app_client.post("/evaluate", json=request)

    # Pydantic validates the model name at request time
    assert response.status_code == 422

    result = response.json()
    first_error = result["detail"][0]
    assert first_error["type"] == "literal_error"
    assert first_error["loc"] == ["config", "model"]


def test_rag_eval_invalid_matcher_via_api(app_client):
    """Test RagEvalMetric error handling for invalid matcher name via API."""
    request = {
        "metric_name": "aidial_rag_eval.rag_eval",
        "config": {
            "matcher": "regex",
            "model": "gemini-2.5-flash-lite",
        },
        "input": {
            "question": "What is Python?",
            "ground_truth_answer": "Python is a programming language.",
            "ground_truth_facts": ["Python is a programming language."],
            "context": ["Python is a high-level programming language."],
            "answer": "Python is used for software development.",
        },
    }

    response = app_client.post("/evaluate", json=request)

    # Pydantic validates the model name at request time
    assert response.status_code == 422

    result = response.json()
    first_error = result["detail"][0]
    assert first_error["type"] == "literal_error"
    assert first_error["loc"] == ["config", "matcher"]
    assert first_error["msg"] == "Input should be 'citation' or 'exact'"
