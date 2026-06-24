import json

import pytest
from pydantic import SecretStr

from aidial_admin_evaluation_metrics.app_config import DialConfig
from aidial_admin_evaluation_metrics.dial.llm_client import create_dial_factory
from aidial_admin_evaluation_metrics.metric import MetricsDescription
from aidial_admin_evaluation_metrics.metrics import create_metrics_registry

registry = create_metrics_registry(
    create_dial_factory(
        DialConfig(dial_url="http://test", dial_api_key=SecretStr("test"))
    )
)


def _get_metric_descriptions():
    return registry.get_all_descriptions()


def _has_type_discriminator(schema) -> bool:
    if not isinstance(schema, dict):
        return False

    # Direct discriminator (pydantic discriminated union)
    disc = schema.get("discriminator")
    if isinstance(disc, dict) and disc.get("propertyName") == "type":
        return True

    # Check oneOf/anyOf entries for type.const or enum with value/error
    for comb in ("oneOf", "anyOf"):
        entries = schema.get(comb)
        if isinstance(entries, list):
            has_value = False
            has_error = False
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                props = entry.get("properties")
                if not isinstance(props, dict):
                    continue
                t = props.get("type")
                if isinstance(t, dict):
                    const = t.get("const")
                    enum = t.get("enum")
                    if const == "value" or (
                        isinstance(enum, list) and "value" in enum
                    ):
                        has_value = True
                    if const == "error" or (
                        isinstance(enum, list) and "error" in enum
                    ):
                        has_error = True
            if has_value and has_error:
                return True

    return False


@pytest.mark.parametrize(
    "metric_description",
    _get_metric_descriptions(),
)
def test_validate_metric_descriptions(metric_description: MetricsDescription):
    """Check that every metric description output schema is subschema of MetricResults"""

    properties = metric_description.output_schema.get("properties") or {}
    if not properties:
        pytest.fail(
            f"Metric '{metric_description.name}' has no output properties"
        )

    for prop_name, prop_schema in properties.items():
        if _has_type_discriminator(prop_schema):
            continue

        # If schema is a $ref or complex composition, include the schema in the failure message
        pretty = json.dumps(prop_schema, indent=2)
        pytest.fail(
            (
                f"Metric '{metric_description.name}' output property '{prop_name}' does not include a 'type'"
                " discriminator or equivalent oneOf entries with 'type' consts ('value' and 'error').\n"
                f"Property schema:\n{pretty}"
            )
        )
