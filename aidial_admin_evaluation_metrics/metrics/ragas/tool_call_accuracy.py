"""Tool Call Accuracy metric backed by ragas."""

from pydantic import BaseModel, Field
from ragas.messages import AIMessage as RagasAIMessage
from ragas.messages import ToolCall as RagasToolCall
from ragas.metrics.collections import ToolCallAccuracy as RagasToolCallAccuracy

from aidial_admin_evaluation_metrics.metrics.common.base_metric import (
    MetricExample,
)
from aidial_admin_evaluation_metrics.metrics.common.types import ToolCallInput
from aidial_admin_evaluation_metrics.metrics.ragas.base import (
    BaseDeterministicRagasMetric,
    BaseRagasMetric,
)
from aidial_admin_evaluation_metrics.metrics.ragas.output_types import (
    RagasDetails,
    RagasScoreField,
)


class ToolCallAccuracyMetric(BaseDeterministicRagasMetric):
    name: str = "ragas.tool_call_accuracy"
    display_name: str = "Ragas: Tool Call Accuracy"
    description: str = (
        "Measures how accurately an LLM agent makes tool calls compared to reference tool calls. "
        "Evaluates sequence alignment (whether calls appear in the expected order when strict_order=True) "
        "and argument accuracy (exact match of each argument). "
        "This is a rule-based metric — no LLM access is required for scoring. "
        "For more details: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/#tool-call-accuracy"
    )

    class Config(BaseDeterministicRagasMetric.Config):
        strict_order: bool = Field(
            default=True,
            description=(
                "When True (default), tool calls must appear in the same order as the reference. "
                "When False, order is ignored and only the set of calls matters."
            ),
        )

    class Input(BaseModel):
        tools_called: list[ToolCallInput] = Field(
            description="Ordered list of tool calls made by the agent"
        )
        reference_tool_calls: list[ToolCallInput] = Field(
            description="Expected tool calls that a correct agent should have made"
        )

    examples = [
        MetricExample(
            config=Config(strict_order=True),
            input=Input(
                tools_called=[
                    ToolCallInput(
                        name="get_weather", args={"location": "Paris"}
                    ),
                    ToolCallInput(
                        name="get_weather", args={"location": "Berlin"}
                    ),
                ],
                reference_tool_calls=[
                    ToolCallInput(
                        name="get_weather", args={"location": "Paris"}
                    ),
                    ToolCallInput(
                        name="get_weather", args={"location": "Berlin"}
                    ),
                ],
            ),
            expected_output=BaseRagasMetric.Output(
                score=RagasScoreField(value=1.0, details=RagasDetails())
            ),
        )
    ]

    def _build_deterministic_ragas_metric(self, config: Config) -> RagasToolCallAccuracy:  # type: ignore[override]
        return RagasToolCallAccuracy(strict_order=config.strict_order)

    def _build_score_kwargs(self, input: Input) -> dict:  # type: ignore[override]
        user_input = [
            RagasAIMessage(
                content="",
                tool_calls=[
                    RagasToolCall(name=tc.name, args=tc.args)
                    for tc in input.tools_called
                ],
            ),
        ]
        reference_tool_calls = [
            RagasToolCall(name=tc.name, args=tc.args)
            for tc in input.reference_tool_calls
        ]
        return {
            "user_input": user_input,
            "reference_tool_calls": reference_tool_calls,
        }
