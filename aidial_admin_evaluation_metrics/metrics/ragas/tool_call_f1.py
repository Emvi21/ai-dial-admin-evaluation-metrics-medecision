"""Tool Call F1 metric backed by ragas."""

from pydantic import BaseModel, Field
from ragas.messages import AIMessage as RagasAIMessage
from ragas.messages import ToolCall as RagasToolCall
from ragas.metrics.collections import ToolCallF1 as RagasToolCallF1

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


class ToolCallF1Metric(BaseDeterministicRagasMetric):
    name: str = "ragas.tool_call_f1"
    display_name: str = "Ragas: Tool Call F1"
    description: str = (
        "Measures the F1 score between the agent's tool calls and the reference tool calls. "
        "Treats each (tool_name, args) pair as a set element, computing precision, recall, "
        "and their harmonic mean (F1). Order is ignored — only set membership matters. "
        "This is a rule-based metric — no LLM access is required for scoring. "
        "For more details: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/#tool-call-f1"
    )

    class Input(BaseModel):
        tools_called: list[ToolCallInput] = Field(
            description="Tool calls made by the agent"
        )
        reference_tool_calls: list[ToolCallInput] = Field(
            description="Expected tool calls that a correct agent should have made"
        )

    examples = [
        MetricExample(
            config=BaseDeterministicRagasMetric.Config(),
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

    def _build_deterministic_ragas_metric(self, config: BaseDeterministicRagasMetric.Config) -> RagasToolCallF1:  # type: ignore[override]
        return RagasToolCallF1()

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
