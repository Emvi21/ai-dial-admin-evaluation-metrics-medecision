"""G-Eval metric backed by deepeval."""

from enum import Enum
from typing import Annotated, List, Optional

from deepeval.metrics import GEval as DeepEvalGEval
from deepeval.metrics.g_eval.utils import Rubric as DeepEvalRubric
from deepeval.metrics.g_eval.utils import validate_and_sort_rubrics
from deepeval.test_case import LLMTestCaseParams
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from aidial_admin_evaluation_metrics.app_config import DEFAULT_MODEL
from aidial_admin_evaluation_metrics.metrics.common.base_metric import (
    MetricExample,
)
from aidial_admin_evaluation_metrics.metrics.deepeval.base import (
    BaseDeepEvalMetric,
)
from aidial_admin_evaluation_metrics.metrics.deepeval.deepeval_llm import (
    DialDeepEvalLLM,
)
from aidial_admin_evaluation_metrics.metrics.deepeval.input_types import (
    ActualOutput,
    DeepevalInput,
    OptionalRetrievalContext,
)
from aidial_admin_evaluation_metrics.metrics.deepeval.output_types import (
    DeepEvalDetails,
    DeepEvalScoreField,
)

_ScoreInt = Annotated[int, Field(ge=0, le=10)]


class Rubric(BaseModel):
    """A scoring rubric entry that maps a score range to an expected outcome."""

    start: Annotated[
        _ScoreInt,
        Field(
            description="Start of the score band (0–10 inclusive, must be ≤ end)."
        ),
    ]

    end: Annotated[
        _ScoreInt,
        Field(
            description="End of the score band (0–10 inclusive, must be ≥ start)."
        ),
    ]

    expected_outcome: Annotated[
        str,
        Field(
            description=(
                "Description of what the LLM output should look like "
                "to receive a score within this range."
            ),
        ),
    ]

    @model_validator(mode="after")
    def start_le_end(self) -> Self:
        if self.start > self.end:
            raise ValueError(
                f"start ({self.start}) must be less than or equal to end ({self.end})."
            )
        return self


def _to_deepeval_rubrics(
    rubrics: Optional[List[Rubric]],
) -> Optional[List[DeepEvalRubric]]:
    if rubrics is None:
        return None
    return [
        DeepEvalRubric(
            score_range=(r.start, r.end), expected_outcome=r.expected_outcome
        )
        for r in rubrics
    ]


class GEvalParam(str, Enum):
    """Subset of LLMTestCaseParams supported by GEvalMetric's Input model."""

    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"
    EXPECTED_OUTPUT = "expected_output"
    CONTEXT = "context"
    RETRIEVAL_CONTEXT = "retrieval_context"


class GEvalMetric(BaseDeepEvalMetric):
    name: str = "deepeval.g_eval"
    display_name: str = "DeepEval: G-Eval"
    description: str = (
        "G-Eval uses LLM-as-a-judge with chain-of-thought (CoT) reasoning to evaluate "
        "LLM outputs against any custom criteria. Provide either a natural-language "
        "criteria description or explicit evaluation_steps (but not both), along with "
        "the evaluation_params that list which LLMTestCase fields the criteria references. "
        "Optionally supply a rubric list to constrain scoring to specific score bands. "
        "Requires LLM access via DIAL. "
        "For more details: https://deepeval.com/docs/metrics-llm-evals"
    )

    class Config(BaseDeepEvalMetric.Config):
        eval_name: Annotated[
            str,
            Field(
                description=(
                    "A short label for this G-Eval instance (e.g. 'Correctness'). "
                    "Used in logs and error messages."
                ),
            ),
        ]

        criteria: Annotated[
            Optional[str],
            Field(
                default=None,
                description=(
                    "Natural-language description of what to evaluate. "
                    "Exactly one of criteria or evaluation_steps must be provided."
                ),
            ),
        ] = None

        evaluation_steps: Annotated[
            Optional[List[str]],
            Field(
                default=None,
                description=(
                    "Explicit step-by-step instructions for the LLM judge. "
                    "Exactly one of criteria or evaluation_steps must be provided."
                ),
            ),
        ] = None

        evaluation_params: Annotated[
            List[GEvalParam],
            Field(
                description=(
                    "The input fields that the criteria/evaluation_steps reference. "
                    "Only fields listed here are included in the evaluation prompt. "
                    "Accepted values: "
                    + ", ".join(f'"{p.value}"' for p in GEvalParam)
                ),
            ),
        ]

        rubric: Annotated[
            Optional[List[Rubric]],
            Field(
                default=None,
                description=(
                    "Optional list of scoring rubric entries, each mapping a score range "
                    "to an expected outcome. When provided, the score is constrained to "
                    "the range spanned by the rubric entries (instead of the default 0–10). "
                    "Entries must not have overlapping score_range values."
                ),
            ),
        ] = None

        @model_validator(mode="after")
        def validate_criteria_xor_steps(self) -> Self:
            has_criteria = (
                self.criteria is not None and self.criteria.strip() != ""
            )
            has_steps = (
                self.evaluation_steps is not None
                and len(self.evaluation_steps) > 0
            )
            if has_criteria == has_steps:
                raise ValueError(
                    "Exactly one of 'criteria' or 'evaluation_steps' must be provided, not both and not neither."
                )
            return self

        @model_validator(mode="after")
        def validate_rubric_ranges(self) -> Self:
            validate_and_sort_rubrics(_to_deepeval_rubrics(self.rubric))
            return self

    class Input(BaseModel):
        input: DeepevalInput
        actual_output: ActualOutput
        expected_output: Annotated[
            Optional[str],
            Field(
                default=None,
                description="The expected or ground-truth output used for evaluation",
            ),
        ] = None
        context: Annotated[
            Optional[List[str]],
            Field(
                default=None,
                description="Background context for the evaluation",
            ),
        ] = None
        retrieval_context: OptionalRetrievalContext = None

    def build_metric(
        self, config: BaseDeepEvalMetric.Config, deepeval_llm: DialDeepEvalLLM
    ) -> DeepEvalGEval:
        """Construct a GEval metric, mapping config fields to GEval's constructor."""
        assert isinstance(config, GEvalMetric.Config)
        return DeepEvalGEval(
            name=config.eval_name,
            evaluation_params=[
                LLMTestCaseParams(p.value) for p in config.evaluation_params
            ],
            criteria=config.criteria,
            evaluation_steps=config.evaluation_steps,
            rubric=_to_deepeval_rubrics(config.rubric),
            model=deepeval_llm,
            threshold=config.threshold,
            strict_mode=config.strict_mode,
            async_mode=False,
            verbose_mode=False,
        )

    examples = [
        MetricExample(
            config=Config(
                model=DEFAULT_MODEL,
                threshold=0.5,
                eval_name="Correctness",
                criteria=(
                    "Determine whether the actual output is factually correct "
                    "based on the expected output."
                ),
                evaluation_params=[
                    GEvalParam.ACTUAL_OUTPUT,
                    GEvalParam.EXPECTED_OUTPUT,
                ],
            ),
            input=Input(
                input="What is the diameter of the Earth?",
                actual_output="The diameter of the Earth is approximately 12,742 kilometers.",
                expected_output="The diameter of the Earth is approximately 12,742 kilometers.",
            ),
            expected_output=BaseDeepEvalMetric.Output(
                score=DeepEvalScoreField(
                    value=1.0,
                    details=DeepEvalDetails(
                        reason="The actual output exactly matches the expected output.",
                        verbose_logs="Criteria:\nDetermine whether the actual output is factually correct based on the expected output.",
                    ),
                )
            ),
        )
    ]
