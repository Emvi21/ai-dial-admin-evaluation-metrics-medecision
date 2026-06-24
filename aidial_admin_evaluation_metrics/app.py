import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator, TypeVar

import anyio
from fastapi import Body, FastAPI, Response
from fastapi.exceptions import HTTPException
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, TypeAdapter, ValidationError

from aidial_admin_evaluation_metrics.api.evaluate import (
    EvaluationRequest,
    EvaluationResponse,
    build_request_examples,
    build_response_examples,
)
from aidial_admin_evaluation_metrics.api.health import HealthResponse
from aidial_admin_evaluation_metrics.api.metrics_list import MetricsResponse
from aidial_admin_evaluation_metrics.app_config import AppSettings
from aidial_admin_evaluation_metrics.dial.llm_client import (
    dial_factory_lifespan,
)
from aidial_admin_evaluation_metrics.metrics import create_metrics_registry
from aidial_admin_evaluation_metrics.metrics.common.base_metric import (
    BaseMetric,
)
from aidial_admin_evaluation_metrics.metrics.common.registry import (
    MetricsRegistry,
)
from aidial_admin_evaluation_metrics.metrics.common.types import MetricResults

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=BaseModel)

tracer = trace.get_tracer(__name__)


def _validate_field(model: type[T], data: Any, field_name: str) -> T:
    """Validate data with a Pydantic model, adding field context to errors.

    This function is designed for FastAPI request handling contexts.
    It validates incoming data and enriches validation errors with field
    context (e.g., 'config' or 'input') to make error messages clearer.

    Args:
        model: The Pydantic model class to validate against
        data: The data to validate
        field_name: The field name to prepend to error locations

    Returns:
        The validated model instance

    Raises:
        HTTPException: 422 error with field context in error locations
    """
    try:
        return model.model_validate(data)
    except ValidationError as e:
        # Use Pydantic's own JSON serialization to get safe, serializable error
        # dicts. e.errors() may contain non-serializable objects in 'ctx' (e.g.
        # ValueError instances from @model_validators), which would crash the
        # JSON response serialization. json.loads(e.json()) converts them to
        # their string representations, which are always JSON-safe.
        errors = [
            {**error, "loc": [field_name] + error["loc"]}
            for error in json.loads(e.json())
        ]
        raise HTTPException(status_code=422, detail=errors)


def _prepare_metrics_response(registry: MetricsRegistry) -> bytes:
    """Prepare the cached response for the /metrics endpoint."""
    descriptions = registry.get_all_descriptions()
    response = MetricsResponse(metrics=descriptions)
    return response.model_dump_json().encode("utf-8")


@asynccontextmanager
async def _evaluation_slot(
    limiter: anyio.CapacityLimiter, max_queue_backlog: int
) -> AsyncIterator[None]:
    """Acquire an evaluation slot, rejecting requests when the backlog is too long."""
    if limiter.statistics().tasks_waiting >= max_queue_backlog:
        raise HTTPException(
            status_code=503,
            detail="Evaluation queue is full. Please try again later.",
        )
    async with limiter:
        yield


@asynccontextmanager
async def _metric_span(
    eval_request: EvaluationRequest, metric: BaseMetric
) -> AsyncIterator[None]:
    """Context manager that starts an OpenTelemetry span for a metric evaluation.

    Accepts the original `EvaluationRequest` and the metric instance to allow
    richer attributes to be attached to the span.

    Usage:
        async with _metric_span(eval_request, metric):
            await metric.evaluate_async(...)
    """
    metric_class = metric.__class__.__name__

    with tracer.start_as_current_span("metric.evaluate") as span:
        span.set_attribute("metric.name", eval_request.metric_name)
        span.set_attribute("metric.class", metric_class)
        try:
            yield
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def create_app(app_config: AppSettings) -> FastAPI:
    _request_examples: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with dial_factory_lifespan(
            app_config.dial_config
        ) as dial_factory:
            registry = create_metrics_registry(dial_factory, app_config.metrics)
            app.state.registry = registry
            _request_examples.update(build_request_examples(registry))
            json_schema_extra = EvaluationResponse.model_config.get(
                "json_schema_extra"
            )
            if isinstance(json_schema_extra, dict):
                json_schema_extra["examples"] = build_response_examples(
                    registry
                )
            app.state.metrics_response_bytes = _prepare_metrics_response(
                registry
            )
            app.state.eval_limiter = anyio.CapacityLimiter(
                app_config.app.max_concurrent_evaluations
            )
            yield

    app = FastAPI(
        title="ai-dial-admin-evaluation-metrics",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def measure_request_time(request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        logger.debug(
            "%s %s completed in %.4f seconds",
            request.method,
            request.url,
            process_time,
        )
        return response

    @app.get("/metrics", response_model=MetricsResponse)
    async def metrics():
        # List of metrics does not change on runtime, but is requested frequently by clients,
        # so we cache the serialized response in app state for fast retrieval.
        return Response(
            content=app.state.metrics_response_bytes,
            media_type="application/json",
        )

    @app.post("/evaluate")
    async def evaluate(
        eval_request: Annotated[
            EvaluationRequest,
            Body(openapi_examples=_request_examples),
        ],
    ) -> EvaluationResponse:
        registry: MetricsRegistry = app.state.registry
        try:
            metric = registry.get_metric_instance(eval_request.metric_name)
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown metric: {eval_request.metric_name}",
            )

        config_cls = registry.get_config_cls(eval_request.metric_name)
        config = _validate_field(config_cls, eval_request.config, "config")
        input = _validate_field(metric.Input, eval_request.input, "input")

        async with _evaluation_slot(
            app.state.eval_limiter, app_config.app.max_queue_backlog
        ):
            async with _metric_span(eval_request, metric):
                output = await metric.evaluate_async(config, input)
        return EvaluationResponse(
            metric_name=eval_request.metric_name,
            output=TypeAdapter(MetricResults).validate_python(
                output.model_dump()
            ),
        )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse()

    return app
