from typing import Optional

import aidial_admin_evaluation_metrics.metrics.maf as maf_metrics
from aidial_admin_evaluation_metrics.app_config import MetricsSettings
from aidial_admin_evaluation_metrics.dial.llm_client import DialFactory
from aidial_admin_evaluation_metrics.metrics.common.base_metric import BaseMetric
from aidial_admin_evaluation_metrics.metrics.common.registry import MetricsRegistry


def _create_instances(
    settings: MetricsSettings,
    dial_factory: DialFactory,
) -> list[BaseMetric]:
    """Explicitly construct every metric instance with its resolved settings."""
    s = settings.maf
    return [
        maf_metrics.MAFPipelineMetric(dial_factory, s),
        maf_metrics.ClinicalSummaryMetric(dial_factory, s),
    ]


def create_metrics_registry(
    dial_factory: DialFactory,
    settings: Optional[MetricsSettings] = None,
) -> MetricsRegistry:
    """Create a new MetricsRegistry with the given settings and factory."""
    resolved = settings or MetricsSettings()
    return MetricsRegistry(metrics=_create_instances(resolved, dial_factory))


__all__ = ["create_metrics_registry"]
