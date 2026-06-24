from typing import List

from pydantic import BaseModel

from aidial_admin_evaluation_metrics.metric import MetricsDescription


class MetricsResponse(BaseModel):
    metrics: List[MetricsDescription]
