from typing import Any

from pydantic import BaseModel


class MetricsDescription(BaseModel):
    name: str
    display_name: str
    description: str
    config_schema: dict[str, Any]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
