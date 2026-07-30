from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from aidial_admin_evaluation_metrics.log_config import (
    init_logging_and_telemetry,
)

# We need to initialize logging and telemetry before everything else
# to be able to capture logs from the initialization process
init_logging_and_telemetry()

import uvicorn  # noqa: E402

from aidial_admin_evaluation_metrics.app import create_app  # noqa: E402
from aidial_admin_evaluation_metrics.app_config import AppSettings  # noqa: E402

app_config = AppSettings()
app = create_app(app_config)

# We do not have the auto-instrumentation with aidial_sdk
# We need to manually instrument the FastAPI app
FastAPIInstrumentor.instrument_app(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)  # noqa: S104
