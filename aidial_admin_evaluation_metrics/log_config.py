import logging
from enum import StrEnum

from aidial_sdk import configure_root_logger
from aidial_sdk.telemetry.init import init_telemetry
from aidial_sdk.telemetry.types import TelemetryConfig
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class LogLevel(StrEnum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


# LogConfig is not part of the AppConfig, because we want to be able to
# configure logging before anything else is initialized
class LogConfig(BaseSettings):
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Log level for the application.",
    )
    log_level_override: dict[str, LogLevel] = Field(
        default={},
        description="Allows to override log level for specific modules.",
    )


def configure_log_levels():
    log_config = LogConfig()
    logging.getLogger().setLevel(log_config.log_level)

    # Logging of the log config is done with the default logger level
    # to be able to debug logging issues
    logger.info(f"Log config: {log_config.model_dump_json()}")

    for name, level in log_config.log_level_override.items():
        logging.getLogger(name).setLevel(level)

    # Making the uvicorn logger delegate logging to the root logger
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers = []
    uvicorn_logger.propagate = True

    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.handlers = []
    uvicorn_access_logger.propagate = True


def init_logging_and_telemetry():
    # We want to initialize logging and telemetry before everything else
    # to be able to capture logs from the initialization process
    # The app instrumentation is done in main.py after the app is created
    init_telemetry(app=None, config=TelemetryConfig())
    configure_root_logger()

    # We do not have the auto-instrumentation with aidial_sdk
    # We need to manually instrument what the SDK is not instrumenting for us
    AsyncioInstrumentor().instrument()
    ThreadingInstrumentor().instrument()

    configure_log_levels()
