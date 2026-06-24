"""
Script to generate API documentation from FastAPI OpenAPI schema.
Uses only FastAPI's built-in tools - no external dependencies needed.
"""
import argparse
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from pydantic import SecretStr

from aidial_admin_evaluation_metrics.app import create_app
from aidial_admin_evaluation_metrics.app_config import AppSettings


def export_openapi_schema(app: FastAPI, output_path: Path) -> None:
    """Export OpenAPI schema to JSON file."""
    openapi_schema = app.openapi()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="\n") as f:
        json.dump(openapi_schema, f, indent=2)

    print(f"✓ OpenAPI schema exported to: {output_path}")


def generate_swagger_ui_html(
    app_title: str, openapi_filename: str, output_path: Path
) -> None:
    """Generate standalone Swagger UI HTML using FastAPI's built-in function."""
    html_content = get_swagger_ui_html(
        openapi_url=f"./{openapi_filename}",
        title=f"{app_title} - Swagger UI",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(html_content.body)

    print(f"✓ Swagger UI HTML generated: {output_path}")


def generate_redoc_html(
    app_title: str, openapi_filename: str, output_path: Path
) -> None:
    """Generate standalone ReDoc HTML using FastAPI's built-in function."""
    html_content = get_redoc_html(
        openapi_url=f"./{openapi_filename}",
        title=f"{app_title} - ReDoc",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(html_content.body)

    print(f"✓ ReDoc HTML generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate API documentation from FastAPI OpenAPI schema"
    )
    parser.add_argument(
        "--openapi",
        type=Path,
        required=True,
        help="Output path for OpenAPI JSON schema",
    )
    parser.add_argument(
        "--swagger",
        type=Path,
        help="Output path for Swagger UI HTML (optional)",
    )
    parser.add_argument(
        "--redoc",
        type=Path,
        help="Output path for ReDoc HTML (optional)",
    )

    args = parser.parse_args()

    # Create app once to get its configuration
    app = create_app(
        AppSettings(
            dial_url="https://dial.example.com",
            dial_api_key=SecretStr("test-key"),
        )
    )

    export_openapi_schema(app, args.openapi)
    openapi_filename = args.openapi.name

    if args.swagger:
        generate_swagger_ui_html(app.title, openapi_filename, args.swagger)

    if args.redoc:
        generate_redoc_html(app.title, openapi_filename, args.redoc)


if __name__ == "__main__":
    main()
