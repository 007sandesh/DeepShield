"""
DeepShield CLI — Command-line interface.

Beautiful, informative CLI for deepfake detection with:
- Rich terminal output
- Progress indicators
- Export options
- Batch processing
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    name="deepshield",
    help="🛡️  DeepShield — AI-Powered Deepfake Detection",
    add_completion=False,
)
console = Console()


@app.command()
def detect(
    file: str = typer.Argument(..., help="Path to image or video file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Export results to JSON file"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Detection threshold (0-1)"),
    device: str = typer.Option("auto", "--device", "-d", help="Compute device (auto/cpu/cuda)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed model output"),
    no_explain: bool = typer.Option(False, "--no-explain", help="Skip explanation generation"),
):
    """
    Analyze a file for deepfake detection.

    Supports both images (jpg, png, webp) and videos (mp4, avi, mov).
    """
    file_path = Path(file)
    if not file_path.exists():
        console.print(f"[red]✗ File not found: {file}[/red]")
        raise typer.Exit(1)

    # Detect file type
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    video_exts = {".mp4", ".avi", ".mov", ".webm", ".mkv"}

    suffix = file_path.suffix.lower()
    is_video = suffix in video_exts
    is_image = suffix in image_exts

    if not is_image and not is_video:
        console.print(f"[red]✗ Unsupported file type: {suffix}[/red]")
        raise typer.Exit(1)

    # Header
    console.print()
    console.print(
        Panel(
            "[bold cyan]🛡️  DeepShield — Deepfake Detection[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print(f"  📄 File: {file_path.name}")
    console.print(f"  📦 Type: {'Video' if is_video else 'Image'}")
    console.print()

    # Run detection
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing...", total=None)

        try:
            from src.core.pipelines.image_pipeline import ImageDetectionPipeline
            from src.core.pipelines.video_pipeline import VideoDetectionPipeline

            if is_video:
                pipeline = VideoDetectionPipeline(device=device)
                pipeline.initialize()
                progress.update(task, description="Running video analysis...")
                result = pipeline.detect(str(file_path))
            else:
                pipeline = ImageDetectionPipeline(device=device)
                pipeline.initialize()
                progress.update(task, description="Running image analysis...")
                result = pipeline.detect(str(file_path))
                result = result.to_dict()

            progress.update(task, description="Complete!", completed=1)

        except Exception as e:
            progress.update(task, description="Failed!")
            console.print(f"\n[red]✗ Detection failed: {e}[/red]")
            raise typer.Exit(1)

    # Display results
    _display_result(result, verbose, threshold)

    # Export if requested
    if output:
        output_path = Path(output)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        console.print(f"\n[green]✓ Results exported to {output_path}[/green]")


@app.command()
def batch(
    directory: str = typer.Argument(..., help="Directory containing files to analyze"),
    output: str = typer.Option("batch_results.json", "--output", "-o", help="Output JSON file"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Scan subdirectories"),
    device: str = typer.Option("auto", "--device", "-d", help="Compute device"),
):
    """Batch analyze all files in a directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        console.print(f"[red]✗ Not a directory: {directory}[/red]")
        raise typer.Exit(1)

    # Collect files
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    video_exts = {".mp4", ".avi", ".mov", ".webm"}

    pattern = "**/*" if recursive else "*"
    files = [
        f for f in dir_path.glob(pattern)
        if f.suffix.lower() in image_exts | video_exts
    ]

    if not files:
        console.print("[yellow]No supported files found[/yellow]")
        raise typer.Exit(0)

    console.print(f"\n📁 Found {len(files)} files to analyze\n")

    from src.core.pipelines.image_pipeline import ImageDetectionPipeline
    import cv2

    pipeline = ImageDetectionPipeline(device=device)
    pipeline.initialize()

    results = []
    with Progress() as progress:
        task = progress.add_task("Processing...", total=len(files))

        for file_path in files:
            progress.update(task, description=f"Analyzing {file_path.name}...")

            try:
                if file_path.suffix.lower() in image_exts:
                    import cv2
                    image = cv2.imread(str(file_path))
                    if image is not None:
                        result = pipeline.detect(image)
                        results.append({
                            "file": str(file_path),
                            **result.to_dict(),
                        })
                else:
                    results.append({
                        "file": str(file_path),
                        "error": "Video batch not yet supported",
                    })
            except Exception as e:
                results.append({
                    "file": str(file_path),
                    "error": str(e),
                })

            progress.advance(task)

    # Summary
    real_count = sum(1 for r in results if r.get("prediction") == "REAL")
    fake_count = sum(1 for r in results if r.get("prediction") == "FAKE")
    error_count = sum(1 for r in results if "error" in r)

    table = Table(title="Batch Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")
    table.add_row("Total Files", str(len(results)))
    table.add_row("Authentic", f"[green]{real_count}[/green]")
    table.add_row("Deepfake", f"[red]{fake_count}[/red]")
    table.add_row("Errors", f"[yellow]{error_count}[/yellow]")
    console.print(table)

    # Save
    with open(output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"\n[green]✓ Results saved to {output}[/green]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of workers"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
):
    """Start the DeepShield API server."""
    import uvicorn

    console.print(
        Panel(
            "[bold cyan]🛡️  DeepShield API Server[/bold cyan]\n"
            f"  Host: {host}\n"
            f"  Port: {port}\n"
            f"  Workers: {workers}\n"
            f"  Docs: http://{host}:{port}/docs",
            border_style="cyan",
        )
    )

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


@app.command()
def info():
    """Show DeepShield system information."""
    import torch

    table = Table(title="🛡️  DeepShield System Info")
    table.add_column("Component", style="cyan")
    table.add_column("Status", justify="right")

    # PyTorch
    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA Available", "✓ Yes" if torch.cuda.is_available() else "✗ No")

    if torch.cuda.is_available():
        table.add_row("GPU", torch.cuda.get_device_name(0))
        table.add_row("GPU Memory", f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Models
    from src.core.models.base import ModelRegistry
    models = ModelRegistry.list_models()
    for name, mtype in models.items():
        table.add_row(f"Model: {name}", mtype)

    console.print(table)


def _display_result(result: dict, verbose: bool, threshold: float):
    """Display detection result with rich formatting."""
    prediction = result.get("prediction", "UNKNOWN")
    confidence = result.get("confidence", 0)

    # Color based on prediction
    if prediction in ("FAKE", "fake"):
        color = "red"
        icon = "🚨"
    elif prediction in ("REAL", "real"):
        color = "green"
        icon = "✅"
    else:
        color = "yellow"
        icon = "❓"

    # Main result panel
    console.print()
    console.print(
        Panel(
            f"[bold {color}]{icon}  {prediction}[/bold {color}]\n\n"
            f"Confidence: [bold]{confidence:.1%}[/bold]\n"
            f"Probability: Real {result.get('probability', {}).get('real', 0):.1%} | "
            f"Fake {result.get('probability', {}).get('fake', 0):.1%}",
            title="Detection Result",
            border_style=color,
        )
    )

    # Explanation
    if result.get("explanation"):
        console.print(f"\n💡 [bold]Explanation:[/bold] {result['explanation']}")

    # Faces
    if "faces_detected" in result:
        console.print(f"👤 Faces detected: {result['faces_detected']}")

    # Model breakdown
    if verbose and "model_results" in result:
        console.print("\n[bold]Model Breakdown:[/bold]")
        table = Table(show_header=True)
        table.add_column("Model", style="cyan")
        table.add_column("Prediction")
        table.add_column("Confidence", justify="right")
        table.add_column("Time", justify="right")

        for name, model_result in result["model_results"].items():
            if "error" in model_result:
                table.add_row(name, "[red]ERROR[/red]", "-", "-")
            else:
                pred = model_result.get("prediction", "?")
                conf = model_result.get("confidence", 0)
                time_ms = model_result.get("inference_time_ms", 0)
                pred_color = "red" if pred.upper() == "FAKE" else "green"
                table.add_row(
                    name,
                    f"[{pred_color}]{pred}[/{pred_color}]",
                    f"{conf:.1%}",
                    f"{time_ms:.1f}ms",
                )

        console.print(table)

    # Performance
    if "performance" in result:
        perf = result["performance"]
        total = perf.get("total_time_ms", 0)
        console.print(f"\n⏱️  Total time: {total:.0f}ms")

    console.print()


if __name__ == "__main__":
    app()
