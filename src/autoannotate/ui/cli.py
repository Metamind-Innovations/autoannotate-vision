import click
from pathlib import Path
import logging
from typing import Literal, cast
from rich.console import Console
from rich.logging import RichHandler

from autoannotate.core.embeddings import EmbeddingExtractor
from autoannotate.core.clustering import ClusteringEngine
from autoannotate.core.organizer import DatasetOrganizer
from autoannotate.ui.interactive import InteractiveLabelingSession
from autoannotate.utils.image_loader import ImageLoader

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    pass


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--n-clusters", "-n", type=int, help="Number of clusters (required for kmeans/spectral)"
)
@click.option(
    "--method",
    "-m",
    type=click.Choice(["kmeans", "hdbscan", "spectral", "dbscan"]),
    default="kmeans",
    help="Clustering method",
)
@click.option(
    "--model",
    type=click.Choice(["clip", "dinov2", "dinov2-large", "siglip2"]),
    default="dinov2",
    help="Embedding model",
)
@click.option(
    "--batch-size", "-b", type=int, default=32, help="Batch size for embedding extraction"
)
@click.option("--recursive", "-r", is_flag=True, help="Search for images recursively")
@click.option("--reduce-dims/--no-reduce-dims", default=True, help="Apply dimensionality reduction")
@click.option(
    "--n-samples", type=int, default=5, help="Number of representative samples per cluster"
)
@click.option("--copy/--symlink", default=True, help="Copy files or create symlinks")
@click.option("--create-splits", is_flag=True, help="Create train/val/test splits")
@click.option(
    "--export-format",
    type=click.Choice(["csv", "json"]),
    default="csv",
    help="Export labels format",
)
def annotate(
    input_dir: Path,
    output_dir: Path,
    n_clusters: int,
    method: str,
    model: str,
    batch_size: int,
    recursive: bool,
    reduce_dims: bool,
    n_samples: int,
    copy: bool,
    create_splits: bool,
    export_format: str,
):
    try:
        console.print("[bold blue]AutoAnnotate-Vision[/bold blue] - SOTA Image Auto-Annotation\n")

        if method in ["kmeans", "spectral"] and n_clusters is None:
            raise click.UsageError(f"--n-clusters is required for {method} clustering")

        console.print(f"[cyan]Loading images from:[/cyan] {input_dir}")
        loader = ImageLoader(input_dir, recursive=recursive)
        images, image_paths = loader.load_images()
        console.print(f"[green]✓[/green] Loaded {len(images)} images\n")

        console.print(f"[cyan]Extracting embeddings using {model}...[/cyan]")
        extractor = EmbeddingExtractor(
            model_name=cast(Literal["clip", "dinov2", "dinov2-large", "siglip2"], model),
            batch_size=batch_size,
        )
        embeddings = extractor(images)
        console.print(f"[green]✓[/green] Extracted embeddings: {embeddings.shape}\n")

        console.print(f"[cyan]Clustering with {method}...[/cyan]")
        clusterer = ClusteringEngine(
            method=cast(Literal["kmeans", "hdbscan", "spectral", "dbscan"], method),
            n_clusters=n_clusters,
            reduce_dims=reduce_dims,
        )
        labels = clusterer.fit_predict(embeddings)
        stats = clusterer.get_cluster_stats(labels)
        console.print("[green]✓[/green] Clustering complete\n")

        session = InteractiveLabelingSession()
        session.display_cluster_stats(stats)

        console.print("\n[cyan]Getting representative samples...[/cyan]")
        representatives = clusterer.get_representative_indices(
            embeddings, labels, n_samples=n_samples
        )
        console.print(
            f"[green]✓[/green] Found representatives for {len(representatives)} clusters\n"
        )

        class_names = session.label_all_clusters(image_paths, labels, representatives, stats, output_dir)

        session.display_labeling_summary(class_names, labels)

        if not class_names:
            console.print("[yellow]No clusters were labeled. Exiting.[/yellow]")
            return

        console.print("\n[cyan]Organizing dataset...[/cyan]")
        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(
            image_paths, labels, class_names, copy_files=copy, create_symlinks=not copy
        )
        console.print(f"[green]✓[/green] Dataset organized in {output_dir}\n")

        console.print(f"[cyan]Exporting labels to {export_format}...[/cyan]")
        labels_file = organizer.export_labels_file(format=export_format)
        console.print(f"[green]✓[/green] Labels exported to {labels_file}\n")

        if create_splits:
            console.print("[cyan]Creating train/val/test splits...[/cyan]")
            organizer.create_split()
            console.print(f"[green]✓[/green] Created splits in {output_dir / 'splits'}\n")

        session.show_completion_message(output_dir)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        logger.exception("Full traceback:")
        raise click.Abort()


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--recursive", "-r", is_flag=True, help="Search recursively")
def validate(input_dir: Path, recursive: bool):
    console.print("[bold blue]Validating images...[/bold blue]\n")

    loader = ImageLoader(input_dir, recursive=recursive)
    image_paths = loader.load_image_paths()

    valid = 0
    invalid = []

    for img_path in image_paths:
        if loader.validate_image(img_path):
            valid += 1
        else:
            invalid.append(img_path)

    console.print(f"[green]Valid images:[/green] {valid}")
    console.print(f"[red]Invalid images:[/red] {len(invalid)}")

    if invalid:
        console.print("\n[yellow]Invalid files:[/yellow]")
        for img_path in invalid:
            console.print(f"  - {img_path}")


def main():
    cli()


if __name__ == "__main__":
    main()
