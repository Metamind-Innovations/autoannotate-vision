from pathlib import Path
from typing import Dict, List
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from autoannotate.ui.html_preview import generate_cluster_preview_html, open_html_in_browser


class InteractiveLabelingSession:

    def __init__(self):
        self.console = Console()
        self.class_names = {}

    def display_cluster_stats(self, stats: Dict):
        table = Table(title="Clustering Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Total Clusters", str(stats["n_clusters"]))
        table.add_row("Unclustered Images", str(stats["n_noise"]))
        table.add_row("Total Images", str(stats["total_samples"]))

        self.console.print(table)

        if stats["cluster_sizes"]:
            size_table = Table(title="Cluster Sizes", show_header=True)
            size_table.add_column("Cluster ID", style="cyan")
            size_table.add_column("Size", style="magenta")

            for cluster_id, size in sorted(stats["cluster_sizes"].items()):
                size_table.add_row(str(cluster_id), str(size))

            self.console.print(size_table)

    def get_class_name_for_cluster(
        self,
        cluster_id: int,
        image_paths: List[Path],
        representative_indices: np.ndarray,
        cluster_size: int,
    ):
        self.console.print(
            Panel(
                f"[bold]Labeling Cluster {cluster_id}[/bold]\n"
                f"Cluster size: {cluster_size} images\n"
                f"Representative samples: {len(representative_indices)} images",
                style="green",
            )
        )

        # Generate and open HTML preview
        self.console.print("[cyan]Generating HTML preview...[/cyan]")
        html_path = generate_cluster_preview_html(
            cluster_id=cluster_id,
            image_paths=image_paths,
            indices=representative_indices,
            cluster_size=cluster_size,
            output_path=Path(f"cluster_{cluster_id}_preview.html"),
        )

        self.console.print(f"[green]✓ Preview generated: {html_path}[/green]")
        self.console.print("[yellow]Opening preview in browser...[/yellow]\n")

        open_html_in_browser(html_path)

        self.console.print(
            "[bold cyan]Review the images in your browser, then return here to label.[/bold cyan]\n"
        )
        self.console.print("[yellow]Options:[/yellow]")
        self.console.print("  1. Enter a class name for this cluster")
        self.console.print("  2. Type 'skip' to skip this cluster")
        self.console.print("  3. Type 'noise' to mark as unclustered\n")

        while True:
            class_name = Prompt.ask(
                f"[bold green]Class name for Cluster {cluster_id}[/bold green]", default=""
            ).strip()

            if not class_name:
                if Confirm.ask("No name entered. Skip this cluster?", default=True):
                    return None
                continue

            if class_name.lower() == "skip":
                return None

            if class_name.lower() == "noise":
                return None

            if class_name in self.class_names.values():
                if not Confirm.ask(
                    f"[yellow]Warning: '{class_name}' already used. Continue?[/yellow]",
                    default=False,
                ):
                    continue

            if Confirm.ask(f"Confirm class name: [bold]'{class_name}'[/bold]?", default=True):
                return class_name

    def label_all_clusters(
        self,
        image_paths: List[Path],
        labels: np.ndarray,
        representative_indices: Dict[int, np.ndarray],
        cluster_stats: Dict,
    ):
        self.console.print("\n[bold blue]Starting Interactive Labeling Session[/bold blue]\n")

        sorted_clusters = sorted(
            cluster_stats["cluster_sizes"].items(), key=lambda x: x[1], reverse=True
        )

        for cluster_id, cluster_size in sorted_clusters:
            if cluster_id not in representative_indices:
                continue

            class_name = self.get_class_name_for_cluster(
                cluster_id, image_paths, representative_indices[cluster_id], cluster_size
            )

            if class_name:
                self.class_names[cluster_id] = class_name
                self.console.print(
                    f"[green]✓ Cluster {cluster_id} labeled as '{class_name}'[/green]\n"
                )
            else:
                self.console.print(f"[yellow]⊘ Cluster {cluster_id} skipped[/yellow]\n")

        return self.class_names

    def display_labeling_summary(self, class_names: Dict[int, str], labels: np.ndarray):
        self.console.print("\n[bold green]Labeling Summary[/bold green]\n")

        table = Table(show_header=True)
        table.add_column("Cluster ID", style="cyan")
        table.add_column("Class Name", style="magenta")
        table.add_column("Images", style="yellow")

        for cluster_id, class_name in sorted(class_names.items()):
            n_images = np.sum(labels == cluster_id)
            table.add_row(str(cluster_id), class_name, str(n_images))

        n_unlabeled = sum(1 for label in labels if label not in class_names and label != -1)
        n_noise = np.sum(labels == -1)

        if n_unlabeled > 0:
            table.add_row("-", "[dim]Unlabeled[/dim]", f"[dim]{n_unlabeled}[/dim]")
        if n_noise > 0:
            table.add_row("-1", "[dim]Unclustered[/dim]", f"[dim]{n_noise}[/dim]")

        self.console.print(table)

        total_labeled = sum(np.sum(labels == cid) for cid in class_names.keys())
        total_images = len(labels)
        coverage = (total_labeled / total_images) * 100 if total_images > 0 else 0

        self.console.print(
            f"\n[bold]Coverage:[/bold] {coverage:.1f}% ({total_labeled}/{total_images} images)"
        )

    def show_completion_message(self, output_dir: Path):
        self.console.print("\n" + "=" * 60)
        self.console.print(
            Panel(
                f"[bold green]✓ Auto-annotation Complete![/bold green]\n\n"
                f"Output directory: [cyan]{output_dir}[/cyan]\n"
                f"Images organized by class (original filenames preserved)\n"
                f"Check metadata.json for detailed information",
                style="green",
            )
        )
        self.console.print("=" * 60 + "\n")
