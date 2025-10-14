from pathlib import Path
from autoannotate import AutoAnnotator


def main():
    
    input_directory = Path("./data/unlabeled_images")
    output_directory = Path("./data/annotated_images")
    
    annotator = AutoAnnotator(
        input_dir=input_directory,
        output_dir=output_directory,
        model="dinov2",
        clustering_method="kmeans",
        n_clusters=10,
        batch_size=32,
        recursive=False,
        reduce_dims=True
    )
    
    result = annotator.run_full_pipeline(
        n_samples=5,
        copy_files=True,
        create_splits=True,
        export_format="csv"
    )
    
    print(f"\n{'='*60}")
    print(f"Annotation Complete!")
    print(f"{'='*60}")
    print(f"Total images processed: {result['n_images']}")
    print(f"Number of classes: {result['n_clusters']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()