from pathlib import Path
from autoannotate import AutoAnnotator
from autoannotate.core.embeddings import EmbeddingExtractor
from autoannotate.core.clustering import ClusteringEngine
from autoannotate.utils.image_loader import ImageLoader
import numpy as np


def example_step_by_step():
    
    print("\n" + "="*60)
    print("STEP-BY-STEP ANNOTATION PIPELINE")
    print("="*60 + "\n")
    
    annotator = AutoAnnotator(
        input_dir="./data/images",
        output_dir="./data/organized",
        model="dinov2-large",
        clustering_method="hdbscan",
        reduce_dims=True
    )
    
    print("1. Loading images...")
    images, paths = annotator.load_images()
    print(f"   ✓ Loaded {len(images)} images\n")
    
    print("2. Extracting embeddings...")
    embeddings = annotator.extract_embeddings()
    print(f"   ✓ Extracted embeddings: {embeddings.shape}\n")
    
    print("3. Clustering images...")
    labels = annotator.cluster()
    stats = annotator.get_cluster_stats()
    print(f"   ✓ Found {stats['n_clusters']} clusters")
    print(f"   ✓ Unclustered: {stats['n_noise']} images\n")
    
    print("4. Interactive labeling...")
    class_names = annotator.interactive_labeling(n_samples=7)
    print(f"   ✓ Labeled {len(class_names)} classes\n")
    
    if class_names:
        print("5. Organizing dataset...")
        annotator.organize_dataset(copy_files=True)
        print("   ✓ Dataset organized\n")
        
        print("6. Creating splits...")
        annotator.create_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        print("   ✓ Splits created\n")
        
        print("7. Exporting labels...")
        labels_file = annotator.export_labels(format="json")
        print(f"   ✓ Labels exported to {labels_file}\n")
    
    print("="*60 + "\n")


def example_custom_pipeline():
    
    print("\n" + "="*60)
    print("CUSTOM PIPELINE WITH DIRECT API")
    print("="*60 + "\n")
    
    input_dir = Path("./data/raw_images")
    
    loader = ImageLoader(input_dir, recursive=True)
    images, image_paths = loader.load_images()
    print(f"Loaded {len(images)} images")
    
    extractor = EmbeddingExtractor(model_name="clip", batch_size=16)
    embeddings = extractor.extract_batch(images)
    print(f"Extracted embeddings: {embeddings.shape}")
    
    clusterer = ClusteringEngine(
        method="spectral",
        n_clusters=15,
        reduce_dims=True,
        target_dims=50
    )
    
    labels = clusterer.fit_predict(embeddings)
    stats = clusterer.get_cluster_stats(labels)
    
    print(f"\nClustering Results:")
    print(f"  - Clusters: {stats['n_clusters']}")
    print(f"  - Cluster sizes: {stats['cluster_sizes']}")
    
    representatives = clusterer.get_representative_indices(
        embeddings, labels, n_samples=3
    )
    
    print(f"\nRepresentative samples:")
    for cluster_id, indices in representatives.items():
        print(f"  Cluster {cluster_id}: {len(indices)} samples")
        for idx in indices:
            print(f"    - {image_paths[idx].name}")
    
    print("\n" + "="*60 + "\n")


def example_comparing_methods():
    
    print("\n" + "="*60)
    print("COMPARING CLUSTERING METHODS")
    print("="*60 + "\n")
    
    loader = ImageLoader("./data/images", recursive=False)
    images, paths = loader.load_images()
    
    extractor = EmbeddingExtractor(model_name="dinov2", batch_size=32)
    embeddings = extractor.extract_batch(images)
    
    methods = [
        ("kmeans", {"n_clusters": 10}),
        ("hdbscan", {}),
        ("spectral", {"n_clusters": 10}),
    ]
    
    results = {}
    
    for method_name, kwargs in methods:
        print(f"Testing {method_name}...")
        
        clusterer = ClusteringEngine(method=method_name, reduce_dims=True, **kwargs)
        labels = clusterer.fit_predict(embeddings)
        stats = clusterer.get_cluster_stats(labels)
        
        results[method_name] = stats
        print(f"  ✓ Clusters: {stats['n_clusters']}, Noise: {stats['n_noise']}\n")
    
    print("\nComparison Summary:")
    print("-" * 60)
    for method, stats in results.items():
        print(f"{method:12} | Clusters: {stats['n_clusters']:3} | Noise: {stats['n_noise']:4}")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "step":
            example_step_by_step()
        elif mode == "custom":
            example_custom_pipeline()
        elif mode == "compare":
            example_comparing_methods()
        else:
            print("Usage: python advanced_usage.py [step|custom|compare]")
    else:
        print("Choose an example:")
        print("  python advanced_usage.py step       - Step-by-step pipeline")
        print("  python advanced_usage.py custom     - Custom pipeline with direct API")
        print("  python advanced_usage.py compare    - Compare clustering methods")