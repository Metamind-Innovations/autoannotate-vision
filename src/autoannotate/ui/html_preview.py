from pathlib import Path
from typing import List, Optional
import webbrowser
import numpy as np


def generate_cluster_preview_html(
    cluster_id: int,
    image_paths: List[Path],
    indices: np.ndarray,
    cluster_size: int,
    output_path: Optional[Path] = None,
) -> Path:

    if output_path is None:
        output_path = Path(f"cluster_{cluster_id}_preview.html")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Cluster {cluster_id} - Image Preview</title>
        <style>
            body {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                padding: 20px;
                margin: 0;
            }}
            .header {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                padding: 30px;
                border-radius: 15px;
                margin-bottom: 30px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }}
            h1 {{
                margin: 0;
                font-size: 2.5em;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            }}
            .stats {{
                font-size: 1.2em;
                margin-top: 10px;
                opacity: 0.9;
            }}
            .gallery {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .image-card {{
                background: rgba(255, 255, 255, 0.15);
                backdrop-filter: blur(10px);
                border-radius: 12px;
                overflow: hidden;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            }}
            .image-card:hover {{
                transform: translateY(-10px) scale(1.02);
                box-shadow: 0 12px 35px rgba(0, 0, 0, 0.4);
            }}
            .image-card img {{
                width: 100%;
                height: 250px;
                object-fit: cover;
                display: block;
            }}
            .image-name {{
                padding: 15px;
                text-align: center;
                font-size: 0.9em;
                background: rgba(0, 0, 0, 0.3);
                word-break: break-all;
            }}
            .instruction {{
                background: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                border-left: 4px solid #4CAF50;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 Cluster {cluster_id}</h1>
            <div class="stats">
                📊 Total Images in Cluster: <strong>{cluster_size}</strong><br>
                👁️ Showing Representative Samples: <strong>{len(indices)}</strong>
            </div>
        </div>
        
        <div class="instruction">
            <strong>📝 Instructions:</strong> Review these sample images and return to the terminal to enter a class name for this cluster.
        </div>
        
        <div class="gallery">
    """

    for idx, img_idx in enumerate(indices, 1):
        img_path = image_paths[img_idx]
        img_uri = img_path.absolute().as_uri()

        html_content += f"""
            <div class="image-card">
                <img src="{img_uri}" alt="{img_path.name}" loading="lazy">
                <div class="image-name">
                    <strong>Sample {idx}</strong><br>
                    {img_path.name}
                </div>
            </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


def open_html_in_browser(html_path: Path):
    webbrowser.open(html_path.absolute().as_uri())
