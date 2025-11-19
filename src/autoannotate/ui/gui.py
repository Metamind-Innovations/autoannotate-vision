import sys
from pathlib import Path
import threading

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    tk = None  # type: ignore

from autoannotate import AutoAnnotator


class AutoAnnotateGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AutoAnnotate-Vision - Image Clustering & Labeling")
        self.root.geometry("700x450")
        self.root.resizable(False, False)

        # Variables
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.n_clusters = tk.IntVar(value=5)
        self.model_choice = tk.StringVar(value="dinov2")

        self.create_widgets()

    def create_widgets(self):
        # Title
        title = tk.Label(
            self.root,
            text="AutoAnnotate-Vision",
            font=("Arial", 20, "bold"),
            fg="#667eea",
        )
        title.pack(pady=15)

        # Frame for inputs
        frame = tk.Frame(self.root, padx=20, pady=5)
        frame.pack(fill="both", expand=True)

        # Input Folder
        tk.Label(frame, text="Input Folder (Images):", font=("Arial", 11, "bold")).grid(
            row=0, column=0, sticky="w", pady=8
        )
        tk.Entry(frame, textvariable=self.input_folder, width=45, font=("Arial", 10)).grid(
            row=0, column=1, padx=10
        )
        tk.Button(
            frame,
            text="Browse...",
            command=self.browse_input,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 9),
        ).grid(row=0, column=2)

        # Output Folder
        tk.Label(frame, text="Output Folder:", font=("Arial", 11, "bold")).grid(
            row=1, column=0, sticky="w", pady=8
        )
        tk.Entry(frame, textvariable=self.output_folder, width=45, font=("Arial", 10)).grid(
            row=1, column=1, padx=10
        )
        tk.Button(
            frame,
            text="Browse...",
            command=self.browse_output,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 9),
        ).grid(row=1, column=2)

        # Number of Classes
        tk.Label(frame, text="Number of Classes:", font=("Arial", 11, "bold")).grid(
            row=2, column=0, sticky="w", pady=8
        )
        tk.Spinbox(
            frame, from_=2, to=50, textvariable=self.n_clusters, width=10, font=("Arial", 10)
        ).grid(row=2, column=1, sticky="w", padx=10)

        # Model Selection
        tk.Label(frame, text="Model:", font=("Arial", 11, "bold")).grid(
            row=3, column=0, sticky="w", pady=8
        )
        model_combo = ttk.Combobox(
            frame,
            textvariable=self.model_choice,
            values=["clip", "dinov2", "dinov2-large", "siglip2"],
            state="readonly",
            width=20,
            font=("Arial", 10),
        )
        model_combo.grid(row=3, column=1, sticky="w", padx=10)

        # Status Label
        self.status_label = tk.Label(
            self.root, text="Ready to start", font=("Arial", 10), fg="blue"
        )
        self.status_label.pack(pady=8)

        # Progress Bar
        self.progress = ttk.Progressbar(self.root, length=500, mode="indeterminate")
        self.progress.pack(pady=8)

        # Run Button - MUCH BIGGER AND MORE VISIBLE
        self.run_button = tk.Button(
            self.root,
            text="▶ Start Auto-Annotation",
            command=self.run_annotation,
            bg="#667eea",
            fg="white",
            font=("Arial", 16, "bold"),
            width=25,
            height=2,
            relief="raised",
            bd=3,
            cursor="hand2",
        )
        self.run_button.pack(pady=15, ipady=15)  # Added ipady for internal padding

    def browse_input(self):
        folder = filedialog.askdirectory(title="Select Input Folder with Images")
        if folder:
            self.input_folder.set(folder)

    def browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder.set(folder)

    def update_status(self, message, color="blue"):
        self.status_label.config(text=message, fg=color)
        self.root.update()

    def run_annotation(self):
        input_dir = self.input_folder.get()
        output_dir = self.output_folder.get()

        if not input_dir or not output_dir:
            messagebox.showerror("Error", "Please select both input and output folders!")
            return

        # Run in separate thread to avoid freezing GUI
        thread = threading.Thread(target=self.annotation_process, args=(input_dir, output_dir))
        thread.daemon = True
        thread.start()

    def annotation_process(self, input_dir, output_dir):
        try:
            self.run_button.config(state="disabled")
            self.progress.start()

            self.update_status("Initializing...", "blue")

            annotator = AutoAnnotator(
                input_dir=Path(input_dir),
                output_dir=Path(output_dir),
                model=self.model_choice.get(),
                clustering_method="kmeans",
                n_clusters=self.n_clusters.get(),
                batch_size=16,
                reduce_dims=True,
            )

            self.update_status("Loading images...", "blue")
            images, paths = annotator.load_images()
            self.update_status(f"✓ Loaded {len(images)} images", "green")
            if len(images) < self.n_clusters.get() * 3:
                n_clusters = self.n_clusters.get()
                response = messagebox.askyesno(
                    "Small Dataset Warning",
                    f"You have {len(images)} images but requested {n_clusters} clusters.\n\n"
                    f"Recommended: At least {n_clusters * 3} images for good clustering.\n\n"
                    f"Continue anyway?",
                )
                if not response:
                    self.update_status("Cancelled by user", "orange")
                    return

            self.update_status("Extracting embeddings (this may take a while)...", "blue")
            annotator.extract_embeddings()
            self.update_status("✓ Embeddings extracted", "green")

            self.update_status("Clustering images...", "blue")
            annotator.cluster()
            stats = annotator.get_cluster_stats()
            self.update_status(f"✓ Found {stats['n_clusters']} clusters", "green")

            self.progress.stop()
            self.update_status("Opening HTML preview for labeling...", "orange")

            # Interactive labeling (will open HTML previews)
            class_names = annotator.interactive_labeling(n_samples=6)

            if class_names:
                self.update_status("Organizing dataset...", "blue")
                self.progress.start()

                annotator.organize_dataset(class_names=class_names, copy_files=True)
                annotator.export_labels(format="csv")

                self.progress.stop()
                self.update_status("✓ Complete!", "green")

                messagebox.showinfo(
                    "Success",
                    f"✓ Annotation Complete!\n\n"
                    f"Processed: {len(images)} images\n"
                    f"Classes: {len(class_names)}\n"
                    f"Output: {output_dir}\n\n"
                    f"Images are organized in class folders with original filenames preserved.",
                )
            else:
                self.update_status("No classes labeled", "orange")
                messagebox.showwarning("Warning", "No clusters were labeled.")

        except Exception as e:
            self.progress.stop()
            self.update_status(f"Error: {str(e)}", "red")
            messagebox.showerror("Error", f"An error occurred:\n\n{str(e)}")

        finally:
            self.run_button.config(state="normal")
            self.progress.stop()


def main():
    """Entry point for the GUI application."""
    if tk is None:
        print("ERROR: tkinter is not installed!", file=sys.stderr)
        print(file=sys.stderr)
        print(
            "The GUI requires tkinter, which is part of Python's standard library", file=sys.stderr
        )
        print("but must be installed separately on some systems:", file=sys.stderr)
        print(file=sys.stderr)
        print("  Ubuntu/Debian:    sudo apt-get install python3-tk", file=sys.stderr)
        print("  RHEL/CentOS:      sudo yum install python3-tkinter", file=sys.stderr)
        print("  Fedora:           sudo dnf install python3-tkinter", file=sys.stderr)
        print("  Arch Linux:       sudo pacman -S tk", file=sys.stderr)
        print(file=sys.stderr)
        print("On Windows and macOS, tkinter is usually included with Python.", file=sys.stderr)
        print(
            "If it's missing, you may need to reinstall Python with tkinter support.",
            file=sys.stderr,
        )
        sys.exit(1)

    root = tk.Tk()
    _ = AutoAnnotateGUI(root)  # noqa: F841
    root.mainloop()


if __name__ == "__main__":
    main()
