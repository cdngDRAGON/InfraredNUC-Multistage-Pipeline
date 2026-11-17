"""web_workbench/server.py - MATLAB-Style Web Workbench HTTP Server.

Provides a full-stack REST API and single-page application server for interactive
infrared nonuniformity correction, comparative baseline benchmarking, and
automated relational input/output mapping.

Usage:
  python3 -m web_workbench.server --port 8080
  python3 -m web_workbench.server --seed-examples
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import mimetypes
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image

from common.io_mapping_manager import IOMappingManager


class WorkbenchRequestHandler(BaseHTTPRequestHandler):
    """Multi-threaded HTTP handler for MATLAB-Style Web Workbench."""

    def __init__(self, *args, **kwargs):
        self.root_dir = Path(".").resolve()
        self.io_manager = IOMappingManager(self.root_dir)
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        """Concise server logging."""
        sys.stderr.write(f"[{self.log_date_time_string()}] {self.command} {self.path} - {format % args}\n")

    def send_json(self, data: Any, status: int = 200) -> None:
        """Send JSON response."""
        encoded = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded)

    def send_file_response(self, file_path: Path) -> None:
        """Serve a static file with appropriate MIME type and cache headers."""
        if not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, f"File not found: {file_path.name}")
            return

        mime_type, _ = mimetypes.guess_type(str(file_path))
        mime_type = mime_type or "application/octet-stream"

        try:
            content = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Error reading file: {e}")

    def do_OPTIONS(self) -> None:
        """Handle CORS pre-flight."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        # 1. Root: Serves index.html
        if path == "/" or path == "/index.html":
            tmpl_path = self.root_dir / "web_workbench" / "templates" / "index.html"
            self.send_file_response(tmpl_path)
            return

        # 2. API: List Datasets
        if path == "/api/datasets":
            self.handle_api_datasets()
            return

        # 3. API: List Mappings
        if path == "/api/mappings":
            mappings = self.io_manager.list_mappings()
            self.send_json(mappings)
            return

        # 4. Static Files: /files/<relative_path>
        if path.startswith("/files/"):
            rel_str = unquote(path[len("/files/") :])
            target_path = (self.root_dir / rel_str).resolve()
            # Security check: must reside inside root_dir
            if not str(target_path).startswith(str(self.root_dir)):
                self.send_error(HTTPStatus.FORBIDDEN, "Access denied.")
                return
            self.send_file_response(target_path)
            return

        self.send_error(HTTPStatus.NOT_FOUND, f"Endpoint not found: {path}")

    def do_POST(self) -> None:
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            payload = {}

        if path == "/api/process":
            self.handle_api_process(payload)
            return

        if path == "/api/batch_process":
            self.handle_api_batch_process(payload)
            return

        self.send_error(HTTPStatus.NOT_FOUND, f"Unknown POST endpoint: {path}")

    def handle_api_datasets(self) -> None:
        """Return list of available real IR datasets and scenes."""
        manifest_csv = self.root_dir / "data" / "dataset_manifest.csv"
        records = []

        if manifest_csv.exists():
            with open(manifest_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    records.append({
                        "name": Path(r["saved_path"]).name,
                        "dataset": r.get("dataset_group", "General"),
                        "path": r["saved_path"],
                        "width": int(r.get("width", 256)),
                        "height": int(r.get("height", 256)),
                        "roughness_rho": float(r.get("roughness_rho", 0.0)),
                    })

        self.send_json({"datasets": records, "count": len(records)})

    def handle_api_process(self, payload: Dict[str, Any]) -> None:
        """Process an infrared image through the two-stage NUC pipeline."""
        image_path = payload.get("image_path")
        image_b64 = payload.get("image_base64")
        filename = payload.get("filename", "input.png")
        source_dataset = payload.get("source_dataset", "custom_upload")
        engine = payload.get("engine", "proposed_ours")
        noise_sigma = float(payload.get("noise_sigma", 1.0))
        max_iter_s1 = int(payload.get("max_iter_s1", 40))
        run_baselines = bool(payload.get("run_baselines", False))

        try:
            clean_gt = None
            if image_b64:
                img_bytes = base64.b64decode(image_b64)
                run_id, in_path, img_arr = self.io_manager.ingest_input(
                    img_bytes, filename=filename, source_dataset=source_dataset
                )
            elif image_path:
                full_path = self.root_dir / image_path
                if not full_path.exists():
                    self.send_json({"status": "error", "error": f"Image file not found: {image_path}"}, status=400)
                    return
                # Check if there is an associated clean reference
                if "degraded_scenes" in image_path:
                    clean_name = full_path.stem.replace("_case1", "").replace("_case2", "").replace("_case3", "") + ".png"
                    clean_cand = self.root_dir / "data" / "clean_scenes" / clean_name
                    if clean_cand.exists():
                        clean_gt = np.array(Image.open(clean_cand).convert("L"), dtype=np.float64)
                elif "clean_scenes" in image_path:
                    # Degrade it using case 1 noise
                    clean_gt = np.array(Image.open(full_path).convert("L"), dtype=np.float64)
                    deg_cand = self.root_dir / "data" / "degraded_scenes" / f"{full_path.stem}_case1.png"
                    if deg_cand.exists():
                        full_path = deg_cand

                run_id, in_path, img_arr = self.io_manager.ingest_input(
                    full_path, filename=filename, source_dataset=source_dataset
                )
            else:
                self.send_json({"status": "error", "error": "No image_path or image_base64 provided."}, status=400)
                return

            params = {
                "engine": engine,
                "noise_sigma": noise_sigma,
                "max_iter_s1": max_iter_s1,
                "fast_mode": False,
            }

            metadata = self.io_manager.process_and_map(
                run_id=run_id,
                input_arr=img_arr,
                clean_gt=clean_gt,
                source_dataset=source_dataset,
                input_filename=filename,
                params=params,
                run_baselines=run_baselines,
                n_jobs=-1,
            )

            self.send_json({"status": "success", "metadata": metadata})

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_json({"status": "error", "error": str(e)}, status=500)

    def handle_api_batch_process(self, payload: Dict[str, Any]) -> None:
        """Execute batch processing over a collection of images."""
        dataset_group = payload.get("dataset_group", "yohann_mire")
        max_images = int(payload.get("max_images", 5))

        manifest_csv = self.root_dir / "data" / "dataset_manifest.csv"
        matching_paths = []
        if manifest_csv.exists():
            with open(manifest_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("dataset_group") == dataset_group:
                        matching_paths.append(r["saved_path"])

        selected = matching_paths[:max_images]
        results = []

        for p_str in selected:
            p = self.root_dir / p_str
            if p.exists():
                run_id, in_path, img_arr = self.io_manager.ingest_input(p, filename=p.name, source_dataset=dataset_group)
                meta = self.io_manager.process_and_map(
                    run_id=run_id,
                    input_arr=img_arr,
                    source_dataset=dataset_group,
                    input_filename=p.name,
                    params={"engine": "proposed_ours", "fast_mode": True},
                )
                results.append(meta)

        self.send_json({"status": "success", "processed_count": len(results), "runs": results})


# ==============================================================================
# Helper: Pre-Seeding 6 Canonical Benchmark Runs
# ==============================================================================

def seed_examples(root_dir: Union[str, Path] = ".") -> None:
    """Pre-seed 6 canonical benchmark runs into uploaded_inputs/ and uploaded_outputs/."""
    print("=" * 70)
    print("  Pre-Seeding 6 Benchmark Runs into uploaded_inputs/ <-> uploaded_outputs/")
    print("=" * 70)

    mgr = IOMappingManager(root_dir)
    clean_dir = Path(root_dir) / "data" / "clean_scenes"
    deg_dir = Path(root_dir) / "data" / "degraded_scenes"

    scene_stems = [
        ("Img1_Checkerboard_Yohann", "Yohann_Mire"),
        ("Img2_IndoorMonitor_Yohann", "Yohann_Mire"),
        ("Img3_BuildingSkyline_He", "DLS_NUC"),
        ("Img4_HousePedestrian_He", "DLS_NUC"),
        ("Img5_FacadeWindows_TNO", "TNO_Thermal"),
        ("Img6_StreetCars_LLVIP", "LLVIP_Thermal"),
    ]

    for stem, ds_name in scene_stems:
        clean_file = clean_dir / f"{stem}.png"
        deg_file = deg_dir / f"{stem}_case1.png"

        if not clean_file.exists() or not deg_file.exists():
            print(f"Skipping {stem}: files missing.")
            continue

        print(f"Processing seed run for {stem}...")
        clean_arr = np.array(Image.open(clean_file).convert("L"), dtype=np.float64)
        deg_arr = np.array(Image.open(deg_file).convert("L"), dtype=np.float64)

        run_id, in_path, img_arr = mgr.ingest_input(
            deg_arr,
            filename=f"{stem}_case1.png",
            source_dataset=ds_name,
        )

        meta = mgr.process_and_map(
            run_id=run_id,
            input_arr=img_arr,
            clean_gt=clean_arr,
            source_dataset=ds_name,
            input_filename=f"{stem}_case1.png",
            params={"engine": "proposed_ours", "fast_mode": True},
            run_baselines=True,
            n_jobs=-1,
        )

        print(f"  [OK] Run ID: {run_id} | Final PSNR: {meta['metrics']['final']['PSNR']:.2f} dB | SSIM: {meta['metrics']['final']['SSIM']:.4f}")

    print("\nPre-seeding completed successfully!")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="MATLAB-Style Web Workbench Server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default 8080)")
    parser.add_argument("--seed-examples", action="store_true", help="Pre-seed 6 canonical benchmark runs")

    args = parser.parse_args()

    if args.seed_examples:
        seed_examples()
        print("\nSeed example generation finished. Exiting as requested.")
        return

    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, WorkbenchRequestHandler)
    print("=" * 70)
    print("  MATLAB-Style Web Workbench Running")
    print(f"  URL: http://{args.host}:{args.port}/")
    print("  Mapping Database: uploaded_outputs/input_output_mapping.csv")
    print("  Press Ctrl+C to terminate.")
    print("=" * 70)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Web Workbench server.")
        httpd.server_close()


if __name__ == "__main__":
    main()
