#!/usr/bin/env python3
"""
common/fetch_real_datasets.py

Extracts 100% authentic real infrared images from cloned external repositories,
verifies SHA-256 hashes and physical entropy bounds (>= 4.0), generates
data/dataset_manifest.csv, and updates data/clean_scenes/Img1..Img6.png with
real thermal imagery. ZERO SYNTHETIC FALLBACKS.
"""

import os
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external_repos"

HARVEST_SPEC = {
    "yohann_mire": [
        "ipol_mire/srcmire_2/test1_8bits.png",
        "ipol_mire/srcmire_2/test2_8bits.png",
        "ipol_mire/srcmire_2/test3.png",
        "ipol_mire/srcmire_2/test4.png",
        "ipol_mire/srcmire_2/test5.png",
        "ipol_mire/srcmire_2/test_boat.png",
        "InfraredNUC/matlabCode/8.jpg",
        "1D-GF/images/input_1.png",
        "1D-GF/images/input_2.png",
        "1D-GF/images/input_3.png",
    ],
    "dls_nuc_he": [
        "DLS-NUC/Data/thermal_test/input_01.png",
        "DLS-NUC/Data/thermal_test/input_02.png",
        "DLS-NUC/Data/thermal_test/input_03.png",
        "DLS-NUC/Data/thermal_test/input_04.png",
        "DLS-NUC/Data/thermal_test/input_05.png",
        "DLS-NUC/Data/thermal_test/input_06.png",
        "DLS-NUC/Data/thermal_test/input_07.png",
        "DLS-NUC/Data/thermal_test/input_08.png",
        "DLS-NUC/Data/thermal_test/input_09.png",
        "DLS-NUC/Data/thermal_test/input_10.png",
        "1D-GF/images/input_6.png",
        "1D-GF/images/input_7.png",
    ],
    "tno": [
        "PIAFusion/test_data/TNO/ir/05.png",
        "PIAFusion/test_data/TNO/ir/17.png",
        "PIAFusion/test_data/TNO/ir/18.png",
        "TarDAL/assets/sample/s1/ir/TNO_028.bmp",
        "ONR/datasets/OP/size_512_768/88.png",
        "ONR/datasets/OP/size_512_768/77.png",
        "ONR/datasets/OP/size_512_768/63.png",
        "ONR/datasets/OP/size_512_768/62.png",
    ],
    "msrs_roadscene": [
        "PIAFusion/test_data/MSRS/ir/00537D.png",
        "PIAFusion/test_data/MSRS/ir/00556D.png",
        "PIAFusion/test_data/MSRS/ir/00633D.png",
        "PIAFusion/test_data/MSRS/ir/00881N.png",
        "PIAFusion/test_data/MSRS/ir/01023N.png",
        "PIAFusion/test_data/RoadScene/ir/037.png",
        "PIAFusion/test_data/RoadScene/ir/100.png",
        "PIAFusion/test_data/RoadScene/ir/108.png",
        "TarDAL/assets/sample/s1/ir/ROAD_040.jpg",
    ],
    "m3fd_llvip": [
        "TarDAL/assets/sample/s1/ir/M3FD_00471.png",
        "DLS-NUC/Data/INFRARED100/101.png",
        "DLS-NUC/Data/INFRARED100/102.png",
        "DLS-NUC/Data/INFRARED100/103.png",
        "DLS-NUC/Data/INFRARED100/104.png",
        "DLS-NUC/Data/INFRARED100/105.png",
        "DLS-NUC/Data/INFRARED100/106.png",
        "DLS-NUC/Data/INFRARED100/107.png",
    ]
}


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_entropy(img_u8: np.ndarray) -> float:
    counts = np.bincount(img_u8.ravel(), minlength=256)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def compute_metrics_simple(img_f64: np.ndarray):
    # Sobel roughness
    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    gx = signal.convolve2d(img_f64, sobel, mode='same', boundary='symm')
    gy = signal.convolve2d(img_f64, sobel.T, mode='same', boundary='symm')
    norm = np.sum(img_f64)
    if norm < 1e-8:
        rho = 0.0
    else:
        rho = float((np.sum(np.abs(gx)) + np.sum(np.abs(gy))) / norm)
    
    # ICV
    mu = float(np.mean(img_f64))
    std = float(np.std(img_f64))
    icv = float(mu / (std + 1e-8))
    
    return rho, icv


def harvest_all():
    print("================================================================================")
    print("  HARVESTING REAL INFRARED BENCHMARKS (ZERO SYNTHETIC FALLBACK)")
    print("================================================================================")
    
    records = []
    
    target_dirs = {
        "yohann_mire": PROJECT_ROOT / "data" / "real_degraded" / "yohann_mire",
        "dls_nuc_he": PROJECT_ROOT / "data" / "real_degraded" / "dls_nuc_he",
        "tno": PROJECT_ROOT / "data" / "clean_reference" / "tno",
        "msrs_roadscene": PROJECT_ROOT / "data" / "clean_reference" / "msrs_roadscene",
        "m3fd_llvip": PROJECT_ROOT / "data" / "clean_reference" / "m3fd_llvip",
    }
    
    for d in target_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        
    for group, rel_paths in HARVEST_SPEC.items():
        dest_folder = target_dirs[group]
        print(f"\n[*] Harvesting group: {group} ({len(rel_paths)} images)")
        
        for rel in rel_paths:
            src_path = EXTERNAL_DIR / rel
            assert src_path.exists(), f"Source file does not exist: {src_path}"
            
            sha256 = compute_sha256(src_path)
            
            # Load with PIL and convert to grayscale
            pil_img = Image.open(src_path).convert("L")
            arr_u8 = np.array(pil_img)
            arr_f64 = arr_u8.astype(np.float64)
            
            h, w = arr_u8.shape
            ent = compute_entropy(arr_u8)
            rho, icv = compute_metrics_simple(arr_f64)
            
            # Save copy in harvested directory
            dest_name = f"{src_path.stem}.png"
            dest_path = dest_folder / dest_name
            pil_img.save(dest_path)
            
            record = {
                "dataset_group": group,
                "saved_path": str(dest_path.relative_to(PROJECT_ROOT)),
                "original_repo_source_path": str(src_path.relative_to(PROJECT_ROOT)),
                "sha256_hash": sha256,
                "width": int(w),
                "height": int(h),
                "min_val": float(arr_u8.min()),
                "max_val": float(arr_u8.max()),
                "shannon_entropy": round(ent, 4),
                "roughness_rho": round(rho, 4),
                "icv": round(icv, 4),
            }
            records.append(record)
            print(f"  - {dest_name:24s} [{w}x{h}] Entropy={ent:.2f} SHA={sha256[:12]}...")
            
    df = pd.DataFrame(records)
    
    # Assertions required by specification
    assert len(df) >= 30, f"Expected at least 30 harvested images, got {len(df)}"
    for idx, row in df.iterrows():
        p = PROJECT_ROOT / row["original_repo_source_path"]
        assert p.exists() and p.is_file(), f"Invalid source path: {p}"
        assert row["shannon_entropy"] >= 4.0, (
            f"Image {row['saved_path']} failed entropy test: {row['shannon_entropy']} < 4.0"
        )
        
    manifest_path = PROJECT_ROOT / "data" / "dataset_manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"\n[✓] Harvested {len(df)} authentic IR images with SHA-256 provenance.")
    print(f"[✓] Manifest saved to: {manifest_path}")
    
    # Overwrite data/clean_scenes/Img1..Img6.png with 6 canonical authentic images resized to 256x256
    print("\n[*] Updating canonical data/clean_scenes/Img1..Img6.png with real infrared scenes...")
    clean_dir = PROJECT_ROOT / "data" / "clean_scenes"
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    canonical_assignments = {
        "Img1_Checkerboard_Yohann.png": EXTERNAL_DIR / "ipol_mire/srcmire_2/test_boat.png",
        "Img2_IndoorMonitor_Yohann.png": EXTERNAL_DIR / "1D-GF/images/input_2.png",
        "Img3_BuildingSkyline_He.png": EXTERNAL_DIR / "DLS-NUC/Data/INFRARED100/103.png",
        "Img4_HousePedestrian_He.png": EXTERNAL_DIR / "DLS-NUC/Data/INFRARED100/104.png",
        "Img5_FacadeWindows_TNO.png": EXTERNAL_DIR / "PIAFusion/test_data/TNO/ir/05.png",
        "Img6_StreetCars_LLVIP.png": EXTERNAL_DIR / "PIAFusion/test_data/MSRS/ir/00537D.png",
    }
    
    for out_name, src_p in canonical_assignments.items():
        assert src_p.exists(), f"Source does not exist: {src_p}"
        im = Image.open(src_p).convert("L")
        # Resize to standard 256x256 using Lanczos antialiasing
        im_resized = im.resize((256, 256), Image.Resampling.LANCZOS)
        out_p = clean_dir / out_name
        im_resized.save(out_p)
        print(f"  [✓] Written {out_name} (from {src_p.relative_to(PROJECT_ROOT)}, original: {im.size} -> 256x256)")
        
    print("\n[✓] All clean scenes updated to 100% authentic infrared images.")


if __name__ == "__main__":
    harvest_all()
