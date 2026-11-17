#!/usr/bin/env python3
"""
scripts/clone_and_inspect_repos.py

Clones the author and baseline repositories, downloads IPOL mire archives,
enumerates all source and dataset files, and inspects authors' MATLAB/C++ implementations.
"""

import os
import sys
import subprocess
import urllib.request
import re
from pathlib import Path

REPOS = {
    "InfraredNUC": "https://github.com/ImageVisioner/InfraredNUC.git",
    "IRdenoise": "https://github.com/ImageVisioner/IRdenoise.git",
    "Image_Fusion_Evaluation_modify": "https://github.com/ImageVisioner/Image_Fusion_Evaluation_modify.git",
    "DLS-NUC": "https://github.com/hezw2016/DLS-NUC.git",
    "1D-GF": "https://github.com/hezw2016/1D-GF.git",
    "ONR": "https://github.com/Kuangxd/ONR.git",
    "PIAFusion": "https://github.com/Linfeng-Tang/PIAFusion.git",
    "SwinFusion": "https://github.com/Linfeng-Tang/SwinFusion.git",
    "TarDAL": "https://github.com/JinyuanLiu-CV/TarDAL.git",
}

IPOL_BASE = "https://www.ipol.im/pub/art/2012/glmt-mire/"
IPOL_FILES = [
    "article_lr.pdf",
    "src.tar.gz",
    "src.zip",
]

TARGET_EXTS = {".m", ".cpp", ".h", ".py", ".mat", ".bmp", ".png", ".jpg", ".tif", ".raw", ".tiff"}


def clone_repo(name: str, url: str, dest_dir: Path):
    target = dest_dir / name
    if target.exists() and any(target.iterdir()):
        print(f"[+] Repo '{name}' already exists at {target}. Skipping clone.")
        return target
    
    print(f"[*] Cloning {name} from {url}...")
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        res = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(target)],
            env=env,
            capture_output=True,
            text=True,
            timeout=180
        )
        if res.returncode == 0:
            print(f"[✓] Successfully cloned {name}.")
        else:
            print(f"[!] Warning: failed to clone {name}: {res.stderr.strip()}")
    except Exception as e:
        print(f"[!] Error cloning {name}: {e}")
    return target


def download_ipol(dest_dir: Path):
    ipol_dir = dest_dir / "ipol_mire"
    ipol_dir.mkdir(parents=True, exist_ok=True)
    
    # Try fetching index page to discover files
    try:
        req = urllib.request.Request(
            IPOL_BASE,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            links = re.findall(r'href=["\']([^"\']+)["\']', html)
            for link in links:
                if any(link.endswith(ext) for ext in [".tar.gz", ".zip", ".pdf", ".png", ".bmp", ".tif", ".jpg", ".pgm"]):
                    full_url = urllib.parse.urljoin(IPOL_BASE, link)
                    fname = os.path.basename(urllib.parse.urlparse(full_url).path)
                    if fname and not (ipol_dir / fname).exists():
                        try:
                            print(f"[*] Downloading IPOL asset {fname}...")
                            urllib.request.urlretrieve(full_url, ipol_dir / fname)
                        except Exception as e:
                            print(f"[!] Failed to download {fname}: {e}")
    except Exception as e:
        print(f"[!] Could not scrape IPOL index: {e}")
        # Try direct known files
        for fname in IPOL_FILES:
            out_file = ipol_dir / fname
            if not out_file.exists():
                try:
                    full_url = IPOL_BASE + fname
                    print(f"[*] Attempting download of {full_url}...")
                    urllib.request.urlretrieve(full_url, out_file)
                    print(f"[✓] Downloaded {fname}")
                except Exception as ex:
                    print(f"[!] Could not download {fname}: {ex}")


def enumerate_files(dest_dir: Path):
    found_files = []
    for root, _, files in os.walk(dest_dir):
        # Skip .git directories
        if "/.git" in root or root.endswith("/.git"):
            continue
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in TARGET_EXTS:
                full_path = Path(root) / f
                found_files.append((ext, full_path))
    return found_files


def main():
    root = Path(__file__).resolve().parent.parent
    dest_dir = root / "data" / "external_repos"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print("================================================================================")
    print("  CLONING EXTERNAL REPOSITORIES & DOWNLOADING REAL IR DATASETS")
    print("================================================================================")
    
    for name, url in REPOS.items():
        clone_repo(name, url, dest_dir)
        
    download_ipol(dest_dir)
    
    print("\n================================================================================")
    print("  ENUMERATING FILES ACROSS EXTERNAL REPOSITORIES")
    print("================================================================================")
    
    files = enumerate_files(dest_dir)
    print(f"[✓] Found {len(files)} target files ({sorted(TARGET_EXTS)})")
    
    by_ext = {}
    for ext, path in files:
        by_ext.setdefault(ext, []).append(path)
        
    for ext, p_list in sorted(by_ext.items()):
        print(f"  {ext:6s}: {len(p_list):4d} files")
        
    # Specifically inspect MATLAB and C++ files in key repos
    print("\n================================================================================")
    print("  KEY AUTHORS' MATLAB / C++ FILES:")
    print("================================================================================")
    for ext in [".m", ".cpp", ".h"]:
        for p in by_ext.get(ext, []):
            rel = p.relative_to(dest_dir)
            if any(k in str(rel) for k in ["InfraredNUC", "Image_Fusion_Evaluation", "1D-GF", "DLS-NUC"]):
                print(f"  - {rel} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
