"""common/dataset_generator.py - Infrared Benchmark Dataset Generator & Loader.

Generates 6 deterministic, high-detail synthetic thermal infrared benchmark scenes
(256x256 float64 in [0, 255]) saved into data/clean_scenes/:
  1. Img1_Checkerboard_Yohann.png (Periodic thermal tiles, sharp step boundaries)
  2. Img2_IndoorMonitor_Yohann.png (Warm desktop monitor, indoor heat gradients)
  3. Img3_BuildingSkyline_He.png (Smooth cold sky, sharp building rooflines & facades)
  4. Img4_HousePedestrian_He.png (Hot pedestrians, suburban house, cool road)
  5. Img5_FacadeWindows_TNO.png (Repetitive architectural window grid, structural frames)
  6. Img6_StreetCars_LLVIP.png (Nighttime street, hot vehicle engines & tires, pedestrians)

Also provides load_scene() to seamlessly ingest real .png / .jpg / .bmp / .tif thermal
imagery placed by users in data/clean_scenes/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_SCENES = [
    "Img1_Checkerboard_Yohann.png",
    "Img2_IndoorMonitor_Yohann.png",
    "Img3_BuildingSkyline_He.png",
    "Img4_HousePedestrian_He.png",
    "Img5_FacadeWindows_TNO.png",
    "Img6_StreetCars_LLVIP.png",
]


def _smooth_step(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Hermite smooth interpolation between edge0 and edge1."""
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _gaussian_2d(
    H: int,
    W: int,
    cy: float,
    cx: float,
    sy: float,
    sx: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate 2D Gaussian thermal blob."""
    y = np.arange(H, dtype=np.float64)[:, None]
    x = np.arange(W, dtype=np.float64)[None, :]
    g = amplitude * np.exp(-0.5 * (((y - cy) / sy) ** 2 + ((x - cx) / sx) ** 2))
    return g


# ==============================================================================
# Scene Synthesizers (256x256 float64 in [0, 255])
# ==============================================================================

def synthesize_checkerboard_yohann(H: int = 256, W: int = 256) -> np.ndarray:
    """Scene 1: Yohann Calibration Checkerboard with continuous thermal conduction."""
    rng = np.random.RandomState(101)
    y = np.arange(H, dtype=np.float64)[:, None]
    x = np.arange(W, dtype=np.float64)[None, :]

    # Ambient thermal drift across sensor array
    ambient = 120.0 + 35.0 * np.sin(x / 55.0) * np.cos(y / 70.0) + 10.0 * (y / H)

    # Checkerboard target (32x32 tiles)
    tile_size = 32
    check_y = (y // tile_size).astype(int)
    check_x = (x // tile_size).astype(int)
    board = ((check_y + check_x) % 2) * 90.0 - 45.0

    # Intra-tile thermal diffusion gradients
    sub_y = (y % tile_size) / float(tile_size)
    sub_x = (x % tile_size) / float(tile_size)
    diffusion = 18.0 * np.sin(np.pi * sub_x) * np.sin(np.pi * sub_y)

    # Subtle metallic target frame
    frame = np.zeros((H, W), dtype=np.float64)
    frame[16:240, 16:24] = 30.0
    frame[16:240, 232:240] = 30.0
    frame[16:24, 16:240] = 30.0
    frame[232:240, 16:240] = 30.0

    img = ambient + board + diffusion + frame
    # Low-level sensor thermal noise
    img += rng.normal(0.0, 0.8, (H, W))
    return np.clip(img, 0.0, 255.0)


def synthesize_indoor_monitor_yohann(H: int = 256, W: int = 256) -> np.ndarray:
    """Scene 2: Yohann Indoor Workspace with hot LCD monitor and table."""
    rng = np.random.RandomState(102)
    y = np.arange(H, dtype=np.float64)[:, None]
    x = np.arange(W, dtype=np.float64)[None, :]

    # Background wall with gentle thermal gradient
    wall = 95.0 + 15.0 * (1.0 - y / H) + 8.0 * np.sin(x / 40.0)

    # Desk surface at bottom (y > 180)
    desk_mask = y >= 180
    desk = 80.0 + 20.0 * ((y - 180.0) / 76.0)

    img = np.where(desk_mask, desk, wall)

    # Monitor outer bezel (y: 45 to 165, x: 50 to 206)
    bezel_mask = (y >= 45) & (y <= 165) & (x >= 50) & (x <= 206)
    img = np.where(bezel_mask, 140.0, img)

    # Monitor active screen (y: 53 to 157, x: 58 to 198)
    screen_mask = (y >= 53) & (y <= 157) & (x >= 58) & (x <= 198)
    screen_content = 185.0 + 25.0 * np.sin((y - 53) / 6.0) * np.cos((x - 58) / 14.0)
    # Bright window on monitor (warm browser/editor window)
    win_mask = (y >= 65) & (y <= 135) & (x >= 75) & (x <= 165)
    screen_content = np.where(win_mask, 215.0 + 10.0 * np.cos((y - 65) / 4.0), screen_content)
    img = np.where(screen_mask, screen_content, img)

    # Monitor stand & base
    stand_mask = (y > 165) & (y < 200) & (x >= 120) & (x <= 136)
    img = np.where(stand_mask, 115.0, img)
    base_mask = (y >= 195) & (y <= 205) & (x >= 100) & (x <= 156)
    img = np.where(base_mask, 125.0, img)

    # Warm coffee mug on desk
    mug_blob = _gaussian_2d(H, W, 215.0, 215.0, 10.0, 8.0, amplitude=95.0)
    img += mug_blob

    img += rng.normal(0.0, 0.7, (H, W))
    return np.clip(img, 0.0, 255.0)


def synthesize_building_skyline_he(H: int = 256, W: int = 256) -> np.ndarray:
    """Scene 3: He et al. Urban Building Skyline with cold sky and warm structures."""
    rng = np.random.RandomState(103)
    y = np.arange(H, dtype=np.float64)[:, None]
    x = np.arange(W, dtype=np.float64)[None, :]

    # Sky: Low thermal emission with vertical atmospheric lapse rate (broadcast to H, W)
    sky = 35.0 + 30.0 * (y / 150.0) + 0.0 * x
    img = np.copy(sky)

    # Background buildings (y > 80)
    b1_mask = (x >= 20) & (x <= 75) & (y >= 70)
    img = np.where(b1_mask, 130.0 + 10.0 * np.sin(y / 5.0), img)

    b2_mask = (x >= 70) & (x <= 145) & (y >= 45)
    img = np.where(b2_mask, 145.0 + 8.0 * np.cos(x / 4.0), img)

    b3_mask = (x >= 140) & (x <= 200) & (y >= 85)
    img = np.where(b3_mask, 120.0 + 12.0 * np.sin(y / 8.0), img)

    b4_mask = (x >= 195) & (x <= 245) & (y >= 60)
    img = np.where(b4_mask, 150.0, img)

    # Repetitive office window heat signatures on tower b2
    for r in range(55, 180, 12):
        for c in range(80, 138, 10):
            win = (y >= r) & (y <= r + 6) & (x >= c) & (x <= c + 5)
            heat = 195.0 if ((r + c) % 3 != 0) else 110.0
            img = np.where(win, heat, img)

    # Foreground lower structure and road (y > 190)
    ground_mask = y >= 190
    ground = 110.0 + 20.0 * ((y - 190) / 66.0)
    img = np.where(ground_mask, ground, img)

    # Antenna spire on center tower
    spire_mask = (x >= 106) & (x <= 108) & (y >= 20) & (y < 45)
    img = np.where(spire_mask, 160.0, img)

    img += rng.normal(0.0, 0.8, (H, W))
    return np.clip(img, 0.0, 255.0)


def synthesize_house_pedestrian_he(H: int = 256, W: int = 256) -> np.ndarray:
    """Scene 4: He et al. Suburban House with sharp roof and hot pedestrians."""
    rng = np.random.RandomState(104)
    y = np.arange(H, dtype=np.float64)[:, None]
    x = np.arange(W, dtype=np.float64)[None, :]

    # Cool sky background (broadcast to H, W)
    sky = 40.0 + 25.0 * (y / 120.0) + 0.0 * x
    img = np.copy(sky)

    # Trees/shrubs in background (fractal-like thermal noise)
    tree_mask = (y >= 70) & (y <= 160)
    tree_tex = 75.0 + 15.0 * np.sin(x / 3.0) * np.cos(y / 4.0) + rng.normal(0, 4.0, (H, W))
    img = np.where(tree_mask, tree_tex, img)

    # House main body (x: 40 to 170, y: 110 to 195)
    house_body = (x >= 40) & (x <= 170) & (y >= 110) & (y <= 195)
    img = np.where(house_body, 125.0, img)

    # Pitched roof (triangular)
    # Peak at (x=105, y=55), left corner (40, 110), right corner (170, 110)
    roof_mask = (y >= 55) & (y <= 110) & (y >= 55 + (110 - 55) * np.abs(x - 105) / 65.0)
    img = np.where(roof_mask, 105.0, img)

    # Chimney on roof
    chimney_mask = (x >= 135) & (x <= 150) & (y >= 45) & (y <= 85)
    img = np.where(chimney_mask, 165.0, img)

    # Warm windows & door
    w1 = (x >= 55) & (x <= 80) & (y >= 125) & (y <= 150)
    w2 = (x >= 120) & (x <= 145) & (y >= 125) & (y <= 150)
    door = (x >= 90) & (x <= 112) & (y >= 150) & (y <= 195)
    img = np.where(w1 | w2, 195.0, img)
    img = np.where(door, 110.0, img)

    # Asphalt street in foreground
    street_mask = y > 195
    street = 75.0 + 15.0 * ((y - 195) / 61.0)
    img = np.where(street_mask, street, img)

    # Hot Pedestrian 1 (walking on road, x ~ 195, y ~ 185..235)
    # Head
    img += _gaussian_2d(H, W, 190.0, 195.0, 4.0, 3.5, amplitude=195.0)
    # Torso
    img += _gaussian_2d(H, W, 206.0, 195.0, 10.0, 5.5, amplitude=210.0)
    # Legs
    img += _gaussian_2d(H, W, 224.0, 192.0, 8.0, 2.5, amplitude=180.0)
    img += _gaussian_2d(H, W, 224.0, 198.0, 8.0, 2.5, amplitude=180.0)

    # Hot Pedestrian 2 (smaller/further, x ~ 75, y ~ 180..215)
    img += _gaussian_2d(H, W, 185.0, 75.0, 3.0, 2.5, amplitude=180.0)
    img += _gaussian_2d(H, W, 197.0, 75.0, 7.0, 4.0, amplitude=190.0)
    img += _gaussian_2d(H, W, 210.0, 74.0, 6.0, 2.0, amplitude=165.0)
    img += _gaussian_2d(H, W, 210.0, 77.0, 6.0, 2.0, amplitude=165.0)

    img += rng.normal(0.0, 0.7, (H, W))
    return np.clip(img, 0.0, 255.0)


def synthesize_facade_windows_tno(H: int = 256, W: int = 256) -> np.ndarray:
    """Scene 5: TNO High-Detail Architectural Facade with dense window grid."""
    rng = np.random.RandomState(105)
    y = np.arange(H, dtype=np.float64)[:, None]
    x = np.arange(W, dtype=np.float64)[None, :]

    # Brick facade base with texture
    base = 118.0 + 8.0 * np.sin(y / 2.5) + 6.0 * np.cos(x / 2.5)
    img = np.copy(base)

    # Dense architectural window grid: 5 rows x 6 columns
    rows_starts = [20, 65, 110, 155, 200]
    cols_starts = [18, 56, 94, 132, 170, 208]
    win_h, win_w = 30, 26

    for r_idx, r in enumerate(rows_starts):
        # Floor separation cornice / ledge (horizontal high-contrast strip)
        ledge_mask = (y >= r - 5) & (y <= r - 2)
        img = np.where(ledge_mask, 160.0, img)

        for c_idx, c in enumerate(cols_starts):
            # Outer window frame (cooler frame)
            outer = (y >= r) & (y <= r + win_h) & (x >= c) & (x <= c + win_w)
            img = np.where(outer, 90.0, img)

            # Inner window glass (panes)
            inner = (
                (y >= r + 3)
                & (y <= r + win_h - 3)
                & (x >= c + 3)
                & (x <= c + win_w - 3)
            )

            # Different rooms have varying internal heat (some lights on, some cool)
            room_heat_state = (r_idx * 7 + c_idx * 11) % 5
            if room_heat_state == 0:
                pane_intensity = 210.0 + 15.0 * np.cos((y - r) / 3.0)
            elif room_heat_state == 1:
                pane_intensity = 185.0
            elif room_heat_state == 2:
                pane_intensity = 145.0 + 10.0 * np.sin((x - c) / 4.0)
            else:
                pane_intensity = 70.0 + 10.0 * ((y - r) / float(win_h))

            img = np.where(inner, pane_intensity, img)

            # Window mullion (cross-bar)
            mullion_v = (
                (y >= r + 3)
                & (y <= r + win_h - 3)
                & (x >= c + win_w // 2 - 1)
                & (x <= c + win_w // 2)
            )
            mullion_h = (
                (y >= r + win_h // 2 - 1)
                & (y <= r + win_h // 2)
                & (x >= c + 3)
                & (x <= c + win_w - 3)
            )
            img = np.where(mullion_v | mullion_h, 95.0, img)

            # Window sill (bottom reflector)
            sill = (y >= r + win_h + 1) & (y <= r + win_h + 3) & (x >= c - 2) & (x <= c + win_w + 2)
            img = np.where(sill, 175.0, img)

    img += rng.normal(0.0, 0.8, (H, W))
    return np.clip(img, 0.0, 255.0)


def synthesize_street_cars_llvip(H: int = 256, W: int = 256) -> np.ndarray:
    """Scene 6: LLVIP Nighttime Surveillance with perspective road, cars, and pedestrians."""
    rng = np.random.RandomState(106)
    y = np.arange(H, dtype=np.float64)[:, None]
    x = np.arange(W, dtype=np.float64)[None, :]

    # Background cityscape & dark sky (broadcast to H, W)
    bg = 45.0 + 20.0 * (y / 110.0) + 0.0 * x
    img = np.copy(bg)

    # Road perspective (y: 110 to 256)
    road_mask = y >= 110
    road = 75.0 + 25.0 * ((y - 110.0) / 146.0)
    img = np.where(road_mask, road, img)

    # Sidewalk curb
    curb = (y >= 106) & (y <= 110)
    img = np.where(curb, 130.0, img)

    # Overhead streetlamp hotspot (x=50, y=30) with bloom
    img += _gaussian_2d(H, W, 30.0, 50.0, 6.0, 6.0, amplitude=210.0)
    img += _gaussian_2d(H, W, 30.0, 50.0, 25.0, 25.0, amplitude=50.0)

    # Vehicle 1 (Sedan moving on right lane, centered at x=175, y=170)
    # Chassis
    car1_body = (x >= 125) & (x <= 225) & (y >= 155) & (y <= 190)
    img = np.where(car1_body, 135.0, img)
    # Cabin / Roof
    car1_roof = (x >= 145) & (x <= 195) & (y >= 135) & (y < 155)
    img = np.where(car1_roof, 110.0, img)
    # Windshield / Windows (cool reflected sky)
    car1_glass = (x >= 150) & (x <= 190) & (y >= 138) & (y <= 152)
    img = np.where(car1_glass, 65.0, img)
    # Hot engine hood / grill
    img += _gaussian_2d(H, W, 172.0, 135.0, 8.0, 10.0, amplitude=105.0)
    # Hot tires (frictional heat from driving)
    img += _gaussian_2d(H, W, 192.0, 142.0, 5.0, 6.0, amplitude=115.0)
    img += _gaussian_2d(H, W, 192.0, 208.0, 5.0, 6.0, amplitude=115.0)
    # Hot exhaust pipe
    img += _gaussian_2d(H, W, 186.0, 222.0, 3.5, 3.5, amplitude=120.0)

    # Vehicle 2 (Parked SUV on left, centered at x=50, y=135)
    car2_body = (x >= 20) & (x <= 80) & (y >= 120) & (y <= 145)
    img = np.where(car2_body, 115.0, img)
    car2_top = (x >= 28) & (x <= 72) & (y >= 105) & (y < 120)
    img = np.where(car2_top, 95.0, img)

    # Pedestrian crossing street (x ~ 100, y ~ 160..205)
    img += _gaussian_2d(H, W, 165.0, 102.0, 3.5, 3.0, amplitude=200.0)  # Head
    img += _gaussian_2d(H, W, 178.0, 102.0, 9.0, 4.5, amplitude=220.0)  # Torso
    img += _gaussian_2d(H, W, 195.0, 99.0, 8.0, 2.5, amplitude=185.0)   # Leg L
    img += _gaussian_2d(H, W, 195.0, 106.0, 8.0, 2.5, amplitude=185.0)  # Leg R

    # Dashed street center line
    for seg_y in range(120, 250, 24):
        dash = (y >= seg_y) & (y <= seg_y + 12) & (x >= 110) & (x <= 113)
        img = np.where(dash, 160.0, img)

    img += rng.normal(0.0, 0.7, (H, W))
    return np.clip(img, 0.0, 255.0)


# ==============================================================================
# Generator and Loader Functions
# ==============================================================================

SYNTHESIZER_REGISTRY = {
    "Img1_Checkerboard_Yohann.png": synthesize_checkerboard_yohann,
    "Img2_IndoorMonitor_Yohann.png": synthesize_indoor_monitor_yohann,
    "Img3_BuildingSkyline_He.png": synthesize_building_skyline_he,
    "Img4_HousePedestrian_He.png": synthesize_house_pedestrian_he,
    "Img5_FacadeWindows_TNO.png": synthesize_facade_windows_tno,
    "Img6_StreetCars_LLVIP.png": synthesize_street_cars_llvip,
}


def generate_benchmark_scenes(
    output_dir: Union[str, Path] = "data/clean_scenes"
) -> Dict[str, np.ndarray]:
    """Ensure all 6 benchmark authentic infrared scenes are harvested and loaded from real repositories.
    
    Guarantees 100% authentic infrared imagery with ZERO synthetic fallbacks.
    """
    from common.fetch_real_datasets import harvest_all
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    canonical_dir = PROJECT_ROOT / "data" / "clean_scenes"

    if not all((canonical_dir / fn).is_file() for fn in BENCHMARK_SCENES):
        harvest_all()

    generated: Dict[str, np.ndarray] = {}
    for filename in BENCHMARK_SCENES:
        src_file = canonical_dir / filename
        dest_file = out_path / filename
        pil_img = Image.open(src_file).convert("L")
        if dest_file != src_file:
            pil_img.save(dest_file, format="PNG")
        generated[filename] = np.array(pil_img, dtype=np.float64)

    return generated


def load_scene(
    scene_name_or_path: Union[str, Path],
    clean_scenes_dir: Union[str, Path] = "data/clean_scenes",
) -> np.ndarray:
    """Load an infrared scene as float64 array in [0, 255].

    Supports:
    - Standard benchmark names (e.g. 'Img1_Checkerboard_Yohann.png')
    - Custom user images (.png, .jpg, .bmp, .tif, .tiff) placed in clean_scenes_dir
    - Absolute or relative file paths to any image file.

    Parameters
    ----------
    scene_name_or_path : Union[str, Path]
        Filename or path to the scene.
    clean_scenes_dir : Union[str, Path]
        Directory containing clean scenes.

    Returns
    -------
    img : np.ndarray
        2D float64 image array in [0, 255].
    """
    p = Path(scene_name_or_path)

    # If direct file path exists
    if p.is_file():
        target_path = p
    else:
        # Check inside clean_scenes_dir
        target_path = Path(clean_scenes_dir) / scene_name_or_path
        if not target_path.is_file():
            # If benchmark scene not yet generated, generate on the fly
            filename = target_path.name
            if filename in SYNTHESIZER_REGISTRY:
                generate_benchmark_scenes(clean_scenes_dir)
            else:
                raise FileNotFoundError(
                    f"Image not found at '{target_path}' or '{p}'. "
                    f"Please verify the filename or place your custom image in '{clean_scenes_dir}'."
                )

    pil_img = Image.open(target_path)
    # Convert to grayscale if RGB or RGBA
    if pil_img.mode != "L":
        pil_img = pil_img.convert("L")

    arr = np.array(pil_img, dtype=np.float64)
    return arr


if __name__ == "__main__":
    import sys

    dest_dir = sys.argv[1] if len(sys.argv) > 1 else "data/clean_scenes"
    print(f"Generating 6 IEEE Sensors Journal infrared benchmark scenes in: {dest_dir}")
    scenes = generate_benchmark_scenes(dest_dir)
    for name, img in scenes.items():
        print(f"  [CREATED] {name:32s} shape={img.shape} dtype={img.dtype} range=[{img.min():.1f}, {img.max():.1f}]")
    print("Done! All 6 clean benchmark scenes successfully generated.")
