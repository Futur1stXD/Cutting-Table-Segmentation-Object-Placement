"""
Cutting-Table v4.9.1 — Multi-Object Distribution + Enhanced 2D Nesting
────────────────────────────────────────────
• Uses Intel RealSense camera RGB stream
• Detects ArUco frame and corrects perspective
• Pure edge-based segmentation: raw Canny edges + dilated edges
• Finds sheets of material in the specified size range
• Click on objects to automatically detect their dimensions
• Places rectangles and circles of various sizes (mm) across multiple objects
• Distributes pieces optimally across selected objects
• Shows utilization coefficient and outputs
  `[ERROR]` for pieces too large, `[WARN]` if space is insufficient

ALGORITHM IMPROVEMENTS (v4.9.1):
• IMPROVED CIRCLE PLACEMENT: 2mm boundary margin for circles prevents edge violations
• STRICTER CIRCLE VALIDATION: 8-point perimeter check (7/8 must be inside)
• BETTER SORTING: Rectangles placed first, then circles (easier to fit rects)
• TOLERANT RECT VALIDATION: Inset corner checks for numerical precision
• REDUCED FALSE ERRORS: Boundary checks use appropriate tolerances

Previous (v4.9.0):
• TRUE BOTTOM-LEFT-FILL (BLF): Proper BLF scan prioritizing lowest Y, then lowest X
• SKYLINE TRACKING: Efficient strip-packing with skyline height tracking
• REDUCED SPACING: Tighter packing with 0.3mm spacing
• NEW SORTING STRATEGIES: Added FFDH, width-fit, longest-side sorting (15 total)

• Controls:
  - Esc – exit, r – reset background, c – clear selection
"""

from __future__ import annotations

import cv2
import math
import sys
import io
import time
from collections import deque
import pyrealsense2 as rs
from typing import Optional
from multiprocessing import Pool, cpu_count
import os
from functools import lru_cache

# ═══════════════════════════════════════════════════════════════════════════
# GPU Acceleration Setup: Try CuPy (GPU) first, fallback to NumPy (CPU)
# ═══════════════════════════════════════════════════════════════════════════
USE_GPU = True  # Set to False to force CPU usage
GPU_AVAILABLE = False
xp = None  # Will be cupy or numpy

try:
    if USE_GPU:
        import cupy as cp
        # Test if GPU is actually available
        device = cp.cuda.Device(0)
        device.compute_capability  # Test access
        xp = cp
        GPU_AVAILABLE = True

        # Get GPU info safely with multiple fallback options
        try:
            # Method 1: Direct name attribute
            gpu_name = device.name.decode('utf-8') if isinstance(device.name, bytes) else str(device.name)
        except:
            try:
                # Method 2: Get from CUDA runtime
                gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
            except:
                # Method 3: Just use device ID
                gpu_name = f"CUDA Device {device.id}"

        try:
            gpu_memory = device.mem_info[1] / (1024**3)  # Total memory in GB
        except:
            gpu_memory = 0.0

        print(f'[INFO] 🚀 GPU ACCELERATION ENABLED')
        print(f'[INFO] GPU: {gpu_name}')
        if gpu_memory > 0:
            print(f'[INFO] GPU Memory: {gpu_memory:.1f} GB')
    else:
        raise ImportError("GPU disabled by user")
except (ImportError, Exception) as e:
    import numpy as np
    xp = np
    GPU_AVAILABLE = False
    if USE_GPU:
        error_msg = str(e)
        print(f'[INFO] GPU not available: {error_msg}')
        print('[INFO] Using CPU (NumPy) instead')
        if 'No module named' in error_msg:
            print('[INFO] To enable GPU: pip install cupy-cuda12x (or cupy-cuda11x)')
    else:
        print('[INFO] GPU disabled, using CPU (NumPy)')

# Keep numpy import for compatibility with existing code
import numpy as np

# Helper functions for GPU/CPU array conversion
def to_gpu(arr):
    """Move array to GPU if available."""
    if GPU_AVAILABLE and not isinstance(arr, xp.ndarray):
        return xp.asarray(arr)
    return arr

def to_cpu(arr):
    """Move array to CPU (convert to numpy)."""
    if GPU_AVAILABLE and isinstance(arr, xp.ndarray):
        return xp.asnumpy(arr)
    return arr

def ensure_cpu_array(arr):
    """Ensure array is on CPU for compatibility."""
    return to_cpu(arr) if GPU_AVAILABLE else arr

# GPU-optimized array operations
def create_zeros(shape, dtype=xp.uint8):
    """Create zeros array on GPU if available."""
    return xp.zeros(shape, dtype=dtype)

def create_ones(shape, dtype=xp.uint8):
    """Create ones array on GPU if available."""
    return xp.ones(shape, dtype=dtype)

def gpu_all(condition):
    """GPU-accelerated all() operation."""
    return xp.all(condition)

def gpu_any(condition):
    """GPU-accelerated any() operation."""
    return xp.any(condition)

def gpu_where(condition):
    """GPU-accelerated where() operation."""
    return xp.where(condition)

# OpenCV-compatible operations (must use NumPy)
def cv2_fillPoly_gpu_aware(grid, points, value):
    """Fill polygon with GPU awareness - converts to CPU for OpenCV."""
    if GPU_AVAILABLE:
        # Convert to CPU for OpenCV operation
        grid_cpu = to_cpu(grid)
        cv2.fillPoly(grid_cpu, [points], value)
        # Convert back to GPU
        return to_gpu(grid_cpu)
    else:
        cv2.fillPoly(grid, [points], value)
        return grid

def cv2_circle_gpu_aware(grid, center, radius, value, thickness=-1):
    """Draw circle with GPU awareness - converts to CPU for OpenCV."""
    if GPU_AVAILABLE:
        grid_cpu = to_cpu(grid)
        cv2.circle(grid_cpu, center, radius, value, thickness)
        return to_gpu(grid_cpu)
    else:
        cv2.circle(grid, center, radius, value, thickness)
        return grid

def cv2_dilate_gpu_aware(grid, kernel, iterations=1):
    """Dilate with GPU awareness - converts to CPU for OpenCV."""
    if GPU_AVAILABLE:
        grid_cpu = to_cpu(grid)
        result = cv2.dilate(grid_cpu, kernel, iterations=iterations)
        return to_gpu(result)
    else:
        return cv2.dilate(grid, kernel, iterations=iterations)

# Try to import Numba for JIT compilation (optional speedup)
try:
    from numba import jit
    NUMBA_AVAILABLE = True
    print('[INFO] Numba JIT compiler available - enabling accelerated math operations')
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator if Numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# ───── Camera / marker settings ───────────────────────────
# Use Intel RealSense camera
ARUCO_DICT = cv2.aruco.DICT_4X4_50
ARUCO_SIZE_MM = 120  # Will be updated from startup prompt (physical marker side length)
MIN_OBJECT_AREA = 150
SIZE_TOLERANCE = 0.15  # Size tolerance for detection
# Add frame smoothing settings
STABILIZATION_BUFFER_SIZE = 3  # Reduced for better responsiveness
RECALCULATION_DELAY = 5       # Reduced for better responsiveness
PACKING_ATTEMPTS = 25         # Maximum attempts for densest packing (reduced for speed)
PARALLEL_PACKING_ATTEMPTS = 16  # Attempts when using multiprocessing (fewer needed due to parallelism)
PACKING_CELL_MM = 1.0         # Default grid resolution in mm (will be adaptive)
MIN_PACKING_CELL_MM = 0.15    # Ultra-fine precision for maximum space utilization
MAX_PACKING_CELL_MM = 1.0     # Tighter maximum for better accuracy
# Orientation bias weight for packing heuristics (0 = off)
PACKING_ORIENTATION_BIAS_WEIGHT = 0.7  # Strong rotation exploration for best fit
# Always show input shapes on each selected object without a separate preview toggle
SHOW_INPUT_ON_SELECTED = True
# Penalize growth of the overall occupied envelope (lower area is better)
PACKING_ENVELOPE_WEIGHT = 0.1  # Minimize penalty to maximize density
FAST_MODE = True  # If True, reduce attempts for speed; toggle with 'f'
# Slight polygon expansion when rasterizing to the grid (mm) – helps fit when
# edges are a bit under-segmented; kept small to avoid overfitting
ALLOWED_MARGIN_MM = 1.0  # Minimal margin for tightest packing

# ═══════════════════════════════════════════════════════════════════════════
# ⚡ PERFORMANCE OPTIMIZATION SETTINGS
# ═══════════════════════════════════════════════════════════════════════════
# 🔹 CPU Multi-processing (8 workers = 5-7x faster)
NUM_WORKERS = 8  # Number of parallel workers for packing (0 = auto-detect CPU count)
ENABLE_MULTIPROCESSING = True  # Enable parallel packing optimization

# 🔹 GPU Acceleration (change USE_GPU at line 48 to enable CuPy)
#    Install: pip install cupy-cuda12x  (or cupy-cuda11x for older CUDA)
#    Provides: 2-5x additional speedup for array operations on NVIDIA GPUs

# ───── Optional GPU acceleration (OpenCV CUDA) ────────────
ENABLE_CUDA = False
try:
    if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        cv2.cuda.setDevice(0)
        ENABLE_CUDA = True
        print('[INFO] CUDA detected: enabling GPU acceleration for segmentation and packing where possible')
    else:
        print('[INFO] OpenCV CUDA not available or no device detected — using CPU path')
except Exception as e:
    ENABLE_CUDA = False
    print(f'[WARN] CUDA check failed: {e}. Using CPU path.')
 
# ───── Enhanced Segmentation Parameters ─────────────────────
SEGMENTATION_CONFIG = {
    # Contour validation settings
    'use_contour_validation': False,  # Set to False to disable contour validation (FIXED: disabled by default)
    'use_bounding_box_only': False,  # Use real contour by default; toggle with 'b'
    # Edge detection parameters
    'canny_low': 30,
    'canny_high': 100,
    
    # Threshold levels for multi-scale analysis
    'thresh_low': 15,
    'thresh_med': 25,
    'thresh_high': 35,
    'weighted_thresh': 20,
    'gradient_thresh': 30,
    
    # Gaussian blur parameters for multi-scale
    'blur_fine_size': 3,
    'blur_fine_sigma': 0.5,
    'blur_med_size': 7,
    'blur_med_sigma': 1.5,
    'blur_coarse_size': 15,
    'blur_coarse_sigma': 3.0,
    
    # Adaptive threshold parameters
    'adaptive_block_size': 21,
    'adaptive_c': 10,
    
    # Scale combination weights
    'weight_fine': 0.4,
    'weight_medium': 0.4,
    'weight_coarse': 0.2,
    
    # Morphological kernel sizes
    'kernel_small': 5,
    'kernel_medium': 9,
    'kernel_large': 13,
    
    # Object connection parameters
    'connection_strength': 10,  # Size of kernel for connecting nearby objects (lower = preserve edges)
    'connection_iterations': 1,  # Number of closing iterations for connection
    
    # Contour filtering
    'min_contour_area': 100,
    'convex_hull_ratio': 0.7,
    
    # Enable/disable different features (simplified mode: background + raw edges only)
    'use_edges': True,
    'use_gradient': False,  # Disabled in simplified mode
    'use_convex_hull': False,  # Disabled in simplified mode
    
    'debug_mode': False  # Set to True to see intermediate results
}

# ───── Global variables for interaction ───────────────────
selected_objects = []  # Can select multiple objects now
all_objects = []
click_position = None
transform_matrix = None
cutting_pieces = []      # [(width, height, shape), ...] where shape is 'rect' or 'circle'
remaining_pieces = []    # Pieces remaining to be placed
distribution_stable = False  # Flag to indicate stable distribution
last_selection_count = 0     # Track selection changes
csv_shapes_placed = False    # Flag to track if CSV shapes have been placed once
# Add variables for stabilization
frame_counter = 0
obj_position_history = []
# Cache for per-object best packings to avoid recalculations each frame
_packing_cache = {}
# Additional cache tracking variables for optimization
_last_objects_signature = None
_last_pieces_signature = None
_last_frame_processed = -1
_cache_valid = False
# Object change tracking for cache invalidation
_objects_hash_cache = {}
_objects_position_cache = {}
_last_objects_count = 0
# Distribution cache for avoiding repeated calculations
_distribution_cache = {}
_last_distribution_key = None
# Suppress repeated warning messages in main loop
_logged_boundary_warnings = set()

# ───── Interactive drawing (arbitrary geometry) ───────────────────
DRAW_MODE = False                 # Toggle with 'g'
DRAW_TYPE = None                  # 'poly' | 'circle' | 'arc'
current_shape = None              # Temporary shape being drawn
user_drawn_shapes = []            # Persisted shapes (stored in rect coordinates, pixels)
mouse_pos_screen = None           # Last mouse position in screen pixels
mouse_pos_rect = None             # Last mouse position in rect pixels
inverse_transform_matrix = None   # For screen->rect mapping
last_finalized_shape = None       # Reference to last appended shape

# Shape library and placement state
LIB_FILE = 'shapes_library.json'
shapes_library = []               # Stored in mm units
PLACE_MODE = False
place_idx = 0
place_rotation_deg = 0.0
CURRENT_SCALE_MM_PER_PX = None    # Updated each frame
PLACEMENT_SCALE_MULT = 1.0        # Mouse wheel adjustable in place mode
CURRENT_TABLE_W_MM = None
CURRENT_TABLE_H_MM = None
SHOW_DIMENSIONS_ON_MAIN = False   # Hide mm labels on main window
LIB_SHAPE_MAP = {}                # {(w_key,h_key): [poly_mm,...]}
LIB_SHAPE_BY_NAME = {}            # {name: poly_data} - for looking up shapes by name
SHOW_LIBRARY_RECT = False         # Draw only original shape overlay (no bbox) when available
EXPECTED_OBJECTS_COUNT = None     # Desired number of objects to select in lib placement mode

# ───── Web logging support ─────────────────────────────────
LOG_BUFFER = deque(maxlen=2000)
_ORIG_STDOUT = sys.stdout
_TEE_INSTALLED = False

class _StdoutTee(io.TextIOBase):
    def __init__(self, inner):
        self.inner = inner
        self._buf = ''
    def write(self, s):
        try:
            self.inner.write(s)
        except Exception:
            pass
        self._buf += s
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            try:
                LOG_BUFFER.append((time.time(), line))
            except Exception:
                pass
        return len(s)
    def flush(self):
        try:
            self.inner.flush()
        except Exception:
            pass

def start_log_capture() -> None:
    global _TEE_INSTALLED, _ORIG_STDOUT
    if not _TEE_INSTALLED:
        sys.stdout = _StdoutTee(sys.stdout)
        _TEE_INSTALLED = True

def get_logs_since(ts: float) -> list[tuple[float, str]]:
    """Return log lines with timestamps > ts."""
    return [entry for entry in list(LOG_BUFFER) if entry[0] > ts]

def rotate_point_xy(x: float, y: float, angle_deg: float) -> tuple[float, float]:
    """Rotate point (x, y) by angle_deg degrees around origin."""
    rad = math.radians(angle_deg)
    c = math.cos(rad)
    s = math.sin(rad)
    return (x * c - y * s, x * s + y * c)

def calculate_optimal_rotation_angles(width_mm: float, height_mm: float, 
                                    free_rect_width: float, free_rect_height: float,
                                    shape_type: str = 'rect') -> list[float]:
    """Calculate optimal rotation angles for a piece based on available space geometry.
    
    Returns list of angles in degrees, ordered by preference (best fit first).
    Uses intelligent angle selection based on aspect ratios and space utilization.
    """
    if shape_type == 'circle':
        return [0.0]  # Circles don't need rotation
    
    angles = []
    
    # Standard orientations - most critical
    angles.append(0.0)    # Original orientation
    angles.append(90.0)   # 90-degree rotation
    
    # Calculate aspect ratios
    piece_aspect = width_mm / height_mm if height_mm > 0 else 1.0
    space_aspect = free_rect_width / free_rect_height if free_rect_height > 0 else 1.0
    
    # Only add intermediate angles if there's a significant mismatch
    # This reduces computation while maintaining flexibility
    aspect_diff = abs(piece_aspect - space_aspect)
    if aspect_diff > 0.3:  # Lowered threshold for earlier angle exploration
        # Add 45-degree angles for better space utilization
        angles.extend([45.0, 135.0])
        
        # Add fine-tuned angles only for very elongated pieces
        if piece_aspect > 2.5 or piece_aspect < 0.4:  # Very elongated pieces
            angles.extend([30.0, 60.0, 120.0, 150.0])
    
    # Calculate fit scores for each angle and sort by preference
    angle_scores = []
    for angle in angles:
        # Calculate rotated dimensions
        if angle == 0.0:
            rot_w, rot_h = width_mm, height_mm
        elif angle == 90.0:
            rot_w, rot_h = height_mm, width_mm
        else:
            # For arbitrary angles, calculate bounding box
            rad = math.radians(angle)
            cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))
            rot_w = width_mm * cos_a + height_mm * sin_a
            rot_h = width_mm * sin_a + height_mm * cos_a
        
        # Check if piece fits in available space
        if rot_w <= free_rect_width and rot_h <= free_rect_height:
            # Calculate fit quality score (lower is better)
            waste_area = (free_rect_width * free_rect_height) - (rot_w * rot_h)
            aspect_match = abs((rot_w / rot_h) - space_aspect) if rot_h > 0 else float('inf')
            
            # Prefer orientations that minimize waste and match space aspect ratio
            score = waste_area + (aspect_match * 100.0)
            
            # Bonus for standard orientations (0° and 90°)
            if angle in [0.0, 90.0]:
                score -= 50.0
            
            angle_scores.append((score, angle))
    
    # Sort by score and return angles
    angle_scores.sort(key=lambda x: x[0])
    return [angle for _, angle in angle_scores]

def calculate_rotation_efficiency(width_mm: float, height_mm: float, angle_deg: float,
                                container_width: float, container_height: float) -> float:
    """Calculate rotation efficiency score for a piece at given angle.
    
    Returns efficiency score (0.0 to 1.0, higher is better).
    Considers space utilization, boundary alignment, and geometric fit.
    """
    if angle_deg == 0.0:
        rot_w, rot_h = width_mm, height_mm
    elif angle_deg == 90.0:
        rot_w, rot_h = height_mm, width_mm
    else:
        # Calculate bounding box for arbitrary angle
        rad = math.radians(angle_deg)
        cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))
        rot_w = width_mm * cos_a + height_mm * sin_a
        rot_h = width_mm * sin_a + height_mm * cos_a
    
    # Check if piece fits
    if rot_w > container_width or rot_h > container_height:
        return 0.0
    
    # Calculate space utilization
    piece_area = width_mm * height_mm
    container_area = container_width * container_height
    utilization = piece_area / container_area if container_area > 0 else 0.0
    
    # Calculate aspect ratio match
    piece_aspect = rot_w / rot_h if rot_h > 0 else 1.0
    container_aspect = container_width / container_height if container_height > 0 else 1.0
    aspect_match = 1.0 / (1.0 + abs(piece_aspect - container_aspect))
    
    # Calculate boundary alignment bonus
    boundary_bonus = 0.0
    tolerance = 1.0  # mm
    if abs(rot_w - container_width) < tolerance:
        boundary_bonus += 0.2
    if abs(rot_h - container_height) < tolerance:
        boundary_bonus += 0.2
    
    # Combine factors
    efficiency = (utilization * 0.5) + (aspect_match * 0.3) + (boundary_bonus * 0.2)
    
    # Bonus for standard angles
    if angle_deg in [0.0, 90.0]:
        efficiency += 0.1
    
    return min(1.0, efficiency)

def load_shapes_library() -> None:
    global shapes_library, LIB_SHAPE_BY_NAME, LIB_SHAPE_MAP
    try:
        import json, os
        if os.path.exists(LIB_FILE):
            with open(LIB_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    shapes_library = data
                    # Pre-populate LIB_SHAPE_BY_NAME and LIB_SHAPE_MAP for all library shapes
                    loaded_count = 0
                    for lib_shape in shapes_library:
                        try:
                            name = lib_shape.get('name', '')
                            pts_mm = lib_shape.get('pts_mm', [])
                            if name and pts_mm:
                                bbox = lib_shape.get('meta', {}).get('bbox_mm', {})
                                width = bbox.get('w', 0)
                                height = bbox.get('h', 0)
                                poly_data = {
                                    'name': name,
                                    'pts': pts_mm,
                                    'closed': bool(lib_shape.get('closed', True)),
                                    'segments': lib_shape.get('segments'),
                                    'bbox_w': width,
                                    'bbox_h': height
                                }
                                LIB_SHAPE_BY_NAME[name] = poly_data
                                if width > 0 and height > 0:
                                    key = (round(float(width), 2), round(float(height), 2))
                                    LIB_SHAPE_MAP.setdefault(key, []).append(poly_data)
                                    loaded_count += 1
                        except Exception as e:
                            pass  # Skip shapes that fail to process
                    print(f"[INFO] Shapes library loaded: {loaded_count} shapes with valid bbox")
                    print(f"[INFO] LIB_SHAPE_BY_NAME keys: {list(LIB_SHAPE_BY_NAME.keys())}")
                else:
                    shapes_library = []
        else:
            shapes_library = []
    except Exception as e:
        print(f"[WARN] Failed to load shapes library: {e}")
        shapes_library = []

def save_shapes_library() -> None:
    try:
        import json
        with open(LIB_FILE, 'w', encoding='utf-8') as f:
            json.dump(shapes_library, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Shapes library saved to {LIB_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to save shapes library: {e}")

def convert_drawn_shape_to_mm(shape: dict, scale_mm_per_px: float, name: str) -> dict | None:
    """Convert a drawn shape in rect pixels to a library entry in mm (anchored at 0,0).

    Also stores 'meta' with scale and side-lengths to preserve intended dimensions.
    """
    try:
        if shape['type'] == 'poly':
            pts_px = shape.get('pts', [])
            if len(pts_px) < 2:
                return None
            pts_mm = [(float(x) * scale_mm_per_px, float(y) * scale_mm_per_px) for (x, y) in pts_px]
            # Anchor to (0,0)
            minx = min(p[0] for p in pts_mm)
            miny = min(p[1] for p in pts_mm)
            pts_mm = [(p[0] - minx, p[1] - miny) for p in pts_mm]
            is_closed = bool(shape.get('closed', True))
            # Edge lengths snapshot (mm) and explicit line segments
            edge_lengths = []
            segments = []
            # BBox in mm (after anchoring)
            try:
                xs = [p[0] for p in pts_mm]
                ys = [p[1] for p in pts_mm]
                bbox_w_mm = float(max(xs) - min(xs)) if xs else 0.0
                bbox_h_mm = float(max(ys) - min(ys)) if ys else 0.0
            except Exception:
                bbox_w_mm = 0.0
                bbox_h_mm = 0.0
            if len(pts_mm) >= 2:
                count = len(pts_mm) if is_closed else len(pts_mm) - 1
                for i in range(max(0, count)):
                    ax, ay = pts_mm[i]
                    bx, by = pts_mm[(i+1) % len(pts_mm)]
                    length_mm = float(((bx-ax)**2 + (by-ay)**2) ** 0.5)
                    edge_lengths.append(length_mm)
                    # Connections only between adjacent segments; do not auto-connect others
                    # Segment ids are 1-based for readability
                    seg_id = i + 1
                    connects_to = []
                    if count > 1:
                        # previous segment id (if exists in open poly or for closed)
                        if is_closed or i > 0:
                            connects_to.append(((i - 1) % count) + 1)
                        # next segment id (if exists)
                        if is_closed or i < count - 1:
                            connects_to.append(((i + 1) % count) + 1)
                    segments.append({
                        'id': seg_id,
                        'from_mm': [float(ax), float(ay)],
                        'to_mm': [float(bx), float(by)],
                        'length_cm': float(length_mm) / 10.0,
                        'connects_to': connects_to,
                    })
            return {
                'name': name,
                'type': 'poly',
                'pts_mm': pts_mm,
                'closed': is_closed,
                'segments': segments,
                'meta': {
                    'scale_mm_per_px': float(scale_mm_per_px),
                    'edge_lengths_mm': edge_lengths,
                    'edge_lengths_cm': [float(x)/10.0 for x in edge_lengths],
                    'bbox_mm': {'w': bbox_w_mm, 'h': bbox_h_mm},
                    'bbox_cm': {'w': bbox_w_mm/10.0, 'h': bbox_h_mm/10.0},
                    'units': 'mm',
                }
            }
        if shape['type'] == 'circle':
            r_px = float(shape.get('radius_px', 0.0))
            diameter_mm = 2.0 * r_px * scale_mm_per_px
            return {
                'name': name,
                'type': 'circle',
                'diameter_mm': diameter_mm,
                'diameter_cm': float(diameter_mm) / 10.0,
                'meta': {
                    'scale_mm_per_px': float(scale_mm_per_px),
                    'units': 'mm'
                }
            }
        if shape['type'] == 'arc':
            r_px = float(shape.get('radius_px', 0.0))
            radius_mm = r_px * scale_mm_per_px
            a0 = float(shape.get('start_deg', 0.0))
            a1 = float(shape.get('end_deg', 0.0))
            ccw = bool(shape.get('ccw', True))
            # Store sweep angle (absolute) and orientation ccw; center becomes anchor
            if ccw:
                if a1 < a0:
                    a1 += 360.0
                ang = a1 - a0
            else:
                if a1 > a0:
                    a1 -= 360.0
                ang = a0 - a1
            arc_len_mm = float(math.radians(abs(ang)) * radius_mm)
            return {
                'name': name,
                'type': 'arc',
                'radius_mm': radius_mm,
                'angle_deg': float(abs(ang)),
                'ccw': ccw,
                'radius_cm': float(radius_mm) / 10.0,
                'arc_length_cm': float(arc_len_mm) / 10.0,
                'meta': {
                    'scale_mm_per_px': float(scale_mm_per_px),
                    'units': 'mm'
                }
            }
    except Exception as e:
        print(f"[ERROR] Failed to convert shape to mm: {e}")
    return None

# ───── Persistent calibration (marker size) ─────────────────────────
CALIB_FILE = 'calibration.json'

def load_calibration() -> None:
    global ARUCO_SIZE_MM
    try:
        import json, os
        if os.path.exists(CALIB_FILE):
            with open(CALIB_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'aruco_size_mm' in data:
                    ARUCO_SIZE_MM = float(data['aruco_size_mm'])
                    print(f"[INFO] Loaded marker size: {ARUCO_SIZE_MM:.1f} mm")
    except Exception as e:
        print(f"[WARN] Failed to load calibration: {e}")

def save_calibration() -> None:
    try:
        import json
        with open(CALIB_FILE, 'w', encoding='utf-8') as f:
            json.dump({'aruco_size_mm': float(ARUCO_SIZE_MM)}, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved marker size: {ARUCO_SIZE_MM:.1f} mm")
    except Exception as e:
        print(f"[WARN] Failed to save calibration: {e}")

# ───── Load shapes from CSV file ─────────────────────────────
def load_shapes_from_csv(csv_file_path: str, load_only_pending: bool = True) -> list[tuple[float, float, str, int]]:
    """Load cutting shapes from CSV file.
    
    Expected CSV format:
    width,height,shape_type,status,shape_name,library_status
    or
    width,height,shape_type,status
    or
    width,height,shape_type
    or
    width,height
    
    where shape_type is 'rect', 'circle', or 'poly' (defaults to 'rect')
    status is 'pending' or 'completed' (defaults to 'pending')
    shape_name is the name of shape from shapes_library (optional)
    library_status is 'available' or 'unavailable' (defaults to 'available')
    
    Returns list of (width, height, shape_type, row_index) tuples.
    """
    try:
        import csv
        shapes = []
        # Use utf-8-sig to automatically handle BOM
        with open(csv_file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row_num, row in enumerate(reader, 1):
                if not row or len(row) < 2:
                    continue
                try:
                    # Skip header row
                    first_cell = row[0].strip().lower()
                    if first_cell in ['width', 'w', 'ширина']:
                        continue
                    
                    width = float(row[0].strip())
                    height = float(row[1].strip())
                    shape_type = row[2].strip().lower() if len(row) > 2 else 'rect'
                    status = row[3].strip().lower() if len(row) > 3 else 'pending'
                    shape_name = row[4].strip() if len(row) > 4 and row[4].strip() else ''
                    library_status = row[5].strip().lower() if len(row) > 5 and row[5].strip() else 'available'
                    
                    # Support 'library' as alias for 'poly' (shapes from library)
                    if shape_type == 'library':
                        shape_type = 'poly'
                    if shape_type not in ['rect', 'circle', 'poly']:
                        print(f"[WARN] Row {row_num}: Invalid shape type '{shape_type}', using 'rect'")
                        shape_type = 'rect'
                    
                    if status not in ['pending', 'completed']:
                        print(f"[WARN] Row {row_num}: Invalid status '{status}', using 'pending'")
                        status = 'pending'
                    
                    if library_status not in ['available', 'unavailable']:
                        print(f"[WARN] Row {row_num}: Invalid library_status '{library_status}', using 'available'")
                        library_status = 'available'
                    
                    # Handle shapes from library
                    if shape_type == 'poly' and shape_name:
                        # Load shapes library if not already loaded
                        if not shapes_library:
                            load_shapes_library()
                        
                        # Find shape in library
                        library_shape = None
                        for lib_shape in shapes_library:
                            if lib_shape.get('name') == shape_name:
                                library_shape = lib_shape
                                break
                        
                        if library_shape:
                            # Get bounding box dimensions from library shape
                            bbox = library_shape.get('meta', {}).get('bbox_mm', {})
                            if bbox:
                                width = bbox.get('w', 0)
                                height = bbox.get('h', 0)
                                print(f"[INFO] Loaded from library: {shape_name} ({width:.1f}x{height:.1f} mm) - {status} - {library_status}")
                                
                                # Add polygon data to LIB_SHAPE_MAP for proper rendering
                                try:
                                    pts_mm = library_shape.get('pts_mm', [])
                                    if pts_mm:
                                        key = (round(float(width), 2), round(float(height), 2))
                                        poly_data = {
                                            'name': shape_name,
                                            'pts': pts_mm,
                                            'closed': bool(library_shape.get('closed', True)),
                                            'segments': library_shape.get('segments'),
                                            'bbox_w': width,
                                            'bbox_h': height
                                        }
                                        LIB_SHAPE_MAP.setdefault(key, []).append(poly_data)
                                        # Also store by name for reliable lookup
                                        LIB_SHAPE_BY_NAME[shape_name] = poly_data
                                except Exception as e:
                                    print(f"[WARN] Failed to add polygon data to LIB_SHAPE_MAP: {e}")
                            else:
                                print(f"[WARN] Row {row_num}: Shape '{shape_name}' found in library but no bbox info, skipping")
                                continue
                        else:
                            print(f"[WARN] Row {row_num}: Shape '{shape_name}' not found in library, skipping")
                            continue
                    else:
                        # Regular rect/circle shapes
                        if width <= 0 or height <= 0:
                            print(f"[WARN] Row {row_num}: Invalid dimensions {width}x{height}, skipping")
                            continue
                        print(f"[INFO] Loaded: {width}x{height} mm ({shape_type}) - {status}")
                    
                    # Only load pending shapes if requested
                    if load_only_pending and status != 'pending':
                        continue
                    
                    # Only load available shapes from library
                    if shape_name and library_status == 'unavailable':
                        print(f"[INFO] Skipping unavailable library shape: {shape_name}")
                        continue
                    
                    # Include shape_name as 5th element for poly shapes
                    shapes.append((width, height, shape_type, row_num, shape_name if shape_name else ''))
                    
                except ValueError as e:
                    print(f"[WARN] Row {row_num}: Invalid number format: {e}")
                    continue
                except Exception as e:
                    print(f"[WARN] Row {row_num}: Error: {e}")
                    continue
        
        print(f"[INFO] Successfully loaded {len(shapes)} shapes from CSV (including {sum(1 for s in shapes if s[2] == 'poly')} from shapes library)")
        return shapes
        
    except FileNotFoundError:
        print(f"[ERROR] CSV file not found: {csv_file_path}")
        return []
    except Exception as e:
        print(f"[ERROR] Failed to load CSV file: {e}")
        return []

def mark_shapes_as_completed(csv_file_path: str, completed_ids: list[int]) -> bool:
    """Mark the specified shapes (by row index) as completed in CSV file."""
    try:
        import csv
        import tempfile
        import os
        
        # Read all rows
        rows = []
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        if not rows:
            print("[ERROR] CSV file is empty")
            return False
        
        # Find and mark pending shapes as completed
        marked_count = 0
        for i, row in enumerate(rows):
            # Row indices are 1-based in load_shapes_from_csv (enumerate(reader, 1))
            # But here rows includes header, so rows[0] is header.
            # If load_shapes_from_csv used enumerate(reader, 1), then row 1 is the first data row.
            # In rows list, rows[0] is header, rows[1] is first data row.
            # So row_num from load_shapes_from_csv corresponds to index in rows list?
            # Let's check load_shapes_from_csv: enumerate(reader, 1).
            # If header is present, reader yields header first? No, csv.reader yields all rows.
            # Wait, load_shapes_from_csv does:
            # reader = csv.reader(f)
            # for row_num, row in enumerate(reader, 1):
            # It doesn't skip header explicitly, but it checks `if not row or len(row) < 2: continue`.
            # And `try: width = float(row[0])`. The header will fail this try/except and be skipped.
            # So row_num 1 is likely the header row if it exists.
            # If row_num corresponds to the line number (1-based), then rows[row_num-1] is the row.
            
            if (i + 1) in completed_ids:
                if len(row) >= 4:
                    row[3] = 'completed'
                    marked_count += 1
                    print(f"[INFO] Marked shape at row {i+1} as completed")
                elif len(row) >= 2: # If status column missing, append it
                    while len(row) < 3: row.append('rect')
                    while len(row) < 4: row.append('pending')
                    row[3] = 'completed'
                    marked_count += 1
                    print(f"[INFO] Marked shape at row {i+1} as completed")

        # Write back to file
        with open(csv_file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        print(f"[INFO] Successfully marked {marked_count} shapes as completed")
        return marked_count > 0
        
    except Exception as e:
        print(f"[ERROR] Failed to update CSV file: {e}")
        return False

def prompt_csv_file_path() -> str:
    """Prompt user for CSV file path"""
    print("\n=== Load shapes from CSV ===")
    print("Expected CSV format:")
    print("width,height,shape_type")
    print("100,50,rect")
    print("75,75,circle")
    print("(shape_type defaults to 'rect' if not specified)")
    
    while True:
        try:
            file_path = input("Enter CSV file path (or press Enter for 'example_cuts.csv'): ").strip()
            if not file_path:
                file_path = 'example_cuts.csv'
            
            # Check if file exists
            import os
            if not os.path.exists(file_path):
                print(f"[ERROR] File not found: {file_path}")
                continue
                
            return file_path
            
        except KeyboardInterrupt:
            return ""
        except Exception as e:
            print(f"[ERROR] Invalid input: {e}")

# ───── CSV-based cutting mode ───────────────────────────────
def run_csv_cutting_mode():
    """Run the CSV cutting mode where shapes are loaded from CSV and placed on selected objects"""
    global cutting_pieces, remaining_pieces, selected_objects, all_objects
    global STARTUP_MODE, PLACE_MODE, DRAW_MODE
    
    print("\n=== CSV Cutting Mode ===")
    print("1. Load shapes from CSV file")
    print("2. Select segmented object(s)")
    print("3. Automatic placement with efficient packing")
    print("4. Use 'N' key to show next batch from remaining pieces")
    
    # Load shapes from CSV
    csv_file = prompt_csv_file_path()
    if not csv_file:
        print("[INFO] CSV loading cancelled")
        return
    
    shapes = load_shapes_from_csv(csv_file)
    if not shapes:
        print("[ERROR] No valid shapes loaded from CSV")
        return
    
    # Set up for CSV cutting mode
    cutting_pieces = shapes.copy()
    remaining_pieces = shapes.copy()
    STARTUP_MODE = 'csv_cut'
    PLACE_MODE = False
    DRAW_MODE = False
    
    # Store CSV file path for later use in show_next_csv_batch
    show_next_csv_batch.csv_file_path = csv_file
    
    print(f"[INFO] Loaded {len(shapes)} shapes for cutting")
    print("[INFO] Start camera and select segmented object(s)")
    print("[INFO] Press 'N' to show next batch of pieces")

def distribute_csv_shapes_on_selected_object(scale: float) -> dict:
    """Distribute CSV shapes on the selected segmented object with enhanced boundary validation"""
    global remaining_pieces, selected_objects, all_objects
    
    if not selected_objects or not all_objects:
        return {}
    
    if not remaining_pieces:
        print("[INFO] All CSV pieces have been placed!")
        return {}
    
    print(f"[INFO] Distributing {len(remaining_pieces)} CSV pieces on selected objects")
    
    # Get the first selected object for this batch
    obj_idx = selected_objects[0]  # Use first selected object
    if obj_idx <= 0 or obj_idx > len(all_objects):
        print("[ERROR] Invalid selected object index")
        return {}
    
    obj = all_objects[obj_idx - 1]
    sheet_width_mm, sheet_height_mm = obj['w_mm'], obj['h_mm']
    
    # Enhanced pre-filtering: check both size and area constraints
    fittable_pieces, too_big = split_cuts_to_fit(sheet_width_mm, sheet_height_mm, remaining_pieces)
    
    # Additional area-based filtering for better placement efficiency
    object_area_mm2 = obj.get('real_area_mm2', sheet_width_mm * sheet_height_mm)
    area_filtered_pieces = []
    
    for piece in fittable_pieces:
        w, h, shape = piece[:3]
        piece_area = (math.pi * (w/2.0) * (h/2.0)) if shape == 'circle' else (w * h)
        
        # Skip pieces that are too large for the available area
        if piece_area > object_area_mm2 * 0.8:  # Use 80% of area as threshold
            print(f"[WARN] Piece {w}x{h} mm {shape} area ({piece_area:.1f} mm²) too large for object area ({object_area_mm2:.1f} mm²)")
            continue
        
        area_filtered_pieces.append(piece)
    
    if not area_filtered_pieces:
        print(f"[WARN] No pieces fit on object #{obj_idx} ({sheet_width_mm:.0f}x{sheet_height_mm:.0f} mm) after area filtering")
        return {}
    
    print(f"[INFO] {len(area_filtered_pieces)} pieces can fit on object #{obj_idx} (after area filtering)")
    
    # Use real contour if available
    allowed_polygon_mm = obj.get('allowed_polygon_mm')
    
    # Create packer based on object type
    if allowed_polygon_mm is not None and len(allowed_polygon_mm) >= 3:
        # Calculate adaptive cell size for better packing efficiency
        adaptive_cell_mm = calculate_adaptive_cell_size(area_filtered_pieces, sheet_width_mm, sheet_height_mm)
        packer = PolygonGridPacker(sheet_width_mm, sheet_height_mm, allowed_polygon_mm, cell_mm=adaptive_cell_mm)
    else:
        packer = MaxRects(sheet_width_mm, sheet_height_mm, allowed_polygon_mm)
    
    # Enhanced sorting: prioritize pieces by multiple criteria for better packing
    def enhanced_piece_key(p):
        w, h, shape = p[:3]
        area = (math.pi * (w/2.0) * (h/2.0)) if shape == 'circle' else (w * h)
        # Prioritize: 1) circles first (harder to place), 2) larger area, 3) aspect ratio closer to square
        aspect_penalty = abs(1.0 - (max(w, h) / min(w, h))) if shape == 'rect' else 0
        circle_priority = 0 if shape == 'circle' else 1
        return (circle_priority, -area, aspect_penalty)
    
    sorted_pieces = sorted(area_filtered_pieces, key=enhanced_piece_key)
    
    # Enhanced placement with comprehensive validation and next-batch management
    placed_pieces = []
    failed_pieces = []  # Track pieces that couldn't be placed for next batch
    
    for piece in sorted_pieces:
        w, h, shape = piece[:3]
        piece_id = piece[3] if len(piece) > 3 else -1
        shape_name = piece[4] if len(piece) > 4 else ''
        
        # Enhanced pre-validation for circles
        if shape == 'circle':
            radius = w / 2.0
            
            # Strict boundary pre-check for circles
            if (2 * radius > min(sheet_width_mm, sheet_height_mm)):
                print(f"[WARN] Circle diameter {w} mm too large for object dimensions, adding to next batch")
                failed_pieces.append(piece)
                continue
            
            # Enhanced contour compatibility check for circles
            if allowed_polygon_mm is not None and len(allowed_polygon_mm) >= 3:
                # More comprehensive position testing for circles
                test_positions = [
                    (sheet_width_mm / 2 - radius, sheet_height_mm / 2 - radius),  # center
                    (radius + 2, radius + 2),  # top-left with margin
                    (sheet_width_mm - radius - 2, radius + 2),  # top-right
                    (radius + 2, sheet_height_mm - radius - 2),  # bottom-left
                    (sheet_width_mm - radius - 2, sheet_height_mm - radius - 2),  # bottom-right
                    (sheet_width_mm / 4, sheet_height_mm / 4),  # quarter positions
                    (3 * sheet_width_mm / 4, sheet_height_mm / 4),
                    (sheet_width_mm / 4, 3 * sheet_height_mm / 4),
                    (3 * sheet_width_mm / 4, 3 * sheet_height_mm / 4)
                ]
                
                can_fit_anywhere = False
                for test_x, test_y in test_positions:
                    if (test_x >= radius and test_y >= radius and 
                        test_x + radius <= sheet_width_mm and test_y + radius <= sheet_height_mm):
                        if piece_fits_in_local_contour_mm(test_x - radius, test_y - radius, w, h, shape, allowed_polygon_mm):
                            can_fit_anywhere = True
                            break
                
                if not can_fit_anywhere:
                    print(f"[WARN] Circle {w} mm cannot fit within object contour, adding to next batch")
                    failed_pieces.append(piece)
                    continue
        
        # Try to place the piece with enhanced validation
        print(f"[DEBUG] Attempting to place {shape} {w}x{h} mm on object #{obj_idx}")
        
        placement_successful = False
        if packer.insert(w, h, shape, piece_id=piece_id, shape_name=shape_name):
            # Validate the placement immediately
            last_placed = packer.used[-1]
            placed_x, placed_y, placed_w, placed_h = last_placed[0], last_placed[1], last_placed[2], last_placed[3]
            
            # Enhanced boundary validation with small tolerance
            boundary_valid = True
            margin_mm = 0.5  # Tolerance for practical placement with measurement precision
            
            # Boundary check with small tolerance
            if (placed_x < -margin_mm or placed_y < -margin_mm or 
                placed_x + placed_w > sheet_width_mm + margin_mm or 
                placed_y + placed_h > sheet_height_mm + margin_mm):
                print(f"[WARN] Piece {w}x{h} mm {shape} exceeds boundaries")
                boundary_valid = False
            
            # Enhanced circle boundary validation
            if boundary_valid and shape == 'circle':
                circle_center_x = placed_x + placed_w / 2.0
                circle_center_y = placed_y + placed_h / 2.0
                radius = placed_w / 2.0
                
                if (circle_center_x - radius < -margin_mm or circle_center_y - radius < -margin_mm or
                    circle_center_x + radius > sheet_width_mm + margin_mm or circle_center_y + radius > sheet_height_mm + margin_mm):
                    print(f"[WARN] Circle {w} mm exceeds boundaries")
                    boundary_valid = False
            
            # Enhanced contour validation
            contour_valid = True
            if boundary_valid and allowed_polygon_mm is not None and len(allowed_polygon_mm) >= 3:
                if not piece_fits_in_local_contour_mm(placed_x, placed_y, placed_w, placed_h, shape, allowed_polygon_mm):
                    print(f"[WARN] Piece {w}x{h} mm {shape} violates object contour")
                    contour_valid = False
            
            # Final validation decision
            if boundary_valid and contour_valid:
                placed_pieces.append(piece)
                placement_successful = True
                print(f"[INFO] Successfully placed {w}x{h} mm {shape} at ({placed_x:.1f}, {placed_y:.1f})")
            else:
                # Remove invalid placement and add to failed pieces
                packer.used.pop()
                failed_pieces.append(piece)
                print(f"[WARN] Placement validation failed for {w}x{h} mm {shape}, adding to next batch")
        else:
            # Packer couldn't find space
            failed_pieces.append(piece)
            print(f"[DEBUG] No space found for {w}x{h} mm {shape}, adding to next batch")
    
    # Enhanced next-batch management: ensure proper transfer of failed pieces
    next_batch_pieces = []
    
    # Add failed pieces to next batch with detailed logging
    for failed_piece in failed_pieces:
        if failed_piece not in next_batch_pieces:
            next_batch_pieces.append(failed_piece)
            print(f"[INFO] Transferring {failed_piece[2]} {failed_piece[0]}x{failed_piece[1]} mm to next batch (placement failed)")
    
    # Add pieces that were too big for this object to next batch
    for big_piece in too_big:
        if big_piece not in next_batch_pieces:
            next_batch_pieces.append(big_piece)
            print(f"[INFO] Transferring {big_piece[2]} {big_piece[0]}x{big_piece[1]} mm to next batch (too large for current object)")
    
    # Update object with validated packing result
    obj['packing_result'] = packer.used.copy()
    
    # Remove successfully placed pieces from remaining_pieces
    for placed in placed_pieces:
        if placed in remaining_pieces:
            remaining_pieces.remove(placed)
            print(f"[DEBUG] Removed placed piece {placed[2]} {placed[0]}x{placed[1]} mm from remaining queue")
    
    # Update remaining_pieces with next batch pieces (avoid duplicates)
    for next_piece in next_batch_pieces:
        if next_piece not in remaining_pieces:
            remaining_pieces.append(next_piece)
    
    # Sort remaining pieces for better next batch processing
    remaining_pieces.sort(key=lambda p: (p[2] == 'rect', -(p[0] * p[1])))
    
    # Calculate pieces that couldn't be placed
    total_attempted = len(fittable_pieces)
    successfully_placed = len(placed_pieces)
    failed_to_place = total_attempted - successfully_placed
    
    print(f"[INFO] Successfully placed {successfully_placed}/{total_attempted} pieces on object #{obj_idx}")
    print(f"[INFO] Remaining pieces for next batch: {len(remaining_pieces)}")
    
    if failed_to_place > 0:
        print(f"[INFO] {failed_to_place} pieces will be automatically transferred to next batch for processing")
    
    # Final validation: ensure all placed pieces are within boundaries
    valid_placements = []
    for u in packer.used:
        # Unpack with piece_id support (7 elements)
        ux, uy, uw, uh, urot, ushape = u[:6]
        u_piece_id = u[6] if len(u) > 6 else -1
        # Double-check boundary constraints with strict checking
        margin_mm = 0.0  # Strict checking - no margin allowed
        if (ux >= margin_mm and uy >= margin_mm and 
            ux + uw <= sheet_width_mm - margin_mm and uy + uh <= sheet_height_mm - margin_mm):
            # Check contour fit if available
            if (allowed_polygon_mm is None or len(allowed_polygon_mm) < 3 or 
                piece_fits_in_local_contour_mm(ux, uy, uw, uh, ushape, allowed_polygon_mm)):
                # Preserve piece_id in valid placements
                valid_placements.append([ux, uy, uw, uh, urot, ushape, u_piece_id])
            else:
                print(f"[WARN] Final validation failed for {ushape} at ({ux:.1f}, {uy:.1f}) - removing from placement")
        else:
            print(f"[WARN] Final boundary check failed for {ushape} at ({ux:.1f}, {uy:.1f}) - removing from placement")
    
    # Update with only valid placements
    obj['packing_result'] = valid_placements
    print(f"[INFO] Final validated placements: {len(valid_placements)} pieces on object #{obj_idx}")
    
    # Return success indicator
    return {'success': len(valid_placements) > 0, 'placed_count': len(valid_placements)}

def show_next_csv_batch():
    """Mark current shapes as completed and load next batch of CSV pieces"""
    global remaining_pieces, selected_objects, all_objects, cutting_pieces, csv_shapes_placed, distribution_stable
    global _packing_cache, _distribution_cache, _last_distribution_key
    
    if not selected_objects:
        print("[INFO] No objects selected")
        return
    
    # Collect IDs of placed shapes from all selected objects
    placed_ids = []
    for obj_idx in selected_objects:
        if obj_idx > 0 and obj_idx <= len(all_objects):
            obj = all_objects[obj_idx - 1]
            if 'packing_result' in obj:
                print(f"[DEBUG] Object #{obj_idx} has {len(obj['packing_result'])} placements")
                for placement in obj['packing_result']:
                    print(f"[DEBUG] Placement: {placement}")
                    # Check if placement has ID (7 elements)
                    if len(placement) >= 7:
                        piece_id = placement[6]
                        if piece_id != -1:
                            placed_ids.append(piece_id)
                            print(f"[DEBUG] Collected piece ID: {piece_id}")
    
    print(f"[DEBUG] Total placed IDs collected: {placed_ids}")
    
    if placed_ids:
        # Mark placed shapes as completed in CSV file
        csv_file_path = getattr(show_next_csv_batch, 'csv_file_path', 'example_cuts.csv')
        success = mark_shapes_as_completed(csv_file_path, placed_ids)
        
        if success:
            print(f"[INFO] Marked {len(placed_ids)} shapes as completed in CSV")
        else:
            print(f"[WARN] Failed to mark shapes as completed in CSV")
    else:
        print("[WARN] No piece IDs were collected from placements - shapes may not be marked as completed")
    
    # Clear ALL caches to force fresh computation
    _packing_cache.clear()
    _distribution_cache.clear()
    _last_distribution_key = None
    print("[DEBUG] Cleared all caches for fresh placement")
    
    # Clear current placements from all selected objects
    for obj_idx in selected_objects:
        if obj_idx > 0 and obj_idx <= len(all_objects):
            obj = all_objects[obj_idx - 1]
            obj['packing_result'] = []  # Clear current placements
    
    # Check if there are remaining pieces from current distribution
    if remaining_pieces:
        print(f"[INFO] Using {len(remaining_pieces)} remaining pieces from current batch (not yet placed)")
        # Keep remaining_pieces as is - these are pieces that didn't fit
        csv_shapes_placed = False  # Reset flag to allow new placement
        distribution_stable = False  # Force redistribution
        
        # Display information about remaining shapes
        rect_count = sum(1 for s in remaining_pieces if len(s) >= 3 and s[2] == 'rect')
        circle_count = sum(1 for s in remaining_pieces if len(s) >= 3 and s[2] == 'circle')
        poly_count = sum(1 for s in remaining_pieces if len(s) >= 3 and s[2] == 'poly')
        
        print(f"[INFO] Remaining shape types: {rect_count} rectangles, {circle_count} circles, {poly_count} library shapes")
        print("[INFO] These shapes will be redistributed in next frame")
        return  # Don't load new shapes from CSV yet
    
    # Only load new shapes from CSV if all current pieces are placed
    csv_file_path = getattr(show_next_csv_batch, 'csv_file_path', 'example_cuts.csv')
    new_shapes = load_shapes_from_csv(csv_file_path, load_only_pending=True)
    
    if new_shapes:
        # Update global variables with new batch
        cutting_pieces = new_shapes.copy()
        remaining_pieces = new_shapes.copy()
        csv_shapes_placed = False  # Reset flag to allow new placement
        distribution_stable = False  # Force redistribution
        
        # Display detailed information about loaded shapes
        print(f"\n=== Loaded next batch: {len(new_shapes)} pending shapes ===")
        
        # Count different types of shapes
        rect_count = sum(1 for s in new_shapes if s[2] == 'rect')
        circle_count = sum(1 for s in new_shapes if s[2] == 'circle')
        poly_count = sum(1 for s in new_shapes if s[2] == 'poly')
        
        print(f"[INFO] Shape types: {rect_count} rectangles, {circle_count} circles, {poly_count} library shapes")
        
        # Display library shapes with their names and status
        if poly_count > 0:
            print("\n=== Library Shapes in Current Batch ===")
            # Re-read CSV to get shape names and library status for display
            try:
                import csv
                with open(csv_file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)  # Skip header
                    
                    for row_num, row in enumerate(reader, 2):
                        if len(row) >= 6:
                            status = row[3].strip().lower() if len(row) > 3 else 'pending'
                            shape_name = row[4].strip() if len(row) > 4 and row[4].strip() else ''
                            library_status = row[5].strip().lower() if len(row) > 5 and row[5].strip() else 'available'
                            shape_type = row[2].strip().lower() if len(row) > 2 else 'rect'
                            
                            if (status == 'pending' and shape_type == 'poly' and 
                                shape_name and library_status == 'available'):
                                width = float(row[0].strip()) if row[0].strip() != '0' else 'auto'
                                height = float(row[1].strip()) if row[1].strip() != '0' else 'auto'
                                print(f"  • {shape_name}: {width}x{height} mm (status: {library_status})")
            except Exception as e:
                print(f"[WARN] Could not display library shape details: {e}")
        
        print("\n[INFO] Shapes will be automatically placed on selected objects")
        print("[INFO] Distribution will start in next frame")
    else:
        print("[INFO] No more pending shapes to load from CSV")
        cutting_pieces = []
        remaining_pieces = []

def cleanup_invalid_placements():
    """Force cleanup of any invalid placements that exceed boundaries"""
    global selected_objects, all_objects
    
    if not selected_objects:
        return
    
    print("[INFO] Performing cleanup of invalid placements...")
    
    for obj_idx in selected_objects:
        if obj_idx <= 0 or obj_idx > len(all_objects):
            continue
        
        obj = all_objects[obj_idx - 1]
        packing_result = obj.get('packing_result', [])
        
        if not packing_result:
            continue
        
        # Check each placed piece
        valid_placements = []
        removed_count = 0
        
        for i, placement in enumerate(packing_result):
            # Unpack with piece_id support (7 elements)
            x, y, w, h, rot, shape = placement[:6]
            piece_id = placement[6] if len(placement) > 6 else -1
            is_valid = True
            
            # Check if piece exceeds object boundaries
            safety_margin = 0.0  # No margin - strict boundary checking for cleanup
            
            if shape == 'circle':
                circle_center_x = x + w / 2.0
                circle_center_y = y + h / 2.0
                radius = w / 2.0
                
                # Check if circle exceeds object boundaries with strict checking
                if (circle_center_x - radius < safety_margin or circle_center_y - radius < safety_margin or
                    circle_center_x + radius > obj['w_mm'] - safety_margin or circle_center_y + radius > obj['h_mm'] - safety_margin):
                    is_valid = False
                    removed_count += 1
                
                # Additional check: verify circle fits within real contour (less strict)
                allowed_polygon_mm = obj.get('allowed_polygon_mm')
                if allowed_polygon_mm is not None and len(allowed_polygon_mm) >= 3:
                    if not piece_fits_in_local_contour_mm(x, y, w, h, shape, allowed_polygon_mm):
                        pass  # Silent - contour check might be too strict
            else:
                # Check rectangles with strict checking
                if (x < safety_margin or y < safety_margin or
                    x + w > obj['w_mm'] - safety_margin or y + h > obj['h_mm'] - safety_margin):
                    is_valid = False
                    removed_count += 1
                
                # Additional check: verify rectangle fits within real contour (less strict)
                allowed_polygon_mm = obj.get('allowed_polygon_mm')
                if allowed_polygon_mm is not None and len(allowed_polygon_mm) >= 3:
                    if not piece_fits_in_local_contour_mm(x, y, w, h, shape, allowed_polygon_mm):
                        pass  # Silent - contour check might be too strict
            
            if is_valid:
                # Preserve piece_id when keeping placement
                if piece_id != -1:
                    valid_placements.append([x, y, w, h, rot, shape, piece_id])
                else:
                    valid_placements.append([x, y, w, h, rot, shape])
        
        # Update object with cleaned results
        obj['packing_result'] = valid_placements
        
        if removed_count > 0:
            print(f"[CLEANUP] Removed {removed_count} invalid placements from object #{obj_idx}")
            print(f"[CLEANUP] Valid placements remaining: {len(valid_placements)}")
        else:
            print(f"[CLEANUP] All placements in object #{obj_idx} are valid")

# ───── Startup menu ─────────────────────────────────────────
STARTUP_MODE = 'live_draw'      # 'rc_place', 'live_draw', 'lib_place'
STARTUP_LOAD_INDEX = None

def prompt_startup_mode():
    global STARTUP_MODE, STARTUP_LOAD_INDEX, ARUCO_SIZE_MM
    print("\n=== Startup modes ===")
    print("1 — Rect/Circle placement (input sizes, show on selected objects)")
    print("2 — Live Draw on segmented object (save from Measurements)")
    print("3 — Load from the library and place (choose objects first)")
    print("4 — Load shapes from CSV and auto-place on selected objects")
    try:
        choice = input("Choose mode (1/2/3/4, Enter=1): ").strip()
        # Use saved calibration if present; otherwise keep default (120 mm)
        try:
            load_calibration()
        except Exception:
            pass
        if choice == '1':
            STARTUP_MODE = 'rc_place'
        elif choice == '2' or choice == '':
            STARTUP_MODE = 'live_draw'
        elif choice == '3':
            # Ask how many objects the user wants to work with first
            try:
                raw_n = input("How many objects will you place onto? (Enter=all detected later): ").strip()
                globals()['EXPECTED_OBJECTS_COUNT'] = int(raw_n) if raw_n else None
            except Exception:
                globals()['EXPECTED_OBJECTS_COUNT'] = None

            # Load from library and then choose shapes with quantities
            load_shapes_library()
            if not shapes_library:
                print("[WARN] Library is empty. Falling back to mode 2 (Live Draw).")
                STARTUP_MODE = 'live_draw'
            else:
                print(f"[INFO] Shapes in library: {len(shapes_library)}")
                for i, e in enumerate(shapes_library, 1):
                    print(f"  {i}. {e.get('name','unnamed')} [{e.get('type')}] ")
                try:
                    pieces = []
                    while True:
                        raw = input("Enter shape indices to place (comma-separated, Enter to finish): ").strip()
                        if not raw:
                            if pieces:
                                break
                            else:
                                print("[WARN] No shapes added yet. Please enter indices or press Ctrl+C to cancel.")
                                continue

                        indices = [int(x)-1 for x in raw.split(',') if x.strip().isdigit()]
                        if not indices:
                            print("[WARN] No indices entered. Try again or press Enter to finish.")
                            continue

                        for idx in indices:
                            if idx < 0 or idx >= len(shapes_library):
                                continue
                            qty_raw = input(f"Quantity for '{shapes_library[idx].get('name','unnamed')}' (Enter=1): ").strip()
                            qty = int(qty_raw) if qty_raw else 1
                            for _ in range(max(1, qty)):
                                entry = shapes_library[idx]
                                piece = lib_entry_to_piece(entry)
                                if piece is not None:
                                    pieces.append(piece)
                        print(f"[INFO] Added. Current total pieces: {len(pieces)}")

                    if not pieces:
                        print("[WARN] No valid shapes chosen. Falling back to mode 2 (Live Draw).")
                        STARTUP_MODE = 'live_draw'
                    else:
                        globals()['cutting_pieces'] = pieces
                        globals()['remaining_pieces'] = pieces.copy()
                        print(f"[INFO] Loaded {len(pieces)} piece(s) from library for auto-placement.")
                        STARTUP_MODE = 'lib_place'
                except Exception as e:
                    print(f"[WARN] Invalid input: {e}. Falling back to mode 2 (Live Draw).")
                    STARTUP_MODE = 'live_draw'
        elif choice == '4':
            # CSV cutting mode
            run_csv_cutting_mode()
        else:
            STARTUP_MODE = 'live_draw'
    except KeyboardInterrupt:
        STARTUP_MODE = 'live'

def run_shape_editor(initial_scale_mm_per_px: float | None = None):
    """Simple offline shape editor: draw polygon/circle/arc on a blank canvas, set scale, save to library."""
    import copy
    print("\n=== Shape editor (draw; sizes in centimeters) ===")
    # Единицы редактора: сантиметры (по умолчанию 100 px = 1 см)
    # scale_local хранит мм/px. При зуме колесиком будем менять этот масштаб.
    scale_local = 0.1  # мм/px (100 px = 10 мм = 1 см)
    W, H = 900, 600
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    win = 'Shape-Editor'
    cv2.namedWindow(win)
    editor_current = None
    editor_shapes = []
    editor_type = 'poly'
    snap_px = 10.0

    def nearest_vertex(pt):
        if editor_current and editor_current.get('type') == 'poly':
            for vx, vy in editor_current.get('pts', []):
                if math.hypot(pt[0]-vx, pt[1]-vy) <= snap_px:
                    return (vx, vy)
        return None

    mouse = [0, 0]
    def cb(event, x, y, flags, param):
        nonlocal editor_current, editor_shapes, editor_type, mouse, scale_local
        mouse = [x, y]
        # Масштаб колесиком: Ctrl+колесо для точной настройки
        if event == cv2.EVENT_MOUSEWHEEL:
            # flags > 0: прокрутка вверх
            delta = 1.0 if flags > 0 else -1.0
            factor = 1.1 if delta > 0 else 1/1.1
            # Ограничим разумно (от 0.01 мм/px до 10 мм/px)
            new_scale = max(0.01, min(10.0, float(scale_local) * float(factor)))
            scale_local = new_scale
            return
        if editor_type == 'poly':
            if event == cv2.EVENT_LBUTTONDOWN:
                p = (float(x), float(y))
                if editor_current is None or editor_current.get('type') != 'poly' or editor_current.get('closed', False):
                    editor_current = {'type': 'poly', 'pts': [], 'closed': False}
                # snapping to first vertex to close
                if editor_current['pts']:
                    nv = nearest_vertex(p)
                    if nv is not None:
                        p = nv
                        # if near first and at least 3 points -> close
                        if len(editor_current['pts']) >= 3 and nv == tuple(editor_current['pts'][0]):
                            editor_current['closed'] = True
                            editor_shapes.append(copy.deepcopy(editor_current))
                            editor_current = None
                            return
                editor_current['pts'].append(p)
            elif event == cv2.EVENT_RBUTTONDOWN:
                if editor_current and len(editor_current.get('pts', [])) >= 3:
                    editor_current['closed'] = True
                    editor_shapes.append(copy.deepcopy(editor_current))
                    editor_current = None
        elif editor_type == 'circle':
            if event == cv2.EVENT_LBUTTONDOWN:
                editor_current = {'type': 'circle', 'center': (float(x), float(y)), 'radius_px': 0.0, 'drawing': True}
            elif event == cv2.EVENT_MOUSEMOVE and editor_current and editor_current.get('type') == 'circle' and editor_current.get('drawing'):
                cx, cy = editor_current['center']
                editor_current['radius_px'] = math.hypot(float(x)-cx, float(y)-cy)
            elif event == cv2.EVENT_LBUTTONUP and editor_current and editor_current.get('type') == 'circle':
                editor_current['drawing'] = False
                if editor_current['radius_px'] > 1:
                    editor_shapes.append(copy.deepcopy(editor_current))
                editor_current = None
        elif editor_type == 'arc':
            if event == cv2.EVENT_LBUTTONDOWN:
                if editor_current is None or editor_current.get('type') != 'arc':
                    editor_current = {'type': 'arc', 'phase': 0}
                phase = editor_current.get('phase', 0)
                if phase == 0:
                    editor_current['center'] = (float(x), float(y))
                    editor_current['phase'] = 1
                elif phase == 1:
                    editor_current['start'] = (float(x), float(y))
                    cx, cy = editor_current['center']
                    sx, sy = editor_current['start']
                    editor_current['radius_px'] = max(1.0, math.hypot(sx-cx, sy-cy))
                    editor_current['phase'] = 2
                elif phase == 2:
                    editor_current['end'] = (float(x), float(y))
                    cx, cy = editor_current['center']
                    sx, sy = editor_current['start']
                    ex, ey = editor_current['end']
                    a0 = math.degrees(math.atan2(sy-cy, sx-cx))
                    a1 = math.degrees(math.atan2(ey-cy, ex-cx))
                    editor_current['start_deg'] = a0
                    editor_current['end_deg'] = a1
                    editor_current['ccw'] = True
                    editor_shapes.append(copy.deepcopy(editor_current))
                    editor_current = None

    cv2.setMouseCallback(win, cb)
    info = True
    while True:
        img = canvas.copy()
        # Draw finalized shapes
        for shp in editor_shapes:
            if shp['type'] == 'poly':
                pts = np.array(shp['pts'], dtype=np.int32).reshape(-1,1,2)
                cv2.polylines(img, [pts], shp.get('closed', True), (0,0,255), 2, cv2.LINE_AA)
                n = len(shp['pts'])
                for i in range(n - (0 if shp.get('closed', True) else 1)):
                    a = np.array(shp['pts'][i]); b = np.array(shp['pts'][(i+1)%n])
                    mid = ((a+b)/2.0).astype(int)
                    length_cm = (float(np.linalg.norm(b-a)) * float(scale_local)) / 10.0
                    cv2.putText(img, f"{length_cm:.1f} cm", (int(mid[0])+5, int(mid[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                    cv2.putText(img, f"{length_cm:.1f} cm", (int(mid[0])+5, int(mid[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
            elif shp['type'] == 'circle':
                c = tuple(np.array(shp['center'], dtype=int))
                r = int(max(1, shp['radius_px']))
                cv2.circle(img, c, r, (0,0,255), 2, cv2.LINE_AA)
                dia_cm = (2.0 * float(shp['radius_px']) * float(scale_local)) / 10.0
                cv2.putText(img, f"D={dia_cm:.1f} cm", (c[0]-50, c[1]-r-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                cv2.putText(img, f"D={dia_cm:.1f} cm", (c[0]-50, c[1]-r-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
            elif shp['type'] == 'arc':
                cx, cy = shp['center']; r = shp['radius_px']; a0 = shp['start_deg']; a1 = shp['end_deg']; ccw = shp.get('ccw', True)
                ang0 = a0; ang1 = a1
                if ccw:
                    if ang1 < ang0: ang1 += 360.0
                else:
                    if ang1 > ang0: ang1 -= 360.0
                num = max(16, int(abs(ang1-ang0)))
                pts = []
                for t in np.linspace(ang0, ang1, num=num):
                    rad = math.radians(t)
                    pts.append((cx + r*math.cos(rad), cy + r*math.sin(rad)))
                pts = np.array(pts, dtype=np.int32).reshape(-1,1,2)
                cv2.polylines(img, [pts], False, (0,0,255), 2, cv2.LINE_AA)
                radius_cm = (r * float(scale_local)) / 10.0
                ang = abs(ang1-ang0)
                arc_len_cm = math.radians(ang) * radius_cm
                c = (int(cx), int(cy))
                cv2.putText(img, f"R={radius_cm:.1f} cm ∠{ang:.1f}° L={arc_len_cm:.1f} cm", (c[0]+10, c[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                cv2.putText(img, f"R={radius_cm:.1f} cm ∠{ang:.1f}° L={arc_len_cm:.1f} cm", (c[0]+10, c[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

        # Current shape preview
        if editor_current is not None:
            if editor_current.get('type') == 'poly':
                pts = editor_current.get('pts', [])
                if len(pts) >= 1:
                    pl = np.array(pts, dtype=np.int32).reshape(-1,1,2)
                    cv2.polylines(img, [pl], False, (255,0,0), 1, cv2.LINE_AA)
                    # rubber band
                    a = np.array(pts[-1]); b = np.array(mouse, dtype=float)
                    cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (255,0,0), 1, cv2.LINE_AA)
                    # live length
                    mid = ((a+b)/2.0).astype(int)
                    length_cm = (float(np.linalg.norm(b-a)) * float(scale_local)) / 10.0
                    cv2.putText(img, f"{length_cm:.1f} cm", (int(mid[0])+5, int(mid[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                    cv2.putText(img, f"{length_cm:.1f} cm", (int(mid[0])+5, int(mid[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1)
            elif editor_current.get('type') == 'circle':
                if 'center' in editor_current and editor_current.get('radius_px', 0.0) > 0:
                    c = tuple(np.array(editor_current['center'], dtype=int))
                    r = int(max(1, editor_current['radius_px']))
                    cv2.circle(img, c, r, (255,0,0), 1, cv2.LINE_AA)
            elif editor_current.get('type') == 'arc':
                phase = editor_current.get('phase', 0)
                if phase >= 0 and 'center' in editor_current:
                    c = tuple(np.array(editor_current['center'], dtype=int))
                    cv2.circle(img, c, 3, (255,0,0), -1, cv2.LINE_AA)
                if phase >= 1 and 'start' in editor_current:
                    s = tuple(np.array(editor_current['start'], dtype=int))
                    c = tuple(np.array(editor_current['center'], dtype=int))
                    cv2.line(img, c, s, (255,0,0), 1, cv2.LINE_AA)
                if phase == 2:
                    cx, cy = editor_current['center']
                    sx, sy = editor_current['start']
                    r = editor_current['radius_px']
                    a0 = math.degrees(math.atan2(sy-cy, sx-cx))
                    ex, ey = mouse
                    a1 = math.degrees(math.atan2(ey-cy, ex-cx))
                    ang0 = a0; ang1 = a1
                    if ang1 < ang0: ang1 += 360.0
                    num = max(16, int(abs(ang1-ang0)))
                    pts = []
                    for t in np.linspace(ang0, ang1, num=num):
                        rad = math.radians(t)
                        pts.append((cx + r*math.cos(rad), cy + r*math.sin(rad)))
                    pts = np.array(pts, dtype=np.int32).reshape(-1,1,2)
                    cv2.polylines(img, [pts], False, (255,0,0), 1, cv2.LINE_AA)

        if info:
            cv2.putText(img, "Editor: 1-Poly 2-Circle 3-Arc | Enter/RightClick finish | u-undo | x-cancel | q-save & exit", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            cv2.putText(img, "Editor: 1-Poly 2-Circle 3-Arc | Enter/RightClick finish | u-undo | x-cancel | q-save & exit", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.putText(img, f"Units: cm (100 px = 1 cm). Scale: {scale_local/10.0:.3f} cm/px", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            cv2.putText(img, f"Units: cm (100 px = 1 cm). Scale: {scale_local/10.0:.3f} cm/px", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,0), 1)
        cv2.imshow(win, img)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:  # Esc
            break
        if k == ord('1'): editor_type = 'poly'; editor_current = None
        if k == ord('2'): editor_type = 'circle'; editor_current = None
        if k == ord('3'): editor_type = 'arc'; editor_current = None
        if k == ord('x'): editor_current = None
        if k == ord('u'):
            if editor_current and editor_current.get('type') == 'poly' and editor_current.get('pts'):
                editor_current['pts'].pop()
                if not editor_current['pts']:
                    editor_current = None
            elif editor_shapes:
                editor_shapes.pop()
        if k in (13, 10):
            if editor_current and editor_current.get('type') == 'poly' and len(editor_current.get('pts', [])) >= 3:
                editor_current['closed'] = True
                editor_shapes.append(copy.deepcopy(editor_current))
                editor_current = None
        if k == ord('q'):
            # Save all shapes drawn this session into library
            if not editor_shapes:
                print('[INFO] Нет фигур для сохранения.')
                break
            load_shapes_library()
            for shp in editor_shapes:
                try:
                    name = input("Shape name (Enter=auto): ").strip() or f"shape_{len(shapes_library)+1}"
                except Exception:
                    name = f"shape_{len(shapes_library)+1}"
                entry = convert_drawn_shape_to_mm(shp, float(scale_local), name)
                if entry:
                    shapes_library.append(entry)
                    print(f"[INFO] Added shape '{name}' ({entry.get('type')})")
            save_shapes_library()
            break
    cv2.destroyWindow(win)

def build_shape_instance_from_lib(entry: dict, anchor_rect_px: tuple[float, float], rotation_deg: float, scale_mm_per_px: float) -> dict | None:
    """Build a drawable shape (in rect px) from a library entry (in mm).

    Placement scale multiplier is applied globally via PLACEMENT_SCALE_MULT.
    """
    try:
        ax, ay = float(anchor_rect_px[0]), float(anchor_rect_px[1])
        mul = float(PLACEMENT_SCALE_MULT) if 'PLACEMENT_SCALE_MULT' in globals() else 1.0
        if entry.get('type') == 'poly':
            pts_mm = entry.get('pts_mm', [])
            if not pts_mm:
                return None
            pts_local_px = []
            # Optional auto-normalization based on table size if present
            table_w = globals().get('CURRENT_TABLE_W_MM', None)
            table_h = globals().get('CURRENT_TABLE_H_MM', None)
            for mx, my in pts_mm:
                mmx = float(mx)
                mmy = float(my)
                # If points look abnormally large compared to table (e.g., > 1.5x table), normalize by table width as a safety
                if table_w and table_w > 0 and mmx > 1.5 * table_w:
                    # Interpret stored units as centimeters and convert to mm
                    mmx *= 10.0
                if table_h and table_h > 0 and mmy > 1.5 * table_h:
                    mmy *= 10.0
                px = (mmx / float(scale_mm_per_px)) * mul
                py = (mmy / float(scale_mm_per_px)) * mul
                rx, ry = rotate_point_xy(px, py, rotation_deg)
                pts_local_px.append((ax + rx, ay + ry))
            return { 'type': 'poly', 'pts': pts_local_px, 'closed': bool(entry.get('closed', False)) }
        if entry.get('type') == 'circle':
            dia_mm = float(entry.get('diameter_mm', 0.0))
            # Normalize if suspicious vs table size
            table_w = globals().get('CURRENT_TABLE_W_MM', None)
            table_h = globals().get('CURRENT_TABLE_H_MM', None)
            if table_w and table_h and dia_mm > 1.5 * min(table_w, table_h):
                dia_mm *= 10.0
            r_px = max(0.0, ((dia_mm / 2.0) / float(scale_mm_per_px)) * mul)
            return { 'type': 'circle', 'center': (ax, ay), 'radius_px': r_px }
        if entry.get('type') == 'arc':
            radius_mm = float(entry.get('radius_mm', 0.0))
            angle_deg = float(entry.get('angle_deg', 0.0))
            ccw = bool(entry.get('ccw', True))
            table_w = globals().get('CURRENT_TABLE_W_MM', None)
            table_h = globals().get('CURRENT_TABLE_H_MM', None)
            if table_w and table_h and radius_mm * 2.0 > 1.5 * min(table_w, table_h):
                radius_mm *= 10.0
            r_px = max(0.0, (radius_mm / float(scale_mm_per_px)) * mul)
            start_deg = rotation_deg
            end_deg = start_deg + (angle_deg if ccw else -angle_deg)
            return { 'type': 'arc', 'center': (ax, ay), 'radius_px': r_px, 'start_deg': start_deg, 'end_deg': end_deg, 'ccw': ccw }
    except Exception as e:
        print(f"[ERROR] Failed to build instance from library: {e}")
    return None

def lib_entry_to_piece(entry: dict) -> Optional[tuple[float, float, str]]:
    """Convert a library entry to a rectangular/circular piece (w_mm, h_mm, shape).

    - poly: use oriented bbox width/height in mm
    - circle: diameter as w=h
    - arc: use its bounding box
    """
    try:
        t = entry.get('type')
        if t == 'circle':
            d = float(entry.get('diameter_mm', 0.0))
            if d > 0:
                return (d, d, 'circle')
            return None
        elif t == 'poly':
            pts = entry.get('pts_mm', [])
            if len(pts) < 2:
                return None
            arr = np.array(pts, dtype=np.float64)
            # Use simple axis-aligned bbox in mm
            min_xy = arr.min(axis=0)
            max_xy = arr.max(axis=0)
            w = float(max_xy[0] - min_xy[0])
            h = float(max_xy[1] - min_xy[1])
            if w <= 0 or h <= 0:
                return None
            # Keep original polygon for potential overlay later
            try:
                key = (round(w, 2), round(h, 2))
                poly = np.array(pts, dtype=np.float64).tolist()
                poly_data = {
                    'pts': poly,
                    'closed': bool(entry.get('closed', True)),
                    'segments': entry.get('segments'),
                    'name': entry.get('name', ''),
                    'bbox_w': w,
                    'bbox_h': h
                }
                LIB_SHAPE_MAP.setdefault(key, []).append(poly_data)
                # Also store by name for reliable lookup
                name = entry.get('name', '')
                if name:
                    LIB_SHAPE_BY_NAME[name] = poly_data
            except Exception:
                pass
            return (w, h, 'poly')  # Return 'poly' type, not 'rect'
        elif t == 'arc':
            R = float(entry.get('radius_mm', 0.0))
            ang = float(entry.get('angle_deg', 0.0))
            if R <= 0:
                return None
            # Bounding box of arc ≤ bounding box of full circle of diameter 2R
            d = 2.0 * R
            return (d, d, 'rect')
    except Exception: 
        return None
    return None


def prompt_rc_pieces():
    """Prompt rect/circle pieces for placement mode."""
    global cutting_pieces, remaining_pieces
    cutting_pieces = []
    print("\n=== Rect/Circle Placement ===")
    try:
        num_pieces = int(input("Number of pieces: "))
        for i in range(num_pieces):
            print(f"  → Piece {i+1}")
            shape_type = input("      Shape type (rect/circle): ").lower().strip()
            if shape_type not in ["rect", "circle"]:
                print("[WARN] Invalid shape type. Using rectangle as default.")
                shape_type = "rect"
            if shape_type == "rect":
                w = float(input("      Width (mm): "))
                h = float(input("      Height (mm): "))
                cutting_pieces.append((w, h, 'rect'))
            else:
                d = float(input("      Diameter (mm): "))
                cutting_pieces.append((d, d, 'circle'))
    except Exception as e:
        print(f"[WARN] Invalid input: {e}. Using default test values")
        cutting_pieces = [(100, 100, 'rect'), (100, 100, 'rect'), (100, 100, 'rect'), (100, 100, 'circle'), (50, 50, 'circle')]
    remaining_pieces = cutting_pieces.copy()
    print(f"[INFO] Total pieces: {len(cutting_pieces)}")

# ───── Mouse click handler ────────────────────────────────
def mouse_callback(event, x, y, flags, param):
    global click_position, DRAW_MODE, DRAW_TYPE, current_shape, mouse_pos_screen, mouse_pos_rect, inverse_transform_matrix
    global PLACE_MODE, place_idx, place_rotation_deg, last_finalized_shape, PLACEMENT_SCALE_MULT
    mouse_pos_screen = (x, y)
    # Map current mouse to rect coordinates if possible
    try:
        if inverse_transform_matrix is not None:
            pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
            rect_pt = cv2.perspectiveTransform(pt, inverse_transform_matrix)
            rx = float(rect_pt[0, 0, 0])
            ry = float(rect_pt[0, 0, 1])
            mouse_pos_rect = (rx, ry)
    except Exception:
        mouse_pos_rect = None

    # Mouse wheel in placement mode: scale multiplier
    if PLACE_MODE and event == cv2.EVENT_MOUSEWHEEL:
        delta = 1.0 if flags > 0 else -1.0
        factor = 1.1 if delta > 0 else 1/1.1
        PLACEMENT_SCALE_MULT = max(0.05, min(20.0, float(PLACEMENT_SCALE_MULT) * float(factor)))
        print(f"[INFO] Placement scale: ×{PLACEMENT_SCALE_MULT:.2f}")
        return

    # Placement mode: commit placement on left click
    if PLACE_MODE and event == cv2.EVENT_LBUTTONDOWN and mouse_pos_rect is not None:
        try:
            if shapes_library:
                entry = shapes_library[min(place_idx, len(shapes_library)-1)]
                inst = build_shape_instance_from_lib(entry, mouse_pos_rect, place_rotation_deg, float(CURRENT_SCALE_MM_PER_PX or 1.0))
                if inst:
                    user_drawn_shapes.append(inst)
                    last_finalized_shape = inst
                    print(f"[INFO] Placed shape '{entry.get('name','unnamed')}' at ({mouse_pos_rect[0]:.1f},{mouse_pos_rect[1]:.1f}) rect px")
            return
        except Exception as e:
            print(f"[WARN] Failed to place shape: {e}")
            return

    if DRAW_MODE:
        # Drawing interactions
        if DRAW_TYPE == 'poly':
            if event == cv2.EVENT_LBUTTONDOWN and mouse_pos_rect is not None:
                # Initialize poly
                if current_shape is None or current_shape.get('type') != 'poly' or current_shape.get('closed', False):
                    current_shape = {'type': 'poly', 'pts': [], 'closed': False}
                # Snapping to first point to close polygon
                snap_tol = 10.0  # px in rect space
                p = mouse_pos_rect
                if current_shape['pts']:
                    first = current_shape['pts'][0]
                    if math.hypot(p[0]-first[0], p[1]-first[1]) <= snap_tol and len(current_shape['pts']) >= 3:
                        current_shape['closed'] = True
                        user_drawn_shapes.append(current_shape)
                        last_finalized_shape = current_shape
                        current_shape = None
                        return
                current_shape['pts'].append(p)
            elif event == cv2.EVENT_RBUTTONDOWN:
                if current_shape and current_shape.get('type') == 'poly' and len(current_shape.get('pts', [])) >= 3:
                    current_shape['closed'] = True
                    user_drawn_shapes.append(current_shape)
                    last_finalized_shape = current_shape
                    current_shape = None
        elif DRAW_TYPE == 'circle':
            if event == cv2.EVENT_LBUTTONDOWN and mouse_pos_rect is not None:
                # Start circle with center
                current_shape = {'type': 'circle', 'center': mouse_pos_rect, 'radius_px': 0.0, 'drawing': True}
            elif event == cv2.EVENT_MOUSEMOVE and current_shape and current_shape.get('type') == 'circle' and current_shape.get('drawing') and mouse_pos_rect is not None:
                cx, cy = current_shape['center']
                current_shape['radius_px'] = math.hypot(mouse_pos_rect[0]-cx, mouse_pos_rect[1]-cy)
            elif event == cv2.EVENT_LBUTTONUP and current_shape and current_shape.get('type') == 'circle':
                current_shape['drawing'] = False
                if current_shape['radius_px'] > 1:
                    user_drawn_shapes.append(current_shape)
                    last_finalized_shape = current_shape
                current_shape = None
        elif DRAW_TYPE == 'arc':
            # Three clicks: center -> start -> end
            if event == cv2.EVENT_LBUTTONDOWN and mouse_pos_rect is not None:
                if current_shape is None or current_shape.get('type') != 'arc':
                    current_shape = {'type': 'arc', 'phase': 0}
                phase = current_shape.get('phase', 0)
                if phase == 0:
                    current_shape['center'] = mouse_pos_rect
                    current_shape['phase'] = 1
                elif phase == 1:
                    current_shape['start'] = mouse_pos_rect
                    # radius from center to start
                    cx, cy = current_shape['center']
                    sx, sy = current_shape['start']
                    current_shape['radius_px'] = max(1.0, math.hypot(sx-cx, sy-cy))
                    current_shape['phase'] = 2
                elif phase == 2:
                    current_shape['end'] = mouse_pos_rect
                    # Compute start/end angles in radians
                    cx, cy = current_shape['center']
                    sx, sy = current_shape['start']
                    ex, ey = current_shape['end']
                    a0 = math.degrees(math.atan2(sy-cy, sx-cx))
                    a1 = math.degrees(math.atan2(ey-cy, ex-cx))
                    current_shape['start_deg'] = a0
                    current_shape['end_deg'] = a1
                    # Determine direction so that arc passes through mid-angle defined by mouse move hint; keep CCW by default
                    current_shape['ccw'] = True
                    user_drawn_shapes.append(current_shape)
                    last_finalized_shape = current_shape
                    current_shape = None
        # Do not process selection while in drawing mode
        return

    # Default behavior: selection clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        click_position = (x, y)

# ───── Marker detection and basic functions ────────────────
def detect_corners(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(4, 4))
    gray_image = clahe.apply(gray_image)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    marker_corners, marker_ids, _ = detector.detectMarkers(gray_image)

    return marker_corners, marker_ids


def order_zone(c):
    p=np.concatenate([x[0] for x in c]); s=p.sum(1); d=np.diff(p,1).squeeze()
    z=np.zeros((4,2),np.float32)
    z[[0,1,2,3]]=[p[np.argmin(s)],p[np.argmin(d)],p[np.argmax(s)],p[np.argmax(d)]]
    return z

def mm_per_px(corners):
    edges=[np.linalg.norm(c[0][i]-c[0][(i+1)%4]) for c in corners for i in range(4)]
    return ARUCO_SIZE_MM/np.mean(edges)

def segment(rect, bg):
    """
    Simplified segmentation using background difference and raw edge detection only
    """
    config = SEGMENTATION_CONFIG
    use_cuda = ENABLE_CUDA

    # Convert to grayscale
    if use_cuda:
        try:
            gpu_rect = cv2.cuda_GpuMat()
            gpu_rect.upload(rect)
            gpu_gray = cv2.cuda.cvtColor(gpu_rect, cv2.COLOR_BGR2GRAY)
            # Light Gaussian blur for noise reduction
            gauss = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 1.0)
            gpu_blur = gauss.apply(gpu_gray)
            blur_light = gpu_blur.download()
        except Exception:
            use_cuda = False
            gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
            blur_light = cv2.GaussianBlur(gray, (5, 5), 1.0)
    else:
        gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
        # Light Gaussian blur for noise reduction
        blur_light = cv2.GaussianBlur(gray, (5, 5), 1.0)
    
    # Store individual masks
    individual_masks = {}
    
    # 1. EDGE DETECTION (RAW + DILATED)
    if config['use_edges']:
        if use_cuda:
            try:
                gpu_blur = cv2.cuda_GpuMat()
                gpu_blur.upload(blur_light)
                canny = cv2.cuda.createCannyEdgeDetector(config['canny_low'], config['canny_high'])
                gpu_edges = canny.detect(gpu_blur)
                # Dilate on GPU
                kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                gpu_kernel = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC1, kernel_edge, iterations=2)
                gpu_edges_dil = gpu_kernel.apply(gpu_edges)
                edges_raw = gpu_edges_dil.download()
                edges_dilated = edges_raw.copy()
            except Exception:
                use_cuda = False
                edges_raw = cv2.Canny(blur_light, config['canny_low'], config['canny_high'])
                kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                edges_dilated = cv2.dilate(edges_raw, kernel_edge, iterations=2)
        else:
            # Step 1: Raw Canny edge detection
            edges_raw = cv2.Canny(blur_light, config['canny_low'], config['canny_high'])
            # Step 2: Dilate edges to create solid regions
            kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges_dilated = cv2.dilate(edges_raw, kernel_edge, iterations=2)
        
        # Store both versions
        individual_masks['edges_raw_original'] = edges_raw
        individual_masks['edges_raw'] = edges_dilated  # This is what we use for segmentation
        
        if config['debug_mode']:
            cv2.imshow('Segmentation - edges_raw', edges_raw)
            cv2.imshow('Segmentation - edges_dilated', edges_dilated)
    
    # 2. BACKGROUND DIFFERENCE
    # Simple background difference
    if use_cuda:
        try:
            g_blur = cv2.cuda_GpuMat(); g_bg = cv2.cuda_GpuMat()
            g_blur.upload(blur_light); g_bg.upload(bg)
            g_abs = cv2.cuda.absdiff(g_blur, g_bg)
            diff_bg = g_abs.download()
            _, bg_thresh = cv2.threshold(diff_bg, config['thresh_med'], 255, cv2.THRESH_BINARY)
        except Exception:
            use_cuda = False
            diff_bg = cv2.absdiff(blur_light, bg)
            _, bg_thresh = cv2.threshold(diff_bg, config['thresh_med'], 255, cv2.THRESH_BINARY)
    else:
        diff_bg = cv2.absdiff(blur_light, bg)
        _, bg_thresh = cv2.threshold(diff_bg, config['thresh_med'], 255, cv2.THRESH_BINARY)
    
    # Clean up background difference
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    if use_cuda:
        try:
            g_mask = cv2.cuda_GpuMat(); g_mask.upload(bg_thresh)
            open_f = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, cv2.CV_8UC1, kernel_bg, iterations=1)
            close_f = cv2.cuda.createMorphologyFilter(cv2.MORPH_CLOSE, cv2.CV_8UC1, kernel_bg, iterations=2)
            g_tmp = open_f.apply(g_mask)
            g_tmp = close_f.apply(g_tmp)
            bg_cleaned = g_tmp.download()
        except Exception:
            use_cuda = False
            bg_cleaned = cv2.morphologyEx(bg_thresh, cv2.MORPH_OPEN, kernel_bg, iterations=1)
            bg_cleaned = cv2.morphologyEx(bg_cleaned, cv2.MORPH_CLOSE, kernel_bg, iterations=2)
    else:
        bg_cleaned = cv2.morphologyEx(bg_thresh, cv2.MORPH_OPEN, kernel_bg, iterations=1)
        bg_cleaned = cv2.morphologyEx(bg_cleaned, cv2.MORPH_CLOSE, kernel_bg, iterations=2)
    
    individual_masks['background'] = bg_cleaned
    
    if config['debug_mode']:
        cv2.imshow('Segmentation - background', bg_cleaned)
    
    # 3. USE PURE EDGE-BASED SEGMENTATION (EDGES_RAW + EDGES_DILATED)
    
    # Use only edge-based masks for the cleanest segmentation
    if config['use_edges'] and 'edges_raw' in individual_masks:
        # Start with the dilated edges as they provide better filled regions
        result = individual_masks['edges_raw'].copy()
        
        # The edges_raw is already the dilated version from step 1
        # This gives us clean, accurate object boundaries without background noise
        
        used_techniques = ['edges_raw', 'edges_dilated']
        if config['debug_mode']:
            cv2.imshow('Segmentation - Pure Edges', result)
    else:
        # Fallback to background difference only if edges are disabled
        result = bg_cleaned
        print("[WARN] Edges disabled, using background difference as fallback")
    
    # 4. OPTIMIZED MORPHOLOGICAL CLEANUP
    
    # Light noise reduction to preserve edge accuracy
    result = cv2.medianBlur(result, 3)
    
    # Define kernels
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_connection = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                (config['connection_strength'], config['connection_strength']))
    
    # Step 1: Light noise removal (preserve edge quality)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    if config['debug_mode']:
        cv2.imshow('Segmentation - After Light Cleanup', result)
    
    # Step 2: Only apply connection if user specifically requests it with larger values
    if config['connection_strength'] > 15:  # Only connect if user wants aggressive connection
        # Light closing to connect nearby objects
        if use_cuda:
            try:
                g_res = cv2.cuda_GpuMat(); g_res.upload(result)
                close_f = cv2.cuda.createMorphologyFilter(cv2.MORPH_CLOSE, cv2.CV_8UC1, kernel_connection, iterations=config['connection_iterations'])
                g_res = close_f.apply(g_res)
                result = g_res.download()
            except Exception:
                result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_connection, 
                                     iterations=config['connection_iterations'])
        else:
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_connection, 
                                     iterations=config['connection_iterations'])
        
        if config['debug_mode']:
            cv2.imshow('Segmentation - After Connection', result)
        
        # Light cleanup to remove connection artifacts
        if use_cuda:
            try:
                g_res2 = cv2.cuda_GpuMat(); g_res2.upload(result)
                open_f2 = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, cv2.CV_8UC1, kernel_small, iterations=1)
                g_res2 = open_f2.apply(g_res2)
                result = g_res2.download()
            except Exception:
                result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel_small, iterations=1)
        else:
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        if config['debug_mode']:
            cv2.imshow('Segmentation - After Final Cleanup', result)
    else:
        # Skip aggressive morphological operations to preserve edge quality
        if config['debug_mode']:
            cv2.imshow('Segmentation - Skipped Connection (preserving edges)', result)
    
    # 5. CONNECTED COMPONENTS FILTERING
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(result, connectivity=8)
    
    # Create final clean mask
    clean_mask = np.zeros_like(result)
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area > config['min_contour_area']:
            # Add component to final mask
            clean_mask[labels == i] = 255
    
    if config['debug_mode']:
        cv2.imshow('Segmentation - Final Result', clean_mask)
        cv2.waitKey(1)
    
    return clean_mask

def toggle_debug_mode():
    """Toggle debug mode for segmentation visualization"""
    global SEGMENTATION_CONFIG
    SEGMENTATION_CONFIG['debug_mode'] = not SEGMENTATION_CONFIG['debug_mode']
    status = "enabled" if SEGMENTATION_CONFIG['debug_mode'] else "disabled"
    print(f"[INFO] Segmentation debug mode {status}")
    if not SEGMENTATION_CONFIG['debug_mode']:
        # Close debug windows for our simplified segmentation
        debug_windows = [
            'Segmentation - edges_raw', 'Segmentation - edges_dilated',
            'Segmentation - background', 'Segmentation - Pure Edges',
            'Segmentation - After Light Cleanup', 'Segmentation - After Connection', 
            'Segmentation - After Final Cleanup', 'Segmentation - Skipped Connection (preserving edges)', 
            'Segmentation - Final Result'
        ]
        for window in debug_windows:
            try:
                cv2.destroyWindow(window)
            except:
                pass

def adjust_segmentation_sensitivity(increase=True):
    """Adjust segmentation sensitivity for different lighting conditions"""
    global SEGMENTATION_CONFIG
    factor = 1.1 if increase else 0.9
    
    # Adjust threshold values
    SEGMENTATION_CONFIG['thresh_low'] = max(5, min(50, int(SEGMENTATION_CONFIG['thresh_low'] * factor)))
    SEGMENTATION_CONFIG['thresh_med'] = max(10, min(70, int(SEGMENTATION_CONFIG['thresh_med'] * factor)))
    SEGMENTATION_CONFIG['thresh_high'] = max(15, min(100, int(SEGMENTATION_CONFIG['thresh_high'] * factor)))
    SEGMENTATION_CONFIG['weighted_thresh'] = max(10, min(80, int(SEGMENTATION_CONFIG['weighted_thresh'] * factor)))
    
    direction = "increased" if increase else "decreased"
    print(f"[INFO] Segmentation sensitivity {direction}")
    print(f"[INFO] New thresholds: low={SEGMENTATION_CONFIG['thresh_low']}, med={SEGMENTATION_CONFIG['thresh_med']}, high={SEGMENTATION_CONFIG['thresh_high']}")

def adjust_connection_strength(increase=True):
    """Adjust object connection strength to unite nearby objects"""
    global SEGMENTATION_CONFIG
    
    current = SEGMENTATION_CONFIG['connection_strength']
    if increase:
        new_strength = min(50, current + 5)  # Increase by 5, max 50
    else:
        new_strength = max(5, current - 5)   # Decrease by 5, min 5
    
    SEGMENTATION_CONFIG['connection_strength'] = new_strength
    
    direction = "increased" if increase else "decreased"
    print(f"[INFO] Object connection strength {direction} to {new_strength}")
    if new_strength <= 15:
        print(f"[INFO] Low values (≤15) preserve edge quality, higher values connect distant objects")
    else:
        print(f"[INFO] Higher values connect objects that are further apart but may degrade edge quality")

def toggle_technique(technique_name):
    """Toggle individual segmentation techniques (edges only in simplified mode)"""
    global SEGMENTATION_CONFIG
    
    technique_map = {
        'e': 'use_edges'  # Only edges available in simplified segmentation
    }
    
    if technique_name in technique_map:
        config_key = technique_map[technique_name]
        SEGMENTATION_CONFIG[config_key] = not SEGMENTATION_CONFIG[config_key]
        status = "enabled" if SEGMENTATION_CONFIG[config_key] else "disabled"
        technique_full_name = config_key.replace('use_', '').replace('_', ' ').title()
        print(f"[INFO] {technique_full_name} segmentation {status}")
        
        # Show current active techniques
        active_techniques = []
        if SEGMENTATION_CONFIG.get('use_edges', False):
            active_techniques.append('edges_raw + edges_dilated (pure edge-based)')
        else:
            active_techniques.append('background (fallback only)')
        
        print(f"[INFO] Active techniques: {', '.join(active_techniques)}")
    else:
        print(f"[INFO] Unknown technique key '{technique_name}'. Use: e=edges (pure edge-based segmentation)")

def reset_segmentation_config():
    """Reset segmentation configuration to defaults"""
    global SEGMENTATION_CONFIG
    
    # Reset to default values for simplified segmentation (background + raw edges only)
    SEGMENTATION_CONFIG.update({
        'thresh_low': 15,
        'thresh_med': 25,
        'thresh_high': 35,
        'canny_low': 30,
        'canny_high': 100,
        'use_edges': True,
        'use_gradient': False,  # Disabled in simplified mode
        'use_convex_hull': False,  # Simplified - no convex hull
        'connection_strength': 10,  # Reset connection strength (preserve edges)
        'connection_iterations': 1
    })
    
    print("[INFO] Segmentation configuration reset to defaults (pure edge-based segmentation)")

def order_box(pts):
    c=pts.mean(0); pts=pts[np.argsort(np.arctan2(pts[:,1]-c[1],pts[:,0]-c[0]))]
    return np.roll(pts,-np.argmin(pts.sum(1)),0)

def enhance_contour_for_display(contour, max_points=200):
    """
    Enhance contour for better display by ensuring adequate point density
    while keeping the shape accurate to the mask
    """
    if contour is None or len(contour) < 3:
        return contour
    
    # If contour already has good point density, use it as is
    if len(contour) >= max_points * 0.7:
        return contour
    
    # Interpolate additional points along the contour for smoother display
    contour_points = contour.reshape(-1, 2)
    
    # Calculate perimeter
    perimeter = cv2.arcLength(contour, True)
    
    # Calculate desired spacing between points
    if perimeter > 0:
        desired_spacing = perimeter / max_points
        
        # Resample contour with more uniform spacing
        enhanced_points = []
        enhanced_points.append(contour_points[0])
        
        current_distance = 0
        for i in range(len(contour_points)):
            current_point = contour_points[i]
            next_point = contour_points[(i + 1) % len(contour_points)]
            
            segment_length = np.linalg.norm(next_point - current_point)
            
            # Add intermediate points if segment is long
            if segment_length > desired_spacing * 1.5:
                num_intermediate = int(segment_length / desired_spacing)
                for j in range(1, num_intermediate):
                    t = j / num_intermediate
                    intermediate_point = current_point + t * (next_point - current_point)
                    enhanced_points.append(intermediate_point)
            
            enhanced_points.append(next_point)
        
        # Convert back to contour format
        enhanced_contour = np.array(enhanced_points, dtype=np.float32).reshape(-1, 1, 2)
        return enhanced_contour
    
    return contour

# ───── Convert contour (rect image px) to local OBB coordinates in mm ─────
def contour_to_local_polygon_mm(contour_px, box_px, scale_mm_per_px):
    """Project a contour from rect-image pixels to the oriented bounding box
    local coordinates measured in millimeters.

    The local origin is the top-left corner of the OBB (box_px[0]). Axes follow
    the OBB's width (to the right) and height (down).
    """
    if contour_px is None or len(contour_px) == 0:
        return None

    tl, tr, br, bl = box_px
    tl = tl.astype(np.float32)
    tr = tr.astype(np.float32)
    bl = bl.astype(np.float32)

    width_vec = tr - tl
    height_vec = bl - tl
    width_len = float(np.linalg.norm(width_vec))
    height_len = float(np.linalg.norm(height_vec))
    if width_len <= 1e-6 or height_len <= 1e-6:
        return None

    width_unit = width_vec / width_len
    height_unit = height_vec / height_len

    polygon_mm = []
    pts = contour_px.reshape(-1, 2).astype(np.float32)
    for p in pts:
        v = p - tl
        u_px = float(np.dot(v, width_unit))
        v_px = float(np.dot(v, height_unit))
        polygon_mm.append((u_px * scale_mm_per_px, v_px * scale_mm_per_px))

    return polygon_mm

# ───── Enhanced MaxRects (Best-Area-Fit) ──────────────────────
class MaxRects:
    def __init__(self, W, H, allowed_polygon_mm=None):
        self.free = [[0,0,W,H]]
        self.used = []
        self.width = W
        self.height = H
        # Optional polygon (in mm, object-local coordinates) limiting valid placement area
        self.allowed_polygon_mm = allowed_polygon_mm
    
    def insert(self, w, h, shape_type='rect', rot=True, piece_id=-1, shape_name=None):
        best = None
        
        # For circles, we use the same width and height (diameter)
        is_circle = shape_type == 'circle'
        
        # For circles, rotation doesn't matter
        if is_circle:
            rot = False
        
        # Enhanced rotation optimization: try multiple angles for better fit
        rotation_angles = [False, True] if rot else [False]
        if rot and shape_type == 'rect':
            # For rectangles, try additional angles for optimal packing
            # Test 0°, 90°, and intermediate angles for irregular shapes
            rotation_angles = [False, True]  # Keep simple for now, can be extended
        
        # Check multiple positioning criteria for better packing
        for fi, fr in enumerate(self.free):
            fx, fy, fw, fh = fr
            
            for r in rotation_angles:
                rw, rh = (h, w) if r else (w, h)
                
                # Enhanced fit calculation for optimal rotation selection
                fit_ratio_w = min(rw/fw, fw/rw) if fw > 0 else 0
                fit_ratio_h = min(rh/fh, fh/rh) if fh > 0 else 0
                rotation_efficiency = fit_ratio_w * fit_ratio_h
                
                # For circles, ensure the entire circle fits within the free rectangle
                if is_circle:
                    # For circles, rw and rh are both the diameter
                    # Check if circle fits entirely within the free rectangle
                    radius = rw / 2
                    # Ensure circle center is at least radius distance from all edges
                    # Strict checking - no safety margin allowed
                    safety_margin = 0.0  # No safety margin for strict boundary checking
                    if (fx + radius + safety_margin <= fx + fw - radius - safety_margin and 
                        fy + radius + safety_margin <= fy + fh - radius - safety_margin and
                        fx + radius + safety_margin >= fx + safety_margin and 
                        fy + radius + safety_margin >= fy + safety_margin):
                        fits_in_free = True
                    else:
                        fits_in_free = False
                else:
                    # For rectangles, use original check
                    fits_in_free = (rw <= fw and rh <= fh)
                
                if fits_in_free:
                    # Validate against real object contour (in local mm coordinates)
                    if self.allowed_polygon_mm is not None:
                        if not piece_fits_in_local_contour_mm(fx, fy, rw, rh, shape_type, self.allowed_polygon_mm):
                            # Candidate would cross the real contour; skip it
                            continue
                    # Try different position scoring methods
                    
                    # Score 1: Best area fit (less waste)
                    area_score = fw*fh - rw*rh
                    
                    # Score 2: Best short side fit (touching edges)
                    short_side_fit = min(fw - rw, fh - rh)
                    
                    # Score 3: Best long side fit
                    long_side_fit = max(fw - rw, fh - rh)
                    
                    # Score 4: Distance to bottom-left
                    dist_score = fx + fy
                    
                    # Orientation bias: prefer aligning piece orientation to free-rect aspect (encourages vertical stacking in tall corridors)
                    try:
                        is_vertical_piece = (rh >= rw)
                        is_tall_slot = (fh >= fw)
                        orientation_bonus = 0.0
                        if is_vertical_piece and is_tall_slot:
                            orientation_bonus += PACKING_ORIENTATION_BIAS_WEIGHT * min(fw, fh)
                        if (not is_vertical_piece) and (not is_tall_slot):
                            orientation_bonus += PACKING_ORIENTATION_BIAS_WEIGHT * min(fw, fh)
                    except Exception:
                        orientation_bonus = 0.0

                    # Edge contact bonus: favor placements that hug global sheet boundaries to reduce fragmentation
                    try:
                        edge_contacts = 0
                        if abs(fx - 0.0) < 1e-6:  # global left
                            edge_contacts += 1
                        if abs(fy - 0.0) < 1e-6:  # global top
                            edge_contacts += 1
                        if abs((fx + rw) - self.width) < 1e-6:  # global right
                            edge_contacts += 1
                        if abs((fy + rh) - self.height) < 1e-6:  # global bottom
                            edge_contacts += 1
                        edge_bonus = edge_contacts * (PACKING_ORIENTATION_BIAS_WEIGHT * 2.0)
                    except Exception:
                        edge_bonus = 0.0

                    # Combined score - prioritize area fit but consider other factors
                    # Modified to prefer compact placement for better space utilization
                    # Add bonus for positions close to existing pieces (encourage compactness)
                    proximity_bonus = 0
                    for u in self.used:
                        ux, uy, uw, uh = u[:4]
                        # Calculate distance to existing piece center
                        existing_center_x = ux + uw/2
                        existing_center_y = uy + uh/2
                        new_center_x = fx + rw/2
                        new_center_y = fy + rh/2
                        distance = math.sqrt((new_center_x - existing_center_x)**2 + (new_center_y - existing_center_y)**2)
                        # Add bonus for being reasonably close (encourage compact placement)
                        if distance < max(rw, rh) * 2.0:
                            proximity_bonus += 50  # Small bonus for compact placement
                    
                    # Envelope growth penalty: prefer candidates that keep used items' bbox compact
                    try:
                        if len(self.used) == 0:
                            used_min_x = fx
                            used_min_y = fy
                            used_max_x = fx + rw
                            used_max_y = fy + rh
                        else:
                            used_min_x = min([u[0] for u in self.used] + [fx])
                            used_min_y = min([u[1] for u in self.used] + [fy])
                            used_max_x = max([u[0] + u[2] for u in self.used] + [fx + rw])
                            used_max_y = max([u[1] + u[3] for u in self.used] + [fy + rh])
                        envelope_area = max(0.0, (used_max_x - used_min_x) * (used_max_y - used_min_y))
                    except Exception:
                        envelope_area = 0.0

                    # Enhanced scoring with rotation efficiency and aspect ratio matching
                    rotation_bonus = rotation_efficiency * PACKING_ORIENTATION_BIAS_WEIGHT * 50
                    
                    # Aspect ratio matching bonus
                    piece_aspect = rw / rh if rh > 0 else 1.0
                    slot_aspect = fw / fh if fh > 0 else 1.0
                    aspect_match = 1.0 / (1.0 + abs(piece_aspect - slot_aspect))
                    aspect_bonus = aspect_match * PACKING_ORIENTATION_BIAS_WEIGHT * 25
                    
                    final_score = (area_score * 3) + (short_side_fit * 2) + (long_side_fit * 1) + (dist_score * 0.5)
                    final_score += PACKING_ENVELOPE_WEIGHT * envelope_area
                    # Apply bonuses (lower score is better)
                    final_score -= orientation_bonus
                    final_score -= edge_bonus
                    final_score -= rotation_bonus
                    final_score -= aspect_bonus
                    final_score -= proximity_bonus  # Subtract bonus to encourage compact placement
                    
                    # Check for overlaps with existing placements
                    overlap = False
                    for u in self.used:
                        ux, uy, uw, uh, urot, ushape = u[:6]
                        if ushape == 'circle' and is_circle:
                            # Circle-to-circle overlap check
                            circle1_center = (fx + rw/2, fy + rh/2)
                            circle2_center = (ux + uw/2, uy + uh/2)
                            distance = math.sqrt((circle1_center[0] - circle2_center[0])**2 + 
                                               (circle1_center[1] - circle2_center[1])**2)
                            if distance < (rw/2 + uw/2):  # Compare radii
                                overlap = True
                                break
                        elif ushape == 'circle':
                            # Rectangle-to-circle overlap check
                            circle_center = (ux + uw/2, uy + uh/2)
                            radius = uw/2
                            # Find closest point on rectangle to circle center
                            closest_x = max(fx, min(circle_center[0], fx + rw))
                            closest_y = max(fy, min(circle_center[1], fy + rh))
                            # Check if this point is inside the circle
                            distance = math.sqrt((closest_x - circle_center[0])**2 + 
                                               (closest_y - circle_center[1])**2)
                            if distance < radius:
                                overlap = True
                                break
                        elif is_circle:
                            # Circle-to-rectangle overlap check
                            circle_center = (fx + rw/2, fy + rh/2)
                            radius = rw/2
                            # Find closest point on rectangle to circle center
                            closest_x = max(ux, min(circle_center[0], ux + uw))
                            closest_y = max(uy, min(circle_center[1], uy + uh))
                            # Check if this point is inside the circle
                            distance = math.sqrt((closest_x - circle_center[0])**2 + 
                                               (closest_y - circle_center[1])**2)
                            if distance < radius:
                                overlap = True
                                break
                        else:
                            # Rectangle-to-rectangle overlap check
                            if (fx < ux + uw and fx + rw > ux and 
                                fy < uy + uh and fy + rh > uy):
                                overlap = True
                                break
                    
                    # Only consider this position if there's no overlap
                    if not overlap and (best is None or final_score < best[0]):
                        best = (final_score, fi, fr, rw, rh, r)
        
        if best is None:
            return False
            
        _, fi, (fx, fy, fw, fh), rw, rh, rotated = best
        
        # Double-check that this position doesn't overlap with any existing pieces
        for u in self.used:
            ux, uy, uw, uh, _, ushape = u[:6]
            if ushape == 'circle' and is_circle:
                c1 = (fx + rw/2, fy + rh/2)
                c2 = (ux + uw/2, uy + uh/2)
                if math.hypot(c1[0]-c2[0], c1[1]-c2[1]) < (rw/2 + uw/2):
                    return False
            elif ushape == 'circle':
                # rect vs circle
                cx, cy = (ux + uw/2, uy + uh/2)
                closest_x = max(fx, min(cx, fx + rw))
                closest_y = max(fy, min(cy, fy + rh))
                if math.hypot(closest_x-cx, closest_y-cy) < (uw/2):
                    return False
            elif is_circle:
                cx, cy = (fx + rw/2, fy + rh/2)
                closest_x = max(ux, min(cx, ux + uw))
                closest_y = max(uy, min(cy, uy + uh))
                if math.hypot(closest_x-cx, closest_y-cy) < (rw/2):
                    return False
            else:
                if (fx < ux + uw and fx + rw > ux and 
                    fy < uy + uh and fy + rh > uy):
                    return False
            # Also check for almost-exact coordinate matches to avoid duplicates
            if abs(fx - ux) < 1e-6 and abs(fy - uy) < 1e-6 and abs(rw - uw) < 1e-6 and abs(rh - uh) < 1e-6:
                return False
        
        # For circles, adjust positioning to ensure proper placement within boundaries
        if is_circle:
            # For circles, center should be at (fx + radius, fy + radius) for proper placement
            radius = rw / 2
            circle_center_x = fx + radius
            circle_center_y = fy + radius
            
            # Store the circle with its center coordinates and diameter
            # This ensures proper rendering and boundary checking
            circle_x = fx
            circle_y = fy
            
            # Check that circle fits within container boundaries - REJECT if out of bounds
            margin_mm = 0.5  # Tolerance for practical placement with measurement precision
            if (circle_center_x - radius < -margin_mm or circle_center_y - radius < -margin_mm or
                circle_center_x + radius > self.width + margin_mm or circle_center_y + radius > self.height + margin_mm):
                return False  # REJECT placement instead of allowing with warning
            
            self.used.append([circle_x, circle_y, rw, rh, rotated, shape_type, piece_id, shape_name])
        else:
            self.used.append([fx, fy, rw, rh, rotated, shape_type, piece_id, shape_name])
        
        # More efficient splitting method
        del self.free[fi]
        
        # Split horizontally (right rectangle)
        if fw - rw > 0:
            self.free.append([fx + rw, fy, fw - rw, fh])
            
        # Split vertically (below rectangle)
        if fh - rh > 0:
            self.free.append([fx, fy + rh, fw, fh - rh])
            
        # Also add the corner rectangle if both splits are possible
        if fw - rw > 0 and fh - rh > 0:
            self.free.append([fx + rw, fy + rh, fw - rw, fh - rh])
            
        self._prune()
        return True
        
    def _prune(self):
        # Improved free rectangle merging
        i = 0
        while i < len(self.free):
            j = i + 1
            while j < len(self.free):
                if self._can_merge(self.free[i], self.free[j]):
                    self.free[i] = self._merge(self.free[i], self.free[j])
                    self.free.pop(j)
                else:
                    j += 1
            i += 1
            
        # Remove contained rectangles
        clean = []
        for i, a in enumerate(self.free):
            ax, ay, aw, ah = a
            keep = True
            for j, b in enumerate(self.free):
                if i != j:
                    bx, by, bw, bh = b
                    if ax >= bx and ay >= by and ax + aw <= bx + bw and ay + ah <= by + bh:
                        keep = False
                        break
            if keep:
                clean.append(a)
        self.free = clean
    
    def _can_merge(self, r1, r2):
        # Check if rectangles can be merged
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        
        # Check horizontal adjacency (same height and y)
        if y1 == y2 and h1 == h2 and (x1 + w1 == x2 or x2 + w2 == x1):
            return True
            
        # Check vertical adjacency (same width and x)
        if x1 == x2 and w1 == w2 and (y1 + h1 == y2 or y2 + h2 == y1):
            return True
            
        return False
    
    def _merge(self, r1, r2):
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        
        # Horizontal merge
        if y1 == y2 and h1 == h2:
            if x1 + w1 == x2:
                return [x1, y1, w1 + w2, h1]
            else:
                return [x2, y2, w1 + w2, h1]
                
        # Vertical merge
        else:
            if y1 + h1 == y2:
                return [x1, y1, w1, h1 + h2]
            else:
                return [x2, y2, w1, h1 + h2]

# ───── Greedy First-Fit Decreasing (FFD) MaxRects Implementation ─────────────────────
class MaxRectsGreedy:
    """Greedy First-Fit Decreasing implementation of MaxRects algorithm
    
    This class implements FFD heuristics:
    - First-Fit: Place piece in the first suitable position found
    - Bottom-Left preference for compact packing
    - Fast placement with reduced search overhead
    """
    
    def __init__(self, W, H, allowed_polygon_mm=None):
        self.free = [[0,0,W,H]]
        self.used = []
        self.width = W
        self.height = H
        self.allowed_polygon_mm = allowed_polygon_mm
        # FFD-specific optimizations
        self.placement_cache = {}  # Cache for faster lookups
        self.last_placement_y = 0  # Track last placement for bottom-left preference
    
    def insert(self, w, h, shape_type='rect', rot=True, piece_id=-1, shape_name=None):
        """FFD insertion with greedy first-fit strategy"""
        best = None
        is_circle = shape_type == 'circle'
        
        if is_circle:
            rot = False
        
        rotation_angles = [False, True] if rot else [False]
        
        # FFD Strategy: Sort free rectangles by bottom-left preference
        # This encourages compact packing from bottom-left corner
        sorted_free = sorted(enumerate(self.free), key=lambda x: (x[1][1], x[1][0]))  # Sort by y, then x
        
        for fi, fr in sorted_free:
            fx, fy, fw, fh = fr
            
            for r in rotation_angles:
                rw, rh = (h, w) if r else (w, h)
                
                # Quick fit check
                if is_circle:
                    radius = rw / 2
                    fits_in_free = (fx + rw <= fx + fw and fy + rh <= fy + fh and
                                  fx + radius >= fx and fy + radius >= fy)
                else:
                    fits_in_free = (rw <= fw and rh <= fh)
                
                if not fits_in_free:
                    continue
                
                # Validate against polygon constraint
                if self.allowed_polygon_mm is not None:
                    if not piece_fits_in_local_contour_mm(fx, fy, rw, rh, shape_type, self.allowed_polygon_mm):
                        continue
                
                # FFD Scoring: Prioritize first-fit with bottom-left preference
                # Lower scores are better
                bottom_left_score = fy * 1000 + fx  # Strongly prefer bottom-left
                area_waste = fw * fh - rw * rh  # Minimize waste
                
                # Bonus for staying close to previous placements (encourage compactness)
                y_proximity_bonus = abs(fy - self.last_placement_y) * 10
                
                ffd_score = bottom_left_score + area_waste * 0.1 + y_proximity_bonus
                
                # Check for overlaps (simplified for FFD speed)
                overlap = self._quick_overlap_check(fx, fy, rw, rh, shape_type)
                
                if not overlap:
                    # FFD: Take the first valid position (greedy approach)
                    best = (ffd_score, fi, fr, rw, rh, r)
                    break  # First-fit: don't search further
            
            if best is not None:
                break  # First-fit: stop at first valid rectangle
        
        if best is None:
            return False
        
        _, fi, (fx, fy, fw, fh), rw, rh, rotated = best
        
        # Final overlap validation
        if self._quick_overlap_check(fx, fy, rw, rh, shape_type):
            return False
        
        # Place the piece
        self.used.append([fx, fy, rw, rh, rotated, shape_type, piece_id, shape_name])
        self.last_placement_y = fy  # Update for next placement
        
        # Update free rectangles (simplified splitting for FFD)
        del self.free[fi]
        
        # Simple splitting strategy for FFD
        if fw - rw > 0:
            self.free.append([fx + rw, fy, fw - rw, fh])
        if fh - rh > 0:
            self.free.append([fx, fy + rh, fw, fh - rh])
        
        self._quick_prune()
        return True
    
    def _quick_overlap_check(self, fx, fy, rw, rh, shape_type):
        """Fast overlap checking for FFD"""
        is_circle = shape_type == 'circle'
        
        for u in self.used:
            ux, uy, uw, uh, _, ushape = u[:6]
            if ushape == 'circle' and is_circle:
                # Circle-to-circle
                c1 = (fx + rw/2, fy + rh/2)
                c2 = (ux + uw/2, uy + uh/2)
                if math.hypot(c1[0]-c2[0], c1[1]-c2[1]) < (rw/2 + uw/2):
                    return True
            elif ushape == 'circle':
                # Rectangle-to-circle
                cx, cy = (ux + uw/2, uy + uh/2)
                closest_x = max(fx, min(cx, fx + rw))
                closest_y = max(fy, min(cy, fy + rh))
                if math.hypot(closest_x-cx, closest_y-cy) < (uw/2):
                    return True
            elif is_circle:
                # Circle-to-rectangle
                cx, cy = (fx + rw/2, fy + rh/2)
                closest_x = max(ux, min(cx, ux + uw))
                closest_y = max(uy, min(cy, uy + uh))
                if math.hypot(closest_x-cx, closest_y-cy) < (rw/2):
                    return True
            else:
                # Rectangle-to-rectangle
                if (fx < ux + uw and fx + rw > ux and 
                    fy < uy + uh and fy + rh > uy):
                    return True
        return False
    
    def _quick_prune(self):
        """Simplified pruning for FFD performance"""
        # Remove contained rectangles only (skip merging for speed)
        clean = []
        for i, a in enumerate(self.free):
            ax, ay, aw, ah = a
            keep = True
            for j, b in enumerate(self.free):
                if i != j:
                    bx, by, bw, bh = b
                    if ax >= bx and ay >= by and ax + aw <= bx + bw and ay + ah <= by + bh:
                        keep = False
                        break
            if keep:
                clean.append(a)
        self.free = clean

# ───── Polygon grid-based packer for arbitrary shapes ─────────────────────
class PolygonGridPacker:
    def __init__(self, width_mm: float, height_mm: float, polygon_mm=None, cell_mm: float = 5.0, seed: Optional[int] = None, allowed_grid: Optional[np.ndarray] = None, initial_used: Optional[list] = None):
        self.width_mm = float(width_mm)
        self.height_mm = float(height_mm)
        self.cell_mm = float(max(1.0, cell_mm))
        self.used = []
        self.rng = np.random.RandomState(seed if seed is not None else 0)
        self.allowed_polygon_mm = polygon_mm  # Store polygon for NFP algorithm
        # Use cached allowed grid if provided; else rasterize polygon
        if allowed_grid is not None:
            self.allowed = (allowed_grid.astype(np.uint8)).copy()
            self.grid_shape = self.allowed.shape
            # Validate that allowed grid has enough coverage
            total_cells = self.grid_shape[0] * self.grid_shape[1]
            filled_cells = np.count_nonzero(self.allowed)
            if filled_cells < total_cells * 0.1:
                # Grid too sparse, fill entirely
                self.allowed[:, :] = 1
        else:
            grid_w = int(math.ceil(self.width_mm / self.cell_mm))
            grid_h = int(math.ceil(self.height_mm / self.cell_mm))
            self.grid_shape = (grid_h, grid_w)
            self.allowed = np.zeros(self.grid_shape, dtype=np.uint8)
            if polygon_mm is not None and len(polygon_mm) >= 3:
                pts = np.array([[p[0] / self.cell_mm, p[1] / self.cell_mm] for p in polygon_mm], dtype=np.float32)
                pts = np.round(pts).astype(np.int32)
                cv2.fillPoly(self.allowed, [pts], 1)
                # Validate that polygon filled enough area
                total_cells = grid_h * grid_w
                filled_cells = np.count_nonzero(self.allowed)
                if filled_cells < total_cells * 0.1:
                    # Polygon fill too sparse, use full rectangle
                    self.allowed[:, :] = 1
            else:
                self.allowed[:, :] = 1
        self.occ = np.zeros_like(self.allowed, dtype=np.uint8)
        # Pre-fill occupancy with already placed items
        if initial_used:
            for u in initial_used:
                ux, uy, uw, uh, urot, ushape = u[:6]
                u_piece_id = u[6] if len(u) > 6 else -1  # Preserve piece_id
                gx = int(round(ux / self.cell_mm))
                gy = int(round(uy / self.cell_mm))
                gw = int(max(1, math.ceil(uw / self.cell_mm)))
                gh = int(max(1, math.ceil(uh / self.cell_mm)))
                gx = max(0, min(gx, self.occ.shape[1] - 1))
                gy = max(0, min(gy, self.occ.shape[0] - 1))
                gx2 = max(gx + gw, gx + 1)
                gy2 = max(gy + gh, gy + 1)
                gx2 = min(gx2, self.occ.shape[1])
                gy2 = min(gy2, self.occ.shape[0])
                if ushape == 'circle':
                    rad = max(1, gw // 2)
                    disk = np.zeros((gy2 - gy, gx2 - gx), dtype=np.uint8)
                    cx = min(rad, disk.shape[1]-1)
                    cy = min(rad, disk.shape[0]-1)
                    cv2.circle(disk, (cx, cy), min(rad, cx, cy), 1, -1)
                    self.occ[gy:gy2, gx:gx2][disk == 1] = 1
                else:
                    self.occ[gy:gy2, gx:gx2] = 1
                # Also mirror these into used to keep reporting consistent - PRESERVE piece_id and shape_name
                u_shape_name = u[7] if len(u) > 7 else None
                self.used.append([ux, uy, uw, uh, urot, ushape, u_piece_id, u_shape_name])

    def reset(self):
        self.used = []
        self.occ[:, :] = 0

    def _fits_mask(self, x: int, y: int, w: int, h: int, footprint: np.ndarray) -> bool:
        # Enhanced boundary validation to prevent array indexing errors
        if x < 0 or y < 0 or x + w > self.grid_shape[1] or y + h > self.grid_shape[0]:
            return False
        
        # Additional safety check for footprint dimensions
        if footprint.shape[0] != h or footprint.shape[1] != w:
            return False
            
        try:
            region_allowed = self.allowed[y:y+h, x:x+w]
            region_occ = self.occ[y:y+h, x:x+w]
            return np.all(region_allowed[footprint == 1] == 1) and np.all(region_occ[footprint == 1] == 0)
        except IndexError:
            # Fallback for any unexpected indexing issues
            return False

    def _contact_score(self, x: int, y: int, footprint: np.ndarray) -> float:
        """Ultra-fast contact scoring - simplified for speed"""
        score = 0.0
        
        # Corner bonus (highest priority)
        if x == 0 and y == 0:
            return 50.0
        elif x == 0:
            score = 25.0
        elif y == 0:
            score = 20.0
        
        # Simple adjacency check (skip detailed scanning)
        if len(self.used) > 0:
            score += 5.0
        
        return score

    def _use_nfp_placement(self, w_mm: float, h_mm: float, shape_type: str, piece_id: int = -1) -> bool:
        """Use No-Fit Polygon algorithm for precise placement"""
        if not hasattr(self, 'allowed_polygon_mm') or not self.allowed_polygon_mm or len(self.allowed_polygon_mm) < 3:
            return False  # NFP requires container polygon
        
        try:
            # Convert container polygon to numpy array
            container_poly = _nfp_calculator.polygon_to_points(self.allowed_polygon_mm)
            
            # Convert existing placements to NFP format
            placed_objects = []
            for u in self.used:
                px, py, pw, ph, rotated, pshape = u[:6]
                placed_objects.append((px, py, pw, ph, pshape))
            
            # Check if we already tried this exact piece configuration to avoid infinite loops
            piece_signature = (w_mm, h_mm, shape_type, len(placed_objects))
            if not hasattr(self, '_nfp_attempts'):
                self._nfp_attempts = set()
            
            if piece_signature in self._nfp_attempts:
                print(f"[DEBUG] Skipping NFP for {shape_type} {w_mm}x{h_mm} - already attempted with {len(placed_objects)} objects")
                return False
            
            self._nfp_attempts.add(piece_signature)
            
            # Find valid positions using NFP
            valid_positions = _nfp_calculator.find_valid_positions(
                container_poly, placed_objects, shape_type, w_mm, h_mm, 
                grid_resolution=self.cell_mm * 2  # Use 2x cell resolution for NFP
            )
            
            if not valid_positions:
                return False
            
            # Try the best positions
            for px_mm, py_mm, score in valid_positions[:10]:  # Try top 10 positions
                # Check for duplicate positions to avoid infinite loops
                position_key = (round(px_mm, 1), round(py_mm, 1))
                if not hasattr(self, '_tried_positions'):
                    self._tried_positions = set()
                
                if position_key in self._tried_positions:
                    continue
                
                self._tried_positions.add(position_key)
                
                # Convert to grid coordinates
                gx = int(round(px_mm / self.cell_mm))
                gy = int(round(py_mm / self.cell_mm))
                
                # Create footprint
                if shape_type == 'circle':
                    d_cells = max(2, int(math.ceil(w_mm / self.cell_mm)))
                    r = d_cells // 2
                    size = d_cells
                    footprint = np.zeros((size, size), dtype=np.uint8)
                    cv2.circle(footprint, (r, r), r, 1, -1)
                    fw, fh = size, size
                else:
                    w_cells = max(1, int(math.ceil(w_mm / self.cell_mm)))
                    h_cells = max(1, int(math.ceil(h_mm / self.cell_mm)))
                    footprint = np.ones((h_cells, w_cells), dtype=np.uint8)
                    fw, fh = w_cells, h_cells
                
                # Check if position is valid in grid
                if (gx >= 0 and gy >= 0 and 
                    gx + fw <= self.grid_shape[1] and gy + fh <= self.grid_shape[0]):
                    
                    if self._fits_mask(gx, gy, fw, fh, footprint):
                        # Use exact NFP position (don't slide)
                        if self._place_exact(gx, gy, fw, fh, footprint, w_mm, h_mm, shape_type, False, px_mm, py_mm, piece_id=piece_id):
                            print(f"[INFO] NFP placement successful for {shape_type} {w_mm}x{h_mm} mm at ({px_mm:.1f}, {py_mm:.1f})")
                            # Clear attempt tracking after successful placement
                            if hasattr(self, '_nfp_attempts'):
                                self._nfp_attempts.clear()
                            if hasattr(self, '_tried_positions'):
                                self._tried_positions.clear()
                            return True
            
            return False
            
        except Exception as e:
            print(f"[WARN] NFP placement failed: {e}")
            return False
    
    def _place_exact(self, x: int, y: int, w: int, h: int, footprint: np.ndarray, 
                    piece_w_mm: float, piece_h_mm: float, shape_type: str, rotated: bool,
                    exact_x_mm: float, exact_y_mm: float, piece_id: int = -1, shape_name: str = None) -> bool:
        """Place object at exact millimeter coordinates (for NFP)"""
        # Boundary check
        margin_mm = 0.1
        if (exact_x_mm < -margin_mm or exact_y_mm < -margin_mm or 
            exact_x_mm + piece_w_mm > self.width_mm + margin_mm or 
            exact_y_mm + piece_h_mm > self.height_mm + margin_mm):
            return False
        
        # Additional circle boundary check
        if shape_type == 'circle':
            circle_center_x = exact_x_mm + piece_w_mm / 2.0
            circle_center_y = exact_y_mm + piece_h_mm / 2.0
            radius = piece_w_mm / 2.0
            
            if (circle_center_x - radius < -margin_mm or circle_center_y - radius < -margin_mm or
                circle_center_x + radius > self.width_mm + margin_mm or circle_center_y + radius > self.height_mm + margin_mm):
                return False
        
        # Mark occupancy and add to used list
        self.occ[y:y+h, x:x+w][footprint == 1] = 1
        self.used.append([exact_x_mm, exact_y_mm, piece_w_mm, piece_h_mm, rotated, shape_type, piece_id, shape_name])
        return True

    def _place(self, x: int, y: int, w: int, h: int, footprint: np.ndarray, piece_w_mm: float, piece_h_mm: float, shape_type: str, rotated: bool, piece_id: int = -1, shape_name: str = None, angle_deg: float = 0):
        # Additional safety check before placing
        px = x * self.cell_mm
        py = y * self.cell_mm
        
        # Different margins for different shapes
        if shape_type == 'circle':
            margin_mm = 2.0  # Larger margin for circles to stay away from edges
        else:
            margin_mm = 1.0  # Standard margin for rectangles
        
        # Check if piece fits within container boundaries with margin
        if (px < margin_mm or py < margin_mm or 
            px + piece_w_mm > self.width_mm - margin_mm or 
            py + piece_h_mm > self.height_mm - margin_mm):
            return False  # REJECT placement - too close to boundary
        
        # For circles, additional boundary validation with margin
        if shape_type == 'circle':
            circle_center_x = px + piece_w_mm / 2.0
            circle_center_y = py + piece_h_mm / 2.0
            radius = piece_w_mm / 2.0
            
            # Ensure circle stays away from all edges
            if (circle_center_x - radius < margin_mm or 
                circle_center_y - radius < margin_mm or
                circle_center_x + radius > self.width_mm - margin_mm or 
                circle_center_y + radius > self.height_mm - margin_mm):
                return False  # REJECT circle placement
        
        self.occ[y:y+h, x:x+w][footprint == 1] = 1
        # Save 9 elements: [x, y, w, h, rotated, shape_type, piece_id, shape_name, angle_deg]
        self.used.append([px, py, piece_w_mm, piece_h_mm, rotated, shape_type, piece_id, shape_name, angle_deg])
        return True

    def _slide_towards_origin(self, x: int, y: int, w: int, h: int, footprint: np.ndarray) -> tuple[int, int]:
        # Greedily slide the footprint up and left while it fits. This compacts the layout.
        moved = True
        while moved:
            moved = False
            if x > 0 and self._fits_mask(x - 1, y, w, h, footprint):
                x -= 1
                moved = True
                continue
            if y > 0 and self._fits_mask(x, y - 1, w, h, footprint):
                y -= 1
                moved = True
        return x, y

    def _find_positions(self, footprint: np.ndarray, max_positions: int = 15) -> list[tuple[int, int, float]]:
        """Optimized Bottom-Left-Fill position finding.
        
        Uses efficient strategies:
        1. BLF scan with early exit
        2. Adjacent placement to existing pieces
        3. Corner positions
        """
        fh, fw = footprint.shape
        positions = []
        visited = set()
        
        # Adaptive step size - larger steps for faster search
        min_dim = min(fw, fh)
        if min_dim > 50:  # Large pieces - coarse search
            step_size = max(3, min_dim // 10)
        elif min_dim > 20:  # Medium pieces
            step_size = max(2, min_dim // 8)
        else:  # Small pieces
            step_size = max(1, min_dim // 5)
        
        step_x = max(2, step_size)
        step_y = max(2, step_size)
        
        # 1. TRUE BOTTOM-LEFT SCAN: Find the lowest Y position, then leftmost X
        # This is the core BLF strategy - prioritizes bottom-left placement
        best_blf_pos = None
        best_blf_y = float('inf')
        best_blf_x = float('inf')
        
        for y in range(0, self.allowed.shape[0] - fh + 1, step_y):
            for x in range(0, self.allowed.shape[1] - fw + 1, step_x):
                if self._can_place(x, y, footprint):
                    # BLF priority: lowest Y first, then lowest X
                    if y < best_blf_y or (y == best_blf_y and x < best_blf_x):
                        best_blf_y = y
                        best_blf_x = x
                        best_blf_pos = (x, y)
                    break  # Found position at this Y level, try next Y
            # Early exit if we found a position at Y=0
            if best_blf_y == 0:
                break
        
        if best_blf_pos:
            x, y = best_blf_pos
            # Give BLF position highest score
            score = 500 + self._contact_score(x, y, footprint)
            positions.append((x, y, score))
            visited.add((x, y))
        
        # 2. CORNER POSITIONS (high stability)
        corner_positions = [
            (0, 0),  # Origin - highest priority
            (0, max(0, self.allowed.shape[0] - fh)),  # Bottom-left
            (max(0, self.allowed.shape[1] - fw), 0),  # Top-right
        ]
        
        for x, y in corner_positions:
            if (x, y) not in visited and self._can_place(x, y, footprint):
                # Origin gets highest corner score
                corner_bonus = 300 if (x == 0 and y == 0) else 200
                score = corner_bonus + self._contact_score(x, y, footprint)
                positions.append((x, y, score))
                visited.add((x, y))
        
        # 3. ADJACENT TO EXISTING PIECES (tight packing strategy)
        if len(self.used) > 0:
            # Sort existing pieces by position for systematic adjacency check
            sorted_used = sorted(self.used, key=lambda u: (u[1], u[0]))  # Sort by Y, then X
            
            for u in sorted_used:
                used_x, used_y, used_w, used_h = u[:4]
                grid_x = int(used_x / self.cell_mm)
                grid_y = int(used_y / self.cell_mm)
                grid_w = int(math.ceil(used_w / self.cell_mm))
                grid_h = int(math.ceil(used_h / self.cell_mm))
                
                # Try positions adjacent to existing piece - prioritize right and below
                adjacent_positions = [
                    (grid_x + grid_w, grid_y, 180),      # Right (same row) - high priority
                    (grid_x, grid_y + grid_h, 170),      # Below - high priority
                    (grid_x + grid_w, grid_y + grid_h, 150),  # Diagonal bottom-right
                    (0, grid_y + grid_h, 160),          # Start of next row
                    (grid_x - fw, grid_y, 100),          # Left
                    (grid_x, grid_y - fh, 100),          # Above
                ]
                
                for ax, ay, adj_bonus in adjacent_positions:
                    if (ax, ay) in visited:
                        continue
                    if (0 <= ax <= self.allowed.shape[1] - fw and 
                        0 <= ay <= self.allowed.shape[0] - fh):
                        if self._can_place(ax, ay, footprint):
                            score = adj_bonus + self._contact_score(ax, ay, footprint)
                            positions.append((ax, ay, score))
                            visited.add((ax, ay))
        
        # 4. SKYLINE POSITIONS: Find positions at the "skyline" of placed pieces
        if len(self.used) > 0:
            # Build skyline: for each X column, find the highest Y occupied
            skyline = np.zeros(self.allowed.shape[1], dtype=np.int32)
            for u in self.used:
                ux, uy, uw, uh = u[:4]
                gx_start = int(ux / self.cell_mm)
                gx_end = min(int(math.ceil((ux + uw) / self.cell_mm)), self.allowed.shape[1])
                gy_end = int(math.ceil((uy + uh) / self.cell_mm))
                for gx in range(gx_start, gx_end):
                    skyline[gx] = max(skyline[gx], gy_end)
            
            # Try placing at skyline positions
            for x in range(0, self.allowed.shape[1] - fw + 1, step_x):
                y = max(skyline[x:x+fw]) if x + fw <= len(skyline) else 0
                if y < self.allowed.shape[0] - fh and (x, y) not in visited:
                    if self._can_place(x, y, footprint):
                        score = 120 + self._contact_score(x, y, footprint)
                        positions.append((x, y, score))
                        visited.add((x, y))
        
        # 5. QUICK SCAN for remaining positions (reduced iterations)
        scan_limit = max_positions * 2
        for y in range(0, self.allowed.shape[0] - fh + 1, step_y * 2):
            if len(positions) >= scan_limit:
                break
            for x in range(0, self.allowed.shape[1] - fw + 1, step_x * 2):
                if (x, y) in visited:
                    continue
                if self._can_place(x, y, footprint):
                    # Simplified scoring: prefer bottom-left
                    pos_score = 100 - y * 0.5 - x * 0.2
                    positions.append((x, y, pos_score))
                    visited.add((x, y))
                    
                    if len(positions) >= scan_limit:
                        break
        
        # Sort by score (descending) and return top candidates
        positions.sort(key=lambda p: p[2], reverse=True)
        return positions[:max_positions]

    def _can_place(self, x: int, y: int, footprint: np.ndarray) -> bool:
        """Check if footprint can be placed at position (x, y)"""
        fh, fw = footprint.shape
        
        # Boundary check
        if (x < 0 or y < 0 or 
            x + fw > self.allowed.shape[1] or 
            y + fh > self.allowed.shape[0]):
            return False
        
        # Check against allowed region and occupancy
        region_allowed = self.allowed[y:y+fh, x:x+fw]
        region_occupied = self.occ[y:y+fh, x:x+fw]
        
        # All footprint cells must be in allowed region and not occupied
        footprint_mask = (footprint == 1)
        if not np.all(region_allowed[footprint_mask] == 1):
            return False
        if np.any(region_occupied[footprint_mask] == 1):
            return False
        
        return True

    def _create_poly_footprint(self, w_mm: float, h_mm: float, shape_name: str = None, spacing_cells: int = 0) -> list:
        """Create polygon footprint from library shape data with multiple rotation options.
        
        Returns list of (footprint, is_rotated, actual_w_mm, actual_h_mm, angle_deg) candidates.
        Uses FULL BOUNDING BOX with spacing to guarantee no overlaps.
        """
        candidates = []
        poly_data = None
        
        # Try to find polygon in library by name first
        if shape_name and shape_name in LIB_SHAPE_BY_NAME:
            poly_data = LIB_SHAPE_BY_NAME[shape_name]
        
        # Fallback: search by dimensions with tolerance
        if not poly_data and LIB_SHAPE_BY_NAME:
            tolerance = 10.0
            for name, pdata in LIB_SHAPE_BY_NAME.items():
                bbox_w = pdata.get('bbox_w', 0)
                bbox_h = pdata.get('bbox_h', 0)
                if ((abs(bbox_w - w_mm) <= tolerance and abs(bbox_h - h_mm) <= tolerance) or
                    (abs(bbox_h - w_mm) <= tolerance and abs(bbox_w - h_mm) <= tolerance)):
                    poly_data = pdata
                    break
        
        if poly_data and ('pts_mm' in poly_data or 'pts' in poly_data):
            try:
                # Try pts_mm first (preferred), then pts (legacy)
                pts_key = 'pts_mm' if 'pts_mm' in poly_data else 'pts'
                pts_original = np.array(poly_data[pts_key], dtype=np.float32)
                
                # Try rotation angles: 0, 90, 180, 270 (simpler, more reliable)
                rotation_angles = [0, 90, 180, 270]
                
                for angle_deg in rotation_angles:
                    # Rotate points around centroid
                    if angle_deg == 0:
                        pts_rot = pts_original.copy()
                    else:
                        centroid = pts_original.mean(axis=0)
                        angle_rad = math.radians(angle_deg)
                        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
                        pts_centered = pts_original - centroid
                        pts_rot = np.column_stack([
                            pts_centered[:, 0] * cos_a - pts_centered[:, 1] * sin_a,
                            pts_centered[:, 0] * sin_a + pts_centered[:, 1] * cos_a
                        ]) + centroid
                    
                    # Normalize to 0,0 origin
                    min_xy = pts_rot.min(axis=0)
                    pts_norm = pts_rot - min_xy
                    max_xy = pts_norm.max(axis=0)
                    rot_w_mm = max(1e-6, float(max_xy[0]))
                    rot_h_mm = max(1e-6, float(max_xy[1]))
                    
                    # Add spacing to dimensions
                    spacing_mm = spacing_cells * self.cell_mm
                    total_w_mm = rot_w_mm + spacing_mm * 2
                    total_h_mm = rot_h_mm + spacing_mm * 2
                    
                    # Create footprint grid
                    w_cells = max(1, int(math.ceil(total_w_mm / self.cell_mm)))
                    h_cells = max(1, int(math.ceil(total_h_mm / self.cell_mm)))
                    
                    # Create polygon-shaped footprint (not just bbox)
                    # Shift polygon by spacing to center it in footprint
                    pts_shifted = pts_norm + spacing_mm
                    pts_cells = pts_shifted / self.cell_mm
                    pts_int = np.round(pts_cells).astype(np.int32)
                    
                    footprint = np.zeros((h_cells, w_cells), dtype=np.uint8)
                    cv2.fillPoly(footprint, [pts_int], 1)
                    
                    # Dilate to add safety margin around polygon
                    if spacing_cells > 0:
                        kernel = np.ones((spacing_cells * 2 + 1, spacing_cells * 2 + 1), dtype=np.uint8)
                        footprint = cv2.dilate(footprint, kernel, iterations=1)
                        # Clip to original size
                        footprint = footprint[:h_cells, :w_cells]
                    
                    is_rotated = (angle_deg != 0)
                    # Store actual polygon size (without spacing) for rendering
                    candidates.append((footprint, is_rotated, rot_w_mm, rot_h_mm, angle_deg))
                
                # Sort by bounding box area (smaller is better)
                candidates.sort(key=lambda c: c[2] * c[3])
                
                # Keep top 4 candidates
                candidates = candidates[:4]
                
            except Exception as e:
                print(f"[WARN] Failed to create poly footprint: {e}")
        
        # Fallback to rectangle if no polygon data
        if not candidates:
            # Add spacing to dimensions
            spacing_mm = spacing_cells * self.cell_mm
            total_w = w_mm + spacing_mm * 2
            total_h = h_mm + spacing_mm * 2
            w_cells = max(1, int(math.ceil(total_w / self.cell_mm)))
            h_cells = max(1, int(math.ceil(total_h / self.cell_mm)))
            rect_a = np.ones((h_cells, w_cells), dtype=np.uint8)
            candidates.append((rect_a, False, w_mm, h_mm, 0))  # 0 degrees
            if w_cells != h_cells:
                rect_b = np.ones((w_cells, h_cells), dtype=np.uint8)
                candidates.append((rect_b, True, h_mm, w_mm, 90))  # 90 degrees
        
        return candidates

    def insert(self, w_mm: float, h_mm: float, shape_type: str = 'rect', rot: bool = True, piece_id: int = -1, shape_name: str = None) -> bool:
        # Grid-based placement algorithm
        
        # Safety margin to keep pieces away from boundaries
        # Larger margin for circles since they need more clearance
        if shape_type == 'circle':
            boundary_margin_mm = 2.0  # 2mm margin for circles
            spacing_mm = 0.5  # Slightly more spacing for circles
        elif shape_type == 'poly' or shape_type == 'library':
            boundary_margin_mm = 1.0
            spacing_mm = 3.0  # 3mm spacing for polygons
        else:
            boundary_margin_mm = 1.0  # 1mm margin for rectangles 
            spacing_mm = 0.3  # 0.3mm spacing between pieces
        
        spacing_cells = max(0, int(math.ceil(spacing_mm / self.cell_mm)))
        
        # Precompute footprint(s) in grid cells with spacing
        if shape_type == 'circle':
            d_cells = max(2, int(math.ceil(w_mm / self.cell_mm))) + spacing_cells
            r = d_cells // 2
            size = d_cells
            footprint = np.zeros((size, size), dtype=np.uint8)
            cv2.circle(footprint, (r, r), r, 1, -1)
            # For circles, ensure we have the exact diameter
            candidates = [(footprint, False, w_mm, w_mm, 0)]  # Use same width/height for circles, 0 degrees
            
            # Check if circle diameter fits within container
            if (w_mm > self.width_mm or w_mm > self.height_mm):
                pass  # Will be rejected later by boundary check
        elif shape_type == 'poly' or shape_type == 'library':
            # Use polygon-aware footprint creation
            poly_spacing_cells = max(1, int(math.ceil(3.0 / self.cell_mm)))
            candidates = self._create_poly_footprint(w_mm, h_mm, shape_name, poly_spacing_cells)
        else:
            w_cells = max(1, int(math.ceil(w_mm / self.cell_mm))) + spacing_cells
            h_cells = max(1, int(math.ceil(h_mm / self.cell_mm))) + spacing_cells
            rect_a = np.ones((h_cells, w_cells), dtype=np.uint8)
            candidates = [(rect_a, False, w_mm, h_mm, 0)]  # 0 degrees
            if rot and w_cells != h_cells:
                rect_b = np.ones((w_cells, h_cells), dtype=np.uint8)
                candidates.append((rect_b, True, h_mm, w_mm, 90))  # 90 degrees

        # Fast placement - reduced position candidates
        # Candidates now have 5 elements: (footprint, rotated, pw, ph, angle_deg)
        for candidate in candidates:
            if len(candidate) == 5:
                footprint, rotated, pw, ph, angle_deg = candidate
            else:
                footprint, rotated, pw, ph = candidate
                angle_deg = 0
            fh, fw = footprint.shape
            
            # Use fewer position candidates for speed
            positions = self._find_positions(footprint, max_positions=20)
            
            if not positions:
                continue
            
            # Try positions in order of decreasing score
            for x, y, score in positions:
                # Validate boundaries in millimeters
                px_mm = x * self.cell_mm
                py_mm = y * self.cell_mm
                if (px_mm + pw > self.width_mm or py_mm + ph > self.height_mm):
                    continue
                
                # Try compacted placement first (skip if origin is already close)
                if x > 2 or y > 2:
                    bx, by = self._slide_towards_origin(x, y, fw, fh, footprint)
                    if (bx >= 0 and by >= 0 and 
                        bx + fw <= self.grid_shape[1] and by + fh <= self.grid_shape[0]):
                        bpx_mm = bx * self.cell_mm
                        bpy_mm = by * self.cell_mm
                        if (bpx_mm + pw <= self.width_mm and bpy_mm + ph <= self.height_mm):
                            if self._fits_mask(bx, by, fw, fh, footprint):
                                if self._place(bx, by, fw, fh, footprint, pw, ph, shape_type, rotated, piece_id=piece_id, shape_name=shape_name, angle_deg=angle_deg):
                                    return True
                
                # Place at original position
                if self._fits_mask(x, y, fw, fh, footprint):
                    if self._place(x, y, fw, fh, footprint, pw, ph, shape_type, rotated, piece_id=piece_id, shape_name=shape_name, angle_deg=angle_deg):
                        return True
            
            # Quick fallback scan with larger steps
            fallback_step = max(2, min(fw, fh) // 6)
            for y in range(0, self.grid_shape[0] - fh + 1, fallback_step):
                for x in range(0, self.grid_shape[1] - fw + 1, fallback_step):
                    px_mm = x * self.cell_mm
                    py_mm = y * self.cell_mm
                    if (px_mm + pw <= self.width_mm and py_mm + ph <= self.height_mm):
                        if self._fits_mask(x, y, fw, fh, footprint):
                            if self._place(x, y, fw, fh, footprint, pw, ph, shape_type, rotated, piece_id=piece_id, shape_name=shape_name, angle_deg=angle_deg):
                                return True
        return False

    def shrink_envelope(self, rounds: int = 1) -> None:
        """Simplified post-compaction: slide items towards origin.
        Reduced iterations for speed.
        """
        if not self.used:
            return
            
        # Single round for speed
        item_indices = list(range(len(self.used)))
        item_indices.sort(key=lambda i: self.used[i][0] + self.used[i][1], reverse=True)
        
        for idx in item_indices:
            px, py, wmm, hmm, rotated, shape = self.used[idx][:6]
            gx = int(round(px / self.cell_mm))
            gy = int(round(py / self.cell_mm))
            gw = int(max(1, math.ceil(wmm / self.cell_mm)))
            gh = int(max(1, math.ceil(hmm / self.cell_mm)))
            
            # Build footprint
            if shape == 'circle':
                size = max(2, int(math.ceil(wmm / self.cell_mm)))
                r = size // 2
                footprint = np.zeros((size, size), dtype=np.uint8)
                cv2.circle(footprint, (r, r), r, 1, -1)
                fw, fh = size, size
            else:
                footprint = np.ones((gh, gw), dtype=np.uint8)
                fw, fh = gw, gh
            
            # Clear current position
            self.occ[gy:gy+fh, gx:gx+fw][footprint == 1] = 0
            
            # Quick slide towards origin (max 10 iterations)
            for _ in range(10):
                moved = False
                if gx > 0 and self._fits_mask(gx - 1, gy, fw, fh, footprint):
                    gx -= 1
                    moved = True
                elif gy > 0 and self._fits_mask(gx, gy - 1, fw, fh, footprint):
                    gy -= 1
                    moved = True
            # Mark occupancy and update position
            self.occ[gy:gy+fh, gx:gx+fw][footprint == 1] = 1
            piece_id = self.used[idx][6] if len(self.used[idx]) > 6 else -1
            shape_name = self.used[idx][7] if len(self.used[idx]) > 7 else None
            angle_deg = self.used[idx][8] if len(self.used[idx]) > 8 else 0
            self.used[idx] = [gx * self.cell_mm, gy * self.cell_mm, wmm, hmm, rotated, shape, piece_id, shape_name, angle_deg]
    
    def _find_best_gap(self, current_x: int, current_y: int, fw: int, fh: int, footprint: np.ndarray) -> Optional[tuple[int, int]]:
        """Quick gap search - limited radius for speed"""
        search_radius = min(5, max(fw, fh))
        best_pos = None
        best_score = current_x + current_y  # Manhattan distance from origin
        
        for dy in range(-search_radius, search_radius + 1, 2):
            for dx in range(-search_radius, search_radius + 1, 2):
                new_x = current_x + dx
                new_y = current_y + dy
                
                if (new_x < 0 or new_y < 0 or 
                    new_x + fw > self.grid_shape[1] or new_y + fh > self.grid_shape[0]):
                    continue
                
                score = new_x + new_y
                if score < best_score:
                    best_score = score
                    best_pos = (new_x, new_y)
        
        return best_pos
    
    def _find_max_contact_position(self, current_x: int, current_y: int, fw: int, fh: int, 
                                  footprint: np.ndarray, search_radius: int = 2) -> Optional[tuple[int, int]]:
        """Simplified contact search - very limited for speed"""
        return None  # Skip contact optimization for speed


# ───── Bottom-Left-Fill (BLF) Greedy Packer ─────────────────────
class PolygonGridPackerBLF(PolygonGridPacker):
    """Bottom-Left-Fill greedy implementation extending PolygonGridPacker
    
    This class implements BLF heuristics:
    - Bottom-Left preference: Always try to place pieces as low and left as possible
    - Greedy placement: Take the first valid bottom-left position found
    - Compact packing: Minimize gaps and encourage tight arrangements
    """
    
    def __init__(self, width_mm: float, height_mm: float, polygon_mm=None, cell_mm: float = 5.0, seed: Optional[int] = None, allowed_grid: Optional[np.ndarray] = None, initial_used: Optional[list] = None):
        super().__init__(width_mm, height_mm, polygon_mm, cell_mm, seed, allowed_grid, initial_used)
        # BLF-specific tracking
        self.bottom_line = 0  # Track the current "bottom" line for placement
        self.placement_history = []  # Track placement order for BLF optimization
    
    def insert(self, w_mm: float, h_mm: float, shape_type: str = 'rect', rot: bool = True, piece_id: int = -1, shape_name: str = None) -> bool:
        """BLF insertion with bottom-left greedy strategy"""
        fw = max(1, math.ceil(w_mm / self.cell_mm))
        fh = max(1, math.ceil(h_mm / self.cell_mm))
        
        # Create footprint for the shape
        # rotation_options: list of (grid_w, grid_h, footprint, is_rotated, actual_w_mm, actual_h_mm, angle_deg)
        if shape_type == 'circle':
            footprint = self._create_circle_footprint(fw, fh)
            rot = False  # Circles don't rotate
            rotation_options = [(fw, fh, footprint, False, w_mm, w_mm, 0)]
        elif shape_type == 'poly' or shape_type == 'library':
            # Use polygon-aware footprint with rotations from _create_poly_footprint
            spacing_cells = max(1, int(math.ceil(3.0 / self.cell_mm)))
            candidates = self._create_poly_footprint(w_mm, h_mm, shape_name, spacing_cells)
            if candidates:
                rotation_options = []
                for fp, is_rot, pw_mm, ph_mm, angle_deg in candidates:
                    fp_h, fp_w = fp.shape
                    rotation_options.append((fp_w, fp_h, fp, is_rot, pw_mm, ph_mm, angle_deg))
            else:
                footprint = np.ones((fh, fw), dtype=np.uint8)
                rotation_options = [(fw, fh, footprint, False, w_mm, h_mm, 0)]
                if rot and fw != fh:
                    rot_footprint = np.ones((fw, fh), dtype=np.uint8)
                    rotation_options.append((fh, fw, rot_footprint, True, h_mm, w_mm, 90))
        else:
            footprint = np.ones((fh, fw), dtype=np.uint8)
            rotation_options = [(fw, fh, footprint, False, w_mm, h_mm, 0)]
            if rot and shape_type != 'circle' and fw != fh:
                # Add 90-degree rotation
                rot_footprint = np.ones((fw, fh), dtype=np.uint8)
                rotation_options.append((fh, fw, rot_footprint, True, h_mm, w_mm, 90))
        
        best_position = None
        best_score = float('inf')  # Lower is better (y * 10000 + x prioritizes bottom-left)
        
        for rw, rh, rfootprint, is_rotated, actual_w, actual_h, angle_deg in rotation_options:
            # BLF Strategy: Scan from bottom-left, find first valid position
            # Y goes from 0 (bottom) to max (top), X goes from 0 (left) to max (right)
            for y in range(0, self.grid_shape[0] - rh + 1):  # Bottom to top (0 = bottom)
                for x in range(0, self.grid_shape[1] - rw + 1):  # Left to right
                    if self._fits_mask(x, y, rw, rh, rfootprint):
                        # Score: prioritize bottom-left (lower y, then lower x)
                        score = y * 10000 + x
                        if score < best_score:
                            best_score = score
                            best_position = (x, y, rw, rh, rfootprint, is_rotated, actual_w, actual_h, angle_deg)
                        # Found position for this rotation, try next rotation
                        break
                if best_position and best_score == y * 10000:
                    # Already found bottom-left corner, can't do better with this rotation
                    break
        
        if best_position is None:
            return False
        
        x, y, rw, rh, rfootprint, is_rotated, actual_w_mm, actual_h_mm, angle_deg = best_position
        
        # Place the piece
        self.occ[y:y+rh, x:x+rw][rfootprint == 1] = 1
        
        # Convert back to mm coordinates
        x_mm = x * self.cell_mm
        y_mm = y * self.cell_mm
        # Use actual polygon dimensions, not grid cell dimensions
        w_final = actual_w_mm
        h_final = actual_h_mm
        
        # Include shape_name and angle for accurate polygon identification during rendering
        # Format: [x_mm, y_mm, w_mm, h_mm, is_rotated, shape_type, piece_id, shape_name, angle_deg]
        self.used.append([x_mm, y_mm, w_final, h_final, is_rotated, shape_type, piece_id, shape_name, angle_deg])
        
        # Update BLF tracking
        self.bottom_line = max(self.bottom_line, y + rh)
        self.placement_history.append((x, y, rw, rh))
        
        return True
    
    def _calculate_blf_contact_score(self, x: int, y: int, w: int, h: int, footprint: np.ndarray) -> float:
        """Calculate contact score for BLF - prioritize bottom and left contacts"""
        contact_score = 0.0
        
        # Bottom contact (highest priority for BLF)
        if y + h >= self.grid_shape[0]:  # Touching bottom edge
            contact_score += 100.0
        else:
            # Check contact with placed pieces below
            for fx in range(w):
                if (x + fx < self.grid_shape[1] and 
                    y + h < self.grid_shape[0] and 
                    footprint[h-1, fx] == 1 and 
                    self.occ[y + h, x + fx] == 1):
                    contact_score += 50.0
        
        # Left contact (secondary priority for BLF)
        if x == 0:  # Touching left edge
            contact_score += 30.0
        else:
            # Check contact with placed pieces to the left
            for fy in range(h):
                if (y + fy < self.grid_shape[0] and 
                    footprint[fy, 0] == 1 and 
                    self.occ[y + fy, x - 1] == 1):
                    contact_score += 20.0
        
        return contact_score
    
    def _calculate_gap_penalty(self, x: int, y: int, w: int, h: int) -> float:
        """Calculate penalty for creating gaps (BLF prefers compact placement)"""
        gap_penalty = 0.0
        
        # Check for gaps below the piece
        for fx in range(w):
            if x + fx < self.grid_shape[1]:
                # Count empty cells below
                empty_below = 0
                for check_y in range(y + h, min(y + h + 3, self.grid_shape[0])):
                    if self.occ[check_y, x + fx] == 0:
                        empty_below += 1
                    else:
                        break
                gap_penalty += empty_below * 5.0
        
        # Check for gaps to the left of the piece
        for fy in range(h):
            if y + fy < self.grid_shape[0]:
                # Count empty cells to the left
                empty_left = 0
                for check_x in range(max(0, x - 3), x):
                    if self.occ[y + fy, check_x] == 0:
                        empty_left += 1
                    else:
                        break
                gap_penalty += empty_left * 3.0
        
        return gap_penalty
    
    def _create_circle_footprint(self, w: int, h: int) -> np.ndarray:
        """Create circular footprint for BLF circle placement"""
        footprint = np.zeros((h, w), dtype=np.uint8)
        center_x, center_y = w // 2, h // 2
        radius = min(center_x, center_y)
        
        for y in range(h):
            for x in range(w):
                if math.hypot(x - center_x, y - center_y) <= radius:
                    footprint[y, x] = 1
        
        return footprint


def calculate_adaptive_cell_size(pieces: list, sheet_width_mm: float, sheet_height_mm: float) -> float:
    """Calculate adaptive grid cell size based on piece sizes and sheet dimensions.
    
    Improved algorithm that balances precision and performance:
    - Smaller cells for small pieces (better accuracy)
    - Larger cells for uniform large pieces (faster computation)
    - Dynamic adjustment based on piece size distribution
    """
    if not pieces:
        return PACKING_CELL_MM
    
    # Extract piece dimensions
    piece_sizes = []
    for piece in pieces:
        if len(piece) >= 2:
            w, h = piece[0], piece[1]
            piece_sizes.extend([w, h])
    
    if not piece_sizes:
        return PACKING_CELL_MM
    
    # Calculate statistics
    min_size = min(piece_sizes)
    max_size = max(piece_sizes)
    avg_size = sum(piece_sizes) / len(piece_sizes)
    median_size = sorted(piece_sizes)[len(piece_sizes) // 2]
    
    # Base cell size on smallest dimension for precision
    # Use 1/25 to 1/30 of the smallest piece dimension for better accuracy
    precision_based_cell = max(MIN_PACKING_CELL_MM, min_size / 30.0)
    
    # Consider sheet size - balance precision vs performance
    sheet_area = sheet_width_mm * sheet_height_mm
    total_pieces_area = sum(p[0] * p[1] for p in pieces if len(p) >= 2)
    density_ratio = total_pieces_area / sheet_area if sheet_area > 0 else 0.5
    
    # Adjust based on packing density expectations
    if density_ratio > 0.7:  # High density - need more precision
        size_factor = 0.85
    elif density_ratio < 0.3:  # Low density - can use larger cells
        size_factor = 1.15
    else:
        size_factor = 1.0
    
    # Adaptive cell size based on piece size distribution
    size_variation = max_size / min_size if min_size > 0 else 1.0
    
    if size_variation > 5:  # High size variation
        # Use smaller cells for better precision with mixed sizes
        # But also consider median size to avoid extreme small values
        adaptive_cell = max(precision_based_cell, median_size / 25.0) * 0.75
    elif size_variation < 2:  # Uniform sizes
        # Can use larger cells for better performance
        adaptive_cell = precision_based_cell * 1.2
    else:  # Moderate variation
        adaptive_cell = precision_based_cell
    
    # Apply density factor
    adaptive_cell *= size_factor
    
    # Additional optimization: for very small pieces, ensure minimum precision
    if min_size < 20:  # Pieces smaller than 20mm
        adaptive_cell = min(adaptive_cell, MIN_PACKING_CELL_MM * 1.5)
    
    # Clamp to reasonable bounds
    adaptive_cell = max(MIN_PACKING_CELL_MM, min(MAX_PACKING_CELL_MM, adaptive_cell))
    
    # Round to 2 decimal places for consistency
    return round(adaptive_cell, 2)

# ───── Cached helper functions for performance ──────────────
def calculate_piece_area_fast(w, h, is_circle):
    """Fast area calculation with optional JIT compilation."""
    if is_circle:
        return math.pi * (w / 2.0) * (h / 2.0)
    else:
        return w * h

def calculate_piece_perimeter_fast(w, h, is_circle):
    """Fast perimeter calculation with optional JIT compilation."""
    if is_circle:
        return math.pi * w
    else:
        return 2.0 * (w + h)

# Apply JIT if available
if NUMBA_AVAILABLE:
    try:
        calculate_piece_area_fast = jit(nopython=True, cache=True)(calculate_piece_area_fast)
        calculate_piece_perimeter_fast = jit(nopython=True, cache=True)(calculate_piece_perimeter_fast)
    except Exception:
        pass  # If JIT fails, use regular Python

@lru_cache(maxsize=2048)
def get_piece_area(w, h, shape_type):
    """Cached area calculation for pieces."""
    is_circle = (shape_type == 'circle')
    return calculate_piece_area_fast(w, h, is_circle)

@lru_cache(maxsize=2048)
def get_piece_perimeter(w, h, shape_type):
    """Cached perimeter calculation for pieces."""
    is_circle = (shape_type == 'circle')
    return calculate_piece_perimeter_fast(w, h, is_circle)

# ───── Filter for oversized pieces ──────────────────────────
def split_cuts_to_fit(sheet_w, sheet_h, lst):
    ok, big = [], []
    for cut in lst:
        if len(cut) >= 3:  # New format with shape info (and optional ID)
            w, h, shape = cut[:3]
            # For circle, check if diameter fits in sheet
            if shape == "circle":
                # For circles, diameter must fit in both dimensions
                if w <= sheet_w and w <= sheet_h:
                    ok.append(cut)
                else:
                    big.append(cut)
            else:  # Rectangle
                if (w<=sheet_w and h<=sheet_h) or (h<=sheet_w and w<=sheet_h):
                    ok.append(cut)
                else:
                    big.append(cut)
        else:  # Backward compatibility with old format
            w, h = cut
            if (w<=sheet_w and h<=sheet_h) or (h<=sheet_w and w<=sheet_h):
                ok.append(cut + ("rect",))  # Assume it's a rectangle
            else:
                big.append(cut + ("rect",))
    # Sort pieces by area for better packing
    ok.sort(key=lambda piece: piece[0]*piece[1], reverse=True)
    return ok, big

# ───── Check if point is inside polygon ────────────────────
def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

# ───── No-Fit Polygon (NFP) Algorithm for Precise Placement ────────────────────
class NFPCalculator:
    """No-Fit Polygon calculator for precise object placement without overlaps"""
    
    def __init__(self, tolerance_mm: float = 0.1):
        self.tolerance = tolerance_mm
    
    def polygon_to_points(self, polygon_mm: list) -> np.ndarray:
        """Convert polygon to numpy array of points"""
        if not polygon_mm or len(polygon_mm) < 3:
            return np.array([])
        return np.array(polygon_mm, dtype=np.float64)
    
    def rect_to_polygon(self, x: float, y: float, w: float, h: float) -> np.ndarray:
        """Convert rectangle to polygon points (counter-clockwise)"""
        return np.array([
            [x, y],           # top-left
            [x + w, y],       # top-right  
            [x + w, y + h],   # bottom-right
            [x, y + h]        # bottom-left
        ], dtype=np.float64)
    
    def circle_to_polygon(self, cx: float, cy: float, radius: float, segments: int = 16) -> np.ndarray:
        """Convert circle to polygon approximation"""
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        points = np.column_stack([
            cx + radius * np.cos(angles),
            cy + radius * np.sin(angles)
        ])
        return points
    
    def translate_polygon(self, polygon: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """Translate polygon by offset"""
        if len(polygon) == 0:
            return polygon
        return polygon + np.array([dx, dy])
    
    def minkowski_sum_convex(self, poly_a: np.ndarray, poly_b: np.ndarray) -> np.ndarray:
        """Compute Minkowski sum for convex polygons (simplified NFP)"""
        if len(poly_a) == 0 or len(poly_b) == 0:
            return np.array([])
        
        # For each vertex of A, add all vertices of B
        result_points = []
        for pt_a in poly_a:
            for pt_b in poly_b:
                result_points.append(pt_a + pt_b)
        
        if not result_points:
            return np.array([])
        
        points = np.array(result_points)
        
        # Compute convex hull to get the outer boundary
        try:
            from scipy.spatial import ConvexHull
            if len(points) >= 3:
                hull = ConvexHull(points)
                return points[hull.vertices]
        except ImportError:
            # Fallback: use cv2 convex hull
            hull = cv2.convexHull(points.astype(np.float32))
            return hull.reshape(-1, 2)
        
        return points
    
    def compute_nfp_simple(self, stationary_poly: np.ndarray, moving_poly: np.ndarray) -> np.ndarray:
        """Compute simplified No-Fit Polygon
        
        The NFP represents all positions where the reference point of moving_poly
        can be placed such that moving_poly just touches but doesn't overlap stationary_poly.
        """
        if len(stationary_poly) == 0 or len(moving_poly) == 0:
            return np.array([])
        
        # Reflect moving polygon around origin (for Minkowski difference)
        reflected_moving = -moving_poly
        
        # Compute Minkowski sum of stationary with reflected moving
        nfp = self.minkowski_sum_convex(stationary_poly, reflected_moving)
        
        return nfp
    
    def point_in_nfp(self, point: tuple, nfp: np.ndarray) -> bool:
        """Check if a point is inside the No-Fit Polygon (forbidden region)"""
        if len(nfp) == 0:
            return False
        return point_in_polygon(point, nfp.tolist())
    
    def find_valid_positions(self, container_polygon: np.ndarray, placed_objects: list, 
                           new_object_shape: str, new_object_w: float, new_object_h: float,
                           grid_resolution: float = 2.0) -> list:
        """Find valid positions for placing a new object without overlaps
        
        Args:
            container_polygon: Container boundary as polygon points
            placed_objects: List of (x, y, w, h, shape_type) for already placed objects
            new_object_shape: 'rect' or 'circle'
            new_object_w, new_object_h: Dimensions of new object
            grid_resolution: Grid spacing for position sampling in mm
            
        Returns:
            List of (x, y, score) tuples for valid positions
        """
        if len(container_polygon) == 0:
            return []
        
        # Get container bounds
        min_x, min_y = container_polygon.min(axis=0)
        max_x, max_y = container_polygon.max(axis=0)
        
        # Create new object polygon at origin
        if new_object_shape == 'circle':
            radius = new_object_w / 2.0
            new_obj_poly = self.circle_to_polygon(0, 0, radius)
        else:
            new_obj_poly = self.rect_to_polygon(0, 0, new_object_w, new_object_h)
        
        # Compute NFPs for all placed objects
        nfp_polygons = []
        for obj_x, obj_y, obj_w, obj_h, obj_shape in placed_objects:
            if obj_shape == 'circle':
                obj_radius = obj_w / 2.0
                obj_poly = self.circle_to_polygon(obj_x + obj_radius, obj_y + obj_radius, obj_radius)
            else:
                obj_poly = self.rect_to_polygon(obj_x, obj_y, obj_w, obj_h)
            
            nfp = self.compute_nfp_simple(obj_poly, new_obj_poly)
            if len(nfp) > 0:
                nfp_polygons.append(nfp)
        
        # Sample positions on a grid
        valid_positions = []
        
        # Adjust sampling bounds to account for object size
        sample_min_x = min_x
        sample_max_x = max_x - new_object_w
        sample_min_y = min_y  
        sample_max_y = max_y - new_object_h
        
        if sample_max_x < sample_min_x or sample_max_y < sample_min_y:
            return []  # Object too large for container
        
        # Grid sampling
        x_steps = max(1, int((sample_max_x - sample_min_x) / grid_resolution))
        y_steps = max(1, int((sample_max_y - sample_min_y) / grid_resolution))
        
        for i in range(x_steps + 1):
            for j in range(y_steps + 1):
                x = sample_min_x + i * (sample_max_x - sample_min_x) / max(1, x_steps)
                y = sample_min_y + j * (sample_max_y - sample_min_y) / max(1, y_steps)
                
                # Check if position is valid
                if self.is_position_valid(x, y, new_object_w, new_object_h, new_object_shape,
                                        container_polygon, nfp_polygons):
                    
                    # Calculate placement score (prefer bottom-left, near existing objects)
                    score = self.calculate_placement_score(x, y, placed_objects, container_polygon)
                    valid_positions.append((x, y, score))
        
        # Sort by score (higher is better)
        valid_positions.sort(key=lambda pos: pos[2], reverse=True)
        return valid_positions[:50]  # Return top 50 positions
    
    def is_position_valid(self, x: float, y: float, w: float, h: float, shape: str,
                         container_polygon: np.ndarray, nfp_polygons: list) -> bool:
        """Check if a position is valid (inside container, outside all NFPs)"""
        
        # Check container bounds first (quick rejection)
        if shape == 'circle':
            radius = w / 2.0
            center_x, center_y = x + radius, y + radius
            test_points = [(center_x, center_y)]  # Just check center for circles
        else:
            # Check all corners for rectangles
            test_points = [
                (x, y), (x + w, y), (x + w, y + h), (x, y + h)
            ]
        
        # All test points must be inside container
        for px, py in test_points:
            if not point_in_polygon((px, py), container_polygon.tolist()):
                return False
        
        # Reference point (top-left corner) must not be inside any NFP
        for nfp in nfp_polygons:
            if self.point_in_nfp((x, y), nfp):
                return False
        
        return True
    
    def calculate_placement_score(self, x: float, y: float, placed_objects: list, 
                                container_polygon: np.ndarray) -> float:
        """Calculate placement score (higher = better position)"""
        score = 0.0
        
        # Prefer bottom-left positions (gravity effect)
        container_bounds = container_polygon.max(axis=0) - container_polygon.min(axis=0)
        if container_bounds[0] > 0 and container_bounds[1] > 0:
            score += (1.0 - x / container_bounds[0]) * 10  # Prefer left
            score += (y / container_bounds[1]) * 10        # Prefer bottom
        
        # Prefer positions near existing objects (compactness)
        if placed_objects:
            min_distance = float('inf')
            for obj_x, obj_y, obj_w, obj_h, _ in placed_objects:
                # Distance to nearest edge of existing object
                dx = max(0, max(obj_x - (x + 50), x - (obj_x + obj_w)))  # Assume 50mm width for distance calc
                dy = max(0, max(obj_y - (y + 50), y - (obj_y + obj_h)))
                distance = math.sqrt(dx*dx + dy*dy)
                min_distance = min(min_distance, distance)
            
            if min_distance < float('inf'):
                score += max(0, 100 - min_distance)  # Closer is better
        
        return score

# Global NFP calculator instance
_nfp_calculator = NFPCalculator()

# ───── H4NP: Heuristic for Nesting Problems ─────────────────────
# Based on: "A general heuristic for two-dimensional nesting problems with limited-size containers"
# H4NP = Genetic Algorithm + BLF/TOPOS placement algorithm

# Global variable to select packing algorithm
PACKING_ALGORITHM = 'H4NP'  # Options: 'BLF', 'MaxRects', 'H4NP', 'AUTO'

# Global H4NP settings (can be adjusted at runtime)
H4NP_GENERATIONS = 4       # Number of GA generations (reduced for speed)
H4NP_POPULATION_MULT = 1.5 # Population = pieces * this multiplier (min 6)
H4NP_EARLY_STOP = True     # Stop early if all pieces placed

class H4NPChromosome:
    """Represents a solution in the H4NP genetic algorithm.
    
    A chromosome encodes:
    - sequence: Order of pieces for placement
    - rotations: Rotation angle for each piece (0°, 90°, 180°, 270° for regular; more for irregular)
    - container_assignments: Which container each piece goes to (for multi-container problems)
    """
    
    def __init__(self, num_pieces: int, allowed_rotations: list = None):
        """Initialize chromosome with random sequence and rotations"""
        self.num_pieces = num_pieces
        self.sequence = list(range(num_pieces))  # Initial sequence
        self.allowed_rotations = allowed_rotations or [0, 90, 180, 270]
        self.rotations = [0] * num_pieces  # Rotation index for each piece
        self.fitness = float('inf')  # Lower is better (minimizing waste/containers)
        self.placed_count = 0
        self.total_area_used = 0.0
        self.utilization = 0.0
        
    def randomize(self, rng: np.random.RandomState):
        """Randomize the chromosome"""
        rng.shuffle(self.sequence)
        self.rotations = [rng.randint(0, len(self.allowed_rotations)) for _ in range(self.num_pieces)]
        
    def copy(self) -> 'H4NPChromosome':
        """Create a deep copy of this chromosome"""
        new_chrom = H4NPChromosome(self.num_pieces, self.allowed_rotations)
        new_chrom.sequence = self.sequence.copy()
        new_chrom.rotations = self.rotations.copy()
        new_chrom.fitness = self.fitness
        new_chrom.placed_count = self.placed_count
        new_chrom.total_area_used = self.total_area_used
        new_chrom.utilization = self.utilization
        return new_chrom


class H4NPPacker:
    """H4NP (Heuristic for Nesting Problems) algorithm implementation.
    
    Combines:
    1. Genetic Algorithm (GA) for optimizing piece sequence and rotations
    2. BLF (Bottom-Left-Fill) or TOPOS for actual piece placement
    
    Key parameters from the paper:
    - Population size: 100-200
    - Crossover rate: 0.8-0.9
    - Mutation rate: 0.1-0.2
    - Selection: Tournament selection (size 3-5)
    - Crossover: Order crossover (OX) for sequence, uniform for rotations
    """
    
    def __init__(self, width_mm: float, height_mm: float, 
                 polygon_mm: list = None, cell_mm: float = 1.0,
                 population_size: int = 15, generations: int = 10,
                 crossover_rate: float = 0.85, mutation_rate: float = 0.15,
                 tournament_size: int = 3, elitism_count: int = 2,
                 use_blf: bool = True):  # True = BLF, False = TOPOS-like
        """Initialize H4NP packer
        
        Args:
            width_mm, height_mm: Container dimensions
            polygon_mm: Optional irregular container polygon
            cell_mm: Grid cell size for BLF placement
            population_size: GA population size
            generations: Number of GA generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            tournament_size: Size of tournament selection
            elitism_count: Number of best solutions to preserve
            use_blf: If True, use BLF; if False, use TOPOS-like approach
        """
        self.width_mm = float(width_mm)
        self.height_mm = float(height_mm)
        self.polygon_mm = polygon_mm
        self.cell_mm = float(cell_mm)
        
        # GA parameters
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.use_blf = use_blf
        
        # Results
        self.used = []  # Final placed pieces
        self.best_chromosome = None
        self.generation_stats = []  # Track convergence
        
        # Cache
        self._cached_grid = None
        self._pieces = []
        self._rng = np.random.RandomState(42)
        
    def _prepare_container_grid(self):
        """Prepare the container grid for placement validation"""
        grid_w = int(math.ceil(self.width_mm / self.cell_mm))
        grid_h = int(math.ceil(self.height_mm / self.cell_mm))
        self._cached_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
        
        if self.polygon_mm and len(self.polygon_mm) >= 3:
            pts = np.array([[p[0] / self.cell_mm, p[1] / self.cell_mm] 
                          for p in self.polygon_mm], dtype=np.float32)
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(self._cached_grid, [pts], 1)
            
            # Check if polygon filled anything - if not, use full rectangle
            if np.count_nonzero(self._cached_grid) < grid_w * grid_h * 0.1:
                print(f"[H4NP] Polygon fill too small, using full rectangle")
                self._cached_grid[:, :] = 1
            else:
                # Expand slightly for better edge handling
                try:
                    expand_iter = max(0, int(round(ALLOWED_MARGIN_MM / self.cell_mm)))
                    if expand_iter > 0:
                        kernel = np.ones((3, 3), dtype=np.uint8)
                        self._cached_grid = cv2.dilate(self._cached_grid, kernel, iterations=expand_iter)
                except Exception:
                    pass
        else:
            # No polygon - use full rectangle
            self._cached_grid[:, :] = 1
    
    def _get_rotated_dimensions(self, w: float, h: float, shape: str, rotation_idx: int) -> tuple:
        """Get dimensions after rotation"""
        if shape == 'circle':
            return w, h  # Circles don't change with rotation
        
        rotation_angles = [0, 90, 180, 270]
        angle = rotation_angles[rotation_idx % len(rotation_angles)]
        
        if angle == 90 or angle == 270:
            return h, w
        return w, h
    
    def _evaluate_chromosome(self, chromosome: H4NPChromosome) -> float:
        """Evaluate a chromosome using BLF or TOPOS placement
        
        Returns fitness value (lower is better):
        - Primary: Maximize number of placed pieces
        - Secondary: Minimize wasted area (maximize utilization)
        - Tertiary: Minimize bounding box height
        """
        # Create a fresh packer for this evaluation
        if self.use_blf:
            packer = PolygonGridPackerBLF(
                self.width_mm, self.height_mm, self.polygon_mm,
                cell_mm=self.cell_mm, seed=0, allowed_grid=self._cached_grid.copy()
            )
        else:
            # TOPOS-like: Use standard grid packer with different scoring
            packer = PolygonGridPacker(
                self.width_mm, self.height_mm, self.polygon_mm,
                cell_mm=self.cell_mm, seed=0, allowed_grid=self._cached_grid.copy()
            )
        
        placed_area = 0.0
        placed_count = 0
        max_y = 0.0
        
        # Place pieces in chromosome sequence order with rotations
        for idx in chromosome.sequence:
            if idx >= len(self._pieces):
                continue
                
            w, h, shape = self._pieces[idx][:3]
            piece_id = self._pieces[idx][3] if len(self._pieces[idx]) > 3 else idx
            shape_name = self._pieces[idx][4] if len(self._pieces[idx]) > 4 else ''
            
            # Apply rotation
            rot_w, rot_h = self._get_rotated_dimensions(w, h, shape, chromosome.rotations[idx])
            
            # Try to place with preferred rotation first
            placed = packer.insert(rot_w, rot_h, shape, rot=False, piece_id=piece_id, shape_name=shape_name)
            
            # If failed, try alternative rotation (90° flip)
            if not placed and shape != 'circle':
                alt_w, alt_h = rot_h, rot_w
                placed = packer.insert(alt_w, alt_h, shape, rot=True, piece_id=piece_id, shape_name=shape_name)
            
            if not placed:
                # Debug: which piece failed? (only for best chromosome during final reconstruction)
                if getattr(self, '_debug_final', False):
                    print(f"[H4NP-DEBUG] Failed to place piece #{idx}: {shape} {w}x{h}mm (tried {rot_w}x{rot_h} and {rot_h}x{rot_w})")
            
            if placed:
                piece_area = (math.pi * (w/2.0)**2) if shape == 'circle' else (w * h)
                placed_area += piece_area
                placed_count += 1
                
                # Track maximum Y for strip height minimization
                if packer.used:
                    last = packer.used[-1]
                    max_y = max(max_y, last[1] + last[3])
        
        # Calculate fitness (multi-objective)
        container_area = self.width_mm * self.height_mm
        utilization = placed_area / container_area if container_area > 0 else 0
        
        # Fitness: prioritize placing all pieces, then maximize utilization
        # Lower fitness is better
        unplaced_penalty = (len(self._pieces) - placed_count) * 10000
        area_penalty = (1.0 - utilization) * 1000
        height_penalty = (max_y / self.height_mm) * 100 if self.height_mm > 0 else 0
        
        fitness = unplaced_penalty + area_penalty + height_penalty
        
        # Store results in chromosome
        chromosome.fitness = fitness
        chromosome.placed_count = placed_count
        chromosome.total_area_used = placed_area
        chromosome.utilization = utilization
        
        return fitness
    
    def _tournament_select(self, population: list) -> H4NPChromosome:
        """Tournament selection"""
        tournament = [population[self._rng.randint(0, len(population))] 
                     for _ in range(self.tournament_size)]
        return min(tournament, key=lambda c: c.fitness)
    
    def _order_crossover(self, parent1: H4NPChromosome, parent2: H4NPChromosome) -> tuple:
        """Order Crossover (OX) for sequence permutation
        
        Preserves relative order of elements from both parents.
        """
        n = parent1.num_pieces
        child1 = H4NPChromosome(n, parent1.allowed_rotations)
        child2 = H4NPChromosome(n, parent2.allowed_rotations)
        
        # Select two crossover points
        point1 = self._rng.randint(0, n)
        point2 = self._rng.randint(0, n)
        if point1 > point2:
            point1, point2 = point2, point1
        
        # Copy segment from parent1 to child1
        child1_seq = [-1] * n
        child2_seq = [-1] * n
        
        for i in range(point1, point2 + 1):
            child1_seq[i] = parent1.sequence[i]
            child2_seq[i] = parent2.sequence[i]
        
        # Fill remaining positions from other parent
        def fill_remaining(child_seq, parent_seq):
            remaining = [x for x in parent_seq if x not in child_seq]
            j = 0
            for i in range(n):
                if child_seq[i] == -1:
                    child_seq[i] = remaining[j]
                    j += 1
            return child_seq
        
        child1.sequence = fill_remaining(child1_seq, parent2.sequence)
        child2.sequence = fill_remaining(child2_seq, parent1.sequence)
        
        # Uniform crossover for rotations
        for i in range(n):
            if self._rng.random() < 0.5:
                child1.rotations[i] = parent1.rotations[i]
                child2.rotations[i] = parent2.rotations[i]
            else:
                child1.rotations[i] = parent2.rotations[i]
                child2.rotations[i] = parent1.rotations[i]
        
        return child1, child2
    
    def _mutate(self, chromosome: H4NPChromosome):
        """Mutation operators for sequence and rotations"""
        n = chromosome.num_pieces
        
        # Sequence mutation: swap two elements (only if n >= 2)
        if n >= 2 and self._rng.random() < self.mutation_rate:
            i, j = self._rng.choice(n, 2, replace=False)
            chromosome.sequence[i], chromosome.sequence[j] = \
                chromosome.sequence[j], chromosome.sequence[i]
        
        # Rotation mutation: change rotation of one piece
        if self._rng.random() < self.mutation_rate:
            i = self._rng.randint(0, n)
            chromosome.rotations[i] = self._rng.randint(0, len(chromosome.allowed_rotations))
        
        # Additional mutation: insertion (remove and reinsert at new position) - only if n >= 2
        if n >= 2 and self._rng.random() < self.mutation_rate * 0.5:
            i = self._rng.randint(0, n)
            j = self._rng.randint(0, n)
            item = chromosome.sequence.pop(i)
            chromosome.sequence.insert(j, item)
    
    def _simple_placement(self, pieces: list) -> list:
        """Simple BLF placement for 1-2 pieces (no GA needed)"""
        self._prepare_container_grid()
        
        packer = PolygonGridPackerBLF(
            self.width_mm, self.height_mm, self.polygon_mm,
            cell_mm=self.cell_mm, seed=0, allowed_grid=self._cached_grid.copy()
        )
        
        # Sort by area (largest first)
        sorted_pieces = sorted(enumerate(pieces), 
                              key=lambda x: x[1][0] * x[1][1], reverse=True)
        
        for orig_idx, (w, h, shape, *rest) in sorted_pieces:
            piece_id = rest[0] if rest else orig_idx
            shape_name = rest[1] if len(rest) > 1 else ''
            # Try both orientations
            if not packer.insert(w, h, shape, rot=False, piece_id=piece_id, shape_name=shape_name):
                packer.insert(h, w, shape, rot=True, piece_id=piece_id, shape_name=shape_name)
        
        self.used = packer.used
        return self.used
    
    def pack(self, pieces: list) -> list:
        """Main H4NP algorithm: optimize placement using GA
        
        Args:
            pieces: List of (width, height, shape_type, [piece_id]) tuples
            
        Returns:
            List of placed pieces: [x, y, w, h, rotated, shape, piece_id]
        """
        if not pieces:
            return []
        
        self._pieces = pieces
        num_pieces = len(pieces)
        
        # Debug: show container info (simplified)
        container_area = self.width_mm * self.height_mm
        print(f"[H4NP] {num_pieces} pieces, container {self.width_mm:.0f}x{self.height_mm:.0f}mm")
        
        # For very few pieces, skip GA and use simple greedy placement
        if num_pieces <= 2:
            print(f"[H4NP] Only {num_pieces} piece(s), using direct BLF placement")
            return self._simple_placement(pieces)
        
        # Prepare container grid
        self._prepare_container_grid()
        
        # Determine allowed rotations based on piece types
        has_irregular = any(p[2] == 'poly' for p in pieces if len(p) > 2)
        allowed_rotations = [0, 90, 180, 270] if not has_irregular else [0, 45, 90, 135, 180, 225, 270, 315]
        
        # Adaptive cell size: prioritize speed
        min_piece_dim = min(min(p[0], p[1]) for p in pieces)
        # Larger cell size = faster search (3-5mm range for speed)
        optimal_cell = max(3.0, min(5.0, min_piece_dim / 6.0))
        self.cell_mm = optimal_cell
        self._prepare_container_grid()
        
        # Initialize population (minimal logging)
        population = []
        for i in range(self.population_size):
            chrom = H4NPChromosome(num_pieces, allowed_rotations)
            chrom.randomize(self._rng)
            population.append(chrom)
        
        # Add greedy-sorted chromosomes with different strategies
        # 1. Sort by area (largest first)
        greedy_area = H4NPChromosome(num_pieces, allowed_rotations)
        greedy_area.sequence = sorted(range(num_pieces), 
                                      key=lambda i: pieces[i][0] * pieces[i][1], reverse=True)
        population[0] = greedy_area
        
        # 2. Sort by width (widest first) - good for strip packing
        if len(population) > 1:
            greedy_width = H4NPChromosome(num_pieces, allowed_rotations)
            greedy_width.sequence = sorted(range(num_pieces), 
                                           key=lambda i: max(pieces[i][0], pieces[i][1]), reverse=True)
            population[1] = greedy_width
        
        # 3. Sort by height (tallest first)
        if len(population) > 2:
            greedy_height = H4NPChromosome(num_pieces, allowed_rotations)
            greedy_height.sequence = sorted(range(num_pieces), 
                                            key=lambda i: min(pieces[i][0], pieces[i][1]), reverse=True)
            population[2] = greedy_height
        
        # 4. Circles and irregular shapes first (they're harder to fit in gaps)
        if len(population) > 3:
            greedy_circles = H4NPChromosome(num_pieces, allowed_rotations)
            def circle_priority(i):
                shape = pieces[i][2] if len(pieces[i]) > 2 else 'rect'
                area = pieces[i][0] * pieces[i][1]
                # Circles get highest priority (0), then polys (1), then rects (2)
                if shape == 'circle':
                    return (0, -area)
                elif shape == 'poly':
                    return (1, -area)
                else:
                    return (2, -area)
            greedy_circles.sequence = sorted(range(num_pieces), key=circle_priority)
            population[3] = greedy_circles
        
        # 5. Alternate: big items interleaved with small (better gap filling)
        if len(population) > 4:
            greedy_interleave = H4NPChromosome(num_pieces, allowed_rotations)
            by_area = sorted(range(num_pieces), key=lambda i: pieces[i][0] * pieces[i][1], reverse=True)
            interleaved = []
            left, right = 0, len(by_area) - 1
            toggle = True
            while left <= right:
                if toggle:
                    interleaved.append(by_area[left])
                    left += 1
                else:
                    interleaved.append(by_area[right])
                    right -= 1
                toggle = not toggle
            greedy_interleave.sequence = interleaved
            population[4] = greedy_interleave
        
        # Evaluate initial population
        for chrom in population:
            self._evaluate_chromosome(chrom)
        
        # Track best solution
        best_ever = min(population, key=lambda c: c.fitness).copy()
        
        # Main GA loop
        for gen in range(self.generations):
            # Sort by fitness
            population.sort(key=lambda c: c.fitness)
            
            # Track statistics
            best_fitness = population[0].fitness
            avg_fitness = sum(c.fitness for c in population) / len(population)
            best_placed = population[0].placed_count
            self.generation_stats.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'best_placed': best_placed,
                'utilization': population[0].utilization
            })
            
            # Update best ever
            if population[0].fitness < best_ever.fitness:
                best_ever = population[0].copy()
            
            # Early termination if all pieces placed
            if H4NP_EARLY_STOP and best_ever.placed_count == num_pieces:
                print(f"[H4NP] Early stop at gen {gen+1}: all {num_pieces} pieces placed!")
                break
            
            # Create new population
            new_population = []
            
            # Elitism: keep best solutions
            for i in range(self.elitism_count):
                new_population.append(population[i].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)
                
                # Crossover
                if self._rng.random() < self.crossover_rate:
                    child1, child2 = self._order_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                self._mutate(child1)
                self._mutate(child2)
                
                # Evaluate
                self._evaluate_chromosome(child1)
                self._evaluate_chromosome(child2)
                
                new_population.extend([child1, child2])
            
            # Trim to population size
            population = new_population[:self.population_size]
        
        # Use best chromosome to get final placement
        self.best_chromosome = best_ever
        self._debug_final = True  # Enable debug output for final placement
        
        # Reconstruct placement using best chromosome
        if self.use_blf:
            final_packer = PolygonGridPackerBLF(
                self.width_mm, self.height_mm, self.polygon_mm,
                cell_mm=self.cell_mm, seed=0, allowed_grid=self._cached_grid.copy()
            )
        else:
            final_packer = PolygonGridPacker(
                self.width_mm, self.height_mm, self.polygon_mm,
                cell_mm=self.cell_mm, seed=0, allowed_grid=self._cached_grid.copy()
            )
        
        placed_pieces = 0
        failed_pieces = 0
        for idx in best_ever.sequence:
            if idx >= len(self._pieces):
                continue
            w, h, shape = self._pieces[idx][:3]
            piece_id = self._pieces[idx][3] if len(self._pieces[idx]) > 3 else idx
            shape_name = self._pieces[idx][4] if len(self._pieces[idx]) > 4 else ''
            rot_w, rot_h = self._get_rotated_dimensions(w, h, shape, best_ever.rotations[idx])
            
            before_count = len(final_packer.used)
            
            # Try preferred rotation first
            placed = final_packer.insert(rot_w, rot_h, shape, rot=False, piece_id=piece_id, shape_name=shape_name)
            
            # If failed, try alternative rotation
            if not placed and shape != 'circle':
                placed = final_packer.insert(rot_h, rot_w, shape, rot=True, piece_id=piece_id, shape_name=shape_name)
            
            if placed and len(final_packer.used) > before_count:
                placed_pieces += 1
            else:
                failed_pieces += 1
        
        self._debug_final = False  # Disable debug
        
        # Apply post-compaction
        if hasattr(final_packer, 'shrink_envelope'):
            final_packer.shrink_envelope(rounds=2)
        
        self.used = final_packer.used
        
        print(f"[H4NP] Completed: {len(self.used)}/{num_pieces} pieces placed, "
              f"utilization={best_ever.utilization:.1%}, "
              f"generations={len(self.generation_stats)}")
        
        return self.used
    
    def insert(self, w: float, h: float, shape_type: str = 'rect', rot: bool = True, piece_id: int = -1, shape_name: str = None) -> bool:
        """Single-piece insertion interface for compatibility with other packers
        
        Note: For best results, use pack() with all pieces at once.
        This method maintains a piece list and runs H4NP when called.
        """
        self._pieces.append((w, h, shape_type, piece_id, shape_name or ''))
        
        # Re-run packing with all pieces
        self.used = self.pack(self._pieces)
        
        # Check if this piece was placed
        for u in self.used:
            if len(u) > 6 and u[6] == piece_id:
                return True
        return len(self.used) >= len(self._pieces)


def select_best_packer(width_mm: float, height_mm: float, pieces: list, 
                      polygon_mm: list = None, cell_mm: float = 1.0) -> object:
    """Automatically select the best packing algorithm based on problem characteristics
    
    Selection criteria from H4NP paper:
    - BLF: Best for regular shapes (rectangles, simple polygons)
    - TOPOS: Better for irregular/complex shapes
    - H4NP: Best overall when computation time allows
    """
    global PACKING_ALGORITHM
    
    if PACKING_ALGORITHM == 'BLF':
        if polygon_mm and len(polygon_mm) >= 3:
            return PolygonGridPackerBLF(width_mm, height_mm, polygon_mm, cell_mm=cell_mm)
        return MaxRectsGreedy(width_mm, height_mm, polygon_mm)
    
    elif PACKING_ALGORITHM == 'MaxRects':
        if polygon_mm and len(polygon_mm) >= 3:
            return PolygonGridPacker(width_mm, height_mm, polygon_mm, cell_mm=cell_mm)
        return MaxRects(width_mm, height_mm, polygon_mm)
    
    elif PACKING_ALGORITHM == 'H4NP':
        # Use global H4NP settings
        num_pieces = len(pieces)
        pop_size = max(8, num_pieces * H4NP_POPULATION_MULT)
        
        return H4NPPacker(width_mm, height_mm, polygon_mm, cell_mm=cell_mm,
                         population_size=pop_size, generations=H4NP_GENERATIONS)
    
    else:  # AUTO
        # Analyze pieces to choose algorithm
        has_circles = any(p[2] == 'circle' for p in pieces if len(p) > 2)
        has_poly = any(p[2] == 'poly' for p in pieces if len(p) > 2)
        num_pieces = len(pieces)
        
        # For very few pieces, BLF is fastest
        if num_pieces <= 3:
            if polygon_mm and len(polygon_mm) >= 3:
                return PolygonGridPackerBLF(width_mm, height_mm, polygon_mm, cell_mm=cell_mm)
            return MaxRectsGreedy(width_mm, height_mm, polygon_mm)
        
        # For irregular shapes or many pieces, use H4NP
        if has_poly or num_pieces > 10:
            pop_size = max(8, num_pieces * H4NP_POPULATION_MULT)
            return H4NPPacker(width_mm, height_mm, polygon_mm, cell_mm=cell_mm,
                             population_size=pop_size, generations=H4NP_GENERATIONS,
                             use_blf=not has_poly)
        
        # Default: BLF for regular shapes
        if polygon_mm and len(polygon_mm) >= 3:
            return PolygonGridPackerBLF(width_mm, height_mm, polygon_mm, cell_mm=cell_mm)
        return MaxRectsGreedy(width_mm, height_mm, polygon_mm)


def toggle_packing_algorithm():
    """Toggle between packing algorithms"""
    global PACKING_ALGORITHM
    algorithms = ['BLF', 'MaxRects', 'H4NP', 'AUTO']
    current_idx = algorithms.index(PACKING_ALGORITHM) if PACKING_ALGORITHM in algorithms else 0
    next_idx = (current_idx + 1) % len(algorithms)
    PACKING_ALGORITHM = algorithms[next_idx]
    
    descriptions = {
        'BLF': 'Bottom-Left-Fill (fast, good for rectangles)',
        'MaxRects': 'MaxRects Best-Area-Fit (good general purpose)',
        'H4NP': 'Genetic Algorithm + BLF (best quality, slower)',
        'AUTO': 'Automatic selection based on problem'
    }
    print(f"[INFO] Packing algorithm: {PACKING_ALGORITHM} - {descriptions[PACKING_ALGORITHM]}")


def adjust_h4np_generations(increase: bool = True):
    """Adjust H4NP generations count"""
    global H4NP_GENERATIONS
    if increase:
        H4NP_GENERATIONS = min(50, H4NP_GENERATIONS + 5)
    else:
        H4NP_GENERATIONS = max(5, H4NP_GENERATIONS - 5)
    print(f"[INFO] H4NP generations: {H4NP_GENERATIONS} (use +/- to adjust)")

# ───── Check if cutting piece fits within irregular object contour ────────────────────
def piece_fits_in_contour(piece_x, piece_y, piece_w, piece_h, shape_type, contour, scale):
    """
    Check if a cutting piece (rectangle or circle) fits entirely within an irregular object contour
    
    Args:
        piece_x, piece_y: Position of piece in mm
        piece_w, piece_h: Size of piece in mm
        shape_type: 'rect' or 'circle'
        contour: Object contour in pixel coordinates
        scale: mm per pixel conversion factor
    
    Returns:
        bool: True if piece fits entirely within contour
    """
    if contour is None or len(contour) < 3:
        return True  # If no contour, assume it fits (more lenient)
    
    # Convert contour to simplified polygon for faster checking
    contour_points = contour.reshape(-1, 2)
    
    if shape_type == 'circle':
        # For circles, check if the center and a few key points are within the contour
        center_x_px = piece_x / scale
        center_y_px = piece_y / scale
        radius_px = (piece_w / 2) / scale  # piece_w is diameter
        
        # Check center point first (most important)
        if not point_in_polygon((center_x_px, center_y_px), contour_points):
            return False
        
        # Sample fewer points around the circle circumference (more lenient)
        num_samples = 8  # Reduced from 16
        for i in range(num_samples):
            angle = 2 * math.pi * i / num_samples
            check_x = center_x_px + radius_px * math.cos(angle)
            check_y = center_y_px + radius_px * math.sin(angle)
            
            if not point_in_polygon((check_x, check_y), contour_points):
                return False
        
        return True
    
    else:  # rectangle
        # For rectangles, check center and corners with more lenient approach
        center_x_px = (piece_x + piece_w/2) / scale
        center_y_px = (piece_y + piece_h/2) / scale
        
        # Check center point first (most important)
        if not point_in_polygon((center_x_px, center_y_px), contour_points):
            return False
        
        # Check corners with more tolerance
        corners_mm = [
            (piece_x, piece_y),  # top-left
            (piece_x + piece_w, piece_y),  # top-right
            (piece_x + piece_w, piece_y + piece_h),  # bottom-right
            (piece_x, piece_y + piece_h)  # bottom-left
        ]
        
        # Convert to pixels and check each corner
        corners_inside = 0
        for corner_x_mm, corner_y_mm in corners_mm:
            corner_x_px = corner_x_mm / scale
            corner_y_px = corner_y_mm / scale
            
            if point_in_polygon((corner_x_px, corner_y_px), contour_points):
                corners_inside += 1
        
        # More lenient: require at least 3 out of 4 corners to be inside
        return corners_inside >= 3
    
    return True





# ───── Local (object) coordinate containment, units in mm ───────────────────
def piece_fits_in_local_contour_mm(piece_x_mm, piece_y_mm, piece_w_mm, piece_h_mm,
                                   shape_type, polygon_mm):
    """Validate that a piece fits within the object's real contour with tolerance.
    
    Uses stricter boundary checking to ensure pieces stay within the contour.
    """
    if polygon_mm is None or len(polygon_mm) < 3:
        return True

    # Reuse existing polygon point-in-polygon tester
    poly = np.array(polygon_mm, dtype=np.float32)
    
    # Calculate polygon bounds with safety margin
    min_x = min(p[0] for p in polygon_mm)
    max_x = max(p[0] for p in polygon_mm)
    min_y = min(p[1] for p in polygon_mm)
    max_y = max(p[1] for p in polygon_mm)

    if shape_type == 'circle':
        cx = piece_x_mm + piece_w_mm / 2.0
        cy = piece_y_mm + piece_h_mm / 2.0
        radius = piece_w_mm / 2.0
        margin = 2.0  # Safety margin for circles

        # Strict bounds check for circles - must be fully inside with margin
        if (cx - radius < min_x + margin or cy - radius < min_y + margin or
            cx + radius > max_x - margin or cy + radius > max_y - margin):
            return False

        # Check center point - most important
        if not point_in_polygon((cx, cy), poly):
            return False

        # Check 8 key points around the circle for better coverage
        key_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        points_inside = 0
        for angle_deg in key_angles:
            angle = math.radians(angle_deg)
            # Check slightly inside the radius for safety
            px = cx + (radius - 1.0) * math.cos(angle)
            py = cy + (radius - 1.0) * math.sin(angle)
            if point_in_polygon((px, py), poly):
                points_inside += 1
        
        # Require at least 7 out of 8 points inside for circles
        if points_inside < 7:
            return False
        
        return True

    # Rectangle validation - more lenient boundary checking
    margin = 1.0  # Small tolerance for numerical precision
    
    # Check center point first (most important)
    cx = piece_x_mm + piece_w_mm / 2.0
    cy = piece_y_mm + piece_h_mm / 2.0
    if not point_in_polygon((cx, cy), poly):
        return False

    # Check corners with tolerance - allow if most corners are inside
    # Use slightly inset corners for tolerance
    inset = 0.5
    corners = [
        (piece_x_mm + inset, piece_y_mm + inset),  # top-left
        (piece_x_mm + piece_w_mm - inset, piece_y_mm + inset),  # top-right
        (piece_x_mm + piece_w_mm - inset, piece_y_mm + piece_h_mm - inset),  # bottom-right
        (piece_x_mm + inset, piece_y_mm + piece_h_mm - inset),  # bottom-left
    ]
    
    corners_inside = 0
    for px, py in corners:
        if point_in_polygon((px, py), poly):
            corners_inside += 1
    
    # Allow placement if at least 3 out of 4 corners are inside
    if corners_inside < 3:
        return False

    # Bounds check with tolerance
    if (piece_x_mm < -margin or piece_y_mm < -margin or
        piece_x_mm + piece_w_mm > max_x + margin or 
        piece_y_mm + piece_h_mm > max_y + margin):
        return False

    return True

def polygon_area_mm2(polygon_mm):
    """Return area of a polygon in mm^2 using the shoelace formula."""
    if polygon_mm is None or len(polygon_mm) < 3:
        return 0.0
    x = np.array([p[0] for p in polygon_mm], dtype=np.float64)
    y = np.array([p[1] for p in polygon_mm], dtype=np.float64)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def polygon_side_lengths_mm(polygon_mm):
    """Return list of edge lengths (mm) following polygon vertex order."""
    if polygon_mm is None or len(polygon_mm) < 2:
        return []
    pts = np.array(polygon_mm, dtype=np.float64)
    diffs = np.diff(np.vstack([pts, pts[0]]), axis=0)
    return [float(np.linalg.norm(d)) for d in diffs]

# ───── Signatures and per-object packing cache ─────────────────────────────
def pieces_signature(pieces: list[tuple]) -> tuple:
    """Order-insensitive signature for input pieces."""
    norm = []
    for p in pieces:
        if len(p) >= 3:
            w, h, s = p[:3]
        else:
            w, h = p
            s = 'rect'
        norm.append((round(float(w), 3), round(float(h), 3), s))
    norm.sort()
    return tuple(norm)

def polygon_signature(poly_mm: Optional[list], w_mm: float, h_mm: float) -> tuple:
    if poly_mm is None or len(poly_mm) < 3:
        return ('bbox', round(float(w_mm), 2), round(float(h_mm), 2))
    sig = tuple((round(float(x), 1), round(float(y), 1)) for x, y in poly_mm[:: max(1, len(poly_mm)//50) ])
    return ('poly', sig)

def compute_object_hash(obj: dict) -> int:
    """Compute hash for object properties that affect packing"""
    try:
        # Include key properties that affect packing
        hash_data = (
            obj['w_mm'],
            obj['h_mm'],
            obj.get('real_area_mm2', 0),
            tuple(tuple(pt) for pt in obj.get('allowed_polygon_mm', [])) if obj.get('allowed_polygon_mm') else None,
            tuple(obj.get('side_lengths_mm', []))
        )
        return hash(hash_data)
    except Exception:
        return 0

def check_objects_changed() -> bool:
    """Check if objects have changed and invalidate cache if needed"""
    global all_objects, _objects_hash_cache, _objects_position_cache, _last_objects_count
    global _packing_cache, _cache_valid, _distribution_cache, _last_distribution_key
    
    current_count = len(all_objects)
    objects_changed = False
    
    # Check if object count changed
    if current_count != _last_objects_count:
        print(f"[DEBUG] Object count changed: {_last_objects_count} -> {current_count}")
        objects_changed = True
        _last_objects_count = current_count
    
    # Check each object for changes
    for i, obj in enumerate(all_objects):
        obj_id = i + 1
        current_hash = compute_object_hash(obj)
        
        # Check if object properties changed
        if obj_id not in _objects_hash_cache or _objects_hash_cache[obj_id] != current_hash:
            _objects_hash_cache[obj_id] = current_hash
            objects_changed = True
        
        # Check if object position changed (using contour center)
        try:
            moments = cv2.moments(obj['contour'])
            if moments['m00'] != 0:
                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])
                current_pos = (center_x, center_y)
                
                if obj_id not in _objects_position_cache or _objects_position_cache[obj_id] != current_pos:
                    if obj_id in _objects_position_cache:
                        old_pos = _objects_position_cache[obj_id]
                        distance = ((current_pos[0] - old_pos[0])**2 + (current_pos[1] - old_pos[1])**2)**0.5
                        if distance > 5:  # Only consider significant position changes (>5 pixels)
                            objects_changed = True
                    _objects_position_cache[obj_id] = current_pos
        except Exception:
            pass
    
    # Clean up cache for removed objects
    current_obj_ids = set(range(1, current_count + 1))
    cached_obj_ids = set(_objects_hash_cache.keys())
    removed_obj_ids = cached_obj_ids - current_obj_ids
    
    for obj_id in removed_obj_ids:
        _objects_hash_cache.pop(obj_id, None)
        _objects_position_cache.pop(obj_id, None)
        objects_changed = True
    
    # Invalidate caches if objects changed
    if objects_changed:
        _packing_cache.clear()
        _distribution_cache.clear()
        _cache_valid = False
        _last_distribution_key = None
        return True
    
    return False

def create_distribution_cache_key(selected_objects: list, remaining_pieces: list, objects_data: list) -> tuple:
    """Create a cache key for distribution results"""
    try:
        # Include selected object indices
        selected_key = tuple(sorted(selected_objects))
        
        # Include pieces signature
        pieces_key = pieces_signature(remaining_pieces)
        
        # Include relevant object properties that affect distribution
        objects_key = []
        for obj_idx in selected_objects:
            if 0 < obj_idx <= len(objects_data):
                obj = objects_data[obj_idx - 1]
                obj_key = (
                    obj['w_mm'],
                    obj['h_mm'],
                    obj.get('real_area_mm2', 0),
                    len(obj.get('allowed_polygon_mm', [])) if obj.get('allowed_polygon_mm') else 0
                )
                objects_key.append(obj_key)
        
        return (selected_key, pieces_key, tuple(objects_key))
    except Exception:
        return None

def _packing_worker_attempt(args):
    """Worker function for parallel packing attempts.

    This function is called by multiprocessing workers to execute a single
    packing attempt with a specific strategy.

    Args:
        args: Tuple of (attempt_num, sheet_width_mm, sheet_height_mm, allowed_polygon_mm,
                       adaptive_cell_mm, cached_grid, ordered_pieces, fittable_pieces_len)

    Returns:
        Tuple of (count, placed_area, compactness, used_list) or None if failed
    """
    try:
        (attempt, sheet_width_mm, sheet_height_mm, allowed_polygon_mm,
         adaptive_cell_mm, cached_grid, ordered_pieces, fittable_pieces_len) = args

        # Determine which packer to use based on strategy
        strategy_id = attempt % 15
        use_greedy_ffd = strategy_id in [8, 11, 12]
        use_greedy_blf = strategy_id in [9, 10, 13, 14]

        if allowed_polygon_mm is not None and len(allowed_polygon_mm) >= 3:
            if use_greedy_blf:
                packer = PolygonGridPackerBLF(sheet_width_mm, sheet_height_mm, allowed_polygon_mm,
                                            cell_mm=adaptive_cell_mm, seed=attempt, allowed_grid=cached_grid)
            else:
                packer = PolygonGridPacker(sheet_width_mm, sheet_height_mm, allowed_polygon_mm,
                                           cell_mm=adaptive_cell_mm, seed=attempt, allowed_grid=cached_grid)
        elif use_greedy_ffd:
            packer = MaxRectsGreedy(sheet_width_mm, sheet_height_mm, allowed_polygon_mm)
        else:
            packer = MaxRects(sheet_width_mm, sheet_height_mm, allowed_polygon_mm)

        placed_area = 0.0
        placed_ids = set()

        # First pass: try to place all pieces in order
        for piece in ordered_pieces:
            w, h, s = piece[:3]
            piece_id = piece[3] if len(piece) > 3 else -1
            shape_name = piece[4] if len(piece) > 4 else None
            piece_area = (math.pi * (w/2.0) * (h/2.0)) if s == 'circle' else (w * h)
            if packer.insert(w, h, s, piece_id=piece_id, shape_name=shape_name):
                placed_area += piece_area
                placed_ids.add(piece_id)
            if len(packer.used) >= fittable_pieces_len:
                break

        # Gap filling: single pass with smart sorting (reduced for speed)
        max_gap_filling_passes = 2
        for gap_pass in range(max_gap_filling_passes):
            unplaced = [p for p in ordered_pieces if (p[3] if len(p) > 3 else -1) not in placed_ids]
            if not unplaced or len(packer.used) >= fittable_pieces_len:
                break

            pieces_placed_this_pass = 0

            # Simplified sorting
            if gap_pass == 0:
                unplaced_sorted = sorted(unplaced, key=lambda p: p[0] * p[1])
            else:
                circles = [p for p in unplaced if p[2] == 'circle']
                rects = [p for p in unplaced if p[2] != 'circle']
                unplaced_sorted = sorted(circles, key=lambda p: p[0]) + sorted(rects, key=lambda p: p[0] * p[1])

            for piece in unplaced_sorted:
                w, h, s = piece[:3]
                piece_id = piece[3] if len(piece) > 3 else -1
                shape_name = piece[4] if len(piece) > 4 else None
                if piece_id in placed_ids:
                    continue
                piece_area = (math.pi * (w/2.0) * (h/2.0)) if s == 'circle' else (w * h)
                if packer.insert(w, h, s, piece_id=piece_id, shape_name=shape_name):
                    placed_area += piece_area
                    placed_ids.add(piece_id)
                    pieces_placed_this_pass += 1

            if pieces_placed_this_pass == 0:
                break

        # Post compaction for grid packer
        if isinstance(packer, PolygonGridPacker):
            try:
                packer.shrink_envelope(rounds=1)
            except Exception:
                pass
            # Defensive: remove accidental duplicates/overlaps
            filtered = []
            occ = np.zeros_like(packer.allowed, dtype=np.uint8)
            for u in packer.used:
                ux, uy, uw, uh, urot, ushape = u[:6]
                gx = int(round(ux / packer.cell_mm))
                gy = int(round(uy / packer.cell_mm))
                gw = int(max(1, math.ceil(uw / packer.cell_mm)))
                gh = int(max(1, math.ceil(uh / packer.cell_mm)))
                if ushape == 'circle':
                    size = max(2, gw)
                    r = size // 2
                    fp = np.zeros((size, size), dtype=np.uint8)
                    cv2.circle(fp, (r, r), r, 1, -1)
                else:
                    fp = np.ones((gh, gw), dtype=np.uint8)
                gx = max(0, min(gx, occ.shape[1] - fp.shape[1]))
                gy = max(0, min(gy, occ.shape[0] - fp.shape[0]))
                region = occ[gy:gy+fp.shape[0], gx:gx+fp.shape[1]]
                if np.any(region[fp == 1] == 1):
                    continue
                region[fp == 1] = 1
                filtered.append(u)
            packer.used = filtered

        count = len(packer.used)

        # Calculate compactness metric
        compactness = float('inf')
        if packer.used and placed_area > 0:
            min_x = min(u[0] for u in packer.used)
            max_x = max(u[0] + u[2] for u in packer.used)
            min_y = min(u[1] for u in packer.used)
            max_y = max(u[1] + u[3] for u in packer.used)
            bounding_area = (max_x - min_x) * (max_y - min_y)
            compactness = bounding_area / placed_area if placed_area > 0 else float('inf')

        return (count, placed_area, compactness, packer.used.copy())

    except Exception as e:
        # Return None for failed attempts
        return None

def compute_best_packing_for_object(obj: dict, pieces: list[tuple]) -> list:
    """Enhanced multi-start packing algorithm with optimized sorting strategies.
    Respects real segmented polygon when present.
    
    Supports multiple algorithms:
    - BLF: Bottom-Left-Fill (fast, good for rectangles)
    - MaxRects: Best-Area-Fit (good general purpose)
    - H4NP: Genetic Algorithm + BLF (best quality, slower)
    - AUTO: Automatic selection based on problem
    """
    global PACKING_ALGORITHM
    
    sheet_width_mm = obj['w_mm']
    sheet_height_mm = obj['h_mm']
    allowed_polygon_mm = obj.get('allowed_polygon_mm')

    fittable_pieces, unfittable = split_cuts_to_fit(sheet_width_mm, sheet_height_mm, pieces)
    if not fittable_pieces:
        return []
    
    # Calculate adaptive cell size for this packing operation
    adaptive_cell_mm = calculate_adaptive_cell_size(fittable_pieces, sheet_width_mm, sheet_height_mm)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # H4NP MODE: Use Genetic Algorithm for optimal packing
    # ═══════════════════════════════════════════════════════════════════════════
    if PACKING_ALGORITHM == 'H4NP' or (PACKING_ALGORITHM == 'AUTO' and len(fittable_pieces) > 6):
        try:
            # Use global H4NP settings - optimized for speed
            num_pieces = len(fittable_pieces)
            pop_size = max(6, int(num_pieces * H4NP_POPULATION_MULT))
            gens = H4NP_GENERATIONS
            
            # Use larger cell for speed (3mm minimum)
            fast_cell_mm = 3.0
            
            # Check if we have irregular shapes
            has_irregular = any(p[2] == 'poly' for p in fittable_pieces if len(p) > 2)
            
            print(f"[H4NP] Starting genetic optimization for {num_pieces} pieces...")
            
            h4np_packer = H4NPPacker(
                sheet_width_mm, sheet_height_mm,
                polygon_mm=allowed_polygon_mm,
                cell_mm=fast_cell_mm,
                population_size=pop_size,
                generations=gens,
                crossover_rate=0.85,
                mutation_rate=0.15,
                tournament_size=3,
                elitism_count=2,
                use_blf=True  # Always use BLF for speed
            )
            
            h4np_result = h4np_packer.pack(fittable_pieces)
            
            # Check if H4NP found a good solution
            if h4np_result and len(h4np_result) >= len(fittable_pieces) * 0.9:
                print(f"[H4NP] Success: {len(h4np_result)}/{len(fittable_pieces)} pieces placed")
                return h4np_result
            else:
                # H4NP didn't place all pieces, but still use its result
                # Continue to multi-start algorithm below for potential improvement
                h4np_backup = h4np_result
        except Exception as e:
            print(f"[H4NP] Error: {e}, falling back to multi-start algorithm")
            h4np_backup = []
    else:
        h4np_backup = None

    # Enhanced piece analysis for better sorting strategies
    def analyze_pieces(pieces_list):
        circles = [p for p in pieces_list if p[2] == 'circle']
        rects = [p for p in pieces_list if p[2] != 'circle']
        
        total_area = sum((math.pi * (w/2.0) * (h/2.0)) if s == 'circle' else (w * h) for w, h, s, *_ in pieces_list)
        circle_area = sum(math.pi * (w/2.0) * (h/2.0) for w, h, s, *_ in circles)
        rect_area = total_area - circle_area
        
        # Analyze rectangle aspect ratios
        tall_rects = [r for r in rects if r[1] > r[0] * 1.2]  # height > 1.2 * width
        square_rects = [r for r in rects if abs(r[1] - r[0]) <= max(r[0], r[1]) * 0.2]  # nearly square
        wide_rects = [r for r in rects if r[0] > r[1] * 1.2]  # width > 1.2 * height
        
        return {
            'circles': circles,
            'rects': rects,
            'tall_rects': tall_rects,
            'square_rects': square_rects,
            'wide_rects': wide_rects,
            'circle_ratio': circle_area / total_area if total_area > 0 else 0,
            'total_area': total_area
        }
    
    # Greedy Best-Fit Decreasing (BFD) sorting algorithms
    def greedy_bfd_sort(pieces_list):
        """Sort pieces using Best-Fit Decreasing strategy - largest area first with tie-breaking"""
        def bfd_key(piece):
            w, h, shape = piece[:3]
            area = (math.pi * (w/2.0) * (h/2.0)) if shape == 'circle' else (w * h)
            perimeter = (math.pi * w) if shape == 'circle' else (2 * (w + h))
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
            # Primary: area (descending), Secondary: aspect ratio (ascending for better fit), Tertiary: perimeter
            return (-area, aspect_ratio, -perimeter)
        return sorted(pieces_list, key=bfd_key)
    
    def greedy_area_perimeter_ratio_sort(pieces_list):
        """Sort by area-to-perimeter ratio for optimal space utilization"""
        def ratio_key(piece):
            w, h, shape = piece[:3]
            area = (math.pi * (w/2.0) * (h/2.0)) if shape == 'circle' else (w * h)
            perimeter = (math.pi * w) if shape == 'circle' else (2 * (w + h))
            ratio = area / perimeter if perimeter > 0 else 0
            return (-ratio, -area)  # Higher ratio first, then larger area
        return sorted(pieces_list, key=ratio_key)
    
    def greedy_difficulty_sort(pieces_list):
        """Sort by placement difficulty - harder pieces first"""
        def difficulty_key(piece):
            w, h, shape = piece[:3]
            area = (math.pi * (w/2.0) * (h/2.0)) if shape == 'circle' else (w * h)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
            # Circles are easier to place, very elongated pieces are harder
            difficulty = aspect_ratio * (2.0 if shape != 'circle' else 1.0)
            return (-difficulty, -area)  # Harder pieces first, then larger area
        return sorted(pieces_list, key=difficulty_key)
    
    def ffdh_sort(pieces_list, sheet_width):
        """First Fit Decreasing Height (FFDH) - excellent for strip packing.
        Sort by height descending, then try to fit pieces in rows.
        """
        def ffdh_key(piece):
            w, h, shape = piece[:3]
            # For pieces wider than tall, consider rotation
            effective_h = max(w, h)  # Height after optimal rotation
            effective_w = min(w, h)  # Width after optimal rotation
            area = (math.pi * (w/2.0) * (h/2.0)) if shape == 'circle' else (w * h)
            return (-effective_h, -area, -effective_w)
        return sorted(pieces_list, key=ffdh_key)
    
    def width_fit_sort(pieces_list, sheet_width):
        """Sort to maximize horizontal strip utilization.
        Group pieces that sum close to sheet width.
        """
        def width_key(piece):
            w, h, shape = piece[:3]
            # Prefer pieces that are good fractions of sheet width
            effective_w = min(w, h) if w != h else w  # Consider rotation
            fit_score = sheet_width / effective_w if effective_w > 0 else 0
            # Prefer pieces that divide sheet width evenly
            remainder = sheet_width % effective_w if effective_w > 0 else sheet_width
            area = (math.pi * (w/2.0) * (h/2.0)) if shape == 'circle' else (w * h)
            return (remainder, -area)
        return sorted(pieces_list, key=width_key)
    
    def longest_side_sort(pieces_list):
        """Sort by longest side - good for filling corners and edges"""
        def longest_key(piece):
            w, h, shape = piece[:3]
            longest = max(w, h)
            area = (math.pi * (w/2.0) * (h/2.0)) if shape == 'circle' else (w * h)
            return (-longest, -area)
        return sorted(pieces_list, key=longest_key)
    
    piece_analysis = analyze_pieces(fittable_pieces)
    
    # Base sorting: rectangles first (easier to pack), then circles
    # Within each group, sort by area (largest first)
    def piece_key(p):
        w, h, s = p[:3]
        area = (math.pi * (w/2.0) * (h/2.0)) if s == 'circle' else (w * h)
        # Rectangles first (0), then circles (1) - circles are harder to fit
        return (1 if s == 'circle' else 0, -area)
    base_order = sorted(fittable_pieces, key=piece_key)
    
    # Generate greedy-optimized orderings
    bfd_order = greedy_bfd_sort(fittable_pieces)
    ratio_order = greedy_area_perimeter_ratio_sort(fittable_pieces)
    difficulty_order = greedy_difficulty_sort(fittable_pieces)

    best_used: list = []
    best_count = -1
    best_area = -1.0
    best_compactness = float('inf')  # Lower is better

    # Calculate adaptive cell size for this packing operation
    adaptive_cell_mm = calculate_adaptive_cell_size(base_order, sheet_width_mm, sheet_height_mm)
    
    # Cache rasterized polygon grid to reuse across attempts
    cached_grid = None
    if allowed_polygon_mm is not None and len(allowed_polygon_mm) >= 3:
            grid_w = int(math.ceil(sheet_width_mm / adaptive_cell_mm))
            grid_h = int(math.ceil(sheet_height_mm / adaptive_cell_mm))
            cached_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
            pts = np.array([[p[0] / adaptive_cell_mm, p[1] / adaptive_cell_mm] for p in allowed_polygon_mm], dtype=np.float32)
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(cached_grid, [pts], 1)
            
            # Expand mask by ALLOWED_MARGIN_MM (in grid cells)
            try:
                expand_iter = max(0, int(round(ALLOWED_MARGIN_MM / adaptive_cell_mm)))
                if expand_iter > 0:
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    cached_grid = cv2.dilate(cached_grid, kernel, iterations=expand_iter)
            except Exception:
                pass

    # Adaptive attempt count - use fewer attempts when multiprocessing (parallel = more coverage)
    will_use_multiprocessing = ENABLE_MULTIPROCESSING and len(fittable_pieces) > 3

    if will_use_multiprocessing:
        # Use fewer attempts with parallelism (8 workers × 16 attempts = 128 total evaluations)
        base_attempts = max(1, PARALLEL_PACKING_ATTEMPTS // (2 if FAST_MODE else 1))
    else:
        # Use more attempts when sequential
        base_attempts = max(1, PACKING_ATTEMPTS // (3 if FAST_MODE else 1))

    # Fewer attempts for faster packing
    if len(fittable_pieces) <= 3:
        attempts = min(base_attempts, 4)
    elif len(fittable_pieces) <= 6:
        attempts = min(base_attempts, 8)
    elif len(fittable_pieces) <= 12:
        attempts = min(base_attempts, 12)
    else:
        attempts = min(base_attempts, 16)  # Reduced for large problems
    
    # Define enhanced sorting strategies based on piece analysis with greedy optimizations
    def get_sorting_strategy(attempt_num, analysis):
        strategy_id = attempt_num % 15  # Expanded to 15 strategies for more variety
        circles = analysis['circles']
        rects = analysis['rects']
        tall_rects = analysis['tall_rects']
        square_rects = analysis['square_rects']
        wide_rects = analysis['wide_rects']
        
        if strategy_id == 0:
            # Strategy 0: Greedy Best-Fit Decreasing (BFD) - optimal for most cases
            return bfd_order
            
        elif strategy_id == 1:
            # Strategy 1: Greedy area-to-perimeter ratio optimization
            return ratio_order
            
        elif strategy_id == 2:
            # Strategy 2: Greedy difficulty-first placement
            return difficulty_order
            
        elif strategy_id == 3:
            # Strategy 3: Circles first, then tall rectangles by area
            tall_rects.sort(key=lambda r: -(r[0] * r[1]))
            other_rects = [r for r in rects if r not in tall_rects]
            other_rects.sort(key=lambda r: -(r[0] * r[1]))
            return circles + tall_rects + other_rects
            
        elif strategy_id == 4:
            # Strategy 4: Circles first, random rectangles (exploration)
            rng = np.random.RandomState(attempt_num)
            rects_copy = rects.copy()
            rng.shuffle(rects_copy)
            return circles + rects_copy
            
        elif strategy_id == 5:
            # Strategy 5: Large rectangles first, then circles
            rects.sort(key=lambda r: -(r[0] * r[1]))
            return rects + circles
            
        elif strategy_id == 6:
            # Strategy 6: Wide rectangles first (horizontal packing)
            wide_rects.sort(key=lambda r: -(r[0] * r[1]))
            other_rects = [r for r in rects if r not in wide_rects]
            other_rects.sort(key=lambda r: -(r[0] * r[1]))
            return wide_rects + other_rects + circles
            
        elif strategy_id == 7:
            # Strategy 7: Mixed approach - alternate large and small pieces
            all_pieces = base_order.copy()
            large_pieces = all_pieces[:len(all_pieces)//2]
            small_pieces = all_pieces[len(all_pieces)//2:]
            mixed = []
            for i in range(max(len(large_pieces), len(small_pieces))):
                if i < len(large_pieces):
                    mixed.append(large_pieces[i])
                if i < len(small_pieces):
                    mixed.append(small_pieces[i])
            return mixed
            
        elif strategy_id == 8:
            # Strategy 8: Square-optimized packing
            square_rects.sort(key=lambda r: -(r[0] * r[1]))
            non_square = [r for r in rects if r not in square_rects]
            non_square.sort(key=lambda r: -(r[0] * r[1]))
            return circles + square_rects + non_square
            
        elif strategy_id == 9:
            # Strategy 9: Aspect ratio balanced approach
            tall_rects.sort(key=lambda r: (-r[1]/r[0], -(r[0] * r[1])))  # Sort by aspect ratio, then area
            wide_rects.sort(key=lambda r: (-r[0]/r[1], -(r[0] * r[1])))
            return circles + tall_rects + wide_rects + square_rects
            
        elif strategy_id == 10:
            # Strategy 10: Reverse area sorting (small first) - sometimes helps with tight spaces
            all_pieces = base_order.copy()
            all_pieces.reverse()
            return all_pieces
            
        elif strategy_id == 11:
            # Strategy 11: Hybrid greedy - combine BFD with shape-specific optimization
            bfd_sorted = bfd_order.copy()
            # Prioritize circles at the beginning for easier placement
            circles_bfd = [p for p in bfd_sorted if p[2] == 'circle']
            rects_bfd = [p for p in bfd_sorted if p[2] != 'circle']
            return circles_bfd + rects_bfd
        
        elif strategy_id == 12:
            # Strategy 12: FFDH (First Fit Decreasing Height) - excellent for strip packing
            return ffdh_sort(fittable_pieces, sheet_width_mm)
        
        elif strategy_id == 13:
            # Strategy 13: Width-fit optimization for horizontal strips
            return width_fit_sort(fittable_pieces, sheet_width_mm)
        
        else:  # strategy_id == 14
            # Strategy 14: Longest side first - good for corner/edge filling
            return longest_side_sort(fittable_pieces)
    
    # Parallel packing: prepare arguments for all attempts
    use_multiprocessing = ENABLE_MULTIPROCESSING and attempts > 1

    if use_multiprocessing:
        # Determine number of workers
        num_workers = NUM_WORKERS if NUM_WORKERS > 0 else cpu_count()
        num_workers = min(num_workers, attempts)  # Don't use more workers than attempts

        print(f"[INFO] Using {num_workers} parallel workers for {attempts} packing attempts")

        # Prepare arguments for each attempt
        worker_args = []
        for attempt in range(attempts):
            ordered_pieces = get_sorting_strategy(attempt, piece_analysis)
            worker_args.append((
                attempt,
                sheet_width_mm,
                sheet_height_mm,
                allowed_polygon_mm,
                adaptive_cell_mm,
                cached_grid,
                ordered_pieces,
                len(fittable_pieces)
            ))

        # Execute packing attempts in parallel with early stopping
        try:
            with Pool(processes=num_workers) as pool:
                # Use imap_unordered for results as they complete (faster)
                results_iter = pool.imap_unordered(_packing_worker_attempt, worker_args, chunksize=1)

                # Process results as they arrive
                completed_attempts = 0
                for result in results_iter:
                    completed_attempts += 1

                    if result is None:
                        continue

                    count, placed_area, compactness, used_list = result

                    # Enhanced selection criteria: prioritize count, then area, then compactness
                    is_better = (count > best_count or
                                (count == best_count and placed_area > best_area) or
                                (count == best_count and placed_area == best_area and compactness < best_compactness))

                    if is_better:
                        best_count = count
                        best_area = placed_area
                        best_compactness = compactness
                        best_used = used_list

                        # Check for perfect packing - early termination
                        if count == len(fittable_pieces):
                            print(f"[INFO] Perfect packing found after {completed_attempts} attempts - stopping early")
                            pool.terminate()  # Stop all remaining workers
                            break

            if best_count > 0:
                print(f"[INFO] Parallel best: {best_count}/{len(fittable_pieces)} pieces ({completed_attempts} attempts)")

        except Exception as e:
            print(f"[WARN] Parallel packing failed ({e}), falling back to sequential")
            use_multiprocessing = False  # Fallback to sequential for this call

    # Sequential packing (fallback or when multiprocessing is disabled)
    if not use_multiprocessing:
        for attempt in range(attempts):
            # Determine which greedy strategy to use
            strategy_id = attempt % 15
            use_greedy_ffd = strategy_id in [8, 11, 12]
            use_greedy_blf = strategy_id in [9, 10, 13, 14]

            if allowed_polygon_mm is not None and len(allowed_polygon_mm) >= 3:
                if use_greedy_blf:
                    packer = PolygonGridPackerBLF(sheet_width_mm, sheet_height_mm, allowed_polygon_mm,
                                                cell_mm=adaptive_cell_mm, seed=attempt, allowed_grid=cached_grid)
                else:
                    packer = PolygonGridPacker(sheet_width_mm, sheet_height_mm, allowed_polygon_mm,
                                               cell_mm=adaptive_cell_mm, seed=attempt, allowed_grid=cached_grid)
            elif use_greedy_ffd:
                packer = MaxRectsGreedy(sheet_width_mm, sheet_height_mm, allowed_polygon_mm)
            else:
                packer = MaxRects(sheet_width_mm, sheet_height_mm, allowed_polygon_mm)

            # Apply enhanced sorting strategy
            ordered_pieces = get_sorting_strategy(attempt, piece_analysis)

            placed_area = 0.0
            placed_ids = set()

            # First pass: try to place all pieces in order
            for piece in ordered_pieces:
                w, h, s = piece[:3]
                piece_id = piece[3] if len(piece) > 3 else -1
                shape_name = piece[4] if len(piece) > 4 else None
                piece_area = (math.pi * (w/2.0) * (h/2.0)) if s == 'circle' else (w * h)
                if packer.insert(w, h, s, piece_id=piece_id, shape_name=shape_name):
                    placed_area += piece_area
                    placed_ids.add(piece_id)
                if len(packer.used) >= len(ordered_pieces):
                    break

            # Gap filling
            max_gap_filling_passes = 2
            for gap_pass in range(max_gap_filling_passes):
                unplaced = [p for p in ordered_pieces if (p[3] if len(p) > 3 else -1) not in placed_ids]
                if not unplaced or len(packer.used) >= len(ordered_pieces):
                    break

                pieces_placed_this_pass = 0

                if gap_pass == 0:
                    unplaced_sorted = sorted(unplaced, key=lambda p: p[0] * p[1])
                else:
                    circles = [p for p in unplaced if p[2] == 'circle']
                    rects = [p for p in unplaced if p[2] != 'circle']
                    unplaced_sorted = sorted(circles, key=lambda p: p[0]) + sorted(rects, key=lambda p: p[0] * p[1])

                for piece in unplaced_sorted:
                    w, h, s = piece[:3]
                    piece_id = piece[3] if len(piece) > 3 else -1
                    shape_name = piece[4] if len(piece) > 4 else None
                    if piece_id in placed_ids:
                        continue
                    piece_area = (math.pi * (w/2.0) * (h/2.0)) if s == 'circle' else (w * h)
                    if packer.insert(w, h, s, piece_id=piece_id, shape_name=shape_name):
                        placed_area += piece_area
                        placed_ids.add(piece_id)
                        pieces_placed_this_pass += 1

                if pieces_placed_this_pass == 0:
                    break

            # Post compaction
            if isinstance(packer, PolygonGridPacker):
                try:
                    packer.shrink_envelope(rounds=1)
                except Exception:
                    pass
                filtered = []
                occ = np.zeros_like(packer.allowed, dtype=np.uint8)
                for u in packer.used:
                    ux, uy, uw, uh, urot, ushape = u[:6]
                    gx = int(round(ux / packer.cell_mm))
                    gy = int(round(uy / packer.cell_mm))
                    gw = int(max(1, math.ceil(uw / packer.cell_mm)))
                    gh = int(max(1, math.ceil(uh / packer.cell_mm)))
                    if ushape == 'circle':
                        size = max(2, gw)
                        r = size // 2
                        fp = np.zeros((size, size), dtype=np.uint8)
                        cv2.circle(fp, (r, r), r, 1, -1)
                    else:
                        fp = np.ones((gh, gw), dtype=np.uint8)
                    gx = max(0, min(gx, occ.shape[1] - fp.shape[1]))
                    gy = max(0, min(gy, occ.shape[0] - fp.shape[0]))
                    region = occ[gy:gy+fp.shape[0], gx:gx+fp.shape[1]]
                    if np.any(region[fp == 1] == 1):
                        continue
                    region[fp == 1] = 1
                    filtered.append(u)
                packer.used = filtered

            count = len(packer.used)

            compactness = float('inf')
            if packer.used and placed_area > 0:
                min_x = min(u[0] for u in packer.used)
                max_x = max(u[0] + u[2] for u in packer.used)
                min_y = min(u[1] for u in packer.used)
                max_y = max(u[1] + u[3] for u in packer.used)
                bounding_area = (max_x - min_x) * (max_y - min_y)
                compactness = bounding_area / placed_area if placed_area > 0 else float('inf')

            is_better = (count > best_count or
                        (count == best_count and placed_area > best_area) or
                        (count == best_count and placed_area == best_area and compactness < best_compactness))

            if is_better:
                best_count = count
                best_area = placed_area
                best_compactness = compactness
                best_used = packer.used.copy()

            # Early termination
            if count == len(fittable_pieces):
                print(f"[INFO] All pieces placed at attempt {attempt+1}/{attempts}")
                break

            if count >= len(fittable_pieces) * 0.95 and attempt >= 5:
                print(f"[INFO] Near-optimal at attempt {attempt+1}: {count}/{len(fittable_pieces)} pieces")
                break

            if attempt >= 10 and best_count >= len(fittable_pieces) * 0.9:
                print(f"[INFO] Stopping at attempt {attempt+1}: {best_count}/{len(fittable_pieces)} pieces")
                break

    # Use H4NP result directly if available (no need for multi-start comparison)
    if h4np_backup is not None:
        return h4np_backup
    
    return best_used

# ───── Distribute cuts across selected objects ────────────────
def distribute_cuts_across_objects(scale=None):
    global remaining_pieces, selected_objects, all_objects, distribution_stable
    global _distribution_cache, _last_distribution_key
    
    # Do NOT clear packing results here; we may want to respect stable placements across frames
    
    if not selected_objects or not all_objects:
        return {}
    
    # If we've already placed all cuts, nothing to do
    if not remaining_pieces:
        print("[INFO] All pieces have been successfully placed!")
        return {}
    
    # Check cache first to avoid redundant calculations
    cache_key = create_distribution_cache_key(selected_objects, remaining_pieces, all_objects)
    if cache_key and cache_key in _distribution_cache:
        print(f"[DEBUG] Using cached distribution result for {len(remaining_pieces)} pieces")
        cached_result = _distribution_cache[cache_key]
        
        # Restore packing results to objects
        for obj_idx, packing_result in cached_result['object_packings'].items():
            if 0 < obj_idx <= len(all_objects):
                all_objects[obj_idx - 1]['packing_result'] = packing_result
        
        # Update remaining pieces
        remaining_pieces[:] = cached_result['remaining_pieces']
        distribution_stable = cached_result['distribution_stable']
        _last_distribution_key = cache_key
        
        return cached_result['placements']
    
    print(f"[INFO] Distributing {len(remaining_pieces)} pieces across {len(selected_objects)} selected objects")
    
    # Always start with all remaining pieces for new selection
    current_pieces = remaining_pieces.copy()
    
    # Track what's placed on each object
    placements = {idx: [] for idx in selected_objects}
    
    # Collect objects by area to use larger ones first
    sorted_objects = []
    for obj_idx in selected_objects:
        if obj_idx <= 0 or obj_idx > len(all_objects):
            continue
        obj = all_objects[obj_idx - 1]
        sorted_objects.append((obj_idx, obj, obj['w_mm'] * obj['h_mm']))
    
    # Sort by area, largest first for better initial distribution
    sorted_objects.sort(key=lambda x: x[2], reverse=True)
    
    # Try to place as many pieces as possible on each object
    for obj_idx, obj, _ in sorted_objects:
        sheet_width_mm, sheet_height_mm = obj['w_mm'], obj['h_mm']
        fittable_pieces, too_big = split_cuts_to_fit(sheet_width_mm, sheet_height_mm, current_pieces)
        if not fittable_pieces:
            continue
        # Build the allowed polygon in object-local mm coordinates from the real contour
        try:
            use_bbox_only = SEGMENTATION_CONFIG.get('use_bounding_box_only', False)
            allowed_polygon_mm = None if use_bbox_only else obj.get('allowed_polygon_mm')
        except Exception:
            allowed_polygon_mm = None

        # Multi-start optimization with enhanced strategies
        def analyze_pieces(pieces):
            """Analyze piece distribution for optimized sorting strategies"""
            circles = [p for p in pieces if p[2] == 'circle']
            rectangles = [p for p in pieces if p[2] != 'circle']
            
            # Classify rectangles by aspect ratio
            tall_rects = [p for p in rectangles if p[1] > p[0] * 1.2]  # height > 1.2 * width
            wide_rects = [p for p in rectangles if p[0] > p[1] * 1.2]  # width > 1.2 * height
            square_rects = [p for p in rectangles if abs(p[0] - p[1]) <= max(p[0], p[1]) * 0.2]
            
            # Calculate area statistics
            all_areas = [(math.pi * (p[0]/2.0) * (p[1]/2.0)) if p[2] == 'circle' else (p[0] * p[1]) for p in pieces]
            avg_area = sum(all_areas) / len(all_areas) if all_areas else 0
            
            return {
                'circles': circles,
                'rectangles': rectangles,
                'tall_rects': tall_rects,
                'wide_rects': wide_rects,
                'square_rects': square_rects,
                'avg_area': avg_area,
                'total_area': sum(all_areas)
            }
        
        def get_sorting_strategy(pieces, strategy_id):
            """Get optimized sorting strategy based on piece analysis"""
            analysis = analyze_pieces(pieces)
            circles = analysis['circles']
            rectangles = analysis['rectangles']
            tall_rects = analysis['tall_rects']
            wide_rects = analysis['wide_rects']
            square_rects = analysis['square_rects']
            
            if strategy_id == 0:
                # Large circles first, then tall rectangles by area
                circles_sorted = sorted(circles, key=lambda p: -(math.pi * (p[0]/2.0) * (p[1]/2.0)))
                tall_sorted = sorted(tall_rects, key=lambda p: -(p[0] * p[1]))
                other_rects = sorted([p for p in rectangles if p not in tall_rects], key=lambda p: -(p[0] * p[1]))
                return circles_sorted + tall_sorted + other_rects
            
            elif strategy_id == 1:
                # Wide rectangles first, then circles, then remaining
                wide_sorted = sorted(wide_rects, key=lambda p: -(p[0] * p[1]))
                circles_sorted = sorted(circles, key=lambda p: -(math.pi * (p[0]/2.0) * (p[1]/2.0)))
                other_rects = sorted([p for p in rectangles if p not in wide_rects], key=lambda p: -(p[0] * p[1]))
                return wide_sorted + circles_sorted + other_rects
            
            elif strategy_id == 2:
                # Mixed strategy: alternate between large and small pieces
                all_pieces = [(p, (math.pi * (p[0]/2.0) * (p[1]/2.0)) if p[2] == 'circle' else (p[0] * p[1])) for p in pieces]
                all_pieces.sort(key=lambda x: -x[1])  # Sort by area descending
                
                result = []
                large_pieces = [p[0] for p in all_pieces[:len(all_pieces)//2]]
                small_pieces = [p[0] for p in all_pieces[len(all_pieces)//2:]]
                
                for i in range(max(len(large_pieces), len(small_pieces))):
                    if i < len(large_pieces):
                        result.append(large_pieces[i])
                    if i < len(small_pieces):
                        result.append(small_pieces[i])
                return result
            
            elif strategy_id == 3:
                # Square-optimized: squares first, then by perimeter
                squares_sorted = sorted(square_rects, key=lambda p: -(p[0] * p[1]))
                circles_sorted = sorted(circles, key=lambda p: -p[0])  # Sort by diameter
                other_rects = sorted([p for p in rectangles if p not in square_rects], 
                                   key=lambda p: -(2 * (p[0] + p[1])))  # Sort by perimeter
                return squares_sorted + circles_sorted + other_rects
            
            elif strategy_id == 4:
                # Density-based: prioritize pieces with high area-to-perimeter ratio
                def density_score(p):
                    if p[2] == 'circle':
                        area = math.pi * (p[0]/2.0) * (p[1]/2.0)
                        perimeter = math.pi * p[0]
                    else:
                        area = p[0] * p[1]
                        perimeter = 2 * (p[0] + p[1])
                    return area / perimeter if perimeter > 0 else 0
                
                return sorted(pieces, key=density_score, reverse=True)
            
            elif strategy_id == 5:
                # Aspect ratio strategy: group by similar ratios
                def aspect_ratio(p):
                    return max(p[0], p[1]) / min(p[0], p[1]) if min(p[0], p[1]) > 0 else float('inf')
                
                # Group by aspect ratio ranges
                very_thin = [p for p in pieces if aspect_ratio(p) > 3]
                thin = [p for p in pieces if 1.5 < aspect_ratio(p) <= 3]
                square_like = [p for p in pieces if aspect_ratio(p) <= 1.5]
                
                # Sort each group by area
                very_thin.sort(key=lambda p: -((math.pi * (p[0]/2.0) * (p[1]/2.0)) if p[2] == 'circle' else (p[0] * p[1])))
                thin.sort(key=lambda p: -((math.pi * (p[0]/2.0) * (p[1]/2.0)) if p[2] == 'circle' else (p[0] * p[1])))
                square_like.sort(key=lambda p: -((math.pi * (p[0]/2.0) * (p[1]/2.0)) if p[2] == 'circle' else (p[0] * p[1])))
                
                return square_like + thin + very_thin
            
            elif strategy_id == 6:
                # Random with size bias: larger pieces have higher probability to be first
                rng = np.random.RandomState(42 + strategy_id)
                pieces_with_weights = []
                for p in pieces:
                    area = (math.pi * (p[0]/2.0) * (p[1]/2.0)) if p[2] == 'circle' else (p[0] * p[1])
                    weight = area / analysis['avg_area'] if analysis['avg_area'] > 0 else 1
                    pieces_with_weights.append((p, weight))
                
                # Weighted random selection
                result = []
                remaining = pieces_with_weights.copy()
                while remaining:
                    weights = [w for _, w in remaining]
                    total_weight = sum(weights)
                    if total_weight <= 0:
                        result.extend([p for p, _ in remaining])
                        break
                    
                    probs = [w / total_weight for w in weights]
                    idx = rng.choice(len(remaining), p=probs)
                    result.append(remaining[idx][0])
                    remaining.pop(idx)
                
                return result
            
            else:  # strategy_id == 7
                # Hybrid strategy: combine multiple approaches
                # Start with largest pieces of each type
                large_circles = sorted([p for p in circles if (math.pi * (p[0]/2.0) * (p[1]/2.0)) > analysis['avg_area']], 
                                     key=lambda p: -(math.pi * (p[0]/2.0) * (p[1]/2.0)))
                large_rects = sorted([p for p in rectangles if (p[0] * p[1]) > analysis['avg_area']], 
                                   key=lambda p: -(p[0] * p[1]))
                
                # Then medium pieces
                medium_pieces = [p for p in pieces if p not in large_circles and p not in large_rects]
                medium_pieces.sort(key=lambda p: -((math.pi * (p[0]/2.0) * (p[1]/2.0)) if p[2] == 'circle' else (p[0] * p[1])))
                
                return large_circles + large_rects + medium_pieces
        
        best_used = []
        best_count = -1
        best_area = -1.0
        best_compactness = float('inf')

        # Calculate adaptive cell size for this packing operation
        adaptive_cell_mm = calculate_adaptive_cell_size(fittable_pieces, sheet_width_mm, sheet_height_mm)
        
        # Cache rasterized polygon grid to reuse across attempts
        cached_grid = None
        if allowed_polygon_mm is not None:
            temp = PolygonGridPacker(sheet_width_mm, sheet_height_mm, allowed_grid=None, cell_mm=adaptive_cell_mm)
            grid_w = int(math.ceil(sheet_width_mm / adaptive_cell_mm))
            grid_h = int(math.ceil(sheet_height_mm / adaptive_cell_mm))
            cached_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
            pts = np.array([[p[0] / adaptive_cell_mm, p[1] / adaptive_cell_mm] for p in allowed_polygon_mm], dtype=np.float32)
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(cached_grid, [pts], 1)
            try:
                expand_iter = max(0, int(round(ALLOWED_MARGIN_MM / adaptive_cell_mm)))
                if expand_iter > 0:
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    cached_grid = cv2.dilate(cached_grid, kernel, iterations=expand_iter)
            except Exception:
                pass

        attempts = max(1, PACKING_ATTEMPTS // (2 if FAST_MODE else 1))
        for attempt in range(attempts):
            if allowed_polygon_mm is not None:
                # Respect already placed items on this object to keep them stable
                initial_used = obj.get('packing_result', [])
                packer = PolygonGridPacker(sheet_width_mm, sheet_height_mm, allowed_polygon_mm,
                                           cell_mm=adaptive_cell_mm, seed=obj_idx * 100 + attempt,
                                           allowed_grid=cached_grid, initial_used=initial_used)
            else:
                packer = MaxRects(sheet_width_mm, sheet_height_mm, allowed_polygon_mm)

            # Use enhanced sorting strategies (8 different approaches)
            strategy_id = attempt % 8
            ordered_pieces = get_sorting_strategy(fittable_pieces, strategy_id)

            placed_area = 0.0
            placed_ids = set()  # Track placed piece IDs for multi-pass gap filling
            occupied_grid = np.zeros((int(math.ceil(sheet_height_mm / adaptive_cell_mm)), int(math.ceil(sheet_width_mm / adaptive_cell_mm))), dtype=np.uint8)
            
            # Helper function to try placing a piece with validation
            def try_place_piece(piece, packer, occupied_grid, placed_area_ref, placed_ids_ref):
                width, height, shape = piece[:3]
                piece_id = piece[3] if len(piece) > 3 else -1
                shape_name = piece[4] if len(piece) > 4 else None
                if piece_id in placed_ids_ref:
                    return False, placed_area_ref
                piece_area = (math.pi * (width/2.0) * (height/2.0) if shape == 'circle' else width * height)
                if obj.get('real_area_mm2') and piece_area > obj['real_area_mm2']:
                    return False, placed_area_ref
                    
                if packer.insert(width, height, shape, piece_id=piece_id, shape_name=shape_name):
                    # Get the placed piece for validation
                    new_used = packer.used[-1]
                    placed_x, placed_y, placed_w, placed_h = new_used[0], new_used[1], new_used[2], new_used[3]
                    
                    # Strict boundary validation
                    is_valid_placement = True
                    
                    # Check basic boundary constraints with small tolerance
                    margin_mm = 0.5
                    if (placed_x < -margin_mm or placed_y < -margin_mm or 
                        placed_x + placed_w > sheet_width_mm + margin_mm or 
                        placed_y + placed_h > sheet_height_mm + margin_mm):
                        packer.used.pop()
                        return False, placed_area_ref
                    
                    # For circles, additional boundary check
                    if shape == 'circle':
                        circle_center_x = placed_x + placed_w / 2.0
                        circle_center_y = placed_y + placed_h / 2.0
                        radius = placed_w / 2.0
                        if (circle_center_x - radius < -margin_mm or 
                            circle_center_y - radius < -margin_mm or
                            circle_center_x + radius > sheet_width_mm + margin_mm or 
                            circle_center_y + radius > sheet_height_mm + margin_mm):
                            packer.used.pop()
                            return False, placed_area_ref
                    
                    # Check if piece fits within real object contour
                    if allowed_polygon_mm is not None and len(allowed_polygon_mm) >= 3:
                        if not piece_fits_in_local_contour_mm(placed_x, placed_y, placed_w, placed_h, shape, allowed_polygon_mm):
                            packer.used.pop()
                            return False, placed_area_ref
                    
                    # Grid overlap validation
                    gx = int(round(placed_x / adaptive_cell_mm))
                    gy = int(round(placed_y / adaptive_cell_mm))
                    gw = int(max(1, math.ceil(placed_w / adaptive_cell_mm)))
                    gh = int(max(1, math.ceil(placed_h / adaptive_cell_mm)))
                    if gy < 0 or gx < 0 or gy+gh > occupied_grid.shape[0] or gx+gw > occupied_grid.shape[1]:
                        packer.used.pop()
                        return False, placed_area_ref
                    region = occupied_grid[gy:gy+gh, gx:gx+gw]
                    if np.any(region == 1):
                        packer.used.pop()
                        return False, placed_area_ref
                    
                    # All validations passed
                    placed_area_ref += piece_area
                    region[:, :] = 1
                    placed_ids_ref.add(piece_id)
                    return True, placed_area_ref
                return False, placed_area_ref
            
            # First pass: try to place all pieces in sorted order
            for piece in ordered_pieces:
                success, placed_area = try_place_piece(piece, packer, occupied_grid, placed_area, placed_ids)
            
            # Multi-pass gap filling: repeatedly try to place remaining pieces with different strategies
            max_gap_filling_passes = 5
            for gap_pass in range(max_gap_filling_passes):
                unplaced = [p for p in fittable_pieces if (p[3] if len(p) > 3 else -1) not in placed_ids]
                if not unplaced or len(packer.used) >= len(fittable_pieces):
                    break
                
                pieces_placed_this_pass = 0
                
                # Different sorting strategies for gap filling
                if gap_pass == 0:
                    # Smallest area first (fill small gaps)
                    unplaced_sorted = sorted(unplaced, key=lambda p: p[0] * p[1])
                elif gap_pass == 1:
                    # Thin pieces first (high aspect ratio)
                    unplaced_sorted = sorted(unplaced, key=lambda p: max(p[0]/p[1] if p[1] > 0 else 1, p[1]/p[0] if p[0] > 0 else 1), reverse=True)
                elif gap_pass == 2:
                    # Smallest dimension first
                    unplaced_sorted = sorted(unplaced, key=lambda p: min(p[0], p[1]))
                elif gap_pass == 3:
                    # Circles first (fit better in odd spaces)
                    circles = sorted([p for p in unplaced if p[2] == 'circle'], key=lambda p: p[0])
                    rects = sorted([p for p in unplaced if p[2] != 'circle'], key=lambda p: p[0] * p[1])
                    unplaced_sorted = circles + rects
                else:
                    # Try largest pieces that might have been skipped
                    unplaced_sorted = sorted(unplaced, key=lambda p: p[0] * p[1], reverse=True)
                
                for piece in unplaced_sorted:
                    success, placed_area = try_place_piece(piece, packer, occupied_grid, placed_area, placed_ids)
                    if success:
                        pieces_placed_this_pass += 1
                
                if pieces_placed_this_pass == 0:
                    break

            # Post-compaction to tighten envelope further when using grid packer
            if isinstance(packer, PolygonGridPacker):
                try:
                    packer.shrink_envelope(rounds=1)
                except Exception:
                    pass

            count = len(packer.used)
            
            # Calculate compactness metric (bounding box area / occupied area)
            compactness = float('inf')
            if count > 0 and placed_area > 0:
                # Calculate bounding box of all placed pieces
                min_x = min(u[0] for u in packer.used)
                min_y = min(u[1] for u in packer.used)
                max_x = max(u[0] + u[2] for u in packer.used)
                max_y = max(u[1] + u[3] for u in packer.used)
                bbox_area = (max_x - min_x) * (max_y - min_y)
                compactness = bbox_area / placed_area if placed_area > 0 else float('inf')
            
            # Enhanced selection criteria: count > area > compactness
            is_better = False
            if count > best_count:
                is_better = True
            elif count == best_count:
                if placed_area > best_area:
                    is_better = True
                elif abs(placed_area - best_area) < 0.01 and compactness < best_compactness:
                    is_better = True
            
            if is_better:
                best_count = count
                best_area = placed_area
                best_compactness = compactness
                best_used = packer.used.copy()
                
                # Early termination for excellent results
                if count == len(fittable_pieces) and compactness < 1.5:
                    print(f"[INFO] Excellent packing achieved on object #{obj_idx}: {count} pieces, compactness {compactness:.2f}")
                    break

        obj['packing_result'] = best_used
        placements[obj_idx] = [(u[2], u[3], u[5]) for u in best_used]
    # Enhanced remaining pieces management with strict validation
    # Track placed piece IDs to filter them out from remaining pieces
    placed_piece_ids = set()
    
    # Also track counts for pieces without IDs (legacy support)
    non_id_piece_counts = {}
    for p in current_pieces:
        if len(p) < 4 or p[3] == -1:
            w, h, s = p[:3]
            key = (float(w), float(h), s)
            non_id_piece_counts[key] = non_id_piece_counts.get(key, 0) + 1

    # Subtract all placed items across selected objects with enhanced validation
    total_placed = 0
    validated_placements = 0
    
    for obj_idx in selected_objects:
        if obj_idx <= 0 or obj_idx > len(all_objects):
            continue
        
        obj = all_objects[obj_idx - 1]
        used = obj.get('packing_result', [])
        sheet_width_mm, sheet_height_mm = obj['w_mm'], obj['h_mm']
        allowed_polygon_mm = obj.get('allowed_polygon_mm')
        
        # Validate each placed piece before counting it as successfully placed
        validated_used = []
        for u in used:
            ux, uy, uw, uh, urot, ushape = u[:6]
            piece_id = u[6] if len(u) > 6 else -1
            
            # Enhanced boundary validation
            boundary_valid = True
            if (ux < 0 or uy < 0 or ux + uw > sheet_width_mm or uy + uh > sheet_height_mm):
                print(f"[WARN] Removing invalid placement: {ushape} {uw}x{uh} at ({ux:.1f}, {uy:.1f}) exceeds boundaries")
                boundary_valid = False
            
            # Enhanced circle boundary validation
            if boundary_valid and ushape == 'circle':
                circle_center_x = ux + uw / 2.0
                circle_center_y = uy + uh / 2.0
                radius = uw / 2.0
                
                if (circle_center_x - radius < 0 or circle_center_y - radius < 0 or
                    circle_center_x + radius > sheet_width_mm or circle_center_y + radius > sheet_height_mm):
                    print(f"[WARN] Removing invalid circle placement: diameter {uw} at ({ux:.1f}, {uy:.1f}) exceeds boundaries")
                    boundary_valid = False
            
            # Enhanced contour validation
            contour_valid = True
            if boundary_valid and allowed_polygon_mm is not None and len(allowed_polygon_mm) >= 3:
                if not piece_fits_in_local_contour_mm(ux, uy, uw, uh, ushape, allowed_polygon_mm):
                    print(f"[WARN] Removing invalid placement: {ushape} {uw}x{uh} violates object contour")
                    contour_valid = False
            
            # Only count valid placements
            if boundary_valid and contour_valid:
                validated_used.append(u)
                validated_placements += 1
                
                if piece_id != -1:
                    placed_piece_ids.add(piece_id)
                else:
                    # Handle non-ID pieces
                    uw_float, uh_float = float(uw), float(uh)
                    if ushape == 'rect':
                        key_a = (uw_float, uh_float, 'rect')
                        key_b = (uh_float, uw_float, 'rect')
                        if non_id_piece_counts.get(key_a, 0) > 0:
                            non_id_piece_counts[key_a] -= 1
                        elif non_id_piece_counts.get(key_b, 0) > 0:
                            non_id_piece_counts[key_b] -= 1
                    else:
                        key_c = (uw_float, uh_float, 'circle')
                        if non_id_piece_counts.get(key_c, 0) > 0:
                            non_id_piece_counts[key_c] -= 1
        
        # Update object with only validated placements
        obj['packing_result'] = validated_used
        total_placed += len(validated_used)
        
        if len(validated_used) < len(used):
            print(f"[INFO] Object #{obj_idx}: {len(validated_used)}/{len(used)} placements validated")

    # Rebuild remaining pieces list
    new_remaining = []
    
    # 1. Add remaining pieces that have IDs
    for p in current_pieces:
        if len(p) >= 4 and p[3] != -1:
            if p[3] not in placed_piece_ids:
                new_remaining.append(p)
    
    # 2. Add remaining pieces that don't have IDs (based on counts)
    for (w, h, s), cnt in non_id_piece_counts.items():
        for _ in range(max(0, cnt)):
            new_remaining.append((w, h, s))
            
    remaining_pieces[:] = new_remaining
    
    # Enhanced logging with validation results
    total_pieces = len(current_pieces)
    remaining_count = len(remaining_pieces)
    print(f"[INFO] Enhanced placement summary: {validated_placements}/{total_pieces} pieces validated and placed")
    print(f"[INFO] {remaining_count} pieces will be transferred to next batch for processing")
    
    # If pieces remain, they will be automatically processed in the next batch
    if remaining_count > 0:
        print(f"[INFO] Next batch will process {remaining_count} remaining pieces on selected objects")
    
    # If nothing remains, mark distribution as stable
    if len(remaining_pieces) == 0:
        distribution_stable = True
        print(f"[INFO] All pieces successfully placed and validated - distribution stable")
    else:
        # Reset distribution stability to ensure next batch processing
        distribution_stable = False
        print(f"[INFO] Distribution not stable - {remaining_count} pieces pending for next batch")
    
    # Cache the distribution result for future use
    if cache_key:
        object_packings = {}
        for obj_idx in selected_objects:
            if 0 < obj_idx <= len(all_objects):
                obj = all_objects[obj_idx - 1]
                if 'packing_result' in obj:
                    object_packings[obj_idx] = obj['packing_result']
        
        cache_result = {
            'placements': placements,
            'remaining_pieces': remaining_pieces.copy(),
            'distribution_stable': distribution_stable,
            'object_packings': object_packings
        }
        _distribution_cache[cache_key] = cache_result
        _last_distribution_key = cache_key
        print(f"[DEBUG] Cached distribution result for {len(selected_objects)} objects")
    
    return placements

# ───── Draw cutting layout ────────────────────────────────────
def draw_layout(frame, perspective_matrix, box, scale, obj_idx, pieces_for_this_object=None, packing_result=None, start_idx=1, allowed_polygon_mm=None):
    if pieces_for_this_object is None:
        pieces_for_this_object = []
    
    tl, tr, br, bl = box
    width_vector = tr - tl
    height_vector = bl - tl
    width_px = np.linalg.norm(width_vector)
    height_px = np.linalg.norm(height_vector)
    width_unit = width_vector / width_px
    height_unit = height_vector / height_px
    sheet_width_mm = width_px * scale
    sheet_height_mm = height_px * scale
    # Update global table size (this is the detected rectangle size between ArUco markers)
    try:
        global CURRENT_TABLE_W_MM, CURRENT_TABLE_H_MM
        CURRENT_TABLE_W_MM = float(sheet_width_mm)
        CURRENT_TABLE_H_MM = float(sheet_height_mm)
    except Exception:
        pass

    # Update global scale to use for library conversions
    global CURRENT_SCALE_MM_PER_PX
    try:
        CURRENT_SCALE_MM_PER_PX = float(scale)
    except Exception:
        pass

    # Display material dimensions on each side (optional)
    if SHOW_DIMENSIONS_ON_MAIN:
        # Top side
        mid_top = (tl + tr) / 2
        top_out = cv2.perspectiveTransform(np.array([[[*mid_top]]],np.float32), perspective_matrix).astype(int)[0,0]
        cv2.putText(frame, f"{sheet_width_mm:.0f} mm", (top_out[0]-40, top_out[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        # Right side
        mid_right = (tr + br) / 2
        right_out = cv2.perspectiveTransform(np.array([[[*mid_right]]],np.float32), perspective_matrix).astype(int)[0,0]
        cv2.putText(frame, f"{sheet_height_mm:.0f} mm", (right_out[0]+10, right_out[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        # Bottom side
        mid_bottom = (bl + br) / 2
        bottom_out = cv2.perspectiveTransform(np.array([[[*mid_bottom]]],np.float32), perspective_matrix).astype(int)[0,0]
        cv2.putText(frame, f"{sheet_width_mm:.0f} mm", (bottom_out[0]-40, bottom_out[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        # Left side
        mid_left = (tl + bl) / 2
        left_out = cv2.perspectiveTransform(np.array([[[*mid_left]]],np.float32), perspective_matrix).astype(int)[0,0]
        cv2.putText(frame, f"{sheet_height_mm:.0f} mm", (left_out[0]-50, left_out[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Check if we have packing results
    used_items = []
    
    # If we have existing packing results, use them
    if packing_result and len(packing_result) > 0:
        used_items = packing_result
    # Otherwise try to create a new packing from provided cuts
    elif pieces_for_this_object:
        ok, big = split_cuts_to_fit(sheet_width_mm, sheet_height_mm, pieces_for_this_object)
        

        for cut in big:
            if len(cut) >= 3 and cut[2] == "circle":
                print(f"[ERROR] Circle ⌀{cut[0]} mm is larger than the sheet {sheet_width_mm:.0f}×{sheet_height_mm:.0f} mm on object #{obj_idx}")
            else:
                w, h = cut[0], cut[1]
                print(f"[ERROR] {w}×{h} mm is larger than the sheet {sheet_width_mm:.0f}×{sheet_height_mm:.0f} mm on object #{obj_idx}")

        # Use provided allowed polygon (real segmented contour in local mm) when available
        if allowed_polygon_mm is not None and len(allowed_polygon_mm) >= 3:
            adaptive_cell_mm = calculate_adaptive_cell_size(ok, sheet_width_mm, sheet_height_mm)
            packer = PolygonGridPacker(sheet_width_mm, sheet_height_mm, allowed_polygon_mm, cell_mm=adaptive_cell_mm)
        else:
            packer = MaxRects(sheet_width_mm, sheet_height_mm, allowed_polygon_mm)
        not_placed = []
        
        for cut in ok:
            if len(cut) >= 3:  # New format with shape
                w, h, shape = cut[:3]
                # Extract piece_id and shape_name if available
                piece_id = cut[3] if len(cut) > 3 else -1
                shape_name = cut[4] if len(cut) > 4 else None
                if not packer.insert(w, h, shape, shape_name=shape_name):
                    not_placed.append(cut)
            else:  # Old format (backwards compatibility)
                w, h = cut
                if not packer.insert(w, h, "rect"):
                    not_placed.append(cut)
        
        if not_placed:
            for cut in not_placed:
                if len(cut) >= 3 and cut[2] == "circle":
                    print(f"[WARN] Not enough space for circle ⌀{cut[0]} mm on object #{obj_idx}")
                else:
                    w, h = cut[0], cut[1]
                    print(f"[WARN] Not enough space for {w}×{h} mm on object #{obj_idx}")
                
        used_items = packer.used

    # Calculate utilization vs real polygon area when possible
    if used_items:
        total_area = 0
        for u in used_items:
            x, y, w, h, rot, shape = u[:6]
            if shape == "circle":
                radius = w / 2
                total_area += math.pi * radius * radius
            else:
                total_area += w * h
        # Estimate real object area via local mm polygon if available
        try:
            local_poly_mm = contour_to_local_polygon_mm(box.reshape(-1, 2), box, scale)
            obj_area_mm2 = polygon_area_mm2(local_poly_mm) if local_poly_mm else (sheet_width_mm * sheet_height_mm)
        except Exception:
            obj_area_mm2 = sheet_width_mm * sheet_height_mm
        utilization = total_area / max(obj_area_mm2, 1e-6)
    else:
        utilization = 0
    
    cv2.putText(frame, f"Util={utilization:.1%}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    
    # Draw each piece (packer already ensures valid non-overlapping placements)
    for i, u in enumerate(used_items):
        x, y, w, h, rot, shape = u[:6]
        try:
            # Calculate continuous item index (start_idx + current item index)
            idx = start_idx + i
            
            # Pre-check: skip drawing pieces that exceed object boundaries
            skip_drawing = False
            
            # Check rectangular pieces
            if x < 0 or y < 0 or x + w > sheet_width_mm or y + h > sheet_height_mm:
                skip_drawing = True
            
            # Additional check for circles
            if not skip_drawing and shape == "circle":
                circle_center_x = x + w / 2.0
                circle_center_y = y + h / 2.0
                radius = w / 2.0
                
                # Check if circle exceeds object boundaries with strict margin
                margin_mm = 0.0  # No margin - strict boundary checking
                if (circle_center_x - radius < margin_mm or circle_center_y - radius < margin_mm or
                    circle_center_x + radius > sheet_width_mm - margin_mm or circle_center_y + radius > sheet_height_mm - margin_mm):
                    skip_drawing = True
            
            # Skip drawing if piece exceeds boundaries
            if skip_drawing:
                continue
            
            # Draw differently based on shape
            if shape == "circle":
                # For circles, we need to calculate the center and radius
                # x, y, w, h are already in mm from packing result
                center_x_mm, center_y_mm = x + w/2, y + h/2
                radius_mm = w/2  # w is diameter in mm
                
                # Convert center from mm to pixels within object coordinates
                center_x_px = center_x_mm / scale
                center_y_px = center_y_mm / scale
                radius_px = radius_mm / scale
                
                # Convert center point from object coordinates to image coordinates
                center_obj = tl + width_unit*center_x_px + height_unit*center_y_px
                center_out = cv2.perspectiveTransform(np.array([[[*center_obj]]],np.float32), perspective_matrix).astype(int)[0][0]
                
                # Calculate radius in screen pixels more accurately
                # Use both width and height unit vectors to get proper scaling
                radius_vec_x = width_unit * radius_px
                radius_vec_y = height_unit * radius_px
                radius_screen = int(max(1, np.linalg.norm(radius_vec_x + radius_vec_y) / 2))
                
                # Ensure minimum radius for visibility
                radius_screen = max(3, radius_screen)
                
                # Draw circle outline with proper thickness
                cv2.circle(frame, center_out, radius_screen, (0,0,255), 2)
                
                # Fill with translucent color
                overlay = frame.copy()
                cv2.circle(overlay, center_out, radius_screen, (0,0,255), -1)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                
                # Draw cutting line (dotted circle) with better spacing
                for angle in range(0, 360, 15):  # Draw dots every 15 degrees for cleaner look
                    rad = math.radians(angle)
                    dot_x = int(center_out[0] + radius_screen * math.cos(rad))
                    dot_y = int(center_out[1] + radius_screen * math.sin(rad))
                    cv2.circle(frame, (dot_x, dot_y), 1, (255,255,255), -1)
                
                # Display diameter with the diameter symbol (⌀)
                diameter_text = f"{int(w)}mm"
                cv2.putText(frame, "D=" + diameter_text, 
                           (center_out[0] - 25, center_out[1] - radius_screen - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.putText(frame, "D=" + diameter_text, 
                           (center_out[0] - 25, center_out[1] - radius_screen - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                
                # Display the item number in the center
                cv2.putText(frame, str(idx), tuple(center_out),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, str(idx), tuple(center_out),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
                
                # Check if circle is within object boundaries
                if (center_x_mm - radius_mm < 0 or center_y_mm - radius_mm < 0 or
                    center_x_mm + radius_mm > sheet_width_mm or center_y_mm + radius_mm > sheet_height_mm):
                    print(f"[WARN] Circle #{idx} may be outside object boundaries!")
                    print(f"[WARN] Object size: {sheet_width_mm:.1f}x{sheet_width_mm:.1f} mm")
                    print(f"[WARN] Circle bounds: ({center_x_mm-radius_mm:.1f}, {center_y_mm-radius_mm:.1f}) to ({center_x_mm+radius_mm:.1f}, {center_y_mm+radius_mm:.1f})")
            elif shape == "poly" or shape == "library":
                # For polygonal shapes from drawing mode or library
                
                # Try to get the actual polygon points from the library or packing result
                drew_poly = False
                
                # Get shape_name and rotation angle from packing result if available
                piece_shape_name = u[7] if len(u) > 7 else None
                piece_angle_deg = u[8] if len(u) > 8 else 0
                
                try:
                    polys = None
                    
                    # First: try exact lookup by shape_name (most accurate)
                    if piece_shape_name and piece_shape_name in LIB_SHAPE_BY_NAME:
                        polys = [LIB_SHAPE_BY_NAME[piece_shape_name]]
                    
                    # Fallback: Look for polygon data in LIB_SHAPE_MAP by dimensions
                    if not polys:
                        key = (round(float(w), 2), round(float(h), 2))
                        rev_key = (round(float(h), 2), round(float(w), 2))
                        polys = LIB_SHAPE_MAP.get(key) or LIB_SHAPE_MAP.get(rev_key)
                    
                    # Debug: log search attempt
                    if not polys:
                        # Try with less precision
                        key1 = (round(float(w), 1), round(float(h), 1))
                        rev_key1 = (round(float(h), 1), round(float(w), 1))
                        polys = LIB_SHAPE_MAP.get(key1) or LIB_SHAPE_MAP.get(rev_key1)
                    
                    # Fallback: search LIB_SHAPE_BY_NAME by matching dimensions with tolerance
                    if not polys and LIB_SHAPE_BY_NAME:
                        tolerance = 10.0  # Increased tolerance for dimension matching
                        for name, poly_data in LIB_SHAPE_BY_NAME.items():
                            bbox_w = poly_data.get('bbox_w', 0)
                            bbox_h = poly_data.get('bbox_h', 0)
                            # Check if dimensions match within tolerance (considering rotation)
                            if ((abs(bbox_w - w) <= tolerance and abs(bbox_h - h) <= tolerance) or
                                (abs(bbox_h - w) <= tolerance and abs(bbox_w - h) <= tolerance)):
                                polys = [poly_data]
                                break
                    
                    if polys:
                        stored = polys[0]  # use first stored polygon
                        # Try pts_mm first (preferred), then pts (legacy)
                        pts_key = 'pts_mm' if 'pts_mm' in stored else 'pts'
                        arr = np.array(stored[pts_key], dtype=np.float32)
                        
                        # Apply rotation if angle is non-zero
                        if piece_angle_deg != 0:
                            centroid = arr.mean(axis=0)
                            angle_rad = math.radians(piece_angle_deg)
                            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
                            arr_centered = arr - centroid
                            arr = np.column_stack([
                                arr_centered[:, 0] * cos_a - arr_centered[:, 1] * sin_a,
                                arr_centered[:, 0] * sin_a + arr_centered[:, 1] * cos_a
                            ])
                            # Normalize to 0,0 origin after rotation
                            arr = arr - arr.min(axis=0)
                        
                        # Get bounding box of (rotated) polygon (mm)
                        min_xy = arr.min(axis=0)
                        max_xy = arr.max(axis=0)
                        pw = max(1e-6, float(max_xy[0] - min_xy[0]))
                        ph = max(1e-6, float(max_xy[1] - min_xy[1]))
                        
                        # Map each point into this rect mm (x,y are mm of top-left of the placed rect)
                        norm = (arr - min_xy) / np.array([pw, ph], dtype=np.float32)
                        pts_mm = np.stack([x + norm[:,0]*float(w), y + norm[:,1]*float(h)], axis=1)
                        
                        # Convert mm→rect-px along OBB axes
                        pts_px_rel = np.stack([pts_mm[:,0]/float(scale), pts_mm[:,1]/float(scale)], axis=1)
                        pts_img = []
                        for ux, uy in pts_px_rel:
                            pt = tl + width_unit*ux + height_unit*uy
                            pts_img.append(pt)
                        pts_img = np.array(pts_img, dtype=np.float32).reshape(-1,1,2)
                        pts_screen = cv2.perspectiveTransform(pts_img, perspective_matrix).astype(int)
                        
                        # Draw the polygon outline
                        is_closed = bool(stored.get('closed', True))
                        cv2.polylines(frame, [pts_screen], is_closed, (0,0,255), 3, cv2.LINE_AA)
                        
                        # Fill with translucent color if closed
                        if is_closed:
                            overlay = frame.copy()
                            cv2.fillPoly(overlay, [pts_screen], (0,0,255))
                            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                        
                        # Draw segments if available
                        segs = stored.get('segments')
                        if segs:
                            for seg in segs:
                                if seg['id'] > 0 and seg['id'] <= len(pts_screen):
                                    a = pts_screen[seg['id']-1][0]
                                    next_id = (seg['id'] % len(pts_screen)) if is_closed else min(seg['id'], len(pts_screen)-1)
                                    b = pts_screen[next_id][0]
                                    cv2.line(frame, tuple(a), tuple(b), (0,0,255), 3, cv2.LINE_AA)
                        
                        drew_poly = True
                        
                        # Calculate center for label
                        center_mm = np.mean(pts_mm, axis=0)
                        center_px_rel = center_mm / scale
                        center_obj = tl + width_unit*center_px_rel[0] + height_unit*center_px_rel[1]
                        center_out = cv2.perspectiveTransform(np.array([[[*center_obj]]], np.float32), perspective_matrix).astype(int)[0,0]
                        
                except Exception as e:
                    print(f"[DEBUG] Poly draw exception: {e}")  # Could not draw polygon from library data
                
                # Fallback: draw as rectangle if polygon data not available
                if not drew_poly:
                    # Debug: print why polygon was not found
                    if idx == start_idx:  # Only print for first poly to avoid spam
                        print(f"[DEBUG] Poly fallback to rect: w={w:.2f}, h={h:.2f}, shape={shape}")
                        print(f"[DEBUG] LIB_SHAPE_BY_NAME keys: {list(LIB_SHAPE_BY_NAME.keys())}")
                        print(f"[DEBUG] LIB_SHAPE_MAP keys: {list(LIB_SHAPE_MAP.keys())}")
                    x_rel, y_rel = x/scale, y/scale
                    p0 = tl + width_unit*x_rel + height_unit*y_rel
                    p1 = p0 + width_unit*(w/scale)
                    p2 = p1 + height_unit*(h/scale)
                    p3 = p0 + height_unit*(h/scale)
                    
                    poly = np.array([[p0,p1,p2,p3]], np.float32).reshape(-1,1,2)
                    out = cv2.perspectiveTransform(poly, perspective_matrix).astype(int)
                    cv2.polylines(frame, [out], True, (0,0,255), 3, cv2.LINE_AA)
                    
                    # Calculate center for label
                    center_obj = (p0+p1+p2+p3)/4
                    center_out = cv2.perspectiveTransform(np.array([[[*center_obj]]], np.float32), perspective_matrix).astype(int)[0,0]
                
                # Display the item number
                cv2.putText(frame, str(idx), tuple(center_out),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, str(idx), tuple(center_out),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
                
                # Display dimensions
                dim_text = f"{w:.0f}x{h:.0f} mm (poly)"
                cv2.putText(frame, dim_text, 
                           (center_out[0]-40, center_out[1]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.putText(frame, dim_text, 
                           (center_out[0]-40, center_out[1]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            else:
                # For rectangles - existing code
                x_rel, y_rel = x/scale, y/scale
                p0 = tl + width_unit*x_rel + height_unit*y_rel
                p1 = p0 + width_unit*(w/scale)
                p2 = p1 + height_unit*(h/scale)
                p3 = p0 + height_unit*(h/scale)
                
                poly = np.array([[p0,p1,p2,p3]], np.float32).reshape(-1,1,2)
                out = cv2.perspectiveTransform(poly, perspective_matrix).astype(int)
                # Draw bbox only if allowed
                if SHOW_LIBRARY_RECT:
                    cv2.polylines(frame, [out], True, (0,0,255), 3)
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [out], (0,0,255,128))
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

                # Try to overlay original library polygon scaled to this rect
                drew_cut_lines = False
                try:
                    key = (round(float(w), 2), round(float(h), 2))
                    rev_key = (round(float(h), 2), round(float(w), 2))
                    polys = LIB_SHAPE_MAP.get(key) or LIB_SHAPE_MAP.get(rev_key)
                    
                    # Fallback: search LIB_SHAPE_BY_NAME by matching dimensions with tolerance
                    if not polys and LIB_SHAPE_BY_NAME:
                        tolerance = 5.0  # mm tolerance for dimension matching (allows for rounding in packing)
                        for name, poly_data in LIB_SHAPE_BY_NAME.items():
                            bbox_w = poly_data.get('bbox_w', 0)
                            bbox_h = poly_data.get('bbox_h', 0)
                            # Check if dimensions match within tolerance (considering rotation)
                            if ((abs(bbox_w - w) <= tolerance and abs(bbox_h - h) <= tolerance) or
                                (abs(bbox_h - w) <= tolerance and abs(bbox_w - h) <= tolerance)):
                                polys = [poly_data]
                                break
                    
                    if polys:
                        stored = polys[0]  # use first stored
                        # Try pts_mm first (preferred), then pts (legacy)
                        pts_key = 'pts_mm' if 'pts_mm' in stored else 'pts'
                        arr = np.array(stored[pts_key], dtype=np.float32)
                        # bbox of stored poly (mm)
                        min_xy = arr.min(axis=0)
                        max_xy = arr.max(axis=0)
                        pw = max(1e-6, float(max_xy[0] - min_xy[0]))
                        ph = max(1e-6, float(max_xy[1] - min_xy[1]))
                        # map each point into this rect mm (x,y are mm of top-left of the placed rect)
                        norm = (arr - min_xy) / np.array([pw, ph], dtype=np.float32)
                        pts_mm = np.stack([x + norm[:,0]*float(w), y + norm[:,1]*float(h)], axis=1)
                        # convert mm→rect-px along OBB axes
                        pts_px_rel = np.stack([pts_mm[:,0]/float(scale), pts_mm[:,1]/float(scale)], axis=1)
                        pts_img = []
                        for ux, uy in pts_px_rel:
                            pt = tl + width_unit*ux + height_unit*uy
                            pts_img.append(pt)
                        pts_img = np.array(pts_img, dtype=np.float32).reshape(-1,1,2)
                        pts_screen = cv2.perspectiveTransform(pts_img, perspective_matrix).astype(int)
                        # Draw only the stored segments (line-by-line) to keep open polylines intact
                        segs = stored.get('segments')
                        if segs:
                            for seg in segs:
                                a = pts_screen[seg['id']-1][0]
                                b = pts_screen[(seg['id'] % len(pts_screen))][0]
                                cv2.line(frame, tuple(a), tuple(b), (0,0,255), 3, cv2.LINE_AA)
                            drew_cut_lines = True
                        else:
                            cv2.polylines(frame, [pts_screen], bool(stored.get('closed', True)), (0,0,255), 3, cv2.LINE_AA)
                            drew_cut_lines = True
                except Exception as _e:
                    pass
                # Fallback: draw rectangle perimeter as cut lines when no library lines found
                if not drew_cut_lines:
                    cv2.polylines(frame, [out], True, (0,0,255), 3, cv2.LINE_AA)
                
                # Calculate and display center point
                ctr = (p0+p1+p2+p3)/4
                ctr_out = cv2.perspectiveTransform(np.array([[[*ctr]]], np.float32), perspective_matrix).astype(int)[0,0]
                # Improve visibility of the number
                cv2.putText(frame, str(idx), tuple(ctr_out),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                cv2.putText(frame, str(idx), tuple(ctr_out),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 1)
                
                # Display dimensions in millimeters
                dim_text = f"{w:.0f}x{h:.0f} mm"
                cv2.putText(frame, dim_text, 
                           (out[0][0][0], out[0][0][1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(frame, dim_text, 
                           (out[0][0][0], out[0][0][1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
        except Exception as e:
            print(f"[ERROR] Failed to compute points for item {idx} on object #{obj_idx}: {e}")
            continue

    return utilization, len(used_items), len(pieces_for_this_object)

# ───── Process sheet ─────────────────────────────────────
def process(mask, rect, scale, frame, zone):
    global all_objects, selected_objects, click_position, transform_matrix, inverse_transform_matrix, remaining_pieces, cutting_pieces
    global distribution_stable, last_selection_count, frame_counter, obj_position_history
    global _last_objects_signature, _last_pieces_signature, _last_frame_processed, _cache_valid, _packing_cache
    
    # Save transformation matrix for click processing
    rect_quad = np.array([[0,0],[rect.shape[1]-1,0],[rect.shape[1]-1,rect.shape[0]-1],[0,rect.shape[0]-1]], np.float32)
    zone_quad = zone.astype(np.float32)
    transform_matrix = cv2.getPerspectiveTransform(rect_quad, zone_quad)
    # Inverse mapping for screen→rect (used by drawing tools)
    try:
        inverse_transform_matrix = cv2.getPerspectiveTransform(zone_quad, rect_quad)
    except Exception:
        inverse_transform_matrix = None
    
    # Find all object contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Save old packing results
    old_objects = {i+1: obj.get('packing_result', []) for i, obj in enumerate(all_objects)}
    
    # Increment frame counter
    frame_counter += 1
    
    # Check for object changes and invalidate cache if needed
    objects_changed = check_objects_changed()
    
    # Check if we've already processed this frame to avoid duplicate calculations
    if _last_frame_processed == frame_counter and _cache_valid and not objects_changed:
        # Skip processing if we've already calculated positions for this frame and objects haven't changed
        return
    
    # Only process object contours and recalculate on some frames to reduce jitter
    recalculate = (frame_counter % RECALCULATION_DELAY == 0)
    
    # If we have previous objects and not recalculating, use the previous objects
    if all_objects and not recalculate and len(obj_position_history) > 0:
        # Just keep the old object positions to avoid jitter
        current_objects = all_objects
    else:
        # Clear objects list for recalculation
        current_objects = []
        
        # Process each contour
        for contour in contours:
            if cv2.contourArea(contour) < MIN_OBJECT_AREA:
                continue
            
            # Use actual contour instead of bounding rectangle
            # Get bounding rectangle for size estimation only
            rot_rect = cv2.minAreaRect(contour)
            width_px, height_px = rot_rect[1]
            width_px, height_px = max(width_px, height_px), min(width_px, height_px)
            width_mm, height_mm = width_px*scale, height_px*scale
            
            # Get the actual contour points for shape-based processing
            # Use original contour with minimal simplification to preserve accuracy
            epsilon = 0.005 * cv2.arcLength(contour, True)  # Much less aggressive simplification
            simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Enhance contour for better display quality
            enhanced_contour = enhance_contour_for_display(contour)
            
            # Transform contour coordinates for display - use enhanced contour for better accuracy
            transformed_contour = cv2.perspectiveTransform(enhanced_contour.astype(np.float32), transform_matrix).astype(int)
            
            # Also transform the simplified contour for cutting validation
            transformed_simplified = cv2.perspectiveTransform(simplified_contour.astype(np.float32), transform_matrix).astype(int)
            
            # Also keep bounding box for compatibility with existing cutting algorithms
            box = order_box(cv2.boxPoints(rot_rect))
            transformed_box = cv2.perspectiveTransform(box.reshape(-1,1,2).astype(np.float32), transform_matrix).astype(int)
            
            # Calculate area of the segmented object
            contour_area_px = cv2.contourArea(contour)
            contour_area_mm2 = contour_area_px * (scale ** 2)

            # Build local-mm polygon and side lengths for precise metrics
            try:
                allowed_polygon_mm = contour_to_local_polygon_mm(simplified_contour, box, scale)
                real_area_mm2 = polygon_area_mm2(allowed_polygon_mm) if allowed_polygon_mm is not None else width_mm * height_mm
                side_lengths_mm = polygon_side_lengths_mm(allowed_polygon_mm) if allowed_polygon_mm is not None else []
            except Exception:
                allowed_polygon_mm = None
                real_area_mm2 = width_mm * height_mm
                side_lengths_mm = []
            
            # Add object to list
            obj_idx = len(current_objects) + 1
            packing_result = old_objects.get(obj_idx, [])
            
            # Calculate object center position for better placement
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                center_x_px = int(moments['m10'] / moments['m00'])
                center_y_px = int(moments['m01'] / moments['m00'])
                center_x_mm = center_x_px * scale
                center_y_mm = center_y_px * scale
            else:
                center_x_mm = width_mm / 2
                center_y_mm = height_mm / 2
            
            current_objects.append({
                'contour': contour,
                'simplified_contour': simplified_contour,
                'transformed_contour': transformed_contour,  # Original contour for accurate display
                'transformed_simplified': transformed_simplified,  # Simplified for cutting validation
                'box': box,  # Keep for backward compatibility
                'transformed_box': transformed_box,  # Keep for backward compatibility
                'w_mm': width_mm,
                'h_mm': height_mm,
                'allowed_polygon_mm': allowed_polygon_mm,
                'real_area_mm2': real_area_mm2,
                'side_lengths_mm': side_lengths_mm,
                'packing_result': packing_result  # Restore previous results
            })
        
        # If we have stable objects, apply smoothing
        if len(obj_position_history) == STABILIZATION_BUFFER_SIZE:
            obj_position_history.pop(0)
        
        # Add current objects to history
        if current_objects:
            obj_position_history.append(current_objects)
        
        # Apply position smoothing if we have enough history
        if len(obj_position_history) > 0:
            # Use the most recent calculated objects but smooth their positions
            all_objects = current_objects
    
    # Process mouse click for object selection
    if click_position:
        for i, obj in enumerate(all_objects):
            # Check if click is inside object using simplified contour for stability
            if 'transformed_simplified' in obj:
                # Use simplified contour for more stable click detection
                contour_points = obj['transformed_simplified'].reshape(-1,2)
            elif 'transformed_contour' in obj:
                # Fall back to original contour
                contour_points = obj['transformed_contour'].reshape(-1,2)
            else:
                # Fall back to bounding box
                contour_points = obj['transformed_box'].reshape(-1,2)
                
            if point_in_polygon(click_position, contour_points):
                obj_idx = i + 1  # +1 for display from 1
                
                # Toggle selection
                if obj_idx in selected_objects:
                    selected_objects.remove(obj_idx)
                    print(f"[INFO] Deselected object #{obj_idx}: {obj['w_mm']:.0f}×{obj['h_mm']:.0f} mm")
                else:
                    # Respect pre-chosen expected objects count if provided
                    try:
                        max_objs = globals().get('EXPECTED_OBJECTS_COUNT', None)
                    except Exception:
                        max_objs = None
                    if max_objs is not None and len(selected_objects) >= max_objs:
                        print(f"[WARN] Already selected {len(selected_objects)}/{max_objs} objects — deselect one to choose another")
                    else:
                        selected_objects.append(obj_idx)
                        print(f"[INFO] Selected object #{obj_idx}: {obj['w_mm']:.0f}×{obj['h_mm']:.0f} mm")
                    
                # Mark distribution as unstable on selection change
                distribution_stable = False
                break
        click_position = None  # Reset click position
    
    # Draw all objects using actual contours (green for selected, yellow for others)
    for i, obj in enumerate(all_objects):
        if i >= len(all_objects):  # Safety check
            continue
            
        obj_idx = i + 1
        # Use actual contour if available, otherwise fall back to box
        if 'transformed_contour' in obj:
            contour_to_draw = obj['transformed_contour']
        else:
            contour_to_draw = obj['transformed_box']
            
        # Draw with better visual quality
        if obj_idx in selected_objects:
            # Selected objects: green outline with semi-transparent fill
            cv2.polylines(frame, [contour_to_draw], True, (0,255,0), 3, cv2.LINE_AA)
            # Add semi-transparent fill for selected objects
            overlay = frame.copy()
            cv2.fillPoly(overlay, [contour_to_draw], (0,128,0))
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        else:
            # Unselected objects: yellow outline
            cv2.polylines(frame, [contour_to_draw], True, (0,255,255), 2, cv2.LINE_AA)
            
                # Optional: Draw object number
        if 'transformed_contour' in obj and len(obj['transformed_contour']) > 0:
            # Calculate centroid of contour for label placement
            moments = cv2.moments(contour_to_draw)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                # Draw object number
                cv2.putText(frame, str(obj_idx), (cx-10, cy+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, str(obj_idx), (cx-10, cy+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)
    

    # ───── Draw user-drawn shapes with mm dimensions ─────
    try:
        def rect_pts_to_screen(pts_rect_xy):
            if pts_rect_xy is None or len(pts_rect_xy) == 0:
                return None
            pts = np.array(pts_rect_xy, dtype=np.float32).reshape(-1, 1, 2)
            scr = cv2.perspectiveTransform(pts, transform_matrix).astype(int).reshape(-1, 2)
            return scr

        def draw_text_with_outline(img, text, org, scale_txt=0.5, color=(255,255,255)):
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale_txt, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, color, 1, cv2.LINE_AA)

        # Draw finalized shapes
        for shp in user_drawn_shapes:
            if shp['type'] == 'poly':
                pts_rect = shp['pts']
                if len(pts_rect) < 2:
                    continue
                pts_scr = rect_pts_to_screen(pts_rect)
                if pts_scr is None:
                    continue
                # Outline and fill light
                cv2.polylines(frame, [pts_scr.reshape(-1,1,2)], shp.get('closed', False), (0,0,255), 2, cv2.LINE_AA)
                # Edge lengths suppressed on main window
            elif shp['type'] == 'circle':
                cx, cy = shp['center']
                r_px = float(shp.get('radius_px', 0.0))
                if r_px <= 0.5:
                    continue
                # Center to screen
                ctr_scr = rect_pts_to_screen([(cx, cy)])[0]
                # Radius vector along +x in rect coordinates
                edge_scr = rect_pts_to_screen([(cx + r_px, cy)])[0]
                radius_scr = int(max(1, np.linalg.norm(edge_scr - ctr_scr)))
                cv2.circle(frame, tuple(ctr_scr), radius_scr, (0,0,255), 2, cv2.LINE_AA)
                # Diameter label suppressed on main window
            elif shp['type'] == 'arc':
                cx, cy = shp['center']
                r_px = float(shp.get('radius_px', 0.0))
                a0 = float(shp.get('start_deg', 0.0))
                a1 = float(shp.get('end_deg', 0.0))
                ccw = bool(shp.get('ccw', True))
                if r_px <= 0.5:
                    continue
                # Build a set of points along the arc in rect coords, then transform
                # Normalize angles
                ang0 = a0
                ang1 = a1
                if ccw:
                    if ang1 < ang0:
                        ang1 += 360.0
                else:
                    if ang1 > ang0:
                        ang1 -= 360.0
                num = max(16, int(abs(ang1-ang0)))
                arc_pts_rect = []
                for t in np.linspace(ang0, ang1, num=num):
                    rad = math.radians(t)
                    arc_pts_rect.append((cx + r_px*math.cos(rad), cy + r_px*math.sin(rad)))
                arc_pts_scr = rect_pts_to_screen(arc_pts_rect)
                cv2.polylines(frame, [arc_pts_scr.reshape(-1,1,2)], False, (0,0,255), 2, cv2.LINE_AA)
                # Measurement labels suppressed on main window

        # If auto-placed from library, paint original polygon inside each placed rect for better fidelity
        try:
            if hasattr(np, 'array'):
                for u in [obj for obj in sum([[obj.get('packing_result', [])] for obj in all_objects], [])]:
                    pass
        except Exception:
            pass
            # If this poly came from a library entry (auto-placed), try overlaying its actual polygon scaled into the placed rect (future)
            # Placeholder for future: overlay original poly

        # Placement preview from library (show even without mouse click)
        if PLACE_MODE and shapes_library:
            # Anchor at current mouse if available; else center of rect
            if mouse_pos_rect is not None:
                anchor = mouse_pos_rect
            else:
                anchor = (rect.shape[1] * 0.5, rect.shape[0] * 0.5)
            entry = shapes_library[min(place_idx, len(shapes_library)-1)]
            inst = build_shape_instance_from_lib(entry, anchor, place_rotation_deg, float(scale))
            if inst:
                # Draw instance as preview in blue
                if inst['type'] == 'poly':
                    pts_scr = rect_pts_to_screen(inst['pts'])
                    cv2.polylines(frame, [pts_scr.reshape(-1,1,2)], True, (255,0,0), 2, cv2.LINE_AA)
                elif inst['type'] == 'circle':
                    ctr_scr = rect_pts_to_screen([inst['center']])[0]
                    r_scr = int(max(1, inst['radius_px']))
                    cv2.circle(frame, tuple(ctr_scr), r_scr, (255,0,0), 2, cv2.LINE_AA)
                elif inst['type'] == 'arc':
                    cx, cy = inst['center']
                    r_px = inst['radius_px']
                    a0 = inst['start_deg']
                    a1 = inst['end_deg']
                    ccw = inst.get('ccw', True)
                    ang0 = a0
                    ang1 = a1
                    if ccw:
                        if ang1 < ang0:
                            ang1 += 360.0
                    else:
                        if ang1 > ang0:
                            ang1 -= 360.0
                    num = max(16, int(abs(ang1-ang0)))
                    arc_pts_rect = []
                    for t in np.linspace(ang0, ang1, num=num):
                        rad = math.radians(t)
                        arc_pts_rect.append((cx + r_px*math.cos(rad), cy + r_px*math.sin(rad)))
                    arc_pts_scr = rect_pts_to_screen(arc_pts_rect)
                    cv2.polylines(frame, [arc_pts_scr.reshape(-1,1,2)], False, (255,0,0), 2, cv2.LINE_AA)

        # Draw current in-progress shape preview
        if DRAW_MODE and current_shape is not None:
            if current_shape.get('type') == 'poly':
                pts = current_shape.get('pts', [])
                if len(pts) >= 1:
                    pts_scr = rect_pts_to_screen(pts)
                    cv2.polylines(frame, [pts_scr.reshape(-1,1,2)], False, (255,0,0), 1, cv2.LINE_AA)
                    # Rubber-band to mouse
                    if mouse_pos_rect is not None:
                        tail = rect_pts_to_screen([pts[-1], mouse_pos_rect])
                        p_start = tuple(tail[0])
                        p_end = tuple(tail[1])
                        # Draw direction arrow for current segment
                        cv2.arrowedLine(frame, p_start, p_end, (255,0,0), 2, cv2.LINE_AA, tipLength=0.15)
                        # Length label in mm
                        try:
                            seg_len_mm = float(np.linalg.norm(np.array(mouse_pos_rect, dtype=np.float32) - np.array(pts[-1], dtype=np.float32))) * float(scale)
                            mid = ((tail[0] + tail[1]) // 2).astype(int)
                            cv2.putText(frame, f"{seg_len_mm:.1f} mm", (int(mid[0])+6, int(mid[1])-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                            cv2.putText(frame, f"{seg_len_mm:.1f} mm", (int(mid[0])+6, int(mid[1])-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)
                        except Exception:
                            pass
            elif current_shape.get('type') == 'circle':
                if 'center' in current_shape and current_shape.get('radius_px', 0.0) > 0 and mouse_pos_rect is not None:
                    cx, cy = current_shape['center']
                    ctr_scr = rect_pts_to_screen([(cx, cy)])[0]
                    edge_scr = rect_pts_to_screen([(mouse_pos_rect[0], mouse_pos_rect[1])])[0]
                    radius_scr = int(max(1, np.linalg.norm(edge_scr - ctr_scr)))
                    cv2.circle(frame, tuple(ctr_scr), radius_scr, (255,0,0), 1, cv2.LINE_AA)
            elif current_shape.get('type') == 'arc':
                phase = current_shape.get('phase', 0)
                if phase >= 0 and 'center' in current_shape:
                    ctr_scr = rect_pts_to_screen([current_shape['center']])[0]
                    cv2.circle(frame, tuple(ctr_scr), 3, (255,0,0), -1, cv2.LINE_AA)
                if phase >= 1 and 'start' in current_shape:
                    # Draw radius line to start
                    start_scr = rect_pts_to_screen([current_shape['start']])[0]
                    cv2.line(frame, tuple(ctr_scr), tuple(start_scr), (255,0,0), 1, cv2.LINE_AA)
                if phase == 2 and mouse_pos_rect is not None:
                    # Preview arc to current mouse
                    cx, cy = current_shape['center']
                    sx, sy = current_shape['start']
                    r_px = current_shape['radius_px']
                    a0 = math.degrees(math.atan2(sy - cy, sx - cx))
                    ex, ey = mouse_pos_rect
                    a1 = math.degrees(math.atan2(ey-cy, ex-cx))
                    ang0 = a0
                    ang1 = a1
                    if ang1 < ang0:
                        ang1 += 360.0
                    num = max(16, int(abs(ang1-ang0)))
                    arc_pts_rect = []
                    for t in np.linspace(ang0, ang1, num=num):
                        rad = math.radians(t)
                        arc_pts_rect.append((cx + r_px*math.cos(rad), cy + r_px*math.sin(rad)))
                    arc_pts_scr = rect_pts_to_screen(arc_pts_rect)
                    cv2.polylines(frame, [arc_pts_scr.reshape(-1,1,2)], False, (255,0,0), 1, cv2.LINE_AA)
        
        # Measurements panel with large schematic drawing and labels
        def draw_text_bg(img, text, org, scale_txt=0.6, txt_color=(0,0,0), bg_color=(255,255,255)):
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale_txt, 1)
            x, y = int(org[0]), int(org[1])
            cv2.rectangle(img, (x-4, y-th-6), (x+tw+4, y+4), bg_color, -1)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale_txt, txt_color, 1, cv2.LINE_AA)

        win_name = 'Measurements'
        show_panel = False
        panel = None
        if DRAW_MODE and (current_shape is not None or last_finalized_shape is not None):
            shp = current_shape if current_shape is not None else last_finalized_shape
            st = shp.get('type')
            panel_w, panel_h = 700, 500
            pad = 60
            panel = np.ones((panel_h, panel_w, 3), dtype=np.uint8) * 255
            if st == 'poly':
                pts = shp.get('pts', [])
                if len(pts) >= 2:
                    pts_np = np.array(pts, dtype=np.float32)
                    min_xy = pts_np.min(axis=0)
                    max_xy = pts_np.max(axis=0)
                    size = np.maximum(max_xy - min_xy, 1.0)
                    sx = (panel_w - 2*pad) / float(size[0])
                    sy = (panel_h - 2*pad) / float(size[1])
                    sf = float(min(sx, sy))
                    pts_draw = ((pts_np - min_xy) * sf + pad).astype(np.int32)
                    cv2.polylines(panel, [pts_draw.reshape(-1,1,2)], bool(shp.get('closed', False)), (0,0,200), 3, cv2.LINE_AA)
                    # edge labels
                    closed = bool(shp.get('closed', False))
                    cnt = len(pts_np) if closed else len(pts_np)-1
                    perim = 0.0
                    for i in range(cnt):
                        a_px = pts_np[i]; b_px = pts_np[(i+1)%len(pts_np)]
                        a_d = pts_draw[i]; b_d = pts_draw[(i+1)%len(pts_draw)]
                        L = float(np.linalg.norm(b_px - a_px)) * float(scale)
                        perim += L
                        mid = ((a_d + b_d)/2).astype(int)
                        draw_text_bg(panel, f"{L:.1f} mm", (int(mid[0])+6, int(mid[1])-6))
                    title = 'Polygon — side lengths'
                    cv2.putText(panel, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
                    cv2.putText(panel, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,150), 1, cv2.LINE_AA)
                    if closed:
                        draw_text_bg(panel, f"Perimeter: {perim:.1f} mm", (10, panel_h-20))
                    show_panel = True
                elif len(pts) == 1 and mouse_pos_rect is not None:
                    # Show first segment length immediately while placing second point
                    p0 = np.array(pts[0], dtype=np.float32)
                    p1 = np.array(mouse_pos_rect, dtype=np.float32)
                    seg = np.stack([p0, p1], axis=0)
                    min_xy = seg.min(axis=0)
                    max_xy = seg.max(axis=0)
                    size = np.maximum(max_xy - min_xy, 1.0)
                    sx = (panel_w - 2*pad) / float(max(1e-3, size[0]))
                    sy = (panel_h - 2*pad) / float(max(1e-3, size[1]))
                    sf = float(min(sx, sy))
                    seg_draw = ((seg - min_xy) * sf + pad).astype(np.int32)
                    cv2.line(panel, tuple(seg_draw[0]), tuple(seg_draw[1]), (0,0,200), 3, cv2.LINE_AA)
                    L = float(np.linalg.norm(p1 - p0)) * float(scale)
                    mid = ((seg_draw[0] + seg_draw[1]) / 2).astype(int)
                    draw_text_bg(panel, f"{L:.1f} mm", (int(mid[0])+6, int(mid[1])-6))
                    title = 'Segment length'
                    cv2.putText(panel, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
                    cv2.putText(panel, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,150), 1, cv2.LINE_AA)
                    show_panel = True
            elif st == 'circle':
                r_px = float(shp.get('radius_px', 0.0))
                if r_px > 0:
                    d_px = 2.0 * r_px
                    sf = float(min((panel_w-2*pad)/d_px, (panel_h-2*pad)/d_px))
                    r_draw = int(max(1, round(r_px * sf)))
                    ctr = (panel_w//2, panel_h//2)
                    cv2.circle(panel, ctr, r_draw, (200,0,0), 3, cv2.LINE_AA)
                    d_mm = 2.0 * r_px * float(scale)
                    draw_text_bg(panel, f"Diameter: {d_mm:.1f} mm", (ctr[0]-100, ctr[1]-r_draw-16))
                    title = 'Circle'
                    cv2.putText(panel, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
                    cv2.putText(panel, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,150), 1, cv2.LINE_AA)
                    show_panel = True
            elif st == 'arc':
                r_px = float(shp.get('radius_px', 0.0))
                a0 = float(shp.get('start_deg', 0.0))
                a1 = float(shp.get('end_deg', 0.0))
                ccw = bool(shp.get('ccw', True))
                if r_px > 0:
                    ang0, ang1 = a0, a1
                    if ccw and ang1 < ang0: ang1 += 360.0
                    if (not ccw) and ang1 > ang0: ang1 -= 360.0
                    num = max(32, int(abs(ang1-ang0)))
                    arc_pts = []
                    for t in np.linspace(ang0, ang1, num=num):
                        rad = math.radians(t)
                        arc_pts.append((r_px*math.cos(rad), r_px*math.sin(rad)))
                    arc_np = np.array(arc_pts, dtype=np.float32)
                    min_xy = arc_np.min(axis=0)
                    max_xy = arc_np.max(axis=0)
                    size = np.maximum(max_xy - min_xy, 1.0)
                    sf = float(min((panel_w-2*pad)/size[0], (panel_h-2*pad)/size[1]))
                    arc_draw = ((arc_np - min_xy) * sf + pad).astype(np.int32)
                    cv2.polylines(panel, [arc_draw.reshape(-1,1,2)], False, (0,0,200), 3, cv2.LINE_AA)
                    sweep = abs(ang1-ang0)
                    r_mm = r_px * float(scale)
                    arc_len = math.radians(sweep) * r_mm
                    draw_text_bg(panel, f"Radius: {r_mm:.1f} mm", (10, panel_h-60))
                    draw_text_bg(panel, f"Angle: {sweep:.1f} deg", (10, panel_h-36))
                    draw_text_bg(panel, f"Arc: {arc_len:.1f} mm", (10, panel_h-12))
                    title = 'Arc'
                    cv2.putText(panel, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
                    cv2.putText(panel, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,150), 1, cv2.LINE_AA)
                    show_panel = True

        if show_panel and panel is not None:
            # Save button hint and support for key press 'S'
            cv2.putText(panel, "Press S to save this shape to library", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(panel, "Press S to save this shape to library", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,200), 1, cv2.LINE_AA)
            cv2.imshow(win_name, panel)
        else:
            try:
                cv2.destroyWindow(win_name)
            except Exception:
                pass
    except Exception as e:
        print(f"[WARN] Drawing overlay failed: {e}")

    # Show connection strength when debug mode is on
    if SEGMENTATION_CONFIG.get('debug_mode', False):
        connection_text = f"Connection: {SEGMENTATION_CONFIG['connection_strength']}"
        cv2.putText(frame, connection_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, connection_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 1)
    

    

    
    # Display distribution of cuts on selected objects
    if selected_objects:
        try:
            # If we must always show input on selected objects: pack input shapes independently per object
            if SHOW_INPUT_ON_SELECTED:
                placements = {}
            else:
                # Only recalculate placements if we have remaining cuts
                if remaining_pieces:
                    placements = distribute_cuts_across_objects(scale)
                else:
                    placements = {}
            
            valid_selected_objects = [idx for idx in selected_objects if 0 < idx <= len(all_objects)]
            
            # Check if we need to recreate packing
            selection_changed = len(selected_objects) != last_selection_count
            # Update only when selection count changed. Otherwise, keep previous value to avoid frequent rebuilds
            if selection_changed:
                last_selection_count = len(selected_objects)
            
            if not SHOW_INPUT_ON_SELECTED and selection_changed:
                print("[INFO] Selection changed or distribution not stable - recalculating...")
                # Reset packing results for each object
                for obj_idx in range(1, len(all_objects) + 1):
                    if obj_idx - 1 < len(all_objects):
                        all_objects[obj_idx - 1]['packing_result'] = []
                        
                remaining_pieces = cutting_pieces.copy()
                distribute_cuts_across_objects(scale)
                
                # Check if all pieces are placed
                total_placed = 0
                for obj in all_objects:
                    if 'packing_result' in obj:
                        total_placed += len(obj['packing_result'])
                        
                if total_placed < len(cutting_pieces):
                    print(f"[WARN] Only {total_placed}/{len(cutting_pieces)} pieces placed. Will try again.")
                
                # If all pieces are placed, consider distribution stable
                if len(remaining_pieces) == 0:
                    distribution_stable = True
            
            # Use continuous numbering between objects
            current_item_idx = 1
            
            # Sort selected objects by object index to ensure consistent numbering
            for obj_idx in sorted(valid_selected_objects):
                obj = all_objects[obj_idx - 1]
                # Determine what to draw: either input shapes on this object, or the stored packing results
                if SHOW_INPUT_ON_SELECTED:
                    # In CSV mode, if shapes already placed, use existing packing_result
                    # This prevents redundant recalculation after distribute_csv_shapes_on_selected_object
                    if STARTUP_MODE == 'csv_cut' and csv_shapes_placed and obj.get('packing_result'):
                        packing_result = obj['packing_result']
                    else:
                        # Use remaining_pieces to show current batch, or cutting_pieces if in initial state
                        pieces_to_show = remaining_pieces if remaining_pieces else cutting_pieces
                        # Calculate adaptive cell size for cache key
                        adaptive_cell_mm = calculate_adaptive_cell_size(pieces_to_show, obj['w_mm'], obj['h_mm'])
                        # Build stable key: object geometry + pieces set
                        key = (
                            obj_idx,
                            polygon_signature(obj.get('allowed_polygon_mm'), obj['w_mm'], obj['h_mm']),
                            pieces_signature(pieces_to_show),
                            round(adaptive_cell_mm, 3),
                            round(ALLOWED_MARGIN_MM, 3)
                        )
                        if key not in _packing_cache:
                            _packing_cache[key] = compute_best_packing_for_object(obj, pieces_to_show)
                        packing_result = _packing_cache[key]
                        # If cache returns fewer items than requested, re-run once with a different cell
                        if len(packing_result) < len(pieces_to_show):
                            alt_cell = max(0.5, adaptive_cell_mm * 0.75)
                            alt_grid = None
                            if obj.get('allowed_polygon_mm') is not None and len(obj.get('allowed_polygon_mm')) >= 3:
                                grid_w = int(math.ceil(obj['w_mm'] / alt_cell))
                                grid_h = int(math.ceil(obj['h_mm'] / alt_cell))
                                alt_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
                                pts = np.array([[p[0] / alt_cell, p[1] / alt_cell] for p in obj['allowed_polygon_mm']], dtype=np.float32)
                                pts = np.round(pts).astype(np.int32)
                                cv2.fillPoly(alt_grid, [pts], 1)
                                try:
                                    expand_iter = max(0, int(round(ALLOWED_MARGIN_MM / alt_cell)))
                                    if expand_iter > 0:
                                        kernel = np.ones((3, 3), dtype=np.uint8)
                                        alt_grid = cv2.dilate(alt_grid, kernel, iterations=expand_iter)
                                except Exception:
                                    pass
                            alt_packer = PolygonGridPacker(obj['w_mm'], obj['h_mm'], obj.get('allowed_polygon_mm'), cell_mm=alt_cell, seed=0, allowed_grid=alt_grid)
                            for piece in pieces_to_show:
                                w, h, s = piece[:3]
                                piece_id = piece[3] if len(piece) > 3 else -1
                                shape_name = piece[4] if len(piece) > 4 else None
                                alt_packer.insert(w, h, s, piece_id=piece_id, shape_name=shape_name)
                            if len(alt_packer.used) > len(packing_result):
                                _packing_cache[key] = alt_packer.used
                                packing_result = alt_packer.used
                    
                        # Store packing result in object for later retrieval when pressing Next
                        obj['packing_result'] = packing_result.copy()
                    
                    utilization, used, _ = draw_layout(
                        frame,
                        transform_matrix,
                        obj['box'],
                        scale,
                        obj_idx,
                        [],
                        packing_result,
                        start_idx=current_item_idx,
                        allowed_polygon_mm=obj.get('allowed_polygon_mm')
                    )
                    # Console warning and overlay if not everything fits (only when not using cached CSV result)
                    if not (STARTUP_MODE == 'csv_cut' and csv_shapes_placed):
                        total_needed = len(pieces_to_show) if 'pieces_to_show' in dir() else len(packing_result)
                        if used < total_needed:
                            print(f"[WARN] Object #{obj_idx}: placed {used}/{total_needed} pieces - not enough space for all input shapes")
                else:
                    # Check for packing results
                    packing_result = obj.get('packing_result', [])
                    # Draw layout for this object using the stored packing result and continue numbering
                    # Use the original box coordinates (not transformed) for proper coordinate mapping
                    print(f"[DEBUG] Drawing layout for object #{obj_idx} with {len(packing_result)} packing results")
                    print(f"[DEBUG] Packing result: {packing_result}")
                    print(f"[DEBUG] Object dimensions: {obj['w_mm']}x{obj['h_mm']} mm")
                    utilization, used, total = draw_layout(frame, transform_matrix, obj['box'], scale, obj_idx, 
                                                 [], packing_result, start_idx=current_item_idx)
                

                
                # Update the starting index for the next object
                current_item_idx += used
                
                # Safely calculate and use centroid for text placement
                try:
                    # Get the transformed box in proper shape
                    reshaped_box = obj['transformed_box'].reshape(-1, 2)
                    
                    # Only proceed if we have valid points
                    if reshaped_box.shape[0] > 0:
                        # Calculate centroid
                        centroid = np.mean(reshaped_box, axis=0).astype(int)
                        
                        # Ensure we have both x and y coordinates
                        if len(centroid) >= 2:
                            # Display object information near the object
                            # Determine total to display based on mode
                            if STARTUP_MODE == 'csv_cut' and csv_shapes_placed:
                                total_disp = len(packing_result)  # Use current packing result count
                            elif SHOW_INPUT_ON_SELECTED:
                                _pieces_for_display = remaining_pieces if remaining_pieces else cutting_pieces
                                total_disp = len(_pieces_for_display)
                            else:
                                total_disp = total
                            cv2.putText(frame, f"#{obj_idx}: {used}/{total_disp} pieces", 
                                          (centroid[0]-50, centroid[1]), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                        else:
                            # Fallback text position if centroid calculation fails
                            _pieces_for_display = remaining_pieces if remaining_pieces else cutting_pieces
                            total_disp = len(_pieces_for_display) if SHOW_INPUT_ON_SELECTED else total
                            cv2.putText(frame, f"#{obj_idx}: {used}/{total_disp} pieces", 
                                      (50, 140 + 30*obj_idx), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                    else:
                        # Fallback text position if no valid points
                        _pieces_for_display = remaining_pieces if remaining_pieces else cutting_pieces
                        total_disp = len(_pieces_for_display) if SHOW_INPUT_ON_SELECTED else total
                        cv2.putText(frame, f"#{obj_idx}: {used}/{total_disp} pieces", 
                                  (50, 140 + 30*obj_idx), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                except Exception as e:
                    print(f"[WARN] Error calculating centroid: {e}")
                    # Fallback text position
                    total_disp = len(cutting_pieces) if SHOW_INPUT_ON_SELECTED else total
                    cv2.putText(frame, f"#{obj_idx}: {used}/{total_disp} pieces", 
                              (50, 140 + 30*obj_idx), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        except Exception as e:
            print(f"[ERROR] Error in distribution: {e}")
            import traceback
            traceback.print_exc()  # Print detailed stack trace for debugging
    
                # Display summary information
            if selected_objects:
                # Show selected count with optional target number
                try:
                    max_objs = globals().get('EXPECTED_OBJECTS_COUNT', None)
                except Exception:
                    max_objs = None
                if max_objs is not None:
                    sel_text = f"Selected: {len(selected_objects)}/{max_objs} objects"
                else:
                    sel_text = f"Selected: {len(selected_objects)} objects"
                cv2.putText(frame, sel_text, 
                           (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.putText(frame, f"Remaining: {len(remaining_pieces)}/{len(cutting_pieces)} pieces", 
                           (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                
                # Display placed vs total pieces
                placed_pieces = len(cutting_pieces) - len(remaining_pieces)
                if placed_pieces == len(cutting_pieces):
                    cv2.putText(frame, f"All {placed_pieces} pieces placed successfully!", 
                              (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                else:
                    cv2.putText(frame, f"Placed: {placed_pieces}/{len(cutting_pieces)} pieces", 
                              (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                
                # Show CSV mode specific information
                if STARTUP_MODE == 'csv_cut':
                    cv2.putText(frame, "CSV Mode: Press 'N' for next batch", 
                              (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
    # Update cache tracking variables at the end of processing
    _last_frame_processed = frame_counter
    _cache_valid = True
    
    # Create signatures for objects and pieces to track changes
    if all_objects:
        objects_sig = tuple((obj.get('w_mm', 0), obj.get('h_mm', 0), 
                           polygon_signature(obj.get('allowed_polygon_mm'), obj.get('w_mm', 0), obj.get('h_mm', 0))) 
                          for obj in all_objects)
        _last_objects_signature = objects_sig
    
    if cutting_pieces:
        _last_pieces_signature = pieces_signature(cutting_pieces)

# ───── Main loop ───────────────────────────────────────────
def main():
    global selected_objects, all_objects, remaining_pieces, obj_position_history, frame_counter
    global STARTUP_MODE, STARTUP_LOAD_INDEX, PLACE_MODE, DRAW_MODE, DRAW_TYPE
    global current_shape, user_drawn_shapes, last_finalized_shape
    global place_idx, place_rotation_deg, shapes_library, csv_shapes_placed
    
    try:
        start_log_capture()
        # Prompt startup mode
        prompt_startup_mode()
        if STARTUP_MODE == 'rc_place':
            prompt_rc_pieces()
        elif STARTUP_MODE == 'editor':
            run_shape_editor()
            print('[INFO] Завершен режим редактора. Запуск живого режима...')
        elif STARTUP_MODE == 'lib_place':
            # Load lib ensured in prompt; set place mode and preselect index
            PLACE_MODE = False  # use auto-distribution via SHOW_INPUT_ON_SELECTED

        # Initialize Intel RealSense camera
        print("[INFO] Initializing Intel RealSense camera...")
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Configure RGB stream (you can adjust resolution as needed)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start the pipeline
        try:
            pipeline.start(config)
            print("[INFO] Successfully initialized RealSense camera")
        except Exception as e:
            print(f"[ERROR] Failed to start RealSense camera: {e}")
            print("[INFO] Please ensure RealSense camera is connected and drivers are installed")
            return
        
        cv2.namedWindow("Cutting-Table")
        cv2.setMouseCallback("Cutting-Table", mouse_callback)
        
        bg, det = None, None
        
        while True:
            # Get frames from RealSense camera
            try:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    print("[ERROR] Failed to get color frame")
                    continue
                
                # Convert to numpy array
                frame = np.asanyarray(color_frame.get_data())
            except Exception as e:
                print(f"[ERROR] Failed to read frame from RealSense: {e}")
                continue
                
            corners, ids = detect_corners(frame)
            if ids is None or len(ids) != 4:
                cv2.putText(frame, "Searching for ArUco markers...", (30, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.imshow("Cutting-Table", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
                
            zone = order_zone(corners)
            cv2.polylines(frame, [zone.astype(int)], True, (0,255,0), 1)
            scale = mm_per_px(corners)
            # Keep global mm-per-pixel scale updated for saving shapes from Measurements
            try:
                global CURRENT_SCALE_MM_PER_PX
                CURRENT_SCALE_MM_PER_PX = float(scale)
            except Exception:
                pass
            width_px, height_px = np.linalg.norm(zone[0]-zone[1]), np.linalg.norm(zone[0]-zone[3])
            
            if det is None:
                det = (int(width_px), int(height_px))
                
            H = cv2.getPerspectiveTransform(zone.astype(np.float32),
                np.array([[0,0],[width_px-1,0],[width_px-1,height_px-1],[0,height_px-1]], np.float32))
            rect = cv2.warpPerspective(frame, H, det)

            if bg is None:
                bg = cv2.GaussianBlur(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY), (5,5), 0)
                cv2.putText(frame, "Background set, waiting for objects...", (30, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.imshow("Cutting-Table", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            mask = segment(rect, bg)
            process(mask, rect, scale, frame, zone)
            
            # Live draw: enable drawing only in this startup mode
            if STARTUP_MODE == 'live_draw':
                DRAW_MODE = True
                if DRAW_TYPE is None:
                    DRAW_TYPE = 'poly'

            # CSV cutting mode: automatically place shapes on selected objects (only once)
            if STARTUP_MODE == 'csv_cut' and selected_objects and remaining_pieces and not csv_shapes_placed:
                # Auto-distribute CSV shapes on selected objects using the best packing algorithm
                try:
                    print(f"[INFO] Auto-placing {len(remaining_pieces)} CSV shapes on {len(selected_objects)} selected objects")
                    
                    # Use cached packing result if available (already computed in process/draw_layout)
                    total_placed = 0
                    for obj_idx in selected_objects:
                        if obj_idx <= 0 or obj_idx > len(all_objects):
                            continue
                        obj = all_objects[obj_idx - 1]
                        
                        # Track pieces before placement for reporting
                        pieces_before = len(remaining_pieces)
                        
                        # Check if we already have a packing result from preview
                        packing_result = obj.get('packing_result')
                        
                        if not packing_result:
                            # Only compute if not already cached
                            packing_result = compute_best_packing_for_object(obj, remaining_pieces.copy())
                            obj['packing_result'] = packing_result
                        
                        # Update remaining pieces - remove placed ones
                        placed_ids = set()
                        for placement in packing_result:
                            if len(placement) >= 7:
                                placed_ids.add(placement[6])
                        
                        # Remove placed pieces from remaining
                        remaining_pieces = [p for p in remaining_pieces if len(p) < 4 or p[3] not in placed_ids]
                        
                        total_placed += len(packing_result)
                        print(f"[INFO] Object #{obj_idx}: placed {len(packing_result)}/{pieces_before} pieces")
                        
                        # Report what couldn't be placed
                        unplaced_count = pieces_before - len(packing_result)
                        if unplaced_count > 0:
                            print(f"[WARN] Object #{obj_idx}: {unplaced_count} pieces could NOT be placed (not enough space)")
                    
                    if total_placed > 0:
                        print(f"[INFO] Auto-placed {total_placed} CSV shapes on selected objects")
                        csv_shapes_placed = True  # Mark as placed to prevent repeated placement
                        # CRITICAL: Clear packing cache to force display of new placements
                        _packing_cache.clear()
                        print(f"[DEBUG] Cleared packing cache to display new placements")
                        
                        # Additional validation: check if any placed pieces exceed boundaries
                        for obj_idx in selected_objects:
                            if obj_idx <= 0 or obj_idx > len(all_objects):
                                continue
                            obj = all_objects[obj_idx - 1]
                            packing_result = obj.get('packing_result', [])
                            
                            for i, placement in enumerate(packing_result):
                                # Unpack with piece_id support (7 elements)
                                x, y, w, h, rot, shape = placement[:6]
                                # Boundary validation is done silently now - only log once per piece
                                # Skip repeated warnings in main loop
                                pass
                                    
                except Exception as e:
                    print(f"[ERROR] Failed to auto-place CSV shapes: {e}")

            # If started in place mode with a selected library index
            if STARTUP_MODE == 'place' and PLACE_MODE and STARTUP_LOAD_INDEX is not None:
                # Keep currently selected index; allow hotkeys to change
                pass

            # Minimal instructions
            if STARTUP_MODE == 'csv_cut':
                cv2.putText(frame, "N - next batch | X - cleanup | u - undo", 
                           (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            else:
                cv2.putText(frame, "u - undo", 
                           (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imshow("Cutting-Table", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc
                break
            if key == ord('r'):  # Reset background
                bg = None
                obj_position_history = []  # Reset position history
                frame_counter = 0          # Reset frame counter
                distribution_stable = False  # Reset stability
                print("[INFO] Background reset")
            if key == ord('c'):  # Clear selection
                selected_objects = []
                remaining_pieces = cutting_pieces.copy()  # Reset remaining cuts
                distribution_stable = False  # Reset stability flag
                last_selection_count = 0     # Reset selection count
                print("[INFO] Object selection cleared")
            if key == ord('d'):  # Toggle debug mode
                toggle_debug_mode()
            if key == ord('+') or key == ord('='):  # Increase sensitivity
                adjust_segmentation_sensitivity(increase=True)
            if key == ord('-') or key == ord('_'):  # Decrease sensitivity
                adjust_segmentation_sensitivity(increase=False)
            if key == ord('0'):  # Reset segmentation config
                reset_segmentation_config()
            if key == ord('e'):  # Toggle edge technique (only edges available)
                toggle_technique(chr(key))
            if key == ord('['):  # Decrease connection strength
                adjust_connection_strength(increase=False)
            if key == ord(']'):  # Increase connection strength
                adjust_connection_strength(increase=True)
            if key == ord('f'):  # Toggle fast mode
                global FAST_MODE
                FAST_MODE = not FAST_MODE
                print(f"[INFO] Fast mode {'ON' if FAST_MODE else 'OFF'} (packing attempts: {max(1, PACKING_ATTEMPTS // (2 if FAST_MODE else 1))})")

            if key == ord('a'):  # Toggle packing algorithm
                toggle_packing_algorithm()

            if key == ord('+') or key == ord('='):  # Increase H4NP generations
                adjust_h4np_generations(increase=True)
            
            if key == ord('-') or key == ord('_'):  # Decrease H4NP generations
                adjust_h4np_generations(increase=False)

            if key == ord('b'):  # Toggle bounding box only mode
                SEGMENTATION_CONFIG['use_bounding_box_only'] = not SEGMENTATION_CONFIG.get('use_bounding_box_only', False)
                status = "enabled" if SEGMENTATION_CONFIG['use_bounding_box_only'] else "disabled"
                print(f"[INFO] Bounding box only mode {status}")
            
            if key == ord('n') or key == ord('N'):  # Next batch for CSV mode
                if STARTUP_MODE == 'csv_cut':
                    show_next_csv_batch()
                else:
                    print("[INFO] Press 'N' only works in CSV cutting mode")
            
            if key == ord('x') or key == ord('X'):  # Force cleanup of invalid placements
                if STARTUP_MODE == 'csv_cut':
                    cleanup_invalid_placements()
                else:
                    print("[INFO] Press 'X' only works in CSV cutting mode")

            # ───── Minimal hotkeys ─────
            if key == ord('u'):
                # Undo last vertex for polygon; otherwise undo last shape
                if current_shape and current_shape.get('type') == 'poly' and current_shape.get('pts'):
                    current_shape['pts'].pop()
                    if not current_shape['pts']:
                        current_shape = None
                    print('[INFO] Removed last point')
                elif user_drawn_shapes:
                    user_drawn_shapes.pop()
                    print('[INFO] Removed last shape')
            if key in (ord('s'), ord('S')) and DRAW_MODE and (current_shape is not None or last_finalized_shape is not None):
                try:
                    shp = current_shape if current_shape is not None else last_finalized_shape
                    name = input("Shape name (Enter=auto): ").strip() or f"shape_{len(shapes_library)+1}"
                    entry = None
                    if shp.get('type') == 'poly':
                        # Convert using current mm/px
                        entry = convert_drawn_shape_to_mm({'type':'poly','pts':shp.get('pts',[]),'closed':bool(shp.get('closed',False))}, float(CURRENT_SCALE_MM_PER_PX or 1.0), name)
                    elif shp.get('type') == 'circle':
                        entry = convert_drawn_shape_to_mm({'type':'circle','radius_px':shp.get('radius_px',0.0)}, float(CURRENT_SCALE_MM_PER_PX or 1.0), name)
                    elif shp.get('type') == 'arc':
                        entry = convert_drawn_shape_to_mm({'type':'arc','radius_px':shp.get('radius_px',0.0),'start_deg':shp.get('start_deg',0.0),'end_deg':shp.get('end_deg',0.0),'ccw':shp.get('ccw',True)}, float(CURRENT_SCALE_MM_PER_PX or 1.0), name)
                    if entry:
                        load_shapes_library()
                        shapes_library.append(entry)
                        save_shapes_library()
                        print(f"[INFO] Saved to library as '{name}'")
                    else:
                        print('[WARN] Nothing to save')
                except Exception as e:
                    print(f"[WARN] Save failed: {e}")
            # Hidden hotkeys to switch live draw shape type
            if key == ord('1'):
                DRAW_TYPE = 'poly'
                current_shape = None
                print('[INFO] Live Draw: polygon mode')
            if key == ord('2'):
                DRAW_TYPE = 'circle'
                current_shape = None
                print('[INFO] Live Draw: circle mode')
            if key == ord('3'):
                DRAW_TYPE = 'arc'
                current_shape = None
                print('[INFO] Live Draw: arc mode')
            # No other keys — everything остальное делаем мышью (ПКМ закрывает полигон, щелчок ставит фигуру)

            


    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'pipeline' in locals():
            pipeline.stop()
            print("[INFO] RealSense pipeline stopped")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set multiprocessing start method for compatibility across platforms
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    # Display CPU info
    cpu_count_info = cpu_count()
    worker_count = NUM_WORKERS if NUM_WORKERS > 0 else cpu_count_info
    print(f"[INFO] System has {cpu_count_info} CPU cores")
    print(f"[INFO] Multiprocessing: {'ENABLED' if ENABLE_MULTIPROCESSING else 'DISABLED'} with {worker_count} workers")

    main()
