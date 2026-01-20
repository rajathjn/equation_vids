# Generate Julia set images animating through a list of c-values
import cupy as cp
import time
import os
from PIL import Image
import random

# Constants
CENTER = (0, 0)  # View center in complex plane
ZOOM = 3.0       # Fixed zoom level (R = range of view)
THRESHOLD_SQ = 4.0

WIDTH = 1920
HEIGHT = 1080
MAX_ITER = 500
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_c')

# C value animation parameters
C_START = -1.5 - 0.001j # Initial c value
C_RESOLUTION = 0.001     # Minimum c change per frame (determines frame count)

# For supersampling
AA_LEVEL = 2

# User-defined list of target c-values to animate through
# The animation will go: C_START -> C_POINTS[0] -> C_POINTS[1] -> ...
C_POINTS = [
    # Add your target c-values here
    0
]


def get_coordinates(R: float) -> tuple[float, float, float, float]:
    """Returns the bounds of the complex plane centered at CENTER."""
    px, py = CENTER
    half_r = R / 2
    return (px - half_r, px + half_r, py - half_r, py + half_r)


def bezier_c(t: float, p0: complex, p1: complex, p2: complex) -> complex:
    """Calculate a point on a cubic Bezier curve."""
    return ((1 - t) ** 2) * p0 + (2 * (1 - t) * t) * p1 + (t ** 2) * p2

def get_dynamic_p1( p0: complex, p1: complex) -> complex:
    """Get the dynamic p1 for a cubic Bezier curve."""
    midpoint = (p0 + p1) / 2
    vector = p1 - p0
    perpendicular_vector = complex(-vector.imag, vector.real) * random.uniform( 0.2, 0.5)
    direction = random.choice([1, -1])
    return midpoint + direction * perpendicular_vector


def calculate_frames(c_start: complex, c_end: complex) -> int:
    """Calculate number of frames based on distance and resolution."""
    distance = abs(c_end - c_start)
    return max(1, int(distance / C_RESOLUTION))


def generate_juliaset_gpu(c_value: complex, R: float, width: int, height: int, 
                          max_iter: int, img_name: str, display: bool = False) -> None:
    """
    Generate and save a Julia set image using GPU acceleration.
    """
    width *= AA_LEVEL
    height *= AA_LEVEL

    # Create complex plane on GPU
    x_min, x_max, y_min, y_max = get_coordinates(R)
    real = cp.linspace(x_min, x_max, width, dtype=cp.float64)
    imag = cp.linspace(y_max, y_min, height, dtype=cp.float64)
    
    # In Julia set, the complex plane represents the initial z values
    z_real = cp.broadcast_to(real[cp.newaxis, :], (height, width)).copy()
    z_imag = cp.broadcast_to(imag[:, cp.newaxis], (height, width)).copy()
    
    # C is constant for the entire image
    c_real = cp.full((height, width), c_value.real, dtype=cp.float64)
    c_imag = cp.full((height, width), c_value.imag, dtype=cp.float64)
    
    # Track escape iterations on GPU
    escape_iter = cp.zeros((height, width), dtype=cp.uint16)
    not_escaped = cp.ones((height, width), dtype=cp.bool_)
    
    new_real = cp.empty_like(z_real)
    new_imag = cp.empty_like(z_imag)
    
    for i in range(1, max_iter + 1):
        new_real = z_real * z_real - z_imag * z_imag + c_real
        new_imag = 2 * z_real * z_imag + c_imag
        
        z_real[not_escaped] = new_real[not_escaped]
        z_imag[not_escaped] = new_imag[not_escaped]
        
        z_mag_sq = z_real * z_real + z_imag * z_imag
        newly_escaped = not_escaped & (z_mag_sq > THRESHOLD_SQ)
        escape_iter[newly_escaped] = i
        not_escaped[newly_escaped] = False
        
        if cp.count_nonzero(not_escaped) < width * height * 0.001:
            break
    
    # SMOOTH ITERATION - eliminates color banding
    z_mag_sq_final = z_real * z_real + z_imag * z_imag
    
    # Smooth iteration count: escape_iter + 1 - log2(log2(|z|) / log(bailout))
    # bailout = 2, so log(bailout) = log(2)
    log_zn = cp.log2(cp.abs(z_mag_sq_final))
    smooth_iter = escape_iter.astype(cp.float64) + 1 - cp.log2(cp.maximum(log_zn, 1e-13))
    
    t = smooth_iter * 0.05
    
    # COSINE PALETTE: color(t) = a + b * cos(2π(c*t + d))
    # Critical rule: a + b ≤ 1 and a - b ≥ 0 to avoid clipping
    a = cp.array([0.5, 0.5, 0.5])   # Mid brightness
    b = cp.array([0.5, 0.5, 0.5])   # Full contrast  
    c = cp.array([0.8, 1.0, 1.2])   # Different frequencies → colors
    d = cp.array([0.5, 0.5, 0.5])   # All start at black

    # color = a + b * cos(2*pi* (c * t + d))
    red = a[0] + b[0] * cp.cos(2 * cp.pi * (c[0] * t + d[0]))
    green = a[1] + b[1] * cp.cos(2 * cp.pi * (c[1] * t + d[1]))
    blue = a[2] + b[2] * cp.cos(2 * cp.pi * (c[2] * t + d[2]))
    
    # Clamp to [0, 1] and scale to [0, 255]
    red = (red * 255).clip(0, 255).astype(cp.uint8)
    green = (green * 255).clip(0, 255).astype(cp.uint8)
    blue = (blue * 255).clip(0, 255).astype(cp.uint8)
    
    # Set non-escaped pixels to black
    red[not_escaped] = 0
    green[not_escaped] = 0
    blue[not_escaped] = 0
    
    img = cp.stack([r, g, b], axis=2)
    
    img_cpu = cp.asnumpy(img)
    os.makedirs(DATA_DIR, exist_ok=True)
    (
        Image.fromarray(img_cpu)
            .resize((width // AA_LEVEL, height // AA_LEVEL), Image.Resampling.LANCZOS)
            .save(os.path.join(DATA_DIR, f'{img_name}.png'))
    )
    
    if display:
        Image.fromarray(img_cpu).show()
    
    del img, img_cpu, escape_iter, not_escaped, z_real, z_imag, new_real, new_imag, c_real, c_imag


def main():
    if not C_POINTS:
        print("Error: C_POINTS list is empty. Add target c-values to animate through.")
        return
    
    print(f"Julia Set C Animation")
    print(f"Resolution: {WIDTH}x{HEIGHT}, Max iterations: {MAX_ITER}")
    print(f"Fixed zoom level: {ZOOM}")
    print(f"C resolution: {C_RESOLUTION} per frame")
    print(f"Starting c: {C_START}")
    print(f"Target points: {len(C_POINTS)}")
    
    # Warm up GPU
    print("\nWarming up GPU...")
    generate_juliaset_gpu(C_START, ZOOM, 100, 100, 100, 'warmup', False)
    os.remove(os.path.join(DATA_DIR, 'warmup.png'))
    print("GPU ready!\n")
    
    total_start = time.time()
    frame_index = 0
    c_start = C_START
    
    for segment_idx, c_end in enumerate(C_POINTS):
        segment_frames = calculate_frames(c_start, c_end)
        distance = abs(c_end - c_start)
        
        print(f"Segment {segment_idx + 1}/{len(C_POINTS)}: {c_start:.4f} -> {c_end:.4f}")
        print(f"  Distance: {distance:.4f}, Frames: {segment_frames}")

        c_mid = get_dynamic_p1(c_start, c_end)
        
        for i in range(segment_frames):
            start = time.time()
            
            t = i / max(1, segment_frames - 1) if segment_frames > 1 else 1.0
            current_c = bezier_c(t, c_start, c_mid, c_end)
            
            frame_index += 1
            img_name = f'Juliaset_c_{frame_index:04d}'
            
            generate_juliaset_gpu(current_c, ZOOM, WIDTH, HEIGHT, MAX_ITER, img_name, False)
            cp.get_default_memory_pool().free_all_blocks()
            
            elapsed = time.time() - start
            remaining = (segment_frames - i - 1) * elapsed
            
            if i % 10 == 0 or i == segment_frames - 1:
                print(f"  Frame {i + 1}/{segment_frames} | c={current_c:.4f} | {elapsed:.2f}s | {remaining:.2f}s remaining")
            
            # Remove this break to generate full animation
            break
        
        # Move to next segment
        c_start = c_end
        break  # Remove to process all segments
    
    print(f"\nTotal frames: {frame_index}")
    print(f"Total time: {(time.time() - total_start)/60:.1f} min")


if __name__ == '__main__':
    main()
