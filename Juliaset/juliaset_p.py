# Generate Julia set images for each point in points.json
import cupy as cp
import json
import os
from PIL import Image

# Constants
CENTER = (0, 0)
THRESHOLD_SQ = 4.0
WIDTH = 1920
HEIGHT = 1080
R = 3.0  # Zoom level
MAX_ITER = 1000
POINTS_DIR = os.path.join(os.path.dirname(__file__), 'points')
POINTS_JSON = os.path.join(os.path.dirname(__file__), 'points.json')

# For supersampling
AA_LEVEL = 2


def get_coordinates(R: float) -> tuple[float, float, float, float]:
    """Returns the bounds of the complex plane centered at CENTER."""
    px, py = CENTER
    half_r = R / 2
    return (px - half_r, px + half_r, py - half_r, py + half_r)


def generate_juliaset_gpu(julia_c: complex, width: int, height: int, max_iter: int, 
                          img_name: str) -> None:
    """
    Generate and save a Julia set image using GPU acceleration.
    """
    # Supersampling: render at higher resolution
    width *= AA_LEVEL
    height *= AA_LEVEL

    # Create complex plane on GPU
    x_min, x_max, y_min, y_max = get_coordinates(R)
    real = cp.linspace(x_min, x_max, width, dtype=cp.float64)
    imag = cp.linspace(y_max, y_min, height, dtype=cp.float64)
    
    # In Julia set, the complex plane represents the initial z values
    # Must use .copy() to make writable arrays (broadcast_to returns read-only views)
    z_real = cp.broadcast_to(real[cp.newaxis, :], (height, width)).copy()
    z_imag = cp.broadcast_to(imag[:, cp.newaxis], (height, width)).copy()
    
    # C is constant for the entire image
    c_real = cp.full((height, width), julia_c.real, dtype=cp.float64)
    c_imag = cp.full((height, width), julia_c.imag, dtype=cp.float64)
    
    # Track escape iterations on GPU
    escape_iter = cp.zeros((height, width), dtype=cp.uint16)
    not_escaped = cp.ones((height, width), dtype=cp.bool_)
    
    new_real = cp.empty_like(z_real)
    new_imag = cp.empty_like(z_imag)
    
    for i in range(1, max_iter + 1):
        # Julia iteration: z = z^2 + c
        new_real = z_real * z_real - z_imag * z_imag + c_real
        new_imag = 2 * z_real * z_imag + c_imag
        
        z_real[not_escaped] = new_real[not_escaped]
        z_imag[not_escaped] = new_imag[not_escaped]
        
        z_mag_sq = z_real * z_real + z_imag * z_imag
        newly_escaped = not_escaped & (z_mag_sq > THRESHOLD_SQ)
        escape_iter[newly_escaped] = i
        not_escaped[newly_escaped] = False
        
        # check if very few pixels remain, exit early
        if cp.count_nonzero(not_escaped) < width * height * 0.001:  # <0.1% remain
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
    
    img = cp.stack([red, green, blue], axis=2)
    
    img_cpu = cp.asnumpy(img)
    os.makedirs(POINTS_DIR, exist_ok=True)
    (
        Image.fromarray(img_cpu)
            .resize((width // AA_LEVEL, height // AA_LEVEL), Image.Resampling.LANCZOS)
            .save(os.path.join(POINTS_DIR, f'{img_name}.png'))
    )
    
    # Free up memory
    del img, img_cpu, escape_iter, not_escaped, z_real, z_imag, new_real, new_imag, c_real, c_imag
    
    return


def main():
    # Load points from JSON
    with open(POINTS_JSON, 'r') as f:
        points = json.load(f)
    
    print(f"Generating Julia set images for {len(points)} points")
    print(f"Resolution: {WIDTH}x{HEIGHT}, Max iterations: {MAX_ITER}")
    print(f"Zoom level (R): {R}")
    print(f"Using GPU acceleration with CuPy")
    print(f"Output directory: {POINTS_DIR}\n")
    
    # Warm up GPU
    print("Warming up GPU...")
    test_c = complex(points['a'])
    generate_juliaset_gpu(test_c, 100, 100, 100, 'warmup')
    warmup_file = os.path.join(POINTS_DIR, 'warmup.png')
    if os.path.exists(warmup_file):
        os.remove(warmup_file)
    print("GPU ready!\n")
    
    # Generate image for each point
    for idx, (key, value) in enumerate(points.items(), 1):
        julia_c = complex(value)
        print(f"[{idx}/{len(points)}] Generating {key}.png (c = {julia_c})")
        
        generate_juliaset_gpu(julia_c, WIDTH, HEIGHT, MAX_ITER, key)
        cp.get_default_memory_pool().free_all_blocks()
    
    print(f"\nComplete! {len(points)} images saved to {POINTS_DIR}")


if __name__ == '__main__':
    main()
