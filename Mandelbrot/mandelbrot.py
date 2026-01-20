# Generate an image of Mandelbrot set using GPU acceleration
import cupy as cp
import time
import os
from PIL import Image

# Constants
# CENTER = (-0.7436438870371587, 0.13182590420642563)  # Tante Renate
CENTER = (-1.4002, 0)  # Tante Renate
THRESHOLD_SQ = 4.0  # Pre-computed threshold squared (2.0)
FPS = 60.0
DURATION = 60.0
WIDTH = 1920
HEIGHT = 1080
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# For supersampling
AA_LEVEL = 2


def get_coordinates(R: float) -> tuple[float, float, float, float]:
    """Returns the bounds of the complex plane centered at CENTER."""
    px, py = CENTER
    half_r = R / 2
    return (px - half_r, px + half_r, py - half_r, py + half_r)


def generate_mandelbrot_gpu(R: float, width: int, height: int, max_iter: int, 
                            img_name: str, display: bool = False) -> None:
    """
    Generate and save a Mandelbrot set image using GPU acceleration.
    """
    # Supersampling: render at higher resolution
    width *= AA_LEVEL
    height *= AA_LEVEL

    # Create complex plane on GPU
    x_min, x_max, y_min, y_max = get_coordinates(R)
    real = cp.linspace(x_min, x_max, width, dtype=cp.float64)
    imag = cp.linspace(y_max, y_min, height, dtype=cp.float64)
    
    c_real = cp.broadcast_to(real[cp.newaxis, :], (height, width))
    c_imag = cp.broadcast_to(imag[:, cp.newaxis], (height, width))
    
    # Track escape iterations on GPU
    escape_iter = cp.zeros((height, width), dtype=cp.uint16)
    not_escaped = cp.ones((height, width), dtype=cp.bool_)
    
    z_real = cp.zeros_like(c_real)
    z_imag = cp.zeros_like(c_imag)
    
    new_real = cp.empty_like(c_real)
    new_imag = cp.empty_like(c_imag)
    
    for i in range(1, max_iter + 1):
        # Mandel brot iteration: z = z^2 + c
        # Complex multiplication: (a + bi)(a + bi) = (a^2 - b^2) + 2abi
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
    os.makedirs(DATA_DIR, exist_ok=True)
    (
        Image.fromarray(img_cpu)
            .resize((width // AA_LEVEL, height // AA_LEVEL), Image.Resampling.LANCZOS)
            .save(os.path.join(DATA_DIR, f'{img_name}.png'))
    )
    
    if display:
        Image.fromarray(img_cpu).show()
    
    # Free up memory
    del img, img_cpu, escape_iter, not_escaped, z_real, z_imag, new_real, new_imag, c_real, c_imag
    
    return


def main():
    total_frames = int(FPS * DURATION)
    
    max_iter_start = 1000
    max_iter_end = 5000
    
    R_start = 3.0
    R_end = 1e-13
    
    start_index = 200
        
    print(f"Generating {total_frames} images")
    print(f"Zooming from {R_start} down to {R_end}")
    print(f"Resolution: {WIDTH}x{HEIGHT}, Max iterations: {max_iter_start} to {max_iter_end}")
    print(f"Using GPU acceleration with CuPy")
    
    # Warm up GPU
    print("Warming up GPU...")
    generate_mandelbrot_gpu(R_start, 100, 100, 100, 'warmup', False)
    os.remove(os.path.join(DATA_DIR, 'warmup.png'))
    print("GPU ready!\n")
    
    total_start = time.time()
    
    for i in range(start_index+1, total_frames+1):
        start = time.time()
        
        # This ensures the zoom 'feel' is perfectly constant at FPS
        current_R = R_start * (R_end / R_start) ** (i / total_frames)
        
        # Linearly increase iterations as we go deeper
        current_iter = max_iter_start + (max_iter_end - max_iter_start) * (i / total_frames)
        current_iter = int(current_iter)
        
        img_name = f'Mandelbrot_{i:04d}'
        
        generate_mandelbrot_gpu(current_R, WIDTH, HEIGHT, current_iter, img_name, False)
        cp.get_default_memory_pool().free_all_blocks()
        
        elapsed = time.time() - start
        remaining = (total_frames - i) * elapsed / 60
        
        if i % 10 == 0 or i == total_frames:
            print(f"Frame {i:04d}/{total_frames} | Time: {elapsed:.2f}s | Est. remaining: {remaining:.1f} min")
        
        break
    
    print(f"\nTotal time: {(time.time() - total_start)/60:.1f} min")


if __name__ == '__main__':
    main()


