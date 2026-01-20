# Generate an image of Julia set using GPU acceleration
import cupy as cp
import time
import os
from PIL import Image

# Constants
# JULIA_C = -0.7 + 0.27015j  # Interesting Julia Set constant
JULIA_C = -0.791 + 0.2j
CENTER = (0, 0)
THRESHOLD_SQ = 4.0
FPS = 60.0
DURATION = 60.0
WIDTH = 1920
HEIGHT = 1080
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def get_coordinates(R: float) -> tuple[float, float, float, float]:
    """Returns the bounds of the complex plane centered at CENTER."""
    px, py = CENTER
    half_r = R / 2
    return (px - half_r, px + half_r, py - half_r, py + half_r)


def generate_juliaset_gpu(R: float, width: int, height: int, max_iter: int, 
                          img_name: str, display: bool = False) -> None:
    """
    Generate and save a Julia set image using GPU acceleration.
    """
    # Create complex plane on GPU
    x_min, x_max, y_min, y_max = get_coordinates(R)
    real = cp.linspace(x_min, x_max, width, dtype=cp.float64)
    imag = cp.linspace(y_max, y_min, height, dtype=cp.float64)
    
    # In Julia set, the complex plane represents the initial z values
    # Must use .copy() to make writable arrays (broadcast_to returns read-only views)
    z_real = cp.broadcast_to(real[cp.newaxis, :], (height, width)).copy()
    z_imag = cp.broadcast_to(imag[:, cp.newaxis], (height, width)).copy()
    
    # C is constant for the entire image
    c_real = cp.full((height, width), JULIA_C.real, dtype=cp.float64)
    c_imag = cp.full((height, width), JULIA_C.imag, dtype=cp.float64)
    
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
    
    # Smooth iteration count to remove banding
    # smooth_iter = iter + 1 - log2(log2(|z|))
    z_mag_sq_final = z_real * z_real + z_imag * z_imag
    # Clamp to avoid log of zero or negative
    z_mag_sq_final = cp.maximum(z_mag_sq_final, 1e-13)
    log_zn = cp.log(z_mag_sq_final) / 2  # log(|z|) = log(|z|^2) / 2
    smooth_iter = escape_iter.astype(cp.float64) + 1.0 - cp.log2(cp.maximum(log_zn, 1e-13))
    
    # Normalize to 0-1 range for coloring
    smooth_iter[not_escaped] = 0
    smooth_iter = cp.maximum(smooth_iter, 0)
    
    # HSV-based colormap for vibrant, psychedelic colors
    # Hue cycles through the spectrum, saturation and value create depth
    t = smooth_iter / max_iter  # Normalize to 0-1
    
    # Create smooth, cycling hue with multiple color bands
    # Start from blue (hue=0.66 in HSV, which is 240 degrees)
    hue = (t * 10.0 + 0.66) % 1.0  # Cycle through hues 5 times, starting from blue
    saturation = 1.0 + 0.3 * cp.sin(t * cp.pi * 2)  # Varying saturation
    value = 0.9 + 0.1 * cp.cos(t * cp.pi * 4)  # Slight value variation
    value = cp.clip(value, 0, 1)
    
    # HSV to RGB conversion
    c = value * saturation
    x = c * (1 - cp.abs((hue * 6) % 2 - 1))
    m = value - c
    
    hue_section = (hue * 6).astype(cp.int32) % 6
    
    r = cp.zeros_like(hue)
    g = cp.zeros_like(hue)
    b = cp.zeros_like(hue)
    
    # Section 0: R=C, G=X, B=0
    mask = hue_section == 0
    r[mask] = c[mask]; g[mask] = x[mask]; b[mask] = 0
    # Section 1: R=X, G=C, B=0
    mask = hue_section == 1
    r[mask] = x[mask]; g[mask] = c[mask]; b[mask] = 0
    # Section 2: R=0, G=C, B=X
    mask = hue_section == 2
    r[mask] = 0; g[mask] = c[mask]; b[mask] = x[mask]
    # Section 3: R=0, G=X, B=C
    mask = hue_section == 3
    r[mask] = 0; g[mask] = x[mask]; b[mask] = c[mask]
    # Section 4: R=X, G=0, B=C
    mask = hue_section == 4
    r[mask] = x[mask]; g[mask] = 0; b[mask] = c[mask]
    # Section 5: R=C, G=0, B=X
    mask = hue_section == 5
    r[mask] = c[mask]; g[mask] = 0; b[mask] = x[mask]
    
    r = ((r + m) * 255).astype(cp.uint8)
    g = ((g + m) * 255).astype(cp.uint8)
    b = ((b + m) * 255).astype(cp.uint8)
    
    # Set interior (non-escaped) points to black
    r[not_escaped] = 0
    g[not_escaped] = 0
    b[not_escaped] = 0
    
    # Create image in RGB format (Pillow/MoviePy expect RGB)    
    img = cp.stack([r, g, b], axis=2)
    
    # Transfer to CPU and save with Pillow
    img_cpu = cp.asnumpy(img)
    os.makedirs(DATA_DIR, exist_ok=True)
    Image.fromarray(img_cpu).save(os.path.join(DATA_DIR, f'{img_name}.png'))
    
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
    
    start_index = 0
        
    print(f"Generating {total_frames} images")
    print(f"Zooming from {R_start} down to {R_end}")
    print(f"Resolution: {WIDTH}x{HEIGHT}, Max iterations: {max_iter_start} to {max_iter_end}")
    print(f"Using GPU acceleration with CuPy")
    print(f"Julia Constant: {JULIA_C}")
    
    # Warm up GPU
    print("Warming up GPU...")
    generate_juliaset_gpu(R_start, 100, 100, 100, 'warmup', False)
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
        
        img_name = f'Juliaset_{i:04d}'
        
        generate_juliaset_gpu(current_R, WIDTH, HEIGHT, current_iter, img_name, False)
        cp.get_default_memory_pool().free_all_blocks()
        
        elapsed = time.time() - start
        remaining = (total_frames - i) * elapsed / 60
        
        if i % 10 == 0 or i == total_frames:
            print(f"Frame {i:04d}/{total_frames} | Time: {elapsed:.2f}s | Est. remaining: {remaining:.1f} min")
        
        # Consistent with the Mandelbrot example, stopping after one frame
        # Remove this break to generate the full sequence
        break
    
    print(f"\nTotal time: {(time.time() - total_start)/60:.1f} min")


if __name__ == '__main__':
    main()
