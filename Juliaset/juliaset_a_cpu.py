# Generate Julia set images animating through a list of c-values - CPU version
import numpy as np
import time
import os
from PIL import Image
import random

# Constants
CENTER = (0, 0)  # View center in complex plane
ZOOM = 3.0       # Fixed zoom level (R = range of view)
THRESHOLD_SQ = 4.0

WIDTH = 3840
HEIGHT = 2160
MAX_ITER = 500
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_a')
# For supersampling
AA_LEVEL = 1.5
# Minimum c change per frame (determines frame count)
C_RESOLUTION = 0.001

import json
points: dict[str, str] = json.load(open(os.path.join(os.path.dirname(__file__), 'points.json')))


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
    if vector == 0:
        vector = p1 + complex(0.01, 0.01)
    perpendicular_vector = complex(-vector.imag, vector.real) * random.uniform( 0.2, 0.5)
    direction = random.choice([1, -1])
    return midpoint + direction * perpendicular_vector


def calculate_frames(c_start: complex, c_end: complex) -> int:
    """Calculate number of frames based on distance and resolution."""
    distance = abs(c_end - c_start)
    if distance == 0:
        return 60*2  # Default to 120 frames if no distance
    return max(1, int(distance / C_RESOLUTION))


def generate_juliaset_cpu(c_value: complex, R: float, width: int, height: int, 
                          max_iter: int, img_name: str, display: bool = False) -> None:
    """
    Generate and save a Julia set image using CPU computation.
    """
    width = int( width * AA_LEVEL )
    height = int( height * AA_LEVEL )

    # Create complex plane on CPU
    x_min, x_max, y_min, y_max = get_coordinates(R)
    real = np.linspace(x_min, x_max, width, dtype=np.float64)
    imag = np.linspace(y_max, y_min, height, dtype=np.float64)
    
    # In Julia set, the complex plane represents the initial z values
    z_real = np.broadcast_to(real[np.newaxis, :], (height, width)).copy()
    z_imag = np.broadcast_to(imag[:, np.newaxis], (height, width)).copy()
    
    # C is constant for the entire image
    c_real = np.full((height, width), c_value.real, dtype=np.float64)
    c_imag = np.full((height, width), c_value.imag, dtype=np.float64)
    
    # Track escape iterations on CPU
    escape_iter = np.zeros((height, width), dtype=np.uint16)
    not_escaped = np.ones((height, width), dtype=bool)
    
    new_real = np.empty_like(z_real)
    new_imag = np.empty_like(z_imag)
    
    for i in range(1, max_iter + 1):
        new_real = z_real * z_real - z_imag * z_imag + c_real
        new_imag = 2 * z_real * z_imag + c_imag
        
        z_real[not_escaped] = new_real[not_escaped]
        z_imag[not_escaped] = new_imag[not_escaped]
        
        z_mag_sq = z_real * z_real + z_imag * z_imag
        newly_escaped = not_escaped & (z_mag_sq > THRESHOLD_SQ)
        escape_iter[newly_escaped] = i
        not_escaped[newly_escaped] = False
        
        if np.count_nonzero(not_escaped) < width * height * 0.001:
            break
    
    # SMOOTH ITERATION - eliminates color banding
    z_mag_sq_final = z_real * z_real + z_imag * z_imag
    
    # Smooth iteration count: escape_iter + 1 - log2(log2(|z|) / log(bailout))
    # bailout = 2, so log(bailout) = log(2)
    log_zn = np.log2(np.abs(z_mag_sq_final))
    smooth_iter = escape_iter.astype(np.float64) + 1 - np.log2(np.maximum(log_zn, 1e-13))
    
    t = smooth_iter * 0.05
    
    # COSINE PALETTE: color(t) = a + b * cos(2π(c*t + d))
    # Critical rule: a + b ≤ 1 and a - b ≥ 0 to avoid clipping
    a = np.array([0.5, 0.5, 0.5])   # Mid brightness
    b = np.array([0.5, 0.5, 0.5])   # Full contrast  
    c = np.array([0.8, 1.0, 1.2])   # Different frequencies → colors
    d = np.array([0.5, 0.5, 0.5])   # All start at black
    # color = a + b * cos(2*pi* (c * t + d))
    red = a[0] + b[0] * np.cos(2 * np.pi * (c[0] * t + d[0]))
    green = a[1] + b[1] * np.cos(2 * np.pi * (c[1] * t + d[1]))
    blue = a[2] + b[2] * np.cos(2 * np.pi * (c[2] * t + d[2]))
    
    # Clamp to [0, 1] and scale to [0, 255]
    red = (red * 255).clip(0, 255).astype(np.uint8)
    green = (green * 255).clip(0, 255).astype(np.uint8)
    blue = (blue * 255).clip(0, 255).astype(np.uint8)
    
    # Set non-escaped pixels to black
    red[not_escaped] = 0
    green[not_escaped] = 0
    blue[not_escaped] = 0
    
    img = np.stack([red, green, blue], axis=2)
    
    img_cpu = img
    os.makedirs(DATA_DIR, exist_ok=True)
    (
        Image.fromarray(img_cpu)
            .resize((int(width // AA_LEVEL), int(height // AA_LEVEL)), Image.Resampling.LANCZOS)
            .save(os.path.join(DATA_DIR, f'{img_name}.png'))
    )
    
    if display:
        Image.fromarray(img_cpu).show()
    
    del img, img_cpu, escape_iter, not_escaped, z_real, z_imag, new_real, new_imag, c_real, c_imag


def main(c_start: complex, c_end: complex):
    
    print(f"Julia Set C Animation")
    print(f"Resolution: {WIDTH}x{HEIGHT}, Max iterations: {MAX_ITER}")
    print(f"Fixed zoom level: {ZOOM}")
    print(f"C resolution: {C_RESOLUTION} per frame")

    # Creating a video for every pair of points given start and end
    
    # Warm up CPU
    print("\nWarming up CPU...")
    generate_juliaset_cpu(c_start, ZOOM, 100, 100, 100, 'warmup', False)
    os.remove(os.path.join(DATA_DIR, 'warmup.png'))
    print("CPU ready!\n")

    print(f"Point names: {c_start} to {c_end}\n")
    
    total_start = time.time()
    frame_index = 0
    
    segment_frames = calculate_frames(c_start, c_end)
    distance = abs(c_end - c_start)
    
    print(f"{c_start:.4f} -> {c_end:.4f}")
    print(f"Distance: {distance:.4f}, Frames: {segment_frames}")

    c_mid = get_dynamic_p1(c_start, c_end)
    
    for i in range(segment_frames):
        start = time.time()
        
        t = i / max(1, segment_frames - 1) if segment_frames > 1 else 1.0
        current_c = bezier_c(t, c_start, c_mid, c_end)
        
        frame_index += 1
        img_name = f'Juliaset_a_{frame_index:04d}'
        
        generate_juliaset_cpu(current_c, ZOOM, WIDTH, HEIGHT, MAX_ITER, img_name, False)
        
        elapsed = time.time() - start
        remaining = (segment_frames - i - 1) * elapsed
        
        if i % 10 == 0 or i == segment_frames - 1:
            print(f"  Frame {i + 1}/{segment_frames} | c={current_c:.4f} | {elapsed:.2f}s | {remaining:.2f}s remaining")
            
    print(f"\nTotal frames: {frame_index}")
    print(f"Total time: {(time.time() - total_start)/60:.1f} min")


if __name__ == '__main__':
    
    # for all points in the points.json, create a video for each pair
    point_names = list(points.keys())
    
    # create copies of list for custom start and end
    point_names_start = point_names.copy()
    point_names_end = point_names.copy()
    
    for start_id in point_names_start:
        for end_id in point_names_end:
            
            if os.path.exists(f'videos/{start_id}{end_id}.mp4'):
                print(f"Video {start_id}{end_id}.mp4 already exists. Skipping...")
                continue
            
            print(f"Creating video from {start_id} to {end_id}")
            start_idx: complex = complex(points[start_id])
            end_idx: complex = complex(points[end_id])
            main(start_idx, end_idx)
            
            # create video
            os.system(f'ffmpeg -framerate 60 -i "{DATA_DIR}/Juliaset_a_%04d.png" -c:v libx265 -pix_fmt yuv420p videos/{start_id}{end_id}.mp4')
            
            # check if video created
            if os.path.exists(f'videos/{start_id}{end_id}.mp4'):
                print(f"Video {start_id}{end_id}.mp4 created successfully.")
            else:
                print(f"Error: Video {start_id}{end_id}.mp4 was not created.")
                raise FileNotFoundError(f"Video {start_id}{end_id}.mp4 was not created.")
            
            # remove all images
            for file in os.listdir(DATA_DIR):
                os.remove(os.path.join(DATA_DIR, file))
