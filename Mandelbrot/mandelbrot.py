# Generate an image of Mandelbrot set using GPU acceleration
import cupy as cp
import time
import os
from PIL import Image
from moviepy import ImageSequenceClip, AudioFileClip, ColorClip, concatenate_videoclips
from moviepy import afx, vfx

# Constants
# CENTER = (-0.7436438870371587, 0.13182590420642563)  # Tante Renate
CENTER = (-0.7436438870570570, 0.13182590420637833)  # Tante Renate
THRESHOLD_SQ = 4.0  # Pre-computed threshold squared (2.0)
FPS = 30.0
DURATION = 60.0
WIDTH = 3840
HEIGHT = 2400
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'audio')


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
    
    r = (escape_iter * 1) % 256
    g = (escape_iter * 2) % 256
    b = (escape_iter * 4) % 256
    
    r[not_escaped] = 0
    g[not_escaped] = 0
    b[not_escaped] = 0
    
    # Create image in RGB format (Pillow/MoviePy expect RGB)    
    img = cp.stack([r, g, b], axis=2).astype(cp.uint8)
    
    # Transfer to CPU and save with Pillow
    img_cpu = cp.asnumpy(img)
    os.makedirs(DATA_DIR, exist_ok=True)
    Image.fromarray(img_cpu).save(os.path.join(DATA_DIR, f'{img_name}.png'))
    
    if display:
        Image.fromarray(img_cpu).show()


PADDING_SECONDS = 1.0  # Blank lead-in and lead-out duration
FADE_DURATION = 1.0   # Video fade in/out duration


def create_video(fps: float = 30.0, audio_file: str = None) -> None:
    """Create video from images with optional audio."""
    if not os.path.exists(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
        return
    
    filenames = [f for f in os.listdir(DATA_DIR) if f.endswith('.png')]
    if not filenames:
        print("No PNG files found")
        return
    
    filenames = sorted(filenames, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    image_paths = [os.path.join(DATA_DIR, f) for f in filenames]
    
    print(f"Found {len(filenames)} images")
    print(f"First 10: {filenames[:10]}")
    
    output_path = os.path.join(os.path.dirname(__file__), 'output.mp4')
    
    video = ImageSequenceClip(image_paths, fps=fps)

    # Add blank pads at start and end
    blank = ColorClip(size=video.size, color=(0, 0, 0), duration=PADDING_SECONDS)
    video = concatenate_videoclips([blank, video, blank])

    # Add video fade in/out
    video = video.with_effects(
        [
            vfx.FadeIn(FADE_DURATION),
            vfx.FadeOut(FADE_DURATION)
        ]
    )
    
    if audio_file:
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        if os.path.exists(audio_path):
            print(f"Adding audio: {audio_file}")
            audio = AudioFileClip(audio_path)
            audio = audio.with_effects(
                [
                    afx.AudioLoop(duration=video.duration),
                    afx.AudioFadeIn(FADE_DURATION),
                    afx.AudioFadeOut(FADE_DURATION)
                ]
            )
            video = video.with_audio(audio)
            audio.close()
    
    video.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=fps)
    video.close()
    print(f"Video saved: {output_path}")


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
        
        generate_mandelbrot_gpu(current_R, WIDTH, HEIGHT, current_iter, img_name, True)
        cp.get_default_memory_pool().free_all_blocks()
        
        elapsed = time.time() - start
        remaining = (total_frames - i) * elapsed / 60
        
        if i % 10 == 0 or i == total_frames:
            print(f"Frame {i:04d}/{total_frames} | Time: {elapsed:.2f}s | Est. remaining: {remaining:.1f} min")
        
        break
    
    print(f"\nTotal time: {(time.time() - total_start)/60:.1f} min")


if __name__ == '__main__':
    main()
    # create_video(fps=60.0, audio_file="Euler's Clock.mp3")


