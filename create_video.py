"""Utility functions for creating videos from images."""
import os
from moviepy import ImageSequenceClip, AudioFileClip, ColorClip, concatenate_videoclips
from moviepy import afx, vfx


PADDING_SECONDS = 1.0  # Blank lead-in and lead-out duration
FADE_DURATION = 1.0   # Video fade in/out duration
DATA_DIR = os.path.join(os.path.dirname(__file__), "Mandelbrot", "data")
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")

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
    
    output_path = os.path.join(os.path.dirname(__file__), 'output.mkv')
    
    print("Loading images into video clip (this may take a while)...")
    video = ImageSequenceClip(image_paths, fps=fps)
    print("Images loaded successfully!")

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
    
    video.write_videofile(output_path, codec='libx265', audio_codec='aac', fps=fps)
    video.close()
    print(f"Video saved: {output_path}")

if __name__ == '__main__':
    create_video(fps=60.0)  # , audio_file="Euler's Clock.mp3")
    