from manim import *

# Match your Mandelbrot video resolution
config.pixel_width = 3840
config.pixel_height = 2400
config.frame_width = 16  # Manim's coordinate system width
config.frame_height = 10  # Manim's coordinate system height

class EndCard(Scene):
    def construct(self):
        # Create "Thanks for Watching" text with word wrap
        thanks_text = Text(
            "Thanks for\n  Watching",
            font_size=90,
            #weight=BOLD,
            color=WHITE,
            line_spacing=1.2,
            font="roboto"
        )
        
        # Write "Thanks for Watching"
        self.play(Write(thanks_text), run_time=2)
        self.wait(2)
        
        # Transition to "Please Subscribe"
        self.play(FadeOut(thanks_text), run_time=1)

class StartCard(Scene):
    def construct(self):
        # Create "Thanks for Watching" text with word wrap
        intro_text = Text(
            "   Zoom of the\nMandelbrot Set",
            font_size=90,
            #weight=BOLD,
            color=WHITE,
            line_spacing=1.2,
            font="roboto"
        )
        
        # Write "Thanks for Watching"
        self.play(Write(intro_text), run_time=2)
        self.wait(2)
        
        # Transition to "Please Subscribe"
        self.play(FadeOut(intro_text), run_time=1)
        