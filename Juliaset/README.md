# Julia Set Animation

GPU-accelerated Julia set fractal generator with smooth iteration coloring and supersampling anti-aliasing.

## Cosine Palette Guide

The formula: **`color(t) = a + b × cos(2π × (c × t + d))`**

Where `t` is your normalized iteration value (0 = escaped quickly, higher = escaped slowly).

---

### **Parameter `a` - Base Brightness/Offset**
```
a = [0.5, 0.5, 0.5]  # Current
```
- Sets the **center point** that the color oscillates around
- **Higher a** → brighter overall image
- **Lower a** → darker overall image
- Example: `[0.7, 0.7, 0.7]` makes everything brighter

---

### **Parameter `b` - Contrast/Amplitude**
```
b = [0.5, 0.5, 0.5]  # Current
```
- Controls how much the color **swings** from dark to light
- **Higher b** → more contrast (deeper darks, brighter lights)
- **Lower b** → muted, washed-out colors
- **Critical rule**: `a + b ≤ 1` and `a - b ≥ 0` to avoid clipping
- Example: `[0.3, 0.3, 0.3]` gives softer pastel colors

---

### **Parameter `c` - Frequency (Color Cycles)**
```
c = [0.8, 1.0, 1.2]  # Current
```
- Controls **how many times** each color channel cycles
- **Higher c** → more color bands/stripes
- **Lower c** → fewer, broader color regions
- **Different c values per channel** → Rainbow effect (R, G, B cycle at different rates)
- Example: All `[1.0, 1.0, 1.0]` → uniform gray cycling, no color separation

---

### **Parameter `d` - Phase (Starting Point)**
```
d = [0.5, 0.5, 0.5]  # Current
```
- Controls **where in the cosine wave** each channel starts at t=0
- `d = 0.0` → cos(0) = 1 → starts at **maximum** (a + b)
- `d = 0.5` → cos(π) = -1 → starts at **minimum** (a - b)
- `d = 0.25` → cos(π/2) = 0 → starts at **middle** (a)

**For black background**: Use `d = [0.5, 0.5, 0.5]` with `a = b` so that `a - b = 0`

---

### Quick Reference Table

| Want | Change |
|------|--------|
| **Darker background** | Lower `a`, or set all `d = 0.5` |
| **Brighter fractal** | Higher `a` and `b` |
| **More colors/bands** | Increase `c` values |
| **Fewer colors** | Decrease `c` values |
| **Rainbow effect** | Use different `c` per channel: `[0.8, 1.0, 1.2]` |
| **Warmer (red/orange)** | Higher `a[0]` and `b[0]`, phase shift `d[0]` |
| **Cooler (blue)** | Higher `a[2]` and `b[2]`, phase shift `d[2]` |
| **Psychedelic bands** | Increase `c` dramatically: `[3.0, 4.0, 5.0]` |

---

### Current Values Explained:
```python
a = [0.5, 0.5, 0.5]  # Start at mid-gray oscillation center
b = [0.5, 0.5, 0.5]  # Full swing: 0.5-0.5=0 to 0.5+0.5=1
c = [0.8, 1.0, 1.2]  # R slower, G normal, B faster → rainbow
d = [0.5, 0.5, 0.5]  # All start at minimum → BLACK at t=0
```

At t=0: `0.5 + 0.5 × cos(π) = 0.5 + 0.5 × (-1) = 0` → **BLACK**

As t increases, the cosine waves rise and create colors!

---

## Other Parameters

### `t` Multiplier (line ~104)
```python
t = smooth_iter * 0.05
```
- Controls how much of the color palette is used
- **Lower value (0.01-0.05)** → fewer color cycles, smoother gradients
- **Higher value (0.1-0.5)** → more psychedelic banding patterns

### `AA_LEVEL`
- Supersampling anti-aliasing level
- `AA_LEVEL = 2` → render at 2x resolution, then downscale with LANCZOS
- Higher = smoother edges but slower rendering
