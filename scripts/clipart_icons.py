"""Shared emoji-clipart helper for FarmFederate figure scripts.

Renders full-color emoji glyphs (Segoe UI Emoji, COLR) via Pillow and places
them as square, undistorted cliparts inside matplotlib axes that use 0..1
data coordinates (the convention of all FarmFederate diagram scripts).
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

_FONT_PATH = r"C:\Windows\Fonts\seguiemj.ttf"
_emoji_cache = {}


def render_emoji(char, px=256):
    """Rasterize a full-color emoji glyph, tightly cropped, RGBA numpy array."""
    key = (char, px)
    if key in _emoji_cache:
        return _emoji_cache[key]
    font = ImageFont.truetype(_FONT_PATH, int(px * 0.80))
    canvas_px = px * 2  # generous canvas; we crop to content afterwards
    img = Image.new("RGBA", (canvas_px, canvas_px), (255, 255, 255, 0))
    d = ImageDraw.Draw(img)
    d.text((canvas_px / 2, canvas_px / 2), char, font=font, embedded_color=True, anchor="mm")
    bbox = img.getbbox()
    if bbox is None:
        raise ValueError("emoji rendered blank: %r" % (char,))
    pad = 2
    bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad), min(canvas_px, bbox[2] + pad), min(canvas_px, bbox[3] + pad))
    arr = np.asarray(img.crop(bbox))
    _emoji_cache[key] = arr
    return arr


def emoji_icon(ax, x, y, size, char, scale=1.0, zorder=12):
    """Place an emoji clipart centered in the size x size data-coord box at (x, y).

    The axes are assumed to span 0..1 in both directions.  Because the figures
    are much wider than tall, a size x size box in data coords is a wide
    rectangle on screen; the emoji is drawn as a square (no distortion) whose
    larger display dimension equals the box's smaller display dimension, so it
    always fits inside the slot.  `scale` enlarges/shrinks around the center.
    """
    fig = ax.figure
    pos = ax.get_position()
    w_in, h_in = fig.get_size_inches()
    ppu_x = w_in * fig.dpi * pos.width   # display px per data unit, x
    ppu_y = h_in * fig.dpi * pos.height  # display px per data unit, y
    side_px = min(size * ppu_x, size * ppu_y) * scale

    arr = render_emoji(char, px=256)
    # OffsetImage displays arr_px * zoom * (dpi/72) pixels (dpi_cor default on).
    dpi_factor = fig.dpi / 72.0
    zoom = side_px / (max(arr.shape[0], arr.shape[1]) * dpi_factor)
    im = OffsetImage(arr, zoom=zoom)
    ab = AnnotationBbox(
        im,
        (x + size / 2.0, y + size / 2.0),
        frameon=False,
        zorder=zorder,
        box_alignment=(0.5, 0.5),
        pad=0,
    )
    ax.add_artist(ab)
    return ab


def make_icon(char, scale=1.0, zorder=12):
    """Return an icon-drawing function with the legacy (ax, x, y, s, color) signature."""

    def _draw(ax, x, y, s, color=None):
        emoji_icon(ax, x, y, s, char, scale=scale, zorder=zorder)

    return _draw
