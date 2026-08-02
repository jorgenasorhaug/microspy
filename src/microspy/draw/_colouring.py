#
# Copyright 2026 the microspy developer(s)
#
# This file is part of microspy.
#
# microspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# microspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with microspy. If not, see <http://www.gnu.org/licenses/>.
#
# This code is inspired by exspy : ~exspy._misc.elements.py

"""Convenience functions for creating HyperSpy signals to use as 
navigators with :meth: '~hyperspy.signals.Signal2D.plot'.
"""
from hyperspy.signals import Signal2D
import numpy as np
from skimage.exposure import rescale_intensity
from matplotlib.colors import ListedColormap

DEFAULT_COLOURS = (
    'red',
    'blue',
    'yellow',
    'magenta',
    'green',
    'indigo',
    'darkorange',
    'cyan',
    'pink',
    'yellowgreen',
)

def get_discrete_colour_map(
    colours : list | None,
    bkgr_colour : str | tuple | None = None,
    **kwargs
) -> ListedColormap:
    """Create a custom colourmap. 
    
    Parameters
    ----------
    colours 
        list of colour names or rgb values.
    bkgr_colour
        Colour of the background, i.e. the minimum 
    kwargs
        keyword arguments passed on to :func:
        'matplotlib.colors.ListedColormap'.
    """
    from copy import deepcopy

    if colours == None: colors = DEFAULT_COLORS
    else: colors = deepcopy(colours)
    
    colors = np.array(colors)
    
    # Insert the background colour at the start:
    if bkgr_colour is not None:
        colors = colors[~(colors == bkgr_colour)]
        colors = np.insert(arr=colors, obj=0, values=bkgr_colour)
    
    return ListedColormap(colors, **kwargs)

class Closest_colorname:
    """Copilot's color matcher: given an RGB color (0-255), it finds the 
    closest named CSS4 color from Matplotlib's CSS4_COLORS palette.
    
    Note!
    It converts colors to the CIE Lab color space, which is designed so 
    that numerical distances more closely correspond to how humans perceive 
    color differences. 
    """
    
    def __init__(
        self, 
        rgb_values : tuple | list | None = None,
        hex_string : str | None = None
    ):
        if rgb_values is not None:
            self.rgb = rgb_values
        if hex_string is not None:
            self.hex_string = hex_string
    
    def __repr__(self):
        return "Run :func:'closest_color_name'"
        
    def _srgb_to_linear(self, c : float):
        """Remove the gamma correction
        
        Parameters
        ----------
        c
            Float value to remove gamma correction
        """
        # c in [0,1]
        return pow((c + 0.055) / 1.055, 2.4) if c > 0.04045 else c / 12.92

    def _rgb_to_xyz(
        self, 
        r : int | float, 
        g : int | float, 
        b : int | float
    ):
        """Convert rgb into a color space is a device-independent color 
        representation.
        """
        
        # r,g,b in 0..255 (sRGB, D65)
        r, g, b = [x / 255.0 for x in (r, g, b)]
        
        # Linearize:
        r_lin, g_lin, b_lin = map(self._srgb_to_linear, (r, g, b))
        
        # sRGB → XYZ (D65), matrix from IEC 61966-2-1
        X = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
        Y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
        Z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041
        
        # scale to the conventional 0..100 Lab domain
        return (X * 100.0, Y * 100.0, Z * 100.0)

    def _f(self, t):
        # CIE Lab helper
        delta = 6/29
        if t > (delta**3):
            return t ** (1/3)
        return (t / (3 * delta**2)) + (4/29)

    def _xyz_to_lab(self, X, Y, Z):
        # D65 reference white (2°): from CIE standard
        Xn, Yn, Zn = 95.047, 100.000, 108.883
        fx, fy, fz = self._f(X / Xn), self._f(Y / Yn), self._f(Z / Zn)
        L = 116 * fy - 16 # lightness
        a = 500 * (fx - fy) # green <-> red axis
        b = 200 * (fy - fz) # blue <-> yellow axis
        return (L, a, b)

    def rgb_to_lab(self, rgb):
        """Convenience wrapper"""
        X, Y, Z = self._rgb_to_xyz(*rgb)
        return self._xyz_to_lab(X, Y, Z)

    def deltaE_cie76(self, lab1, lab2):
        """Compute Euclidean distance in Lab space"""
        return ((lab1[0]-lab2[0])**2 + (lab1[1]-lab2[1])**2 + (lab1[2]-lab2[2])**2) ** 0.5

    def hex_to_rgb(self, hex_str):
        """Converts hexagonal string to rgb representation.
        """
        hex_str = hex_str.lstrip('#')
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

    # Optional: simpler but less perceptual metric (toggle in closest_css4_color)
    def rgb_distance(self, c1, c2):
        """
        """
        return sum((a-b)**2 for a,b in zip(c1, c2)) ** 0.5

    def closest_color_name(
        self, 
        rgb : tuple | None = None, 
        use_perceptual : bool = True
    ):
        """
        Find the closest CSS4 color name to an (R,G,B) tuple (0-255).
        
        Parameters
        ----------
        rgb
            tuple of rgb values. If None (default), the attribute rgb is used.
        use_preceptual
            Whether to 
        Returns: (name, matched_rgb, matched_hex, distance)
        """
        from matplotlib.colors import CSS4_COLORS
        
        if rgb is None:
            rgb = self.rgb 
        
        if any(not (0 <= v <= 255) for v in rgb):
            raise ValueError("RGB components must be in 0..255")

        # Precompute palette in both RGB and Lab
        palette = []
        for name, hexval in CSS4_COLORS.items():
            c_rgb = self.hex_to_rgb(hexval)
            c_lab = self.rgb_to_lab(c_rgb) if use_perceptual else None
            palette.append((name, c_rgb, hexval, c_lab))

        if use_perceptual:
            target_lab = self.rgb_to_lab(rgb)
            dist = lambda item: self.deltaE_cie76(target_lab, item[3])
        else:
            dist = lambda item: self.rgb_distance(rgb, item[1])

        best = min(palette, key=dist)
        return best[0], best[1], best[2], dist(best)