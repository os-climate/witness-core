'''
Copyright 2024 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import annotations

import colorsys
import re

import numpy
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor, sRGBColor
from plotly.colors import find_intermediate_color, make_colorscale

# ruff: noqa: E741

# Define a dictionary of color names and their hex codes
CSS3_NAMES_TO_HEX: dict[str, str] = {
    "aliceblue": "#F0F8FF",
    "antiquewhite": "#FAEBD7",
    "aqua": "#00FFFF",
    "aquamarine": "#7FFFD4",
    "azure": "#F0FFFF",
    "beige": "#F5F5DC",
    "bisque": "#FFE4C4",
    "black": "#000000",
    "blanchedalmond": "#FFEBCD",
    "blue": "#0000FF",
    "blueviolet": "#8A2BE2",
    "brown": "#A52A2A",
    "burlywood": "#DEB887",
    "cadetblue": "#5F9EA0",
    "chartreuse": "#7FFF00",
    "chocolate": "#D2691E",
    "coral": "#FF7F50",
    "cornflowerblue": "#6495ED",
    "cornsilk": "#FFF8DC",
    "crimson": "#DC143C",
    "cyan": "#00FFFF",
    "darkblue": "#00008B",
    "darkcyan": "#008B8B",
    "darkgoldenrod": "#B8860B",
    "darkgray": "#A9A9A9",
    "darkgreen": "#006400",
    "darkkhaki": "#BDB76B",
    "darkmagenta": "#8B008B",
    "darkolivegreen": "#556B2F",
    "darkorange": "#FF8C00",
    "darkorchid": "#9932CC",
    "darkred": "#8B0000",
    "darksalmon": "#E9967A",
    "darkseagreen": "#8FBC8F",
    "darkslateblue": "#483D8B",
    "darkslategray": "#2F4F4F",
    "darkturquoise": "#00CED1",
    "darkviolet": "#9400D3",
    "deeppink": "#FF1493",
    "deepskyblue": "#00BFFF",
    "dimgray": "#696969",
    "dodgerblue": "#1E90FF",
    "firebrick": "#B22222",
    "floralwhite": "#FFFAF0",
    "forestgreen": "#228B22",
    "fuchsia": "#FF00FF",
    "gainsboro": "#DCDCDC",
    "ghostwhite": "#F8F8FF",
    "gold": "#FFD700",
    "goldenrod": "#DAA520",
    "gray": "#808080",
    "green": "#008000",
    "greenyellow": "#ADFF2F",
    "honeydew": "#F0FFF0",
    "hotpink": "#FF69B4",
    "indianred": "#CD5C5C",
    "indigo": "#4B0082",
    "ivory": "#FFFFF0",
    "khaki": "#F0E68C",
    "lavender": "#E6E6FA",
    "lavenderblush": "#FFF0F5",
    "lawngreen": "#7CFC00",
    "lemonchiffon": "#FFFACD",
    "lightblue": "#ADD8E6",
    "lightcoral": "#F08080",
    "lightcyan": "#E0FFFF",
    "lightgoldenrodyellow": "#FAFAD2",
    "lightgray": "#D3D3D3",
    "lightgreen": "#90EE90",
    "lightpink": "#FFB6C1",
    "lightsalmon": "#FFA07A",
    "lightseagreen": "#20B2AA",
    "lightskyblue": "#87CEFA",
    "lightslategray": "#778899",
    "lightsteelblue": "#B0C4DE",
    "lightyellow": "#FFFFE0",
    "lime": "#00FF00",
    "limegreen": "#32CD32",
    "linen": "#FAF0E6",
    "magenta": "#FF00FF",
    "maroon": "#800000",
    "mediumaquamarine": "#66CDAA",
    "mediumblue": "#0000CD",
    "mediumorchid": "#BA55D3",
    "mediumpurple": "#9370DB",
    "mediumseagreen": "#3CB371",
    "mediumslateblue": "#7B68EE",
    "mediumspringgreen": "#00FA9A",
    "mediumturquoise": "#48D1CC",
    "mediumvioletred": "#C71585",
    "midnightblue": "#191970",
    "mintcream": "#F5FFFA",
    "mistyrose": "#FFE4E1",
    "moccasin": "#FFE4B5",
    "navajowhite": "#FFDEAD",
    "navy": "#000080",
    "oldlace": "#FDF5E6",
    "olive": "#808000",
    "olivedrab": "#6B8E23",
    "orange": "#FFA500",
    "orangered": "#FF4500",
    "orchid": "#DA70D6",
    "palegoldenrod": "#EEE8AA",
    "palegreen": "#98FB98",
    "paleturquoise": "#AFEEEE",
    "palevioletred": "#DB7093",
    "papayawhip": "#FFEFD5",
    "peachpuff": "#FFDAB9",
    "peru": "#CD853F",
    "pink": "#FFC0CB",
    "plum": "#DDA0DD",
    "powderblue": "#B0E0E6",
    "purple": "#800080",
    "rebeccapurple": "#663399",
    "red": "#FF0000",
    "rosybrown": "#BC8F8F",
    "royalblue": "#4169E1",
    "saddlebrown": "#8B4513",
    "salmon": "#FA8072",
    "sandybrown": "#F4A460",
    "seagreen": "#2E8B57",
    "seashell": "#FFF5EE",
    "sienna": "#A0522D",
    "silver": "#C0C0C0",
    "skyblue": "#87CEEB",
    "slateblue": "#6A5ACD",
    "slategray": "#708090",
    "snow": "#FFFAFA",
    "springgreen": "#00FF7F",
    "steelblue": "#4682B4",
    "tan": "#D2B48C",
    "teal": "#008080",
    "thistle": "#D8BFD8",
    "tomato": "#FF6347",
    "turquoise": "#40E0D0",
    "violet": "#EE82EE",
    "wheat": "#F5DEB3",
    "white": "#FFFFFF",
    "whitesmoke": "#F5F5F5",
    "yellow": "#FFFF00",
    "yellowgreen": "#9ACD32",
}


def patch_asscalar(a):
    return a.item()


setattr(numpy, "asscalar", patch_asscalar)


def calculate_distance(color1: LabColor, color2: LabColor) -> float:
    """
    Calculate perceptual distance between two LAB colors.

    Args:
        color1 (LabColor): First color in LAB space.
        color2 (LabColor): Second color in LAB space.

    Returns:
        float: Perceptual distance between colors.

    Example:
        >>> lab1 = rgb_to_lab((255, 0, 0))
        >>> lab2 = rgb_to_lab((0, 255, 0))
        >>> distance = calculate_distance(lab1, lab2)
        >>> print(f"{distance:.2f}")
        86.61

    """
    return delta_e_cie2000(color1, color2)


def create_colorscale(colors: list[str]) -> list[list[float]]:
    """
    Create a Plotly colorscale from colors.

    Args:
        colors (list[str]): list of color hex codes.

    Returns:
        list[list[float]]: Plotly colorscale.

    Example:
        >>> scale = create_colorscale(["#FF0000", "#00FF00", "#0000FF"])
        >>> print(scale[:2])  # Print first two entries of the colorscale
        [[0.0, '#FF0000'], [0.5, '#00FF00']]

    """
    return make_colorscale(colors)


def interpolate_between_colors(
        start_color: str, end_color: str, num_colors: int
) -> list[str]:
    """
    Interpolate between two specific colors.

    Args:
        start_color (str): Starting color hex code.
        end_color (str): Ending color hex code.
        num_colors (int): Number of colors to interpolate.

    Returns:
        list[str]: list of interpolated color hex codes.

    Example:
        >>> print(interpolate_between_colors("#FF0000", "#0000FF", 3))
        ['#FF0000', '#7F007F', '#0000FF']

    """
    return [
        rgb_to_hex(
            find_intermediate_color(
                hex_to_rgb(start_color),
                hex_to_rgb(end_color),
                i / (num_colors - 1),
            )
        )
        for i in range(num_colors)
    ]


def generate_color_variations(color: str, num_shades: int) -> list[str]:
    """
    Generate lighter and darker variations of a color.

    Args:
        color (str): Base color hex code.
        num_shades (int): Number of shades to generate.

    Returns:
        list[str]: list of generated color hex codes.

    Example:
        >>> print(generate_color_variations("#FF0000", 2))
        ['#000000', '#7F0000', '#FF0000', '#FF7F7F', '#FFFFFF']

    """
    rgb_color = hex_to_rgb(color)

    darker_shades = [
        rgb_to_hex(adjust_color_brightness(rgb_color, -i / num_shades))
        for i in range(1, num_shades + 1)
    ]

    lighter_shades = [
        rgb_to_hex(adjust_color_brightness(rgb_color, i / num_shades))
        for i in range(1, num_shades + 1)
    ]

    return darker_shades[::-1] + [color] + lighter_shades


def rgb_to_lab(rgb: tuple) -> LabColor:
    """
    Convert an RGB tuple (0-255) to LAB color space.

    Args:
        rgb (tuple[int, int, int]): RGB color tuple.

    Returns:
        LabColor: Color in LAB color space.

    Example:
        >>> rgb_to_lab((255, 0, 0))
        LabColor(lab_l=53.23896002513146,lab_a=80.09045298802708,lab_b=67.2013836595967)

    """
    srgb = sRGBColor(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    return convert_color(srgb, LabColor)


def to_hex(color: str) -> str:
    """Converts any color format to hex.

    Args:
        color (str): Color string in supported format (hex or RGB).

    Returns:
        str: Color in hex format.

    Example:
        >>> print(to_hex("rgb(255, 0, 0)"))
        #FF0000
        >>> print(to_hex("#FF0000"))
        #FF0000
    """
    if re.match(r"^#[0-9a-fA-F]{6}$", color):
        return color
    elif re.match(r"^rgb\((\d+),\s*(\d+),\s*(\d+)\)$", color):
        return rgb_string_to_hex(color)
    else:
        raise ValueError(f"Unsupported color format: {color}")


def to_rgb_string(color: str) -> str:
    """Converts any color format to RGB string.

    Args:
        color (str): Color string in supported format (hex, RGB, HSL, or LAB).

    Returns:
        str: Color in RGB string format.

    Example:
        >>> print(to_rgb_string("#FF0000"))
        rgb(255, 0, 0)
        >>> print(to_rgb_string("hsl(0, 100%, 50%)"))
        rgb(255, 0, 0)
    """
    if re.match(r"^#[0-9a-fA-F]{6}$", color):
        return hex_to_rgb_string(color)
    elif re.match(r"^rgb\((\d+),\s*(\d+),\s*(\d+)\)$", color):
        return color
    elif re.match(r"^hsl\((\d+),\s*(\d+)%,\s*(\d+)%\)$", color):
        return hsl_string_to_rgb_string(color)
    elif re.match(r"^lab\(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)$", color):
        return lab_string_to_rgb_string(color)
    else:
        raise ValueError(f"Unsupported color format: {color}")


# Helper methods for conversions
def hls_to_hex(h: float, l: float, s: float) -> str:
    """
    Convert HLS color to hex.

    Args:
        h (float): Hue (0-1).
        l (float): Lightness (0-1).
        s (float): Saturation (0-1).

    Returns:
        str: Hex color code.

    Example:
        >>> print(hls_to_hex(0.0, 0.5, 1.0))
        #FF0000

    """
    rgb = colorsys.hls_to_rgb(h, l, s)
    return rgb_to_hex(tuple(int(x * 255) for x in rgb))


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """
    Convert hex color to RGB tuple.

    Args:
        hex_color (str): Hex color code.

    Returns:
        tuple[int, int, int]: RGB color tuple.

    Example:
        >>> hex_to_rgb("#FF0000")
        (255, 0, 0)

    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i: i + 2], 16) for i in (0, 2, 4))


def hex_to_rgb_string(hex_color: str) -> str:
    """Convert hex color to RGB string format.

    Args:
        hex_color (str): Color in hex format.

    Returns:
        str: Color in RGB string format.

    Example:
        >>> print(hex_to_rgb_string("#FF0000"))
        rgb(255, 0, 0)
    """
    rgb = hex_to_rgb(hex_color)
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


def rgb_string_to_hsl_string(rgb_string: str) -> str:
    """Convert RGB string to HSL string format.

    Args:
        rgb_string (str): Color in RGB string format.

    Returns:
        str: Color in HSL string format.

    Example:
        >>> print(rgb_string_to_hsl_string("rgb(255, 0, 0)"))
        hsl(0, 100%, 50%)
    """
    match = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", rgb_string)
    r, g, b = map(int, match.groups())
    hsl = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    return f"hsl({int(hsl[0] * 360)}, {int(hsl[2] * 100)}%, {int(hsl[1] * 100)}%)"


def rgb_string_to_lab_string(rgb_string: str) -> str:
    """Convert RGB string to LAB color space string format.

    Args:
        rgb_string (str): Color in RGB string format (e.g., "rgb(255, 0, 0)").

    Returns:
        str: Color in LAB string format.

    Example:
        >>> print(rgb_string_to_lab_string("rgb(255, 0, 0)"))
        lab(53.24, 80.09, 67.20)
    """
    match = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", rgb_string)
    r, g, b = map(int, match.groups())
    lab = convert_color(sRGBColor(r / 255.0, g / 255.0, b / 255.0), LabColor)
    return f"lab({lab.lab_l:.2f}, {lab.lab_a:.2f}, {lab.lab_b:.2f})"


def hsl_string_to_rgb_string(hsl_string: str) -> str:
    """Convert HSL string to RGB string format.

    Args:
        hsl_string (str): Color in HSL string format (e.g., "hsl(0, 100%, 50%)").

    Returns:
        str: Color in RGB string format.

    Example:
        >>> print(hsl_string_to_rgb_string("hsl(0, 100%, 50%)"))
        rgb(255, 0, 0)
    """
    match = re.match(r"hsl\((\d+),\s*(\d+)%,\s*(\d+)%\)", hsl_string)
    h, s, l = map(int, match.groups())
    rgb = colorsys.hls_to_rgb(h / 360.0, l / 100.0, s / 100.0)
    return f"rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})"


def lab_string_to_rgb_string(lab_string: str) -> str:
    """Convert LAB string to RGB string format.

    Args:
        lab_string (str): Color in LAB string format (e.g., "lab(53.24, 80.09, 67.20)").

    Returns:
        str: Color in RGB string format.

    Example:
        >>> print(lab_string_to_rgb_string("lab(53.24, 80.09, 67.20)"))
        rgb(250, 0, 6)
    """
    match = re.match(r"lab\(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)", lab_string)
    l, a, b = map(float, match.groups())
    rgb = convert_color(LabColor(l, a, b), sRGBColor)
    return f"rgb({int(rgb.clamped_rgb_r * 255.0)}, {int(rgb.clamped_rgb_g * 255.0)}, {int(rgb.clamped_rgb_b * 255.0)})"


def rgb_to_hex(rgb_color: tuple[int, int, int]) -> str:
    """
    Convert RGB tuple to hex color.

    Args:
        rgb_color (tuple[int, int, int]): RGB color tuple.

    Returns:
        str: Hex color code.

    Raises:
        ValueError: If the input is not a valid RGB tuple.

    Example:
        >>> rgb_to_hex((255, 0, 0))
        '#FF0000'

    """
    if not (len(rgb_color) == 3 and all(0 <= x <= 255 for x in rgb_color)):
        msg = "Input must be a tuple with three integers in the range 0-255"
        raise ValueError(msg)

    return "#{:02X}{:02X}{:02X}".format(*(int(v) for v in rgb_color))


def rgb_string_to_hex(rgb_string: str) -> str:
    """
    Convert an RGB string in the format "rgb(xxx, xxx, xxx)" to a hex color string.

    Args:
        rgb_string (str): RGB color string (e.g., "rgb(255, 0, 0)").

    Returns:
        str: Hex color string (e.g., "#FF0000").

    Example:
        >>> print(rgb_string_to_hex("rgb(255, 0, 0)"))
        #FF0000
    """
    # Extract the integer values from the RGB string
    match = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", rgb_string)
    if not match:
        raise ValueError(f"Invalid RGB string format: {rgb_string}")

    r, g, b = map(int, match.groups())
    # Ensure the values are within the valid range
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        raise ValueError(f"RGB values must be in the range 0-255: {rgb_string}")

    # Format the RGB tuple as a hex color string
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def adjust_color_brightness(
        rgb_color: tuple[int, int, int], factor: float
) -> tuple[int, int, int]:
    """
    Adjust color brightness.

    Args:
        rgb_color (tuple[int, int, int]): RGB color tuple.
        factor (float): Adjustment factor. Negative factor darkens, positive factor lightens.

    Returns:
        tuple[int, int, int]: Adjusted RGB color tuple.

    Example:
        >>> adjust_color_brightness((100, 150, 200), 0.2)
        (131, 171, 211)

    """
    if factor < 0:  # Darken
        black = (0, 0, 0)
        return tuple(
            int(rgb_color[i] + (black[i] - rgb_color[i]) * -factor) for i in range(3)
        )

    # else Lighten
    white = (255, 255, 255)
    return tuple(
        int(rgb_color[i] + (white[i] - rgb_color[i]) * factor) for i in range(3)
    )


def adjust_color_intensity(
        rgb_color: tuple[int, int, int], factor: float
) -> tuple[int, int, int]:
    """
    Adjust color intensity.

    Args:
        rgb_color (tuple[int, int, int]): RGB color tuple.
        factor (float): Adjustment factor. Negative factor decreases intensity, positive factor increases intensity.

    Returns:
        tuple[int, int, int]: Adjusted RGB color tuple.

    Example:
        >>> adjust_color_intensity((100, 150, 200), 0.2)
        (80, 140, 200)
    """
    # Convert RGB to HSV
    r, g, b = [x / 255.0 for x in rgb_color]
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val

    if max_val == min_val:
        h = 0
    elif max_val == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_val == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360

    s = 0 if max_val == 0 else (diff / max_val)
    v = max_val

    # Adjust saturation
    s = min(1, s * (1 + factor)) if factor > 0 else max(0, s * (1 + factor))

    # Convert back to RGB
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    r, g, b = [int((x + m) * 255) for x in (r, g, b)]
    return (r, g, b)


def get_closest_color_name(hex_color: str) -> str | None:
    """
    Get the closest standard color name for a given hex color.

    Args:
        hex_color (str): Hex color code.

    Returns:
        str: Closest color name.

    Example:
        >>> get_closest_color_name("#FF0000")
        'red'

    """
    try:
        # Convert hex code to RGB
        rgb_value = hex_to_rgb(hex_color)
    except ValueError:
        return None

    # Find the closest color
    closest_name = None
    min_distance = float("inf")
    for name, code in CSS3_NAMES_TO_HEX.items():
        color_rgb = hex_to_rgb(code)
        distance = sum((rgb_value[i] - color_rgb[i]) ** 2 for i in range(3))
        if distance < min_distance:
            closest_name = name
            min_distance = distance

    return closest_name


def generate_complementary_palette(base_color: str) -> dict[str, str]:
    """
    Generate a complementary color palette.

    Args:
        base_color (str): Base color in hex format.

    Returns:
        dict[str, str]: dictionary of complementary colors.

    Example:
        >>> print(generate_complementary_palette("#FF0000"))
        {'base': '#FF0000', 'complementary': '#00FEFF', 'analogous_1': '#FF7E00', 'analogous_2': '#FF007E', 'triadic_1': '#00FF00', 'triadic_2': '#0000FF'}

    """
    # Convert hex to HSL
    h, l, s = colorsys.rgb_to_hls(
        *[x / 255.0 for x in hex_to_rgb(base_color)]
    )

    # Complementary color (180 degrees on color wheel)
    complementary_h = (h + 0.5) % 1.0

    # Generate variations
    return {
        "base": base_color,
        "complementary": hls_to_hex(complementary_h, l, s),
        "analogous_1": hls_to_hex((h + 0.083) % 1.0, l, s),
        "analogous_2": hls_to_hex((h - 0.083) % 1.0, l, s),
        "triadic_1": hls_to_hex((h + 0.333) % 1.0, l, s),
        "triadic_2": hls_to_hex((h - 0.333) % 1.0, l, s),
    }


def select_colors_from_list(colors: list[str], n: int) -> list[str]:
    """
    Select n colors maximizing the distance between them.

    Args:
        colors (list[str]): list of colors to select from. In Hex format
        n (int): Number of colors to select.

    Returns:
        list[str]: Selected colors.

    Example:
        >>> colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"]
        >>> print(select_colors_from_list(colors, 3))
        ['#FF0000', '#00FF00', '#0000FF']

    """
    # Convert colors to LAB color space
    lab_colors = [rgb_to_lab(hex_to_rgb(color)) for color in colors]

    # Start with the first color and build the selected list
    selected_colors = [lab_colors[0]]

    for _ in range(1, n):
        max_min_dist = -1
        next_color = None

        for candidate in lab_colors:
            if candidate in selected_colors:
                continue
            # Compute minimum distance to the already selected colors
            min_dist = min(
                calculate_distance(candidate, selected)
                for selected in selected_colors
            )

            if min_dist > max_min_dist:
                max_min_dist = min_dist
                next_color = candidate

        if next_color:
            selected_colors.append(next_color)

    # Convert selected LAB colors back to RGB for the result
    return [colors[lab_colors.index(selected)] for selected in selected_colors]
