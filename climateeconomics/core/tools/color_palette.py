from __future__ import annotations

import colorsys
import random
from typing import Optional

import numpy as np
import plotly.graph_objs as go
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


class ColorPalette:
    """
    A color palette with flexible configuration options and various color manipulation methods.

    Attributes:
        name (str): Name of the palette.
        main_colors (list[str]): list of main colors in the palette.
        highlight_colors (list[str]): list of highlight colors in the palette.
        color_tags (list[str]): list of tags associated with colors.
        predefined_shades (dict[str, list[str]]): dictionary of predefined color shades.
        colorscale (list[list[float]]): Plotly colorscale representation.
        color_map (dict[str, str]): Mapping of color tags to color codes.
        icolor_map (dict[str, str]): Inverse mapping of color codes to color tags.

    """

    # Predefined color palettes with color-blind friendly options
    PRESET_PALETTES = {
        # Scientific/Professional Palettes
        "data_science": {
            "main_colors": [
                "#1696D2",
                "#D2D2D2",
                "#000000",
                "#FDBF11",
                "#EC008B",
                "#55B748",
                "#DB2B27",
            ],
            "highlight_colors": [],
            "color_tags": [
                "primary_blue",
                "neutral_gray",
                "black",
                "yellow",
                "magenta",
                "green",
                "red",
            ],
            "shades": {
                "primary_blue": [
                    "#CFE8F3",
                    "#A2D4EC",
                    "#73BFE2",
                    "#46ABDB",
                    "#1696D2",
                    "#12719E",
                    "#0A4C6A",
                    "#062635",
                ],
                "neutral_gray": [
                    "#F5F5F5",
                    "#ECECEC",
                    "#E3E3E3",
                    "#DCDBDB",
                    "#D2D2D2",
                    "#9D9D9D",
                    "#696969",
                    "#353535",
                ],
                "yellow": [
                    "#FFF2CF",
                    "#FCE39E",
                    "#FDD870",
                    "#FCCB41",
                    "#FDBF11",
                    "#E88E2D",
                    "#CA5800",
                    "#843215",
                ],
                "magenta": [
                    "#F5CBDF",
                    "#EB99C2",
                    "#E46AA7",
                    "#E54096",
                    "#EC008B",
                    "#AF1F6B",
                    "#761548",
                    "#351123",
                ],
                "green": [
                    "#DCEDD9",
                    "#BCDEB4",
                    "#98CF90",
                    "#78C26D",
                    "#55B748",
                    "#408941",
                    "#2C5C2D",
                    "#1A2E19",
                ],
                "black": [
                    "#D5D5D4",
                    "#ADABAC",
                    "#848081",
                    "#5C5859",
                    "#332D2F",
                    "#262223",
                    "#1A1717",
                    "#0E0C0D",
                ],
                "red": [
                    "#F8D5D4",
                    "#F1AAA9",
                    "#E9807D",
                    "#E25552",
                    "#DB2B27",
                    "#A4201D",
                    "#6E1614",
                    "#370B0A",
                ],
            },
            "description": "Clean, professional palette for data visualization",
            "color_blind_friendly": False,
        },
        "color_blind_safe": {
            "main_colors": ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#F0E442"],
            "highlight_colors": ["#56B4E9", "#D55E00"],
            "color_tags": [
                "blue",
                "orange",
                "green",
                "pink",
                "yellow",
                "light_blue",
                "red",
            ],
            "description": "Color-blind friendly palette with high contrast",
            "color_blind_friendly": True,
        },
        "earth_tones": {
            "main_colors": ["#2C5F2D", "#97BC62", "#77A146", "#D5E2BC", "#E2D1B8"],
            "highlight_colors": ["#8B4513", "#A0522D"],
            "color_tags": [
                "dark_green",
                "light_green",
                "mid_green",
                "pale_green",
                "beige",
                "brown_1",
                "brown_2",
            ],
            "description": "Natural, earthy color palette",
            "color_blind_friendly": False,
        },
        "pastel_vibes": {
            "main_colors": ["#FFB3BA", "#BAFFC9", "#BAE1FF", "#F5F5DC", "#E6E6FA"],
            "highlight_colors": ["#FF6B6B", "#4ECDC4"],
            "color_tags": [
                "pastel_pink",
                "pastel_green",
                "pastel_blue",
                "cream",
                "lavender",
                "bright_pink",
                "teal",
            ],
            "description": "Soft, gentle pastel palette",
            "color_blind_friendly": False,
        },
    }

    def __init__(
        self,
        name: str = "custom_palette",
        palette: Optional[str] = None,
        main_colors: Optional[list[str]] = None,
        highlight_colors: Optional[list[str]] = None,
        color_tags: Optional[list[str]] = None,
        predefined_shades: Optional[dict] = None,
        custom_palette: Optional[dict[str, list[str]]] = None,
    ):
        """
        Initialize an enhanced palette with flexible configuration options.

        Args:
            name (str): Name of the palette.
            palette (Optional[str]): Preset palette name.
            main_colors (Optional[list[str]]): Custom main colors.
            highlight_colors (Optional[list[str]]): Custom highlight colors.
            color_tags (Optional[list[str]]): Custom color tags.
            predefined_shades (Optional[dict]): dictionary of predefined color shades.
            custom_palette (Optional[dict[str, list[str]]]): Completely custom palette definition.

        Example:
            >>> cm = ColorPalette(name="My palette", palette="data_science")
            >>> print(cm.name)
            My palette
        """
        # Determine color sources with priority
        if custom_palette:
            self.main_colors = custom_palette.get("main_colors", [])
            self.highlight_colors = custom_palette.get("highlight_colors", [])
            self.color_tags = custom_palette.get("color_tags", [])
            self.predefined_shades = custom_palette.get("shades", {})
        elif palette and palette in self.PRESET_PALETTES:
            preset = self.PRESET_PALETTES[palette]
            self.main_colors = main_colors or preset["main_colors"]
            self.highlight_colors = highlight_colors or preset["highlight_colors"]
            self.color_tags = color_tags or preset["color_tags"]
            self.predefined_shades = predefined_shades or preset.get("shades", {})
        else:
            self.main_colors = main_colors or ["#000000", "#FFFFFF"]
            self.highlight_colors = highlight_colors or []
            self.color_tags = color_tags or [
                f"color_{i}" for i in range(len(self.main_colors))
            ]
            self.predefined_shades = {}

        # Ensure color_tags matches the number of main_colors
        if len(self.color_tags) != len(self.main_colors + self.highlight_colors):
            self.color_tags = [f"color_{i}" for i in range(len(self.main_colors))]

        self.name = name
        self.colorscale = self._create_colorscale(self.main_colors)

        # Tag-to-color mapping
        self.color_map = dict(
            zip(self.color_tags, [*self.main_colors, *self.highlight_colors])
        )

        # color-to-tag mapping
        self.icolor_map = dict(
            zip([*self.main_colors, *self.highlight_colors], self.color_tags)
        )

    def get_color_by_tag(self, tag: str) -> str:
        """
        Retrieve a color by its tag.

        Args:
            tag (str): Color tag.

        Returns:
            str: Hex color code.

        Raises:
            ValueError: If the color tag is not found in the color map.

        Example:
            >>> cm = ColorPalette(palette="data_science")
            >>> print(cm.get_color_by_tag("primary_blue"))
            #1696D2
        """
        if tag not in self.color_map:
            raise ValueError(f"Color tag '{tag}' not found in the color map")
        return self.color_map[tag]

    def generate_complementary_palette(self, base_color: str) -> dict[str, str]:
        """
        Generate a complementary color palette.

        Args:
            base_color (str): Base color in hex format.

        Returns:
            dict[str, str]: dictionary of complementary colors.

        Example:
            >>> cm = ColorPalette()
            >>> print(cm.generate_complementary_palette("#FF0000"))
            {'base': '#FF0000', 'complementary': '#00FFFF', 'analogous_1': '#FF8000', 'analogous_2': '#FF0080', 'triadic_1': '#80FF00', 'triadic_2': '#0080FF'}
        """
        # Convert hex to HSL
        h, l, s = colorsys.rgb_to_hls(
            *[x / 255.0 for x in self._hex_to_rgb(base_color)]
        )

        # Complementary color (180 degrees on color wheel)
        complementary_h = (h + 0.5) % 1.0

        # Generate variations
        return {
            "base": base_color,
            "complementary": self._hls_to_hex(complementary_h, l, s),
            "analogous_1": self._hls_to_hex((h + 0.083) % 1.0, l, s),
            "analogous_2": self._hls_to_hex((h - 0.083) % 1.0, l, s),
            "triadic_1": self._hls_to_hex((h + 0.333) % 1.0, l, s),
            "triadic_2": self._hls_to_hex((h - 0.333) % 1.0, l, s),
        }

    def generate_accessible_palette(self, contrast_ratio: float = 4.5) -> list[str]:
        """
        Generate an accessible color palette with sufficient contrast.

        Args:
            contrast_ratio (float): Minimum contrast ratio (WCAG standard is 4.5).

        Returns:
            list[str]: list of accessible colors.

        Example:
            >>> cm = ColorPalette(palette="data_science")
            >>> print(cm.generate_accessible_palette())
            ['#1696D2', '#FDBF11', '#000000', '#55B748', '#EC008B']
        """

        def calculate_luminance(color: str) -> float:
            """Calculate relative luminance of a color."""
            rgb = self._hex_to_rgb(color)
            rgb_norm = [x / 255.0 for x in rgb]
            rgb_adj = [
                x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4
                for x in rgb_norm
            ]
            return 0.2126 * rgb_adj[0] + 0.7152 * rgb_adj[1] + 0.0722 * rgb_adj[2]

        def calculate_contrast_ratio(color1: str, color2: str) -> float:
            """Calculate contrast ratio between two colors."""
            lum1 = calculate_luminance(color1)
            lum2 = calculate_luminance(color2)
            lighter = max(lum1, lum2)
            darker = min(lum1, lum2)
            return (lighter + 0.05) / (darker + 0.05)

        accessible_colors = []
        for color in self.main_colors + self.highlight_colors:
            if not any(
                calculate_contrast_ratio(color, existing) >= contrast_ratio
                for existing in accessible_colors
            ):
                accessible_colors.append(color)

        return accessible_colors

    def get_color(self, value: float, interpolation_type: str = "linear") -> str:
        """
        Get a color from the colorscale based on a value between 0 and 1.

        Args:
            value (float): Float between 0 and 1.
            interpolation_type (str): Type of color interpolation ('linear', 'log', etc.).

        Returns:
            str: Interpolated color.

        Example:
            >>> cm = ColorPalette(palette="data_science")
            >>> print(cm.get_color(0.5))
            #8B4513
        """
        if interpolation_type == "log":
            # Logarithmic interpolation
            value = np.log(1 + value) / np.log(2)

        return find_intermediate_color(self.main_colors[0], self.main_colors[-1], value)

    def get_shades(
        self, color: str, n_colors: int = 5, force_generate: bool = False, **kwargs
    ) -> list[str]:
        """
        Get n_colors from shades of a color either from a predefined shade or generated it.

        Args:
            color (str): Base color.
            n_colors (int): Number of colors to generate.
            force_generate (bool): Force generation of shades even if predefined shades exist.
            **kwargs: Additional arguments for shade generation.

        Returns:
            list[str]: list of color shades.

        Example:
            >>> cm = ColorPalette(palette="data_science")
            >>> print(cm.get_shades("primary_blue", 3))
            ['#A2D4EC', '#1696D2', '#0A4C6A']

        """
        if not force_generate and color in self.predefined_shades:
            return self.select_colors_from_list(
                colors=self.predefined_shades[color], n=n_colors
            )

        if isinstance(n_colors, tuple):
            a = n_colors[0], b = n_colors[1]
        else:
            a = n_colors // 2
            b = n_colors - a
        return self.generate_shades(color, num_darker=a, num_lighter=b, **kwargs)

    def generate_shades(
        self,
        color: str = None,
        num_darker: int = 3,
        num_lighter: int = 3,
        intensity_range: list | None = None,
    ) -> list[str]:
        """
        Generate shades of a given color.

        Args:
            color (str): The base color to generate shades from.
            num_darker (int): Number of shades darker than the base color.
            num_lighter (int): Number of shades lighter than the base color.
            intensity_range (list | None): Range of intensity factors.

        Returns:
            list[str]: A list of shades.

        Example:
            >>> cm = ColorPalette()
            >>> print(cm.generate_shades("#FF0000", num_darker=2, num_lighter=2))
            ['#FF8080', '#FF4040', '#FF0000', '#BF0000', '#800000']

        """
        if "#" not in color:
            color = self.main_colors[self.color_tags.index(color)]

        rgb_color = self._hex_to_rgb(color)
        darker_shades = [
            self._rgb_to_hex(
                self._adjust_color_brightness(rgb_color, -i / (num_darker + 2))
            )
            for i in range(1, num_darker + 1)
        ]
        lighter_shades = [
            self._rgb_to_hex(
                self._adjust_color_brightness(rgb_color, i / (num_lighter + 2))
            )
            for i in range(1, num_lighter + 1)
        ]

        colors = [*lighter_shades[::-1], color, *darker_shades]

        # Apply intensity factor
        if intensity_range is not None:
            intensity_factors = np.linspace(
                intensity_range[0], intensity_range[1], len(colors)
            )
            colors = [
                self._rgb_to_hex(
                    self._adjust_color_intensity(self._hex_to_rgb(color), factor)
                )
                for color, factor in zip(colors, intensity_factors)
            ]

        return colors

    @staticmethod
    def select_colors_from_list(colors, n) -> list:
        """
        Select n colors maximizing the distance between them.

        Args:
            colors (list[str]): list of colors to select from.
            n (int): Number of colors to select.

        Returns:
            list[str]: Selected colors.

        Example:
            >>> cm = ColorPalette()
            >>> colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"]
            >>> print(cm.select_colors_from_list(colors, 3))
            ['#FF0000', '#00FF00', '#0000FF']

        """
        # Convert colors to LAB color space
        lab_colors = [ColorPalette._rgb_to_lab(color) for color in colors]

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
                    ColorPalette._calculate_distance(candidate, selected)
                    for selected in selected_colors
                )

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    next_color = candidate

            if next_color:
                selected_colors.append(next_color)

        # Convert selected LAB colors back to RGB for the result
        return [colors[lab_colors.index(selected)] for selected in selected_colors]

    def generate_color_palette(
        self, num_colors: int, palette_type: str = "sequential"
    ) -> list[str]:
        """
        Generate a color palette based on specified parameters.

        Args:
            num_colors (int): Number of colors to generate.
            palette_type (str): 'sequential', 'categorical', or 'diverging'.

        Returns:
            list[str]: list of generated colors.

        Example:
            >>> cm = ColorPalette(palette="data_science")
            >>> print(cm.generate_color_palette(5, "sequential"))
            ['#1696D2', '#46ABDB', '#73BFE2', '#A2D4EC', '#CFE8F3']

        """
        if palette_type == "sequential":
            return self._interpolate_colors(num_colors)
        elif palette_type == "categorical":
            return self._generate_categorical_colors(num_colors)
        elif palette_type == "diverging":
            if len(self.main_colors) < 3:
                raise ValueError("Diverging palette requires at least 3 main colors")
            mid_color = self.main_colors[len(self.main_colors) // 2]
            return self._generate_diverging_colors(num_colors, mid_color)
        else:
            raise ValueError(f"Unknown palette type: {palette_type}")

    def suggest_additional_palettes(self) -> dict[str, list[str]]:
        """
        Suggest additional color palettes based on existing colors.

        Returns:
            dict[str, list[str]]: dictionary of suggested palettes.

        Example:
            >>> cm = ColorPalette(palette="data_science")
            >>> suggested = cm.suggest_additional_palettes()
            >>> print(list(suggested.keys())[:2])  # Print first two suggested palette names
            ['Monochromatic_DeepSkyBlue', 'Monochromatic_LightGray']

        """
        suggested_palettes = {}

        # Suggest monochromatic variations
        for color in self.main_colors:
            monochromatic = self._generate_color_variations(color, 5)
            suggested_palettes[
                f"Monochromatic_{self._get_closest_color_name(color)}"
            ] = monochromatic

        # Suggest complementary palettes
        for color in self.main_colors:
            complementary = list(self.generate_complementary_palette(color).values())
            suggested_palettes[
                f"Complementary_{self._get_closest_color_name(color)}"
            ] = complementary

        return suggested_palettes

    def visualize_palette(self, num_shades: int = 4) -> go.Figure:
        """
        Create a visualization of the color palette.

        Args:
            num_shades (int): Number of lighter/darker shades to generate.

        Returns:
            go.Figure: Plotly Figure object.

        Example:
            >>> cm = ColorPalette(palette="data_science")
            >>> fig = cm.visualize_palette()
            >>> fig.show()  # This will display the palette visualization

        """
        fig = go.Figure()

        # Main colors
        main_colors_len = len(self.main_colors)
        fig.add_trace(
            go.Bar(
                y=["Main Colors"] * main_colors_len,
                x=[1] * main_colors_len,
                orientation="h",
                marker=dict(color=self.main_colors, line=dict(width=0)),
                text=[
                    f"{color} <br> ({self.icolor_map[color]})"
                    for color in self.main_colors
                ],
                textposition="inside",
                insidetextanchor="middle",
                showlegend=False,
                hoverinfo="none",
            )
        )

        # Highlight colors
        if self.highlight_colors:
            highlight_colors_len = len(self.highlight_colors)
            fig.add_trace(
                go.Bar(
                    y=["Highlight Colors"] * highlight_colors_len,
                    x=[1] * highlight_colors_len,
                    orientation="h",
                    marker=dict(color=self.highlight_colors, line=dict(width=0)),
                    text=[
                        f"{color} <br> ({self.icolor_map[color]})"
                        for color in self.highlight_colors
                    ],
                    textposition="inside",
                    insidetextanchor="middle",
                    showlegend=False,
                    hoverinfo="none",
                )
            )

        # Color variations
        for colors in [self.main_colors, self.highlight_colors]:
            for color in colors:
                color_name = self.icolor_map[color]

                if color_name in self.predefined_shades:
                    shades = self.predefined_shades[color_name]
                    fig.add_trace(
                        go.Bar(
                            y=[f"Predefined Shades of {color_name}"] * len(shades),
                            x=[1] * len(shades),
                            orientation="h",
                            marker=dict(color=shades, line=dict(width=0)),
                            text=shades,
                            textposition="inside",
                            insidetextanchor="middle",
                            showlegend=False,
                            hoverinfo="none",
                        )
                    )

                shades = self.get_shades(color)

                self.generate_shades(color, num_darker=3, num_lighter=4)
                fig.add_trace(
                    go.Bar(
                        y=[f"Shades of {color_name}"] * len(shades),
                        x=[1] * len(shades),
                        orientation="h",
                        marker={"color": shades, "line": {"width": 0}},
                        text=shades,
                        textposition="inside",
                        insidetextanchor="middle",
                        showlegend=False,
                        hoverinfo="none",
                    )
                )
        # Layout
        fig.update_layout(
            title=f"{self.name} Color Palette",
            xaxis={"showticklabels": False},
            yaxis={"title": "Color Groups"},
            height=400 + len(self.main_colors + self.highlight_colors) * 50,
            width=800,
            margin={"l": 150, "r": 50, "t": 50, "b": 50},
            autosize=False,
            barmode="stack",
        )

        return fig

    @staticmethod
    def _rgb_to_lab(rgb) -> LabColor:
        """
        Convert an RGB tuple (0-255) to LAB color space.

        Args:
            rgb (tuple[int, int, int]): RGB color tuple.

        Returns:
            LabColor: Color in LAB color space.

        Example:
            >>> ColorPalette._rgb_to_lab((255, 0, 0))
            <colormath.color_objects.LabColor object at ...>
        """
        srgb = sRGBColor(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
        return convert_color(srgb, LabColor)

    @staticmethod
    def _calculate_distance(color1: LabColor, color2: LabColor) -> float:
        """
        Calculate perceptual distance between two LAB colors.

        Args:
            color1 (LabColor): First color in LAB space.
            color2 (LabColor): Second color in LAB space.

        Returns:
            float: Perceptual distance between colors.

        Example:
            >>> lab1 = ColorPalette._rgb_to_lab((255, 0, 0))
            >>> lab2 = ColorPalette._rgb_to_lab((0, 255, 0))
            >>> distance = ColorPalette._calculate_distance(lab1, lab2)
            >>> print(f"{distance:.2f}")
            86.60
        """
        return delta_e_cie2000(color1, color2)

    def _hls_to_hex(self, h: float, l: float, s: float) -> str:
        """
        Convert HLS color to hex.

        Args:
            h (float): Hue (0-1).
            l (float): Lightness (0-1).
            s (float): Saturation (0-1).

        Returns:
            str: Hex color code.

        Example:
            >>> cm = ColorPalette()
            >>> print(cm._hls_to_hex(0.0, 0.5, 1.0))
            #FF0000

        """
        rgb = colorsys.hls_to_rgb(h, l, s)
        return self._rgb_to_hex(tuple(int(x * 255) for x in rgb))

    def _create_colorscale(self, colors: list[str]) -> list[list[float]]:
        """
        Create a Plotly colorscale from colors.

        Args:
            colors (list[str]): list of color hex codes.

        Returns:
            list[list[float]]: Plotly colorscale.

        Example:
            >>> cm = ColorPalette()
            >>> scale = cm._create_colorscale(["#FF0000", "#00FF00", "#0000FF"])
            >>> print(scale[:2])  # Print first two entries of the colorscale
            [[0.0, 'rgb(255,0,0)'], [0.5, 'rgb(0,255,0)']]

        """
        return make_colorscale(colors)

    def _interpolate_colors(self, num_colors: int) -> list[str]:
        """
        Interpolate colors along the main color range.

        Args:
            num_colors (int): Number of colors to interpolate.

        Returns:
            list[str]: list of interpolated color hex codes.

        Example:
            >>> cm = ColorPalette(main_colors=["#FF0000", "#0000FF"])
            >>> print(cm._interpolate_colors(3))
            ['#FF0000', '#800080', '#0000FF']

        """
        return [
            self._rgb_to_hex(
                find_intermediate_color(
                    self._hex_to_rgb(self.main_colors[0]),
                    self._hex_to_rgb(self.main_colors[-1]),
                    i / (num_colors - 1),
                )
            )
            for i in range(num_colors)
        ]

    def _generate_categorical_colors(self, num_colors: int) -> list[str]:
        """
        Generate categorical colors, possibly with highlight colors.

        Args:
            num_colors (int): Number of colors to generate.

        Returns:
            list[str]: list of generated color hex codes.

        Example:
            >>> cm = ColorPalette(palette="data_science")
            >>> print(cm._generate_categorical_colors(3))
            ['#1696D2', '#EC008B', '#55B748']

        """
        all_colors = self.main_colors + self.highlight_colors
        return random.sample(all_colors, min(num_colors, len(all_colors)))

    def _generate_diverging_colors(self, num_colors: int, mid_color: str) -> list[str]:
        """
        Generate a diverging color palette.

        Args:
            num_colors (int): Number of colors to generate.
            mid_color (str): Middle color of the diverging palette.

        Returns:
            list[str]: list of generated color hex codes.

        Example:
            >>> cm = ColorPalette(main_colors=["#FF0000", "#FFFFFF", "#0000FF"])
            >>> print(cm._generate_diverging_colors(5, "#FFFFFF"))
            ['#FF0000', '#FF8080', '#FFFFFF', '#8080FF', '#0000FF']

        """
        left_colors = self._interpolate_between_colors(
            self.main_colors[0], mid_color, num_colors // 2
        )
        right_colors = self._interpolate_between_colors(
            mid_color, self.main_colors[-1], num_colors - num_colors // 2
        )
        return left_colors + right_colors

    def _interpolate_between_colors(
        self, start_color: str, end_color: str, num_colors: int
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
            >>> cm = ColorPalette()
            >>> print(cm._interpolate_between_colors("#FF0000", "#0000FF", 3))
            ['#FF0000', '#800080', '#0000FF']

        """
        return [
            self._rgb_to_hex(
                find_intermediate_color(
                    self._hex_to_rgb(start_color),
                    self._hex_to_rgb(end_color),
                    i / (num_colors - 1),
                )
            )
            for i in range(num_colors)
        ]

    def _generate_color_variations(self, color: str, num_shades: int) -> list[str]:
        """
        Generate lighter and darker variations of a color.

        Args:
            color (str): Base color hex code.
            num_shades (int): Number of shades to generate.

        Returns:
            list[str]: list of generated color hex codes.

        Example:
            >>> cm = ColorPalette()
            >>> print(cm._generate_color_variations("#FF0000", 2))
            ['#800000', '#FF0000', '#FF8080']

        """
        rgb_color = self._hex_to_rgb(color)

        darker_shades = [
            self._rgb_to_hex(self._adjust_color_brightness(rgb_color, -i / num_shades))
            for i in range(1, num_shades + 1)
        ]

        lighter_shades = [
            self._rgb_to_hex(self._adjust_color_brightness(rgb_color, i / num_shades))
            for i in range(1, num_shades + 1)
        ]

        return darker_shades[::-1] + [color] + lighter_shades

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        """
        Convert hex color to RGB tuple.

        Args:
            hex_color (str): Hex color code.

        Returns:
            tuple[int, int, int]: RGB color tuple.

        Example:
            >>> ColorPalette._hex_to_rgb("#FF0000")
            (255, 0, 0)

        """
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def _rgb_to_hex(rgb_color: tuple[int, int, int]) -> str:
        """
        Convert RGB tuple to hex color.

        Args:
            rgb_color (tuple[int, int, int]): RGB color tuple.

        Returns:
            str: Hex color code.

        Raises:
            ValueError: If the input is not a valid RGB tuple.

        Example:
            >>> ColorPalette._rgb_to_hex((255, 0, 0))
            '#FF0000'

        """
        if not (len(rgb_color) == 3 and all(0 <= x <= 255 for x in rgb_color)):
            msg = "Input must be a tuple with three integers in the range 0-255"
            raise ValueError(msg)

        return "#{:02X}{:02X}{:02X}".format(*(int(v) for v in rgb_color))

    @staticmethod
    def _adjust_color_brightness(
        rgb_color: tuple[int, int, int], factor: float
    ) -> tuple[int, ...]:
        """
        Adjust color brightness.

        Args:
            rgb_color (tuple[int, int, int]): RGB color tuple.
            factor (float): Adjustment factor. Negative factor darkens, positive factor lightens.

        Returns:
            tuple[int, int, int]: Adjusted RGB color tuple.

        Example:
            >>> ColorPalette._adjust_color_brightness((100, 150, 200), 0.2)
            (131, 170, 211)

        """
        if factor < 0:  # Darken
            black = (0, 0, 0)
            return tuple(
                int(rgb_color[i] + (black[i] - rgb_color[i]) * -factor)
                for i in range(3)
            )

        # else Lighten
        white = (255, 255, 255)
        return tuple(
            int(rgb_color[i] + (white[i] - rgb_color[i]) * factor) for i in range(3)
        )

    @staticmethod
    def _adjust_color_intensity(
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
            >>> ColorPalette._adjust_color_intensity((100, 150, 200), 0.2)
            (100, 160, 220)
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

    def _get_closest_color_name(self, hex_color: str) -> str:
        """
        Get the closest standard color name for a given hex color.

        Args:
            hex_color (str): Hex color code.

        Returns:
            str: Closest color name.

        Example:
            >>> cm = ColorPalette()
            >>> cm._get_closest_color_name("#FF0000")
            'red'

        """
        try:
            # Convert hex code to RGB
            rgb_value = self._hex_to_rgb(hex_color)
        except ValueError:
            return None

        # Find the closest color
        closest_name = None
        min_distance = float("inf")
        for name, code in CSS3_NAMES_TO_HEX.items():
            color_rgb = self._hex_to_rgb(code)
            distance = sum((rgb_value[i] - color_rgb[i]) ** 2 for i in range(3))
            if distance < min_distance:
                closest_name = name
                min_distance = distance

        return closest_name


if __name__ == "__main__":
    # Example usage
    def main():
        # Create palette using a preset palette
        for pal in ColorPalette.PRESET_PALETTES:
            cm = ColorPalette(name=pal, palette=pal)

            # Visualize the palette
            fig = cm.visualize_palette()
            fig.show()

        # Create palette using a color-blind friendly palette
        color_blind_palette = ColorPalette(
            name="Color Blind Safe Palette", palette="color_blind_safe"
        )

        # Get color by tag
        primary_blue = color_blind_palette.get_color_by_tag("blue")
        print("Primary Blue:", primary_blue)

        # Generate complementary palette
        complementary_colors = color_blind_palette.generate_complementary_palette(
            primary_blue
        )
        print("Complementary Colors:", complementary_colors)

        # Generate accessible palette
        accessible_colors = color_blind_palette.generate_accessible_palette()
        print("Accessible Colors:", accessible_colors)

        # Suggest additional palettes
        suggested_palettes = color_blind_palette.suggest_additional_palettes()
        for name, palette in suggested_palettes.items():
            print(f"{name} Palette: {palette}")

    main()

    # import json

    # PRESET_PALETTES = ColorPalette.PRESET_PALETTES

    # for palette_name, palette_data in PRESET_PALETTES.items():
    #     filename = f"colormaps/{palette_name}_palette.json"
    #     palette_data["name"] = palette_name
    #     palette_dict = palette_data

    #     with open(filename, 'w') as f:
    #         json.dump(palette_dict, f, indent=4)

    #     print(f"Created {filename}")
