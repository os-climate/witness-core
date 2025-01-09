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

import random

import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from plotly.colors import find_intermediate_color, make_colorscale

from climateeconomics.core.tools.color_tools import (
    CSS3_NAMES_TO_HEX,
    adjust_color_brightness,
    adjust_color_intensity,
    hex_to_rgb,
    interpolate_between_colors,
    rgb_string_to_hex,
    rgb_to_hex,
    to_hex,
    to_rgb_string,
)

# ruff: noqa: E741


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
        "witness": {
            "main_colors": [
                "#1696D2",
                "#D2D2D2",
                "#000000",
                "#FDBF11",
                "#EC008B",
                "#5c5859",
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
                "space_gray",
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
            "groups": {
                "two": ["primary_blue", "black"],
                "two_2": ["primary_blue", "neutral_gray"],
                "two_3": ["primary_blue", "yellow"],
                "two_4": ["black", "yellow"],
                "two_seq": ["#a2d4ec", "primary_blue"],
                "three": ["primary_blue", "black", "neutral_gray"],
                "three_2": ["primary_blue", "black", "yellow"],
                "three_3": ["primary_blue", "black", "green"],
                "three_4": ["primary_blue", "black", "magenta"],
                "three_seq": ["#a2d4ec", "primary_blue", "#0a4c6a"],
            },
            "description": "Clean, professional palette for data visualization",
            "color_blind_friendly": False,
        }
    }

    def __init__(
        self,
        name: str = "",
        main_colors: list[str] | None | dict = None,
        highlight_colors: list[str] | None = None,
        color_tags: list[str] | None = None,
        predefined_shades: dict | None = None,
        predefined_groups: dict | None = None,
    ):
        """
        Initialize an enhanced palette with flexible configuration options.

        Args:
            name (str): Name of the palette.
            main_colors (list[str]): Custom main colors.
            highlight_colors (Optional[list[str]]): Custom highlight colors.
            color_tags (Optional[list[str]]): Custom color tags.
            predefined_shades (Optional[dict]): dictionary of predefined color shades.

        Example:
            >>> cm = ColorPalette(name="My palette", main_colors=["#000000", "#FFFFFF"])
            >>> print(cm.name)
            My palette

        """
        # Determine color sources with priority
        if isinstance(main_colors, dict):
            if isinstance(highlight_colors, list):
                raise ValueError("highlight colors should be given in same format of main_colors.")

            self.main_colors = list(main_colors.values())
            highlight_colors = highlight_colors or {}
            self.highlight_colors = list(highlight_colors.values())

            self.color_tags = list(main_colors.keys()) + list(highlight_colors.keys())
        else:
            self.main_colors = main_colors or []
            self.highlight_colors = highlight_colors or []

            self.color_tags = color_tags or [
                f"color_{i}" for i in range(len(self.main_colors + self.highlight_colors))
            ]

        self.predefined_shades = predefined_shades or {}
        self.predefined_groups = predefined_groups or {}

        # Ensure color_tags matches the number of main_colors
        if len(self.color_tags) != len(self.main_colors + self.highlight_colors):
            self.color_tags = [f"color_{i}" for i in range(len(self.main_colors + self.highlight_colors))]

        self.name = name
        self.colorscale = make_colorscale(self.main_colors) if main_colors else []

        # Tag-to-color mapping
        self.color_map = dict(
            zip(self.color_tags, [*self.main_colors, *self.highlight_colors])
        )

        # color-to-tag mapping
        self.icolor_map = dict(
            zip([*self.main_colors, *self.highlight_colors], self.color_tags)
        )

        # Convert shades and group colors to hex
        for shade_name, shade_colors in self.predefined_shades.items():
            self.predefined_shades[shade_name] = self.convert_colors_to_format(
                shade_colors
            )
        for group_name, group_colors in self.predefined_groups.items():
            self.predefined_groups[group_name] = self.convert_colors_to_format(
                group_colors
            )

    @classmethod
    def from_dict(cls, palette_name: str, palette_dict: dict) -> ColorPalette:
        """Create a ColorPalete instance from a dict."""

        main_colors = palette_dict.get("main_colors")
        highlight_colors = palette_dict.get("highlight_colors")
        color_tags = palette_dict.get("color_tags")
        predefined_shades = palette_dict.get("shades", {})
        predefined_groups = palette_dict.get("groups", {})

        return cls(
            name=palette_name,
            main_colors=main_colors,
            highlight_colors=highlight_colors,
            color_tags=color_tags,
            predefined_shades=predefined_shades,
            predefined_groups=predefined_groups,
        )

    @classmethod
    def from_preset(cls, preset_name: str, name: str | None = None) -> ColorPalette:
        """Create a ColorPalette instance from a preset palette."""
        if preset_name not in cls.PRESET_PALETTES:
            msg = "Preset does not exist."
            raise ValueError(msg)

        if name is None:
            name = preset_name

        preset = cls.PRESET_PALETTES[preset_name]

        return cls.from_dict(palette_name=name, palette_dict=preset)

    @classmethod
    def from_plotly_template(
        cls,
        template_name: str,
        name: str | None = None,
    ) -> ColorPalette:
        """
        Create a ColorPalette instance from a Plotly template colorway.

        Args:
            template_name (str): Name of the Plotly template.
            name (str): Name for the new ColorPalette instance.

        Returns:
            ColorPalette: A new ColorPalette instance.

        Example:
            >>> cm = ColorPalette.from_plotly_template('plotly', name='Plotly Palette')
            >>> print(cm.main_colors)
            ['#636efa', '#EF553B', '#00cc96', '#ab63fa', '#FFA15A', '#19d3f3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

        """
        # Retrieve the colorway from the specified Plotly template
        if template_name not in pio.templates:
            msg = f"Template '{template_name}' not found in Plotly templates."
            raise ValueError(msg)

        colorway = list(pio.templates[template_name].layout.colorway)
        if not colorway:
            msg = f"Template '{template_name}' does not have a colorway defined."
            raise ValueError(msg)

        if name is None:
            name = template_name

        # Create a ColorPalette instance using the retrieved colorway
        return cls(name=name, main_colors=colorway)

    @classmethod
    def from_plotly_qualitative(
        cls, colorway_name: str, name: str | None = None
    ) -> ColorPalette:
        """
        Create a ColorPalette instance from a Plotly Express qualitative colorway.

        Args:
            colorway_name (str): Name of the Plotly Express qualitative colorway.
            name (str): Name for the new ColorPalette instance.

        Returns:
            ColorPalette: A new ColorPalette instance.

        Example:
            >>> cm = ColorPalette.from_plotly_qualitative('Set1', name='Set1 Palette')
            >>> print(cm.main_colors)
            ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628', '#F781BF', '#999999']
        """
        # Retrieve the colorway from the specified Plotly Express qualitative colorway
        if not hasattr(px.colors.qualitative, colorway_name):
            msg = f"Qualitative colorway '{colorway_name}' not found in Plotly Express."
            raise ValueError(msg)

        colorway = list(getattr(px.colors.qualitative, colorway_name))

        # check if color in rgb() format and convert it to hex format
        if "rgb" in colorway[0]:
            colorway = [rgb_string_to_hex(color) for color in colorway]

        if name is None:
            name = colorway_name

        # Create a ColorPalette instance using the retrieved colorway
        return cls(name=name, main_colors=colorway)

    def register_as_plotly_template(self) -> None:
        """Register palette as a plotly template."""
        pio.templates[self.name] = go.layout.Template(
            layout=go.Layout(colorway=self.main_colors)
        )

    def get_group_by_name(self, group_name: str) -> list[str]:
        if group_name not in self.predefined_groups:
            msg = f"{group_name} is not a valid group name."
            raise ValueError(msg)
        return self.predefined_groups[group_name]

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
            >>> cm = ColorPalette(
            ...     name="My palette",
            ...     main_colors=["#000000", "#FFFFFF"],
            ...     color_tags=["color_1", "color_2"],
            ... )
            >>> print(cm.get_color_by_tag("color_1"))
            #000000

        """
        if tag not in self.color_map:
            msg = f"Color tag '{tag}' not found in the color map"
            raise ValueError(msg)
        return self.color_map[tag]

    def generate_accessible_palette(self, contrast_ratio: float = 4.5) -> list[str]:
        """
        Generate an accessible color palette with sufficient contrast.

        Args:
            contrast_ratio (float): Minimum contrast ratio (WCAG standard is 4.5).

        Returns:
            list[str]: list of accessible colors.

        Example:
            >>> cm = ColorPalette.from_preset(preset_name="witness")
            >>> print(cm.generate_accessible_palette())
            ['#1696D2', '#D2D2D2', '#FDBF11', '#EC008B', '#55B748', '#DB2B27']

        """

        def calculate_luminance(color: str) -> float:
            """Calculate relative luminance of a color."""
            rgb = hex_to_rgb(color)
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
            >>> cm = ColorPalette(main_colors=["#000000", "#FFFFFF"])
            >>> print(cm.get_color(0.5))
            #7F7F7F

        """
        if interpolation_type == "log":
            # Logarithmic interpolation
            value = np.log(1 + value) / np.log(2)

        return rgb_to_hex(
            find_intermediate_color(
                hex_to_rgb(self.main_colors[0]), hex_to_rgb(self.main_colors[-1]), value
            )
        )

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
            >>> cm = ColorPalette(main_colors=["#000000", "#FFFFFF"])
            >>> print(cm.get_shades("color_0", 3))
            ['#7F7F7F', '#3F3F3F', '#000000', '#000000']

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
            ['#FF7F7F', '#FF3F3F', '#FF0000', '#BF0000', '#7F0000']

        """
        if "#" not in color:
            color = self.main_colors[self.color_tags.index(color)]

        rgb_color = hex_to_rgb(color)
        darker_shades = [
            rgb_to_hex(adjust_color_brightness(rgb_color, -i / (num_darker + 2)))
            for i in range(1, num_darker + 1)
        ]
        lighter_shades = [
            rgb_to_hex(adjust_color_brightness(rgb_color, i / (num_lighter + 2)))
            for i in range(1, num_lighter + 1)
        ]

        colors = [*lighter_shades[::-1], color, *darker_shades]

        # Apply intensity factor
        if intensity_range is not None:
            intensity_factors = np.linspace(
                intensity_range[0], intensity_range[1], len(colors)
            )
            colors = [
                rgb_to_hex(adjust_color_intensity(hex_to_rgb(color), factor))
                for color, factor in zip(colors, intensity_factors)
            ]

        return colors

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
            >>> cm = ColorPalette(main_colors=['#000000', '#FFFFFF'])
            >>> print(cm.generate_color_palette(5, "sequential"))
            ['#000000', '#3F3F3F', '#7F7F7F', '#BFBFBF', '#FFFFFF']

        """
        if palette_type == "sequential":
            return self._interpolate_colors(num_colors)
        if palette_type == "categorical":
            return self._generate_categorical_colors(num_colors)
        if palette_type == "diverging":
            if len(self.main_colors) < 3:
                msg = "Diverging palette requires at least 3 main colors"
                raise ValueError(msg)
            mid_color = self.main_colors[len(self.main_colors) // 2]
            return self._generate_diverging_colors(num_colors, mid_color)

        # If none, raise an error
        msg = f"Unknown palette type: {palette_type}"
        raise ValueError(msg)

    def visualize_palette(self, show: bool = True) -> go.Figure:
        """
        Create a visualization of the color palette.

        Returns:
            go.Figure: Plotly Figure object.

        Example:
            >>> cm = ColorPalette.from_preset(preset_name="witness")
            >>> fig = cm.visualize_palette()
        """
        fig = go.Figure()

        # Color variations
        for colors in [self.main_colors, self.highlight_colors]:
            for color in colors:
                color_name = self.icolor_map[color]

                if color_name in self.predefined_shades:
                    shades = self.predefined_shades[color_name]
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
                else:
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

        # Highlight colors
        if self.highlight_colors:
            highlight_colors_len = len(self.highlight_colors)
            fig.add_trace(
                go.Bar(
                    y=["Highlight Colors"] * highlight_colors_len,
                    x=[1] * highlight_colors_len,
                    orientation="h",
                    marker={"color": self.highlight_colors, "line": {"width": 0}},
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
        # Main colors
        main_colors_len = len(self.main_colors)
        fig.add_trace(
            go.Bar(
                y=["Main Colors"] * main_colors_len,
                x=[1] * main_colors_len,
                orientation="h",
                marker={"color": self.main_colors, "line": {"width": 0}},
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

        if show:
            fig.show()

        return fig

    def visualize_groups(self, show: bool = True) -> go.Figure | None:
        """
        Create a visualization of the predefined color groups.

        Returns:
            go.Figure: Plotly Figure object showing all predefined groups.

        Example:
            >>> cm = ColorPalette.from_preset(preset_name="witness")
            >>> fig = cm.visualize_groups()
        """
        if not self.predefined_groups:
            return None

        fig = go.Figure()

        # Add a bar for each group
        for group_name, colors in self.predefined_groups.items():
            # Convert any color names to hex codes
            colors = self.convert_colors_to_format(colors)

            fig.add_trace(
                go.Bar(
                    y=[group_name] * len(colors),
                    x=[1] * len(colors),
                    orientation="h",
                    marker={
                        "color": colors,
                        "line": {"width": 0}
                    },
                    text=[f"{color}" for color in colors],
                    textposition="inside",
                    insidetextanchor="middle",
                    showlegend=False,
                    hoverinfo="none",
                )
            )

        # Layout configuration
        fig.update_layout(
            title=f"{self.name} Color Groups",
            xaxis={
                "showticklabels": False,
                "showgrid": False,
                "zeroline": False,
            },
            yaxis={
                "title": "Groups",
                "gridwidth": 1,
                "gridcolor": "lightgray",
            },
            height=50 + len(self.predefined_groups) * 40,  # Dynamic height based on number of groups
            width=800,
            margin={"l": 150, "r": 50, "t": 50, "b": 50},
            autosize=False,
            barmode="stack",
            plot_bgcolor="white",
        )

        if show:
            fig.show()

        return fig

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
            ['#FF0000', '#7F007F', '#0000FF']

        """
        return [
            rgb_to_hex(
                find_intermediate_color(
                    hex_to_rgb(self.main_colors[0]),
                    hex_to_rgb(self.main_colors[-1]),
                    i / (num_colors - 1),
                )
            )
            for i in range(num_colors)
        ]

    def _generate_categorical_colors(self, num_colors: int) -> list[str]:
        """
        Generate random list of colors, possibly with highlight colors.

        Args:
            num_colors (int): Number of colors to generate.

        Returns:
            list[str]: list of generated color hex codes.

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
            ['#FF0000', '#FFFFFF', '#FFFFFF', '#7F7FFF', '#0000FF']

        """
        left_colors = interpolate_between_colors(
            self.main_colors[0], mid_color, num_colors // 2
        )
        right_colors = interpolate_between_colors(
            mid_color, self.main_colors[-1], num_colors - num_colors // 2
        )
        return left_colors + right_colors

    def convert_colors_to_format(
        self, colors: list[str], target_format: str = "hex"
    ) -> list[str]:
        """
        Convert a list of color strings to a consistent format.

        Args:
            colors (list[str]): List of color strings in various formats.
            target_format (str): Target format for the colors (default is "hex").

        Returns:
            list[str]: List of colors converted to the target format.

        Example:
            >>> cm = ColorPalette.from_preset(name="witness", preset_name="witness")
            >>> colors = [
            ...     "#FF0000",
            ...     "rgb(0, 255, 0)",
            ...     "primary_blue",
            ... ]
            >>> converted_colors = cm.convert_colors_to_format(colors)
            >>> print(converted_colors)
            ['#FF0000', '#00FF00', '#1696D2']
        """
        converted_colors = []
        for color in colors:
            resolved_color = self._resolve_color_name(color)
            if target_format == "hex":
                converted_colors.append(to_hex(resolved_color))
            elif target_format == "rgb":
                converted_colors.append(to_rgb_string(resolved_color))
            else:
                raise ValueError(f"Unsupported target format: {target_format}")
        return converted_colors

    def _resolve_color_name(self, color: str) -> str:
        """
        Resolve a color name to its hex code.

        Args:
            color (str): Color name or color string.

        Returns:
            str: Hex color code.

        Raises:
            ValueError: If the color name is not found.

        Example:
            >>> cm = ColorPalette.from_preset(name="witness", preset_name="witness")
            >>> print(cm._resolve_color_name("primary_blue"))
            #1696D2
        """
        # Check internal mapping first
        if color in self.color_map:
            return self.color_map[color]
        # Check CSS3 colors
        if color in CSS3_NAMES_TO_HEX:
            return CSS3_NAMES_TO_HEX[color]

        return color
        # # If not found, raise an error
        # raise ValueError(f"Color name '{color}' not found in internal mapping or CSS3 colors")
