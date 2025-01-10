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

from typing import TYPE_CHECKING, Generic, TypeVar

from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    TwoAxesInstanciatedChart as BaseTwoAxesInstanciatedChart,
)
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import (
    InstantiatedPlotlyNativeChart as BaseInstantiatedPlotlyNativeChart,
)

from climateeconomics.core.tools.color_map import ColorMap
from climateeconomics.core.tools.colormaps import available_colormaps
from climateeconomics.core.tools.palettes import available_palettes

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from climateeconomics.core.tools.color_palette import ColorPalette

DEFAULT_PALETTE: ColorPalette | None = None
DEFAULT_COLORMAP: ColorMap | None = None


def set_default_palette(
    palette: str | ColorPalette | None = None,
) -> ColorPalette | None:
    """Set default palette to new value.

    Example:
        >>> from climateeconomics.core.tools.color_palette import ColorPalette
        >>> palette = set_default_palette(ColorPalette(name="witness"))
        >>> palette.name
        'witness'
    """
    global DEFAULT_PALETTE

    if isinstance(palette, str):
        if palette in available_palettes:
            DEFAULT_PALETTE = available_palettes[palette]
            return DEFAULT_PALETTE

    DEFAULT_PALETTE = palette
    return DEFAULT_PALETTE


def set_default_colormap(colormap: str | ColorMap | None = None) -> ColorMap | None:
    """Set default colormap to new value.

    Example:
        >>> colormap = set_default_colormap(ColorMap(name="sectors", color_map={}))
        >>> print(colormap.name)
        sectors
    """
    global DEFAULT_COLORMAP

    if isinstance(colormap, str):
        if colormap in available_colormaps:
            DEFAULT_COLORMAP = available_colormaps[colormap]
        return DEFAULT_COLORMAP

    DEFAULT_COLORMAP = colormap
    return DEFAULT_COLORMAP


T = TypeVar("T", bound="ExtendedMixin")


# Mixin class with common additional methods
class ExtendedMixin(Generic[T]):
    color_palette: ColorPalette = None
    color_map: ColorMap = None
    group_name: str = None
    layout_custom_updates: dict = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "color_palette" in kwargs:
            self.set_color_palette(kwargs.get("color_palette"))
        else:
            self.set_color_palette(DEFAULT_PALETTE)

        if "group_name" in kwargs:
            self.set_group(kwargs.get("group_name"))

        if "color_map" in kwargs:
            self.set_color_map(kwargs.get("color_map"))
        else:
            self.set_color_map(DEFAULT_COLORMAP)

    def set_group(self, group_name: str) -> T:
        if self.color_palette is None:
            msg = "No palette as been set yet. Please set it specifying a group."
            raise ValueError(msg)

        if group_name not in self.color_palette.predefined_groups:
            msg = f"{group_name} is not defined in the palette ({self.color_palette.name})."
            raise ValueError(msg)

        self.group_name = group_name

        return self

    def set_color_palette(
        self,
        color_palette: ColorPalette | str | None = None,
        group_name: str | None = None,
    ) -> T:
        """Set color palette."""
        if isinstance(color_palette, str):
            color_palette = color_palette.lower()
            if color_palette not in available_palettes and color_palette:
                possible_palettes = list(available_palettes)
                msg = (
                    f"{color_palette} not available in predefined color palettes. "
                    f"Possible values are: {possible_palettes}"
                )
                raise ValueError(msg)
            self.color_palette = available_palettes[color_palette]
        else:
            self.color_palette = color_palette

        self.group_name = group_name

        return self

    def set_color_map(
        self,
        color_map: dict | ColorMap | str | None = None,
        fill_nonexistent: bool = False,
    ) -> T:
        """Set color map."""

        if isinstance(color_map, str):
            color_map = color_map.lower()
            if color_map not in available_colormaps:
                msg = f"No colormap named {color_map} is available. Possible colormap names are {available_colormaps.keys()}"
                raise ValueError(msg)
            color_map = available_colormaps[color_map]

        elif isinstance(color_map, dict):
            color_map = ColorMap(color_map=color_map, fill_nonexistent=fill_nonexistent)

        self.color_map = color_map

        return self

    def set_layout_custom_updates(self, layout_updates: dict) -> T:
        """Set layout custom updates."""
        self.layout_custom_updates = layout_updates
        return self

    def to_plotly(self, logger=None) -> go.Figure:
        """Convert to plotly figure."""
        fig: go.Figure = super().to_plotly(logger=logger)

        # Set the colorway in the layout of the figure
        if self.color_palette is not None:
            # Check if color palette has enough colors to plot all the traces
            if len(self.color_palette.main_colors) < len(fig.data):
                msg = "Palette does not have enough colors for plotting all the data."
                raise ValueError(msg)

            fig.update_layout(colorway=self.color_palette.main_colors)

        # Force the first N colors to follow the group name, if given
        if self.group_name is not None:
            group_colors = self.color_palette.get_group_by_name(self.group_name)
            len_group = len(group_colors)
            for i, trace in enumerate(fig.data):
                if i >= len_group:
                    break
                trace.update(line={"color": group_colors[i]})

        # Loop through each trace and update the color based on its name
        if self.color_map is not None:
            for trace in fig.data:
                series_name = trace.name
                if series_name in self.color_map:
                    color = self.color_map[series_name]
                elif self.color_map.fill_nonexistent:
                    color = self.color_map.get_color(series_name)
                else:
                    continue

                # Update marker color if trace has marker properties
                if hasattr(trace, "marker"):
                    trace.marker.color = color

                # Update line color if trace has line properties
                if hasattr(trace, "line"):
                    trace.line.color = color

        # Remove vertical lines in x axis
        fig.update_layout(xaxis={"showgrid": False})
        fig.update_yaxes(rangemode="tozero")

        # Update layout with custom layout updates
        if self.layout_custom_updates:
            fig.update_layout(**self.layout_custom_updates)

        return fig


class WITNESSTwoAxesInstanciatedChart(
    ExtendedMixin["WITNESSTwoAxesInstanciatedChart"], BaseTwoAxesInstanciatedChart
):
    pass


class WITNESSInstantiatedPlotlyNativeChart(
    ExtendedMixin["WITNESSInstantiatedPlotlyNativeChart"], BaseInstantiatedPlotlyNativeChart
):
    pass


class TwoAxesInstanciatedChart(
    ExtendedMixin["TwoAxesInstanciatedChart"], BaseTwoAxesInstanciatedChart
):
    pass


class InstantiatedPlotlyNativeChart(
    ExtendedMixin["InstantiatedPlotlyNativeChart"], BaseInstantiatedPlotlyNativeChart
):
    pass
