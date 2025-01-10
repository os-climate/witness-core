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

# ruff: noqa: E741


class ColorMap:
    """Represents a color map between text and colors."""

    def __init__(
        self,
        color_map: dict,
        name: str = "default_name",
        fill_nonexistent: bool = False,
    ) -> None:
        self.name = name
        self.base_mapping = color_map
        self.additional_mapping = {}
        self.fill_nonexistent = fill_nonexistent

    @property
    def all_colors(self) -> list:
        """Return list of all colors.

        Example:
            >>> cmap = ColorMap({"text1": "red", "text2": "blue"})
            >>> print(cmap.all_colors)
            ['red', 'blue']
            """
        return list(
            (self.base_mapping | self.additional_mapping).values()
        )

    @property
    def mapping(self) -> dict:
        """Return extended mapping.

        Example:
            >>> cmap = ColorMap({"text1": "red", "text2": "blue"})
            >>> print(cmap.mapping)
            {'text1': 'red', 'text2': 'blue'}
            >>> cmap.additional_mapping = {"text3": "black"}
            >>> print(cmap.mapping)
            {'text1': 'red', 'text2': 'blue', 'text3': 'black'}

        """
        return self.base_mapping | self.additional_mapping

    def get_colors(self, text_list: list, as_dict: bool = False) -> list | dict:
        """Get colors from list of strings.

        Args:
            text_list (list): Strings to assign a color to.
            as_dict (bool, optional): Whether to return a dict or a list. Defaults to False.

        Returns:
            list | dict: _description_
        """
        if as_dict:
            return {text: self.get_color(text) for text in text_list}
        return [self.get_color(text) for text in text_list]

    def get_color(self, text: str) -> str:
        """Get color from text, based on mapping and additional mapping.

        Args:
            text (str): Text to be used to find the color.

        Returns:
            str: color
        """
        text = text.lower().strip()

        # Initialize additional_mapping if it's None
        if self.additional_mapping is None:
            self.additional_mapping = {}

        # Check if the resource is in the original mapping or additional mapping
        for resource, color in {**self.base_mapping, **self.additional_mapping}.items():
            if resource in text:
                return color

        # If no matching resource found, generate a random color
        while True:
            # Generate a random color
            random_color = f"#{random.randint(0, 0xFFFFFF):06x}"

            # Check if the color is unique
            if random_color not in self.all_colors:
                # Add the new mapping to additional_mapping
                self.additional_mapping[text] = random_color
                return random_color

    # Implementing __contains__ to allow "if something in color_map"
    def __contains__(self, key):
        return key in self.mapping

    # Implementing __iter__ to allow "for color in color_map"
    def __iter__(self):
        return iter(self.mapping)

    # Implementing __getitem__ to allow "color_map['something']"
    def __getitem__(self, key):
        return self.mapping[key]

    # Implementing __add__ to merge two ColorMap instances
    def __add__(self, other) -> ColorMap:
        if not isinstance(other, ColorMap):
            msg = "Can only add another ColorMap instance"
            raise TypeError(msg)

        # Merge the two mappings, keeping the values from the first (self) in case of duplicates
        merged_mapping = self.mapping.copy()
        merged_mapping.update(
            other.mapping
        )  # Updates with the second mapping, keeping first's values for duplicates

        return ColorMap(merged_mapping)

    # Implementing __or__ to merge two ColorMap instances using the | operator
    def __or__(self, other):
        if not isinstance(other, ColorMap):
            msg = "Can only use '|' with another ColorMap instance"
            raise TypeError(msg)

        # Merge the two mappings, keeping the values from the first (self) in case of duplicates
        merged_mapping = self.mapping.copy()
        merged_mapping.update(
            other.mapping
        )  # Updates with the second mapping, keeping first's values for duplicates

        return ColorMap(merged_mapping)
