"""
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
"""

from climateeconomics.core.tools.color_palette import ColorPalette

WITNESSPalette: ColorPalette = ColorPalette.from_dict(
    palette_name="witness",
    palette_dict={
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
    },
)

ColorBlindSafePalette: ColorPalette = ColorPalette.from_dict(
    palette_name="color_blind_safe",
    palette_dict={
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
)

EarthTonesPalette = ColorPalette.from_dict(
    palette_name="earth_tones",
    palette_dict={
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
)

PastelVibesPalette = ColorPalette.from_dict(palette_name="pastel_vibes", palette_dict={
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
})

available_palettes: dict = {
    "witness": WITNESSPalette,
    "cbs": ColorBlindSafePalette,
    "earth_tones": EarthTonesPalette,
    "pastel_vibes": PastelVibesPalette
}
