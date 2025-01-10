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
            "#0a4c6a",
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
            "dark_blue",
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
            "four": ["primary_blue", "black", "neutral_gray", "yellow"],
            "four_2": ["primary_blue", "black", "neutral_gray", "yellow"],
            "four_3": ["primary_blue", "black", "neutral_gray", "magenta"],
            "four_4": ["primary_blue", "black", "yellow", "green"],
            "four_5": ["primary_blue", "black", "yellow", "magenta"],
            "four_seq": ["#cfe8f3", "#73bfe2", "primary_blue", "#0a4c6a"],
            "five": ["primary_blue", "black", "neutral_gray", "yellow", "magenta"],
            "five_2": ["primary_blue", "black", "neutral_gray", "yellow", "dark_blue"],
            "five_3": ["primary_blue", "black", "neutral_gray", "green", "dark_blue"],
            "five_4": ["primary_blue", "black", "yellow", "green", "magenta"],
            "five_seq": ["#cfe8f3", "#73bfe2", "primary_blue", "dark_blue", "black"],
            "six": [
                "primary_blue",
                "black",
                "neutral_gray",
                "yellow",
                "green",
                "magenta",
            ],
            "six_2": [
                "primary_blue",
                "black",
                "neutral_gray",
                "yellow",
                "green",
                "dark_blue",
            ],
            "six_3": [
                "primary_blue",
                "black",
                "dark_blue",
                "yellow",
                "green",
                "magenta",
            ],
            "seven": [
                "primary_blue",
                "black",
                "neutral_gray",
                "yellow",
                "green",
                "magenta",
                "dark_blue",
            ],
            "diverging": [
                "#ca5800",
                "#fdbf11",
                "#fdd870",
                "#fff2cf",
                "#cfe8f3",
                "#73bfe2",
                "#1696d2",
                "#0a4c6a",
            ],
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

PastelVibesPalette = ColorPalette.from_dict(
    palette_name="pastel_vibes",
    palette_dict={
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
)

IBMColorSafePalette = ColorPalette.from_dict(
    palette_name="IBM Color Safe",
    palette_dict={
        "main_colors": [
            "#648fff",
            "#785ef0",
            "#dc267f",
            "#fe6100",
            "#ffb000",
            "#000000",
        ],
        "color_tags": ["ultramarine", "indigo", "magenta", "orange", "gold", "black"],
        "description": "retrieved from https://lospec.com/palette-list/ibm-color-blind-safe",
        "color_blind_friendly": False,
    },
)

CapgeminiPalette = ColorPalette(
    name="Capgemini",
    main_colors={
        "capgemini_blue": "#0070AD",
        "vibrant_blue": "#12ABDB",
        "dark_grey": "#272936",
        "green": "#2EA657",
        "teal": "#00BFBF",
        "peacock": "#0F878A",
        "sapphire": "#14596B",
        "yellow": "#FFB24A",
        "red": "#FF304D",
        "violet": "#BA2980",
        "velvet": "#750D5C",
    },
    predefined_groups={
        "cool": ["green", "teal", "peacock", "sapphire"],
        "warm": ["yellow", "red", "violet", "velvet"],
    },
)

print(CapgeminiPalette.color_map)

PlotlyPalette = ColorPalette.from_plotly_qualitative("Plotly")
ColorBlindSafePalette2 = ColorPalette.from_plotly_qualitative("Safe")

available_palettes: dict = {
    "witness": WITNESSPalette,
    "capgemini": CapgeminiPalette,
    "color_blind_safe": ColorBlindSafePalette,
    "color_blind_safe2": ColorBlindSafePalette2,
    "earth_tones": EarthTonesPalette,
    "pastel_vibes": PastelVibesPalette,
    "plotly": PlotlyPalette,
    "ibm": IBMColorSafePalette,
}

if __name__ == "__main__":
    for pal in available_palettes.values():
        # Visualize the palette
        pal.visualize_palette()
        pal.visualize_groups()
