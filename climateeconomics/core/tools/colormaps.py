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

from climateeconomics.core.tools.color_map import ColorMap

EnergyColorMap: ColorMap = ColorMap(
    name="Energy",
    color_map={
        "electricity": "#ffd700",  # gold (yellow)
        "hydrogen": "#00aaff",  # light blue
        "nuclear": "#ffcc00",  # bright yellow
        "wind": "#add8e6",  # light blue
        "solar": "#ffa500",  # orange
        "geothermal": "#800080",  # purple
        "biomass": "#8b4513",  # saddle brown
        "hydropower": "#4682b4",  # steel blue
        "heat": "#ff0000",  # red
        "wind power": "#00bfff",  # deep sky blue
        "solar power": "#ff6347",  # tomato (red-orange)
        "coal power": "#696969",  # dim gray
        "oil power": "#2f4f4f",  # dark slate gray
        "nuclear power": "#ffd700",  # gold
        "biomass power": "#a0522d",  # sienna
        "geothermal power": "#9932cc",  # dark orchid
        "tidal power": "#008080",  # teal
        "wave power": "#20b2aa",  # light sea green
    },
)

ResourcesColorMap: ColorMap = ColorMap(
    name="resources",
    color_map={
        "coal": "#4b4b4b",  # dark gray (blackish)
        "iron": "#b7410e",  # rust (brownish-red)
        "steel": "#0000ff",  # blue
        "natural gas": "#1e90ff",  # dodger blue
        "oil": "#000000",  # black
        "hydrogen fuel": "#40e0d0",  # turquoise
        "renewable energy": "#32cd32",  # lime green
        # Additional resources
        "natural gas power": "#87ceeb",  # sky blue
        "fuel cell": "#7fffd4",  # aquamarine
        "battery storage": "#dda0dd",  # plum
        "pumped hydro storage": "#4169e1",  # royal blue
        "compressed air energy storage": "#708090",  # slate gray
        "flywheel energy storage": "#cd853f",  # peru
        "thermal energy storage": "#dc143c",  # crimson
        "biofuel": "#bdb76b",  # dark khaki
        "ethanol": "#f4a460",  # sandy brown
        "biodiesel": "#9acd32",  # yellow green
        "methane": "#6b8e23",  # olive drab
        "propane": "#ff69b4",  # hot pink
        "uranium": "#7fff00",  # chartreuse
        "thorium": "#00fa9a",  # medium spring green
        "lithium": "#ff1493",  # deep pink
        "cobalt": "#4682b4",  # steel blue
        "rare earth elements": "#9370db",  # medium purple
        "silicon": "#c0c0c0",  # silver
        "copper": "#b87333",  # copper
        "aluminum": "#848484",  # gray
        "nickel": "#a9a9a9",  # dark gray
        "lead": "#778899",  # light slate gray
        "zinc": "#d3d3d3",  # light gray
        "titanium": "#e6e6fa",  # lavender
        "platinum": "#e5e4e2",  # platinum
        "palladium": "#bcd4e6",  # pale blue
        "wood": "#deb887",  # burlywood
        "paper": "#f5deb3",  # wheat
        "plastic": "#f0e68c",  # khaki
        "glass": "#e0ffff",  # light cyan
        "cement": "#d3d3d3",  # light gray
        "concrete": "#a9a9a9",  # dark gray
        "asphalt": "#2f4f4f",  # dark slate gray
        # Adding water and related resources
        "water": "#0077be",  # ocean blue
        "freshwater": "#87cefa",  # light sky blue
        "saltwater": "#1e90ff",  # dodger blue
        "wastewater": "#8b4513",  # saddle brown
        "desalination": "#48d1cc",  # medium turquoise
        "water treatment": "#5f9ea0",  # cadet blue
        "groundwater": "#66cdaa",  # medium aquamarine
        "surface water": "#4169e1",  # royal blue
        "rainwater": "#b0e0e6",  # powder blue
        "ice": "#f0f8ff",  # alice blue
        "iron ore": "#3b2300",  # very dark brown
        "scrap": "#cc9c54",  # light brown yellow
        "limestone": "#c7ea46",  # light lime green
        "blast furnace gas": "#ff7900",  # orange
    },
)

SectorsColorMap: ColorMap = ColorMap(
    color_map={"Industry": "#000000", "Services": "#1696D2", "Agriculture": "#55B748"}
)

FullColorMap: ColorMap = EnergyColorMap | ResourcesColorMap | SectorsColorMap

available_colormaps: dict = {
    "energy": EnergyColorMap,
    "resources": ResourcesColorMap,
    "sectors": SectorsColorMap,
    "full": FullColorMap
}
