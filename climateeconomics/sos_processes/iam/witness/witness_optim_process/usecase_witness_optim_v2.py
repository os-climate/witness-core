from os.path import join, dirname
import numpy as np
import pandas as pd
import json
from sostrades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):
    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(
            __file__, run_usecase=run_usecase, execution_engine=execution_engine
        )

    def setup_usecase(self):
        sty = self.study_name
        inputs = {}

        self.data_dir = join(dirname(__file__), "data")
        margin = pd.read_csv(join(self.data_dir, "margin.csv"))
        transport_cost = pd.read_csv(join(self.data_dir, "transport_cost.csv"))
        biogas_transport_cost = pd.read_csv(
            join(self.data_dir, "biogas_transport_cost.csv")
        )
        biomass_dry_transport_cost = pd.read_csv(
            join(self.data_dir, "biomass_dry_transport_cost.csv")
        )
        AutothermalReforming_margin = pd.read_csv(
            join(self.data_dir, "AutothermalReforming_margin.csv")
        )
        eq_constraints = []

        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.DesignVariables.design_var_descriptor"
        ] = json.load(open(join(self.data_dir, "design_var_descriptor.json")))
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.DesignVariables.write_xvect"] = False
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.FunctionsManager.function_df"
        ] = pd.read_csv(join(self.data_dir, "function_df.csv"))
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.direct_air_capture.AmineScrubbing.carbon_capture_direct_air_capture_AmineScrubbing_array_mix"
        ] = np.asarray(
            [
                139.82135063123013,
                64.4740511421994,
                42.551739068431104,
                0.45663126472936244,
                4.731963253584282,
                4.357868530831491,
                2.7316377671495724,
                12.22767969513814,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.direct_air_capture.AmineScrubbing.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.direct_air_capture.CalciumPotassiumScrubbing.carbon_capture_direct_air_capture_CalciumPotassiumScrubbing_array_mix"
        ] = np.asarray(
            [
                2.6138572933308715,
                60.7601240990392,
                40.88734848699872,
                39.06563849466754,
                44.12149384200339,
                4.54625257468244,
                2.711205904554535,
                11.85525930571041,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.direct_air_capture.CalciumPotassiumScrubbing.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.flue_gas_capture.CO2Membranes.carbon_capture_flue_gas_capture_CO2Membranes_array_mix"
        ] = np.asarray(
            [
                48.62632709599286,
                81.55174592762631,
                62.13318185698337,
                57.38342772204428,
                50.27206775420003,
                53.53848909630916,
                49.05598695525594,
                27.725959195015744,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.flue_gas_capture.CO2Membranes.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.flue_gas_capture.CalciumLooping.carbon_capture_flue_gas_capture_CalciumLooping_array_mix"
        ] = np.asarray(
            [
                184.6049198736637,
                94.85147549996249,
                60.972964797106165,
                55.93305629094463,
                55.04421925492856,
                53.00857913193146,
                49.64206989003302,
                32.1182943449808,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.flue_gas_capture.CalciumLooping.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.flue_gas_capture.ChilledAmmoniaProcess.carbon_capture_flue_gas_capture_ChilledAmmoniaProcess_array_mix"
        ] = np.asarray(
            [
                132.17884341025885,
                87.22935959970202,
                60.729458680349566,
                55.63785910567908,
                20.2795297556883,
                13.497542291566978,
                10.205481627739669,
                17.53164240688241,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.flue_gas_capture.ChilledAmmoniaProcess.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.flue_gas_capture.MonoEthanolAmine.carbon_capture_flue_gas_capture_MonoEthanolAmine_array_mix"
        ] = np.asarray(
            [
                42.64765450629227,
                80.46557042136918,
                61.89301329122928,
                57.05676613325659,
                16.69199499726616,
                14.363186940552124,
                11.161213843859723,
                18.777182682729723,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.flue_gas_capture.MonoEthanolAmine.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.flue_gas_capture.PiperazineProcess.carbon_capture_flue_gas_capture_PiperazineProcess_array_mix"
        ] = np.asarray(
            [
                155.506820929452,
                91.84902224352525,
                62.581423412348485,
                57.726981461628284,
                22.945841222582768,
                15.843834931617524,
                12.325904508424836,
                19.275813495388817,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.flue_gas_capture.PiperazineProcess.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.flue_gas_capture.PressureSwingAdsorption.carbon_capture_flue_gas_capture_PressureSwingAdsorption_array_mix"
        ] = np.asarray(
            [
                116.47053916052916,
                85.84899780928193,
                62.318647126535126,
                57.574495632638026,
                56.45769698137545,
                54.03266763959089,
                50.67658136379352,
                29.955958414152686,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.flue_gas_capture.PressureSwingAdsorption.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.flue_gas_capture.flue_gas_mean"
        ] = pd.read_csv(join(self.data_dir, "flue_gas_mean.csv"))
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.flue_gas_capture.technologies_list"
        ] = [
            "electricity.CoalGen",
            "electricity.GasTurbine",
            "electricity.CombinedCycleGasTurbine",
            "hydrogen.gaseous_hydrogen.WaterGasShift",
            "methane.FossilGas",
            "solid_fuel.Pelletizing",
            "syngas.CoalGasification",
            "carbon_capture.direct_air_capture.AmineScrubbing",
            "carbon_capture.direct_air_capture.CalciumPotassiumScrubbing",
        ]
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.technologies_list"
        ] = [
            "direct_air_capture.AmineScrubbing",
            "direct_air_capture.CalciumPotassiumScrubbing",
            "flue_gas_capture.CalciumLooping",
            "flue_gas_capture.ChilledAmmoniaProcess",
            "flue_gas_capture.CO2Membranes",
            "flue_gas_capture.MonoEthanolAmine",
            "flue_gas_capture.PiperazineProcess",
            "flue_gas_capture.PressureSwingAdsorption",
        ]
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.transport_cost"
        ] = pd.read_csv(join(self.data_dir, "carbon_capture_transport_cost.csv"))
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_capture.transport_margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_storage.BiomassBuryingFossilization.carbon_storage_BiomassBuryingFossilization_array_mix"
        ] = np.asarray(
            [
                23.8515100953984,
                91.27283425883923,
                82.2881688185994,
                52.55263092234808,
                58.33845417720326,
                20.60876153878768,
                13.953088862518772,
                18.79144995071249,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_storage.BiomassBuryingFossilization.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_storage.DeepOceanInjection.carbon_storage_DeepOceanInjection_array_mix"
        ] = np.asarray(
            [
                35.329834230666584,
                58.04898646408195,
                45.7570318300752,
                44.00683532748709,
                3.6989853618303092,
                1.8414242262639298,
                1.5361638568042357,
                12.599135848510274,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_storage.DeepOceanInjection.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_storage.DeepSalineFormation.carbon_storage_DeepSalineFormation_array_mix"
        ] = np.asarray(
            [
                68.76053077798106,
                63.98531736848887,
                45.25483400804451,
                42.99328980236895,
                2.4866952105072593,
                0.6185168289056874,
                0.45214969633190977,
                11.67477854692657,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_storage.DeepSalineFormation.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_storage.DepletedOilGas.carbon_storage_DepletedOilGas_array_mix"
        ] = np.asarray(
            [
                68.76053077798106,
                63.98531736848887,
                45.25483400804451,
                42.99328980236895,
                2.4866952105072593,
                0.6185168289056874,
                0.45214969633190977,
                11.67477854692657,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_storage.DepletedOilGas.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_storage.EnhancedOilRecovery.carbon_storage_EnhancedOilRecovery_array_mix"
        ] = np.asarray(
            [
                14.909888885149893,
                0.05522192194123814,
                0.10020223690147034,
                0.1773755136898275,
                0.4682830225475374,
                1.348088136834494,
                1.15877031009116,
                10.853501896109048,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_storage.EnhancedOilRecovery.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_storage.GeologicMineralization.carbon_storage_GeologicMineralization_array_mix"
        ] = np.asarray(
            [
                27.06172509624647,
                5.891619973638714,
                0.06771212729631632,
                0.7421933156087999,
                1.2897074188798963,
                0.2516693757611913,
                0.08508539288618691,
                11.221576833622388,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_storage.GeologicMineralization.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_storage.PureCarbonSolidStorage.carbon_storage_PureCarbonSolidStorage_array_mix"
        ] = np.asarray(
            [
                0.447284745497298,
                1e-06,
                0.002683264556133653,
                1.4980920613626854,
                2.560308445167628,
                1.7229507103140953,
                1.2000400734672996,
                11.994115412185069,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_storage.PureCarbonSolidStorage.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_storage.technologies_list"
        ] = [
            "BiomassBuryingFossilization",
            "DeepOceanInjection",
            "DeepSalineFormation",
            "DepletedOilGas",
            "EnhancedOilRecovery",
            "GeologicMineralization",
            "PureCarbonSolidStorage",
        ]
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_storage.transport_cost"
        ] = transport_cost
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CCUS.carbon_storage.transport_margin"
        ] = margin
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.CO2_taxes"] = pd.read_csv(
            join(self.data_dir, "CO2_taxes.csv")
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.Damage.damage_constraint_factor"
        ] = np.asarray(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
        )
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.Damage.tipping_point"] = True
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.all_streams_demand_ratio"
        ] = pd.read_csv(join(self.data_dir, "all_streams_demand_ratio.csv"))
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.biogas.AnaerobicDigestion.biogas_AnaerobicDigestion_array_mix"
        ] = np.asarray(
            [
                1e-06,
                1e-06,
                4.635552701142764,
                9.82228195910671,
                43.34705806149213,
                0.05347317058985979,
                46.30961180618926,
                0.02363947106301918,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.biogas.AnaerobicDigestion.invest_level"
        ] = pd.read_csv(join(self.data_dir, "invest_level.csv"))
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.biogas.AnaerobicDigestion.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.biogas.technologies_list"
        ] = ["AnaerobicDigestion"]
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.biogas.transport_cost"
        ] = biogas_transport_cost
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.biogas.transport_margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.biomass_dry.CropEnergy.biomass_dry_CropEnergy_array_mix"
        ] = np.asarray(
            [
                1e-06,
                1e-06,
                1e-06,
                0.00010570883497285512,
                1.295564047707274,
                7.667025030322828,
                10.91866266488285,
                35.682652894266354,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.biomass_dry.CropEnergy.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.biomass_dry.ManagedWood.biomass_dry_ManagedWood_array_mix"
        ] = np.asarray(
            [
                45.63649359465881,
                0.9551437584905283,
                9.608975835653018,
                69.84248375883024,
                10.953789269359207,
                14.42608491973701,
                23.355122585723137,
                44.55612797898519,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.biomass_dry.ManagedWood.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.biomass_dry.UnmanagedWood.biomass_dry_UnmanagedWood_array_mix"
        ] = np.asarray(
            [
                0.2452944435620282,
                81.25797161777288,
                12.80702772252956,
                11.585131366256569,
                3.2367117168862807,
                1.991702907633112,
                5.1119856380224995,
                37.62657180391165,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.biomass_dry.UnmanagedWood.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.biomass_dry.technologies_list"
        ] = ["ManagedWood", "UnmanagedWood", "CropEnergy"]
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.biomass_dry.transport_cost"
        ] = biomass_dry_transport_cost
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.biomass_dry.transport_margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.BiogasFired.electricity_BiogasFired_array_mix"
        ] = np.asarray(
            [
                83.92075831481002,
                59.035915340448334,
                38.67226411225961,
                37.5210902486365,
                45.64904242274436,
                58.01532870558022,
                55.18748118377263,
                22.634286619668618,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.BiogasFired.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.BiomassFired.electricity_BiomassFired_array_mix"
        ] = np.asarray(
            [
                70.01822726577939,
                54.529299007083026,
                39.31593056894398,
                38.44463967035194,
                2.423066446634825,
                3.4105170826434086,
                2.655223479290449,
                12.214675555528121,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.BiomassFired.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.CoalGen.electricity_CoalGen_array_mix"
        ] = np.asarray(
            [
                67.58092317650102,
                62.02171585032439,
                43.56059009655475,
                52.07106396860074,
                61.11313336860881,
                55.32176964080033,
                47.97852431341532,
                51.64178868258009,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.CoalGen.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.CombinedCycleGasTurbine.electricity_CombinedCycleGasTurbine_array_mix"
        ] = np.asarray(
            [
                84.33900166359467,
                58.44370711365012,
                40.75368977503415,
                38.02311076465773,
                44.15817189611249,
                47.18232945978747,
                5.86471731816006,
                13.07644398046312,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.CombinedCycleGasTurbine.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.GasTurbine.electricity_GasTurbine_array_mix"
        ] = np.asarray(
            [
                89.8531363762152,
                70.11560001055705,
                44.28143753155949,
                35.87111056952168,
                41.434408155121524,
                45.65477544820899,
                5.623004604945519,
                12.83151789034287,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.GasTurbine.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.Geothermal.electricity_Geothermal_array_mix"
        ] = np.asarray(
            [
                58.761381037308475,
                236.8044859337879,
                181.52784185721382,
                112.864307588336,
                137.69700324183077,
                120.18416025155275,
                77.53577143926924,
                54.55251637130297,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.Geothermal.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.Hydropower.electricity_Hydropower_array_mix"
        ] = np.asarray(
            [
                55.44495354831313,
                57.12919785118268,
                7.258777014459715,
                17.743822319216935,
                63.746886285076194,
                57.654879785467244,
                51.25672183820443,
                21.165482126820052,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.Hydropower.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.Nuclear.electricity_Nuclear_array_mix"
        ] = np.asarray(
            [
                83.90238992379614,
                54.571434831564964,
                44.22263951493872,
                7.890275131207737,
                48.917209079823216,
                45.96492833854881,
                22.96339702549017,
                12.648814620883881,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.Nuclear.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.OilGen.electricity_OilGen_array_mix"
        ] = np.asarray(
            [
                61.87563829105607,
                56.56992024008964,
                40.12958825716619,
                46.30489572930916,
                20.251351687729937,
                9.368439187205775,
                5.046421064727701,
                11.898027573758103,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.OilGen.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.SolarPv.electricity_SolarPv_array_mix"
        ] = np.asarray(
            [
                117.40562066822977,
                224.8726695284786,
                127.3838770317134,
                103.44269376801803,
                91.01248683172949,
                117.52494806086243,
                96.84761447658553,
                62.70010705374149,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.SolarPv.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.SolarThermal.electricity_SolarThermal_array_mix"
        ] = np.asarray(
            [
                73.92517752000379,
                49.9452978563094,
                34.50068551527759,
                0.9773100347585842,
                2.8555444629541236,
                37.84602508363191,
                14.281442506195365,
                13.184480303054258,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.SolarThermal.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.WindOffshore.electricity_WindOffshore_array_mix"
        ] = np.asarray(
            [
                89.22692294676204,
                236.29082777470543,
                181.0720230750835,
                111.7092122288184,
                99.43848877765016,
                92.05914213463203,
                62.7001065897335,
                54.952033388438856,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.WindOffshore.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.WindOffshore.transport_cost"
        ] = pd.read_csv(join(self.data_dir, "WindOffshore_transport_cost.csv"))
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.WindOnshore.electricity_WindOnshore_array_mix"
        ] = np.asarray(
            [
                86.31450939944126,
                50.7112595214212,
                0.483404475472591,
                0.4722215955478319,
                49.151885787737754,
                52.34424863424213,
                40.33788971349627,
                17.003084761401126,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.WindOnshore.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.technologies_list"
        ] = [
            "WindOffshore",
            "WindOnshore",
            "SolarPv",
            "SolarThermal",
            "Hydropower",
            "Nuclear",
            "CombinedCycleGasTurbine",
            "GasTurbine",
            "BiogasFired",
            "Geothermal",
            "CoalGen",
            "OilGen",
            "BiomassFired",
        ]
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.transport_cost"
        ] = transport_cost
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.electricity.transport_margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.energy_CO2_emissions"
        ] = pd.read_csv(join(self.data_dir, "energy_CO2_emissions.csv"))
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.energy_prices"
        ] = pd.read_csv(join(self.data_dir, "energy_prices.csv"))
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fuel.biodiesel.Transesterification.fuel_biodiesel_Transesterification_array_mix"
        ] = np.asarray(
            [
                8.048066567378351,
                1e-06,
                3.23131836796386,
                7.130731181701519,
                0.08201406492012599,
                2.759453740242555,
                2.680533751094854,
                35.38363051088449,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fuel.biodiesel.Transesterification.invest_level"
        ] = pd.read_csv(join(self.data_dir, "Transesterification_invest_level.csv"))
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fuel.biodiesel.Transesterification.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fuel.biodiesel.technologies_list"
        ] = ["Transesterification"]
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fuel.biodiesel.transport_cost"
        ] = biogas_transport_cost
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fuel.biodiesel.transport_margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fuel.hydrotreated_oil_fuel.HefaDecarboxylation.fuel_hydrotreated_oil_fuel_HefaDecarboxylation_array_mix"
        ] = np.asarray(
            [
                151.274073342804,
                54.21486536526984,
                37.80189867502031,
                32.67548005686364,
                15.723670478522173,
                10.22703398503077,
                20.656925090635315,
                13.932201938578777,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fuel.hydrotreated_oil_fuel.HefaDecarboxylation.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fuel.hydrotreated_oil_fuel.HefaDeoxygenation.fuel_hydrotreated_oil_fuel_HefaDeoxygenation_array_mix"
        ] = np.asarray(
            [
                163.2431956748762,
                57.13008529579019,
                39.02522849289736,
                33.36685577761835,
                16.804737193751293,
                11.48580325273313,
                21.44051147331653,
                12.361646304614816,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fuel.hydrotreated_oil_fuel.HefaDeoxygenation.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fuel.hydrotreated_oil_fuel.technologies_list"
        ] = ["HefaDecarboxylation", "HefaDeoxygenation"]
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fuel.hydrotreated_oil_fuel.transport_cost"
        ] = biogas_transport_cost
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fuel.hydrotreated_oil_fuel.transport_margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fuel.liquid_fuel.FischerTropsch.fuel_liquid_fuel_FischerTropsch_array_mix"
        ] = np.asarray(
            [
                60.949128711411305,
                58.23489880092028,
                47.01414116987845,
                10.322900070265932,
                7.6048183773849525,
                7.427565767034099,
                5.1115391862728705,
                12.57916362654771,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fuel.liquid_fuel.FischerTropsch.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fuel.liquid_fuel.Refinery.fuel_liquid_fuel_Refinery_array_mix"
        ] = np.asarray(
            [
                232.96734442333837,
                50.688288385869164,
                33.09677889464902,
                0.31712500128971594,
                3.6974749915077942,
                2.545286917954235,
                6.4109759272236095,
                7.057007394951883,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fuel.liquid_fuel.Refinery.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fuel.liquid_fuel.technologies_list"
        ] = ["Refinery", "FischerTropsch"]
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fuel.liquid_fuel.transport_cost"
        ] = biogas_transport_cost
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fuel.liquid_fuel.transport_margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.Electrolysis.AWE.hydrogen_gaseous_hydrogen_Electrolysis_AWE_array_mix"
        ] = np.asarray(
            [
                2.259451048144528,
                1e-06,
                0.03383911861806382,
                5.821360086555711,
                3.385286312918807,
                0.0907378194474129,
                65.35908039128789,
                0.35227555598232146,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.Electrolysis.AWE.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.Electrolysis.PEM.hydrogen_gaseous_hydrogen_Electrolysis_PEM_array_mix"
        ] = np.asarray(
            [
                14.57517244381776,
                1e-06,
                0.7611106651556696,
                7.391587183383655,
                2.443100724767565,
                9.470126699651981,
                0.04878079500709937,
                0.09590066900677005,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.Electrolysis.PEM.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.Electrolysis.SOEC.hydrogen_gaseous_hydrogen_Electrolysis_SOEC_array_mix"
        ] = np.asarray(
            [
                7.478438443367908,
                0.31713415485782387,
                0.37445768836790905,
                0.4104062573951139,
                1.064405951107414,
                32.76039176256654,
                20.986701903345253,
                5.250669642087696,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.Electrolysis.SOEC.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.PlasmaCracking.hydrogen_gaseous_hydrogen_PlasmaCracking_array_mix"
        ] = np.asarray(
            [
                1.4073081178734363,
                0.032345691029281996,
                0.009785666097337258,
                1.0177401255090732,
                37.03920359376068,
                33.45855380884192,
                0.09078636799173338,
                9.523478781832916,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.PlasmaCracking.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.WaterGasShift.hydrogen_gaseous_hydrogen_WaterGasShift_array_mix"
        ] = np.asarray(
            [
                2.3999901142686006,
                1e-06,
                0.9141288372963446,
                19.49763082323134,
                28.18088784140624,
                38.63138259997889,
                39.017257971557754,
                50.18214517505513,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.WaterGasShift.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.technologies_list"
        ] = [
            "WaterGasShift",
            "Electrolysis.SOEC",
            "Electrolysis.PEM",
            "Electrolysis.AWE",
            "PlasmaCracking",
        ]
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.transport_cost"
        ] = pd.read_csv(join(self.data_dir, "gaseous_hydrogen_transport_cost.csv"))
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.transport_margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.hydrogen.liquid_hydrogen.HydrogenLiquefaction.hydrogen_liquid_hydrogen_HydrogenLiquefaction_array_mix"
        ] = np.asarray(
            [
                272.60322690052317,
                101.11461124217934,
                53.88126500425642,
                42.14910983665568,
                0.2860732369234116,
                0.4229841273612259,
                0.2265336125533627,
                9.53398484619864,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.hydrogen.liquid_hydrogen.HydrogenLiquefaction.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.hydrogen.liquid_hydrogen.technologies_list"
        ] = ["HydrogenLiquefaction"]
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.hydrogen.liquid_hydrogen.transport_cost"
        ] = transport_cost
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.hydrogen.liquid_hydrogen.transport_margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.land_demand_df"
        ] = pd.read_csv(join(self.data_dir, "land_demand_df.csv"))
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.methane.FossilGas.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.methane.FossilGas.methane_FossilGas_array_mix"
        ] = np.asarray(
            [
                1.475471831553906,
                0.026606197809582805,
                66.2663990049736,
                31.755704012824054,
                7.54857699465874,
                32.988140153437925,
                17.635601373397755,
                47.921407770174234,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.methane.UpgradingBiogas.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.methane.UpgradingBiogas.methane_UpgradingBiogas_array_mix"
        ] = np.asarray(
            [
                79.98019126800178,
                52.73303631610524,
                1.7814603729883707,
                37.59275825377029,
                32.73521132478098,
                24.266900198636677,
                31.073036172118165,
                50.04276928537637,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.methane.technologies_list"
        ] = ["FossilGas", "UpgradingBiogas"]
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.methane.transport_cost"
        ] = biogas_transport_cost
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.methane.transport_margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.resources_CO2_emissions"
        ] = pd.read_csv(join(self.data_dir, "resources_CO2_emissions.csv"))
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.resources_price"
        ] = pd.read_csv(join(self.data_dir, "resources_price.csv"))
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.solid_fuel.CoalExtraction.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.solid_fuel.CoalExtraction.solid_fuel_CoalExtraction_array_mix"
        ] = np.asarray(
            [
                1e-06,
                1e-06,
                1e-06,
                0.0010214722498108988,
                0.7421210634185418,
                0.32437591237471364,
                6.742130230595968,
                1.369385033285908,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.solid_fuel.Pelletizing.margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.solid_fuel.Pelletizing.solid_fuel_Pelletizing_array_mix"
        ] = np.asarray(
            [
                18.285855655307103,
                22.07868092220524,
                18.951802015972376,
                26.955264780702457,
                1e-06,
                2.16080352445859,
                1e-06,
                0.08860067051595885,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.solid_fuel.technologies_list"
        ] = ["CoalExtraction", "Pelletizing"]
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.solid_fuel.transport_cost"
        ] = biomass_dry_transport_cost
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.solid_fuel.transport_margin"
        ] = margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.syngas.AutothermalReforming.margin"
        ] = AutothermalReforming_margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.syngas.AutothermalReforming.syngas_AutothermalReforming_array_mix"
        ] = np.asarray(
            [
                37.07365060159862,
                32.771220885813214,
                6.021194506010394,
                16.40469709295582,
                26.746625971400867,
                0.1490423783854092,
                1e-06,
                0.25776220487755874,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.syngas.BiomassGasification.margin"
        ] = AutothermalReforming_margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.syngas.BiomassGasification.syngas_BiomassGasification_array_mix"
        ] = np.asarray(
            [
                62.32621119968406,
                54.52265769671962,
                39.324732169394004,
                38.637959339466725,
                35.61285384842871,
                30.641691518810113,
                32.855198347734444,
                48.57072391207981,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.syngas.CoElectrolysis.margin"
        ] = AutothermalReforming_margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.syngas.CoElectrolysis.syngas_CoElectrolysis_array_mix"
        ] = np.asarray(
            [
                55.35997148833697,
                54.19534489633926,
                39.20958157193773,
                0.33551478876853585,
                6.752633880571002,
                7.593206217421089,
                1e-06,
                10.734888467397303,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.syngas.CoalGasification.margin"
        ] = AutothermalReforming_margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.syngas.CoalGasification.syngas_CoalGasification_array_mix"
        ] = np.asarray(
            [
                85.89730910487789,
                83.45545007216619,
                52.95434380804781,
                40.25074801088247,
                40.98433252919498,
                41.646171367323674,
                40.278040302017175,
                50.326007814257686,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.syngas.Pyrolysis.margin"
        ] = AutothermalReforming_margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.syngas.Pyrolysis.syngas_Pyrolysis_array_mix"
        ] = np.asarray(
            [
                1e-06,
                1e-06,
                9.492159653788201,
                11.192570958220637,
                0.42714077395534034,
                1e-06,
                54.349623780111045,
                2.686528948503043,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.syngas.SMR.margin"
        ] = AutothermalReforming_margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.syngas.SMR.syngas_SMR_array_mix"
        ] = np.asarray(
            [
                107.03447118901796,
                46.636584689497774,
                0.5647712498034817,
                38.02177618617164,
                34.26574915692154,
                27.434048508244768,
                0.0460654429809435,
                7.982128383913868,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.syngas.syngas_ratio"
        ] = np.asarray(
            [
                121.98083347174325,
                120.97625569268818,
                99.85797984659354,
                86.41350515804106,
                98.278532292313,
                114.15309802066426,
                126.91349386105847,
                136.41646574790892,
                145.58816523984598,
                154.23760909582052,
                162.92965234804203,
                165.25582002310637,
                168.76569365077467,
                171.73973000029196,
                173.45970636517967,
                175.2671551343192,
                176.3965619590901,
                176.4730248596626,
                175.1422564213159,
                172.61474691037193,
                166.9438696095637,
                160.99158676985934,
                154.57504701610654,
                147.40481498708348,
                139.50981818029928,
                136.15464442601368,
                133.88860798710135,
                126.13802560745499,
                119.11303027342034,
                118.57352275831936,
                115.83195366979209,
                115.86069781492412,
                115.61646945765352,
                114.9146485337076,
                112.10113087800681,
                112.01032629637739,
                109.54547120454843,
                106.64789318411535,
                106.56915284053113,
                106.4886207969072,
                106.41499691689928,
                106.46463457829658,
                106.54637920829401,
                106.66202557341117,
                106.8143491592615,
                107.0040919221097,
                106.99085635855897,
                107.14734198263835,
                107.30560860817495,
                107.50415523645256,
                107.6666625005076,
                107.82668899598032,
                107.97235433405221,
                108.09361765837527,
                108.1804786348236,
                108.22352232848984,
                108.21512350835154,
                108.15065113334235,
                108.02993198736563,
                107.86046259817205,
                107.65765524138014,
                107.44648608116552,
                107.2596684804138,
                107.13403248726149,
                107.10465633435255,
                107.19874252821315,
                107.43058187696118,
                107.79908260846487,
                108.28654972305004,
                108.86009651155473,
                109.48163322791578,
                110.12423172759294,
                110.77093926058096,
                111.38408582581837,
                111.95098120416489,
                112.46423217574842,
                112.91955617219334,
                113.31402119329412,
                113.64497844840386,
                113.90901665593927,
                114.10003879211735,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.syngas.technologies_list"
        ] = [
            "BiomassGasification",
            "SMR",
            "CoalGasification",
            "Pyrolysis",
            "AutothermalReforming",
            "CoElectrolysis",
        ]
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.syngas.transport_cost"
        ] = transport_cost
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.syngas.transport_margin"
        ] = AutothermalReforming_margin
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.Energy_demand.transport_demand"
        ] = pd.read_csv(join(self.data_dir, "transport_demand.csv"))
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.Land.Agriculture.diet_df"
        ] = pd.DataFrame(
            {
                "red meat": [11.02],
                "white meat": [31.11],
                "milk": [79.27],
                "eggs": [9.68],
                "rice and maize": [97.76],
                "potatoes": [32.93],
                "fruits and vegetables": [217.62],
            }
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.Land.Agriculture.other_use_agriculture"
        ] = np.asarray(
            [
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
                0.102,
            ]
        )
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.Land.Forest.CO2_per_ha"] = 4000
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.Land.Forest.initial_emissions"
        ] = 3.21
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.Land.Forest.limit_deforestation_surface"
        ] = 1000
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.Land.Forest.reforestation_cost_per_ha"
        ] = 3800
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.Macroeconomics.CO2_tax_efficiency"
        ] = pd.read_csv(join(self.data_dir, "CO2_tax_efficiency.csv"))
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.Macroeconomics.damage_to_productivity"
        ] = True
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.NormalizationReferences.liquid_hydrogen_percentage"
        ] = np.asarray(
            [
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.NormalizationReferences.total_emissions_ref"
        ] = 39.6
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.Resources.non_modeled_resource_price"
        ] = pd.read_csv(join(self.data_dir, "non_modeled_resource_price.csv"))
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.beta"] = 1.0
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.ccs_list"] = [
            "carbon_capture",
            "carbon_storage",
        ]
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.conso_elasticity"] = 1.45
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.deforested_surface_ctrl"
        ] = np.asarray(
            [
                12.344876509401397,
                12.546566162142291,
                12.60722800913804,
                10.807123906837166,
                8.41103751287496,
                6.369705384851529,
                5.684112464713017,
                4.983330561724365,
            ]
        )
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.economics_df"] = pd.read_csv(
            join(self.data_dir, "economics_df.csv")
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.energy_investment"
        ] = pd.read_csv(join(self.data_dir, "energy_investment.csv"))
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.energy_list"] = [
            "methane",
            "hydrogen.gaseous_hydrogen",
            "biogas",
            "syngas",
            "fuel.liquid_fuel",
            "fuel.hydrotreated_oil_fuel",
            "solid_fuel",
            "biomass_dry",
            "electricity",
            "fuel.biodiesel",
            "hydrogen.liquid_hydrogen",
        ]
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.forest_investment_array_mix"
        ] = np.asarray(
            [
                0.05579290708677141,
                0.003821118916880622,
                1e-06,
                0.3028342934230259,
                2.083726138785509,
                21.41020737523814,
                14.79863862738392,
                23.061300521814786,
            ]
        )
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.frac_damage_prod"] = 0.3
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.gamma"] = 0.5
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.init_gross_output"] = 130.187
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.init_rate_time_pref"] = 0.0
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.is_stream_demand"] = True
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.land_surface_for_food_df"
        ] = pd.read_csv(join(self.data_dir, "land_surface_for_food_df.csv"))
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.population_df"] = pd.read_csv(
            join(self.data_dir, "population_df.csv")
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.red_meat_percentage_ctrl"
        ] = np.asarray(
            [
                6.744039813039008,
                6.741100445236898,
                6.466582371777589,
                6.042643432732087,
                6.622113886069014,
                6.8481270857768575,
                6.822699665037604,
                6.804052317459368,
            ]
        )
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.scaling_factor_energy_investment"
        ] = 100.0
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.share_energy_investment_ctrl"
        ] = np.asarray(
            [
                1.607340131651491,
                1.6172109529577732,
                1.6487205160779008,
                1.6492396911129608,
                1.6482845877663865,
                1.6494600166286537,
                1.6499023235888042,
                1.6350555973964591,
            ]
        )
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.temperature_df"] = pd.read_csv(
            join(self.data_dir, "temperature_df.csv")
        )
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.time_step"] = 1
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.total_food_land_surface"
        ] = pd.read_csv(join(self.data_dir, "total_food_land_surface.csv"))
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.total_investment_share_of_gdp"
        ] = pd.read_csv(join(self.data_dir, "total_investment_share_of_gdp.csv"))
        inputs[
            f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.white_meat_percentage_ctrl"
        ] = np.asarray(
            [
                13.833217685495578,
                13.832852019581965,
                13.798701161115865,
                13.745961905874807,
                13.818049740343177,
                13.95349909247264,
                13.950335846337737,
                13.948016061145209,
            ]
        )
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.year_end"] = 2100
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.WITNESS.year_start"] = 2020
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.epsilon0"] = 1.0
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.linear_solver_MDA_options"] = {
            "tol": 1e-10,
            "max_iter": 50000,
        }
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.linear_solver_MDO_options"] = {
            "tol": 1e-10,
            "max_iter": 10000,
        }
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.max_mda_iter"] = 50
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.sub_mda_class"] = "GSPureNewtonMDA"
        inputs[f"{sty}.WITNESS_MDO.WITNESS_Eval.tolerance"] = 1e-10
        inputs[f"{sty}.WITNESS_MDO.algo"] = "L-BFGS-B"
        inputs[f"{sty}.WITNESS_MDO.algo_options"] = {
            "disp": 30,
            "normalize_design_space": True,
            "maxcor": 464,
            "max_ls_step_nb": 1392,
            "xtol_rel": 1e-16,
            "xtol_abs": 1e-16,
            "ftol_rel": 3e-16,
            "ftol_abs": 3e-16,
            "max_time": 0,
            "pg_tol": 1e-16,
            "max_iter": 700,
        }
        inputs[f"{sty}.WITNESS_MDO.design_space"] = pd.read_csv(
            join(self.data_dir, "design_space.csv"),
            converters={
                "value": eval,
                "lower_bnd": eval,
                "upper_bnd": eval,
                "activated_elem": eval,
            },
        )
        inputs[f"{sty}.WITNESS_MDO.differentiation_method"] = "user"
        inputs[f"{sty}.WITNESS_MDO.eq_constraints"] = eq_constraints
        inputs[f"{sty}.WITNESS_MDO.formulation"] = "DisciplinaryOpt"
        inputs[f"{sty}.WITNESS_MDO.ineq_constraints"] = eq_constraints
        inputs[f"{sty}.WITNESS_MDO.max_iter"] = 800
        inputs[f"{sty}.WITNESS_MDO.objective_name"] = "objective_lagrangian"
        inputs[f"{sty}.WITNESS_MDO.parallel_options"] = {
            "parallel": False,
            "n_processes": 32,
            "use_threading": False,
            "wait_time_between_fork": 0,
        }
        inputs[f"{sty}.epsilon0"] = 1

        return [inputs]


if "__main__" == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run(logger_level="DEBUG", for_test=False)
