# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:33:43 2022

@author: MORERE_M
"""
#Code for computing capital per sector in current $ so need to be converted in 2020 constant $ to be used 

import pandas as pd 
import os
base_path = os.path.dirname(os.path.realpath(__file__))
#import conversion rate 
conversion_data =  pd.read_excel(os.path.join(base_path, 'data', "Exchange_Rates_fordataanalysis.xlsx"))

#first agriculture capital 
data_df = pd.read_excel(os.path.join(base_path, 'data', 'cap_agri_sea.xlsx'))
#Prepare dataframe for multilplication with conversion rate                     
test_cap_agri = data_df.groupby(by = ['country']).sum()
conversion_data = conversion_data.drop(columns = ['Country'])
conversion_data.set_index('Acronym', inplace = True)
cap_agri = test_cap_agri.transpose()
conv_data = conversion_data.transpose()
new_df = pd.DataFrame()

#Multiply two dataframe not in the best way. to convert in $
for i in conv_data.keys():
    if i in cap_agri.keys():
        new_df[i] = conv_data[i]*cap_agri[i]*1e6
new_df2 = new_df.sum(axis=1)
    
cap_sea = pd.read_excel(os.path.join(base_path, 'data', "cap_sea.xlsx"))
test_cap_sea = cap_sea.groupby(by = ['country']).sum()
cap_tot = test_cap_sea.transpose()
new_df_tot = pd.DataFrame()
for i in conv_data.keys():
    if i in cap_tot.keys():
        new_df_tot[i] = conv_data[i]*cap_tot[i]*1e6
new_df2_tot = new_df_tot.sum(axis=1)

#Indus
cap_indus = pd.read_excel(os.path.join(base_path, 'data', "cap_sea_indus_with_energy.xlsx"))
test_cap_indus = cap_indus.groupby(by = ['country']).sum()
cap_tindus = test_cap_indus.transpose()
new_df_indus = pd.DataFrame()
for i in conv_data.keys():
    if i in cap_tindus.keys():
        new_df_indus[i] = conv_data[i]*cap_tindus[i]*1e6
new_df2_indus = new_df_indus.sum(axis=1)

#Services
cap_services =pd.read_excel(os.path.join(base_path, 'data',"cap_service_sea.xlsx"))
test_cap_services = cap_services.groupby(by = ['country']).sum()
cap_tservices = test_cap_services.transpose()
new_df_services = pd.DataFrame()
for i in conv_data.keys():
    if i in cap_tservices.keys():
        new_df_services[i] = conv_data[i]*cap_tservices[i]*1e6
new_df2_services = new_df_services.sum(axis=1)

#Energy
cap_energy = pd.read_excel(os.path.join(base_path, 'data', "cap_energy_sea.xlsx"))
test_cap_energy = cap_energy.groupby(by = ['country']).sum()
cap_tenergy = test_cap_energy.transpose()
new_df_energy = pd.DataFrame()
for i in conv_data.keys():
    if i in cap_tenergy.keys():
        new_df_energy[i] = conv_data[i]*cap_tenergy[i]*1e6
new_df2_energy = new_df_energy.sum(axis=1)

#%%
#detailed energy: sector D
cap_energy_d = cap_energy[cap_energy['code'] == 'D35']
cap_energy_dd = cap_energy_d.groupby(by = ['country']).sum()
cap_tenergy_d = cap_energy_dd.transpose()
new_df_energy_d = pd.DataFrame()
for i in conv_data.keys():
    if i in cap_tenergy_d.keys():
        new_df_energy_d[i] = conv_data[i]*cap_tenergy_d[i]*1e6
new_df2_energy_d = new_df_energy_d.sum(axis=1)
#%%
#detailed energy: sector B 
cap_energy_b = cap_energy[cap_energy['code'] == 'B']
cap_energy_bb = cap_energy_b.groupby(by = ['country']).sum()
cap_tenergy_b = cap_energy_bb.transpose()
new_df_energy_b = pd.DataFrame()
for i in conv_data.keys():
    if i in cap_tenergy_b.keys():
        new_df_energy_b[i] = conv_data[i]*cap_tenergy_b[i]*1e6
new_df2_energy_b = new_df_energy_b.sum(axis=1)
