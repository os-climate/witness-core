'''
Copyright 2022 Airbus SAS

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

import os
import numpy as np
from pathlib import Path
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from sos_trades_core.tools.grad_solvers.validgrad.FDGradient import FDGradient
from sos_trades_core.execution_engine.func_manager.func_manager import FunctionManager

#### FUNCTIONS ####
def compute_all(x, capital_df, energy_df, population_df, pib_base_df):
    mod_func, delta, productivity_df, pib_df, delta_var, delta_pib  = eval_all_v2(x, capital_df, energy_df, population_df, pib_base_df)
    return mod_func

def FD_compute_all(x, capital_df, energy_df, population_df, pib_base_df):
    grad_eval = FDGradient(1j, compute_all, fd_step=1.e-12)
    grad_eval.set_multi_proc(False)
    outputs_grad= grad_eval.grad_f(x, args=(capital_df, energy_df, population_df, pib_base_df))
    return outputs_grad
    
def compute_gross_output_ter(year, population_df, capital_df, energy_df, productivity_df, capital_share, elast_KL_E, energy_share, energy_intensity):
        energy_y = energy_df.loc[year, 'Energy'] #in TWh
        population_y = population_df.loc[year, 'population'] #In millions of people
        capital_y = capital_df.loc[year, 'capital (trill 2011)'] #In trill$
        productivity_y = productivity_df.loc[year, 'productivity']
    # Cobb-Douglas part linking Capital and Labour
        cobb_douglas = ((productivity_y * (capital_y**capital_share) * ((population_y / 1000)
                                                                         ** (1 - capital_share)))) ** ((elast_KL_E - 1) / elast_KL_E)
        energy_part = (((energy_intensity * energy_y)) **((elast_KL_E - 1) / elast_KL_E))
        # 2-level nested CES function, links the Cobb-Douglas and the Energy
        gross_output_ter = ((1 - energy_share) * cobb_douglas + energy_share *
                            energy_part) ** (elast_KL_E / (elast_KL_E - 1))
        return gross_output_ter

def compute_gross_output_ter_relative(year, population_df, capital_df, energy_df, productivity_df, capital_share, elast_KL_E, energy_share, energy_intensity, year_range):
        energy_y = energy_df.loc[year, 'Energy']/energy_df.loc[year_range[0], 'Energy'] #in TWh
        population_y = population_df.loc[year, 'population']/population_df.loc[year_range[0], 'population'] #In millions of people
        capital_y = capital_df.loc[year, 'capital (trill 2011)']/capital_df.loc[year_range[0], 'capital (trill 2011)'] #In trill$
        productivity_y = productivity_df.loc[year, 'productivity']/ productivity_df.loc[year_range[0], 'productivity']
    # Cobb-Douglas part linking Capital and Labour
        cobb_douglas = ((productivity_y * (capital_y**capital_share) * ((population_y / 1000)
                                                                         ** (1 - capital_share)))) ** ((elast_KL_E - 1) / elast_KL_E)
        energy_part = (((energy_intensity * energy_y)) **((elast_KL_E - 1) / elast_KL_E))
        # 2-level nested CES function, links the Cobb-Douglas and the Energy
        gross_output_ter = ((1 - energy_share) * cobb_douglas + energy_share *
                            energy_part) ** (elast_KL_E / (elast_KL_E - 1))
        gross_output_ter_rel = gross_output_ter * 1.36944 #Value at t=0 of pib 
        return gross_output_ter_rel
    
def compute_productivity_growth_rate(year, decline_rate_tfp, productivity_gr_start, year_start):
        t = ((year - year_start) / 1) + 1
        productivity_gr = productivity_gr_start * np.exp(-decline_rate_tfp * 1 * (t - 1))
        return productivity_gr
    
def comp_delta_pib(pib_base, pib_df):
    """ Compute (y_ref - y_comp)^2 and returns a series
    Inputs: 2 dataframes"""
    pib_base_df = pib_base.loc[(pib_base['years']>= 1965) & (pib_base['years']<= 2017)]
    delta = (pib_base_df['pib']/1e12 - pib_df['output'])/(pib_base_df['pib']/1e12)
    #add harder contraints on some points
    delta.loc[pib_base['years']== 1965] = delta.loc[pib_base['years']== 1965]*3.
#     delta.loc[pib_base['years']== 1995] = delta.loc[pib_base['years']== 1995]*3.
    delta.loc[pib_base['years']== 2008] = delta.loc[pib_base['years']== 2008]*3.
    delta.loc[pib_base['years']== 2009] = delta.loc[pib_base['years']== 2009]*3.
    delta.loc[pib_base['years']== 2015] = delta.loc[pib_base['years']== 2015]*3.
    absdelta = np.sign(delta)*delta
    return absdelta

def comp_delta_var(pib_base, pib_df, year_range, var_df):
    """Compute the difference in variation"""
    for year in year_range[1:]: 
        pib_base_y = pib_base.loc[year, 'pib']/1e12
        p_pib_base = pib_base.loc[year-1, 'pib']/1e12
        pib_y = pib_df.loc[year, 'output']
        p_pib = pib_df.loc[year -1, 'output']
        var_pib_base = pib_base_y - p_pib_base
        var_pib_y = pib_y - p_pib
        delta_var = var_pib_base - var_pib_y 
        var_df.loc[year] = delta_var**2
    return var_df
    
def compute_productivity( year, productivity_df):
    p_productivity = productivity_df.loc[year-1, 'productivity'] 
    p_productivity_gr = productivity_df.loc[year-1, 'productivity_gr'] 
    productivity = (p_productivity / (1 - (p_productivity_gr / (5 / 1))))
    return productivity

def eval_all_v2(x, capital_df, energy_df, population_df, pib_base_df):
    alpha = x[0]
    beta = x[1]
    gamma = x[2]
    small_a = x[3]
    b = x[4]
    gr_rate_ener = x[5]
    productivity_start = x[6]
    productivity_gr_start = x[7]
    energy_factor = x[8]
    decline_rate_tfp = x[9]
    #Initialise everything
    pib_base = pib_base_df
    year_range = np.arange(1965, 2018) 
    data = np.zeros(len(year_range))
    productivity_df = pd.DataFrame({'year': year_range, 'productivity': data, 'productivity_gr': data}, index = year_range) 
    productivity_df.loc[year_range[0], 'productivity_gr'] = productivity_gr_start
    productivity_df.loc[year_range[0], 'productivity'] = productivity_start
    pib_df = pd.DataFrame({'year': year_range, 'output': data}, index = year_range)
    delta_var = pd.Series(data, index = year_range)
    energy_intens = pd.Series(data, index= year_range)
    #COmpute productivity
    for year in year_range[1:]:
        productivity_df.loc[year, 'productivity_gr'] = compute_productivity_growth_rate(year, decline_rate_tfp, productivity_gr_start, year_range[0])
        productivity_df.loc[year, 'productivity'] = compute_productivity(year, productivity_df)
    for year in year_range:
        energy_intens[year] = energy_factor*(1+gr_rate_ener)**(year-year_range[0])
    #COmpute pib 
    for year in year_range:
        pib_df.loc[year, 'output'] = compute_gross_output_paper_relative(year, population_df, capital_df, energy_df, productivity_df, 
                                                                energy_intens, alpha, beta, gamma, small_a, b, year_range)
    #Compute( y_ref - y_computed)^2 
    delta_pib = comp_delta_pib(pib_base, pib_df)
    delta_var = comp_delta_var(pib_base, pib_df, year_range, delta_var)
    delta_sum = delta_var #+ delta_pib
    func_manager = FunctionManager()
    #-- reference square due to delta**2
    func_manager.add_function('cst_delta_pib', np.array(delta_pib), FunctionManager.INEQ_CONSTRAINT, weight=1.)
    #func_manager.add_function('cst_delta_var', np.array(delta_var), FunctionManager.INEQ_CONSTRAINT, weight=100./5.**2)
    func_manager.build_aggregated_functions(eps=1e-3)
    mod_func = func_manager.mod_obj
    return mod_func, delta_sum, productivity_df, pib_df, delta_var, delta_pib 

def eval_all_v1(x, capital_df, energy_df, population_df, pib_base_df):
    #Initialise everything
    productivity_start = x[0]
    productivity_gr_start = x[1]
    energy_share = x[2]
    capital_share = x[3]
    elast_KL_E = x[4]
    decline_rate_tfp = x[5]
    energy_intensity = x[6]
    #capital_df = my_args['capital_df']
    #pib_base = my_args['pib_base_df']
    #population_df = my_args['population_df']
    #energ_df = my_args['energy_df']
    pib_base = pib_base_df
    year_range = np.arange(1965, 2018) 
    data = np.zeros(len(year_range))
    productivity_df = pd.DataFrame({'year': year_range, 'productivity': data, 'productivity_gr': data}, index = year_range) 
    productivity_df.loc[year_range[0], 'productivity_gr'] = productivity_gr_start
    productivity_df.loc[year_range[0], 'productivity'] = productivity_start
    pib_df = pd.DataFrame({'year': year_range, 'output': data}, index = year_range)
    #COmpute productivity
    for year in year_range[1:]:
        productivity_df.loc[year, 'productivity_gr'] = compute_productivity_growth_rate(year, decline_rate_tfp, productivity_gr_start, year_range[0])
        productivity_df.loc[year, 'productivity'] = compute_productivity(year, productivity_df)
    #COmpute pib 
    for year in year_range:
        pib_df.loc[year, 'output'] = compute_gross_output_ter_relative(year, population_df, capital_df, energy_df, productivity_df, capital_share, elast_KL_E, energy_share, energy_intensity, year_range)
    #Compute( y_ref - y_computed)^2 
    delta = delta_pib(pib_base, pib_df)
    func_manager = FunctionManager()
    func_manager.add_function('cst_delta', delta, FunctionManager.INEQ_CONSTRAINT, weight=1./75**2)
    func_manager.build_aggregated_functions(eps=1e-3)
    mod_func = func_manager.mod_obj
    return mod_func, delta, productivity_df, pib_df

def compute_gross_output_paper(year, population_df, capital_df, energy_df, productivity_df, energy_intens , alpha, beta, gamma, small_a, b, big_a):
        energy_y = energy_df.loc[year, 'Energy'] #in TWh
        population_y = population_df.loc[year, 'population'] #In millions of people
        capital_y = capital_df.loc[year, 'capital (trill 2011)'] #In trill$
        productivity_y = productivity_df.loc[year, 'productivity']
        energy_intens_y = energy_intens[year]
        #cobb_douglas = (productivity_y * (capital_y**capital_share) * ((population_y)** (1 - capital_share)))
        cobb_douglas = productivity_y*(small_a * (capital_y) **(-alpha) + (1- small_a) * (population_y)**(- alpha))**(-(1/alpha))
        output = big_a*(b*cobb_douglas**(- beta) + (1-b)* (energy_intens_y* (energy_y/population_y)) **(- beta))**(- gamma/ beta)
        return output

def compute_gross_output_paper_relative(year, population_df, capital_df, energy_df, productivity_df, energy_intens, alpha, beta, gamma, small_a, b, year_range):
        energy_y = energy_df.loc[year, 'Energy']/energy_df.loc[year_range[0], 'Energy']  #in TWh
        population_y = population_df.loc[year, 'population']/ population_df.loc[year_range[0], 'population']  #In millions of people
        capital_y = capital_df.loc[year, 'capital (trill 2011)']/ capital_df.loc[year_range[0], 'capital (trill 2011)'] #In trill$
        productivity_y = productivity_df.loc[year, 'productivity']/ productivity_df.loc[year_range[0], 'productivity']
        energy_intens_y = energy_intens[year]/energy_intens[year_range[0]]
        #cobb_douglas = (productivity_y * (capital_y**capital_share) * ((population_y)** (1 - capital_share)))
        cobb_douglas = productivity_y*(small_a * (capital_y) **(-alpha) + (1- small_a) * (population_y)**(- alpha))**(-(1/alpha))
        output = (b*cobb_douglas**(- beta) + (1-b)* (energy_intens_y* (energy_y/population_y)) **(- beta))**(- gamma/ beta)
        output_rel = output*1.36944 #Value at t=0 of pib 
        return output_rel

#parameters value 
productivity_start = 0.6
productivity_gr_start= 0.076
energy_share = 0.8  #0.4
capital_share = 0.3 #0.3
elast_KL_E = 0.5 #0.5
decline_rate_tfp = 0.005
energy_intensity = 2.8/1000


base_path = os.path.dirname(os.path.realpath(__file__))
##Read inputs
energy_df = pd.read_csv(os.path.join(base_path,'data','energy_df.csv'))
energy_df = energy_df.set_index(energy_df['Year'])
pib_base_df = pd.read_csv(os.path.join(base_path,'data','pib_base.csv'))
pib_base_df = pib_base_df.set_index(pib_base_df['years'])
population_df = pd.read_csv(os.path.join(base_path,'data','population_df.csv'))
population_df = population_df.set_index(population_df['year'])
capital_df = pd.read_csv(os.path.join(base_path,'data','capital_df.csv'))
capital_df = capital_df.set_index(capital_df['year'])

my_args = (capital_df, energy_df, population_df, pib_base_df)

#Parameter values v2
alpha = 1.
beta = 1.
gamma = 1.
small_a = 0.5
b =   0.5
energy_factor  = 0.5
gr_rate_ener = 0.001
decline_rate_tfp = 0.005
x2 = [alpha, beta, gamma, small_a, b, gr_rate_ener, productivity_start,productivity_gr_start, energy_factor, decline_rate_tfp]
#x_paper = [-0.04139, -0.0568, 0.2004, 0.6270, 0.9312 , 1., 1., 1.]
# bounds = [(-1.,5.), (-1, 5.), (0.01, 5.),(0.2 ,0.9), (0.3, 0.95), (-0.05, 0.2), (0.1, 1), (-0.1, 0.5), (-5., 5.), (0.0001, 0.01)]
# x_opt, f,d = fmin_l_bfgs_b(compute_all, x2, fprime=FD_compute_all, bounds=bounds, args= my_args, maxfun=10000, approx_grad =0,
#                maxiter=1000, m=len(x2),iprint=1, pgtol=1.e-9, factr=1., maxls=2*len(x2))
# print('x_opt =',x_opt)

##EVAL 
mod_func, delta_sum, productivity_df, pib_df, delta_var, delta_pib = eval_all_v2(x2, capital_df, energy_df, population_df, pib_base_df)
print(pib_df)
# AND PLOT 
#Plot of variation data pib energy
var_pib = pd.Series([0]*len(np.arange(1966, 2018)), index= np.arange(1966, 2018))
var_energy = pd.Series([0]*len(np.arange(1966, 2018)), index = np.arange(1966, 2018))
for year in np.arange(1966, 2018):
    var_pib[year] = (pib_base_df.loc[year, 'pib']/1e12 - pib_base_df.loc[year-1, 'pib']/1e12)/(pib_base_df.loc[year-1, 'pib']/1e12)*100
    var_energy[year] =(energy_df.loc[year, 'Energy'] - energy_df.loc[year-1, 'Energy'])/energy_df.loc[year-1, 'Energy']*100
plt.plot(var_pib, label ='var pib')
plt.plot(var_energy, label = 'var energy')
plt.legend()
plt.show()    
#PIB comparison
plt.plot(pib_df['year'], pib_df['output'], label= 'eval')
plt.plot(pib_base_df['years'], pib_base_df['pib']/1e12, label ='base')
plt.xlabel('year')
plt.ylabel('pib in trill$')
plt.legend()
plt.show()
var_pib2 = pd.Series([0]*len(np.arange(1966, 2018)), index= np.arange(1966, 2018))
var_pib_est = pd.Series([0]*len(np.arange(1966, 2018)), index= np.arange(1966, 2018))
for year in np.arange(1966, 2018):
    var_pib2[year] = pib_base_df.loc[year, 'pib']/1e9 - pib_base_df.loc[year-1, 'pib']/1e9
    var_pib_est[year] = (pib_df.loc[year, 'output'] - pib_df.loc[year-1, 'output'])*1e3

plt.plot(var_pib_est, label = 'computed y-y_1')
plt.plot(var_pib2, label = 'ref y-y-1')
plt.legend()
plt.show()

#Plot energy
fig, ax_left = plt.subplots()
ax_right = ax_left.twinx()
lns1 = ax_left.plot(pib_df['year'], pib_df['output']*1e3, label = 'pib eval')
lns2 = ax_left.plot(pib_base_df['years'], pib_base_df['pib']/1e9, label ='pib base')
lns3 = ax_right.plot(energy_df['Year'], energy_df['Energy'], label = 'Energy', color = 'green')
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax_left.legend(lns, labs, loc=0)
ax_right.set_xlabel('year')
ax_left.set_ylabel("GDP in M$")
ax_right.set_ylabel('Energy in TWh')
plt.show()

