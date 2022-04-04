

from ntpath import join
from posixpath import dirname
import numpy as np
import pandas as pd


class CopperModel :

    YEAR_START = 2020
    YEAR_END = 2100
    DEMAND = 'All_Demand'
    COPPER_STOCK = 'copper_stock'
    COPPER_PRICE = 'copper_price'
    COPPER_RESERVE = 'copper_reserve'
    PRODUCTION = 'production'


    def __init__(self, copper_demand, annual_extraction):
        self.copper_demand = pd.DataFrame() #DataFrame avec la demande par année en cuivre
        self.annual_extraction = annual_extraction

        data_dir = join(dirname(dirname(__file__)), 'sos_wrapping', 'data')

        self.copper_stock =  pd.DataFrame(columns = ['Year', 'Stock']) #pd.read_csv(join(data_dir, 'copper_previous_stock.csv'))  
        self.copper_reserve = pd.DataFrame(columns = ['Year', 'Reserve']) #DataFrame avec la réserve mie à jour chaque année
        self.copper_prod_price = pd.DataFrame(columns = ['Year','Price/Mt', 'Total Price'])#DataFrame avec le prix de la prod maj chaque année.
        self.copper_prod = pd.DataFrame(columns = ['Year', 'Extraction', 'World Production', 'Cumulated World Production'])

    
    

    def compute(self, copper_demand, period_of_exploitation):
        
        self.copper_demand = copper_demand
        years = np.arange(2020, 2101, 1)

        # on rempli la colonne Years
        self.copper_demand['Year'] = years
        self.copper_prod_price['Year'] = years
        self.copper_reserve['Year'] = years
        self.copper_prod['Year'] = years
        self.copper_stock['Year'] = years

        # on met les années en index
        self.copper_demand.index = self.copper_demand['Year'].values
        self.copper_prod_price.index = self.copper_prod_price['Year'].values
        self.copper_stock.index = self.copper_stock['Year'].values
        self.copper_reserve.index = self.copper_reserve['Year'].values
        self.copper_prod.index = self.copper_prod['Year'].values

        # on initialise toutes les autres colonnes
        self.copper_prod['Extraction'] = self.annual_extraction
        self.copper_prod['World Production'] = np.linspace(0,0,len(years))
        self.copper_prod['Cumulated World Production'] = np.linspace(0,0,len(years))
        self.copper_reserve['Reserve'] = np.linspace(0,0,len(years))
        self.copper_prod_price['Total Price'] = np.linspace(0,0,len(years))
        self.copper_stock['Stock'] = np.linspace(0,0,len(years))
        self.copper_prod_price['Price/Mt'] = np.linspace(0,0,len(years))

        



        
        for year in period_of_exploitation :           

            
            #mise à jour des réserves
            self.compute_copper_reserve(year)
            #détermine le stock et la production
            self.compute_copper_stock_and_production(year)
            #calcule le prix de la prod annuelle :
            self.compute_copper_price(year)


    #### méthodes statiques

  
    def compute_copper_reserve(self, year):
        copper_extraction = self.copper_prod.loc[year, 'Extraction']

        if year == self.YEAR_START :
            remainig_copper = 3500
       

        else :
            remainig_copper = self.copper_reserve.loc[year -1, 'Reserve']
        
        
            
        #si on veut extraire plus que ce qui est disponible, on n'extrait que ce qui est dispo
        if remainig_copper < copper_extraction :
            self.copper_reserve.loc[year, 'Reserve']= 0
                                    
            self.copper_prod.loc[year, 'Extraction']= remainig_copper
            

        #Quand on a trop pioché dans les resources on diminue l'extraction   
        elif  remainig_copper < 500 :
            self.copper_prod.loc[year, 'Extraction'] = 0.95 * self.copper_prod.loc[year -1 , 'Extraction']
            self.copper_reserve.loc[year, 'Reserve'] = remainig_copper - self.copper_prod.loc[year, 'Extraction']
                                        
            
        else :
            self.copper_reserve.loc[year, 'Reserve'] = remainig_copper - copper_extraction
        

    def compute_copper_stock_and_production(self, year) :

        if year == self.YEAR_START :
            old_stock = 880
        else :
            old_stock = self.copper_stock.at[year -1, 'Stock']        
       
        
        extraction = self.copper_prod.at[year, 'Extraction']
        copper_demand = self.copper_demand.at[year, 'Demand']

        #stock de l'année précédente auquel on ajoute le minerais extrait et auquel on soustrait la demande. Si demande trop forte et dépasse stock, il n'y en a plus

        new_stock = old_stock + extraction - copper_demand
    

        if new_stock < 0 :
            self.copper_stock.at[year, 'Stock']= 0
                                    
            #s'il n'y a plus de stock, la production sera ce qui a été extrait et ce qu'il restait de stock (prod peut etre nulle)
            self.copper_prod.at[year, 'World Production'] = extraction + old_stock
        else :
            self.copper_stock.at[year, 'Stock'] = new_stock
                                    
            #s'il reste du stock, c'est que la demande a été satsfaite
            self.copper_prod.at[year, 'World Production'] = copper_demand
        
        if year == self.YEAR_START :
            self.copper_prod.at[year, 'Cumulated World Production'] = self.copper_prod.at[year, 'World Production']
        else : 
            self.copper_prod.at[year, 'Cumulated World Production'] = self.copper_prod.at[year -1, 'Cumulated World Production'] + self.copper_prod.at[year, 'World Production']
             


    
    def compute_copper_price (self, year) :

        if year ==  self.YEAR_START :
            self.copper_prod_price.loc[year, 'Price/Mt'] = 9780000

        else :     
            # losrqu'il y a un trop grand écart entre la demande et ce qui est extrait, les prix augmentent
            if self.copper_demand.loc[year, 'Demand'] - self.copper_prod.loc[year, 'Extraction'] > 5 :
                self.copper_prod_price.loc[year, 'Price/Mt'] = self.copper_prod_price.loc[year - 1, 'Price/Mt'] * 1.01
            else :
                self.copper_prod_price.loc[year, 'Price/Mt'] = self.copper_prod_price.loc[year - 1, 'Price/Mt']
        
       
        self.copper_prod_price.loc[year, 'Total Price']= self.copper_prod.loc[year, 'World Production'] *  self.copper_prod_price.loc[year, 'Price/Mt']
        
    
