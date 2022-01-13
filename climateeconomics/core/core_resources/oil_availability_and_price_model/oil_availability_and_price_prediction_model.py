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
#!/usr/bin/env python
# coding: utf-8

# # Oil availability prediction model
#
# This model aims at using existing production and reserves data to
# anticipate oil availability and especially shortage of oil supply and
# predict associated price


# importing required libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
from sklearn import set_config
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import os

warnings.filterwarnings('ignore')


class dataset:

    """
    Create a clean dataset out of bp stat review like format gathering file from `self.source` path


    Attributes:
    - df: the result dataframe used later in models
    - source: path to the data to be loaded

    Methods:
    - load_dataset
    - set_source
    - cleaning_bp_dataset

    """

    df = pd.DataFrame()
    source = os.path.join(os.path.dirname(
        __file__), 'data', 'bp_stats_review_2020_consolidated_dataset_panel_format.csv')

    def __init__(self):
        """
        load and clean default dataset: bp stat review 2020
        """
        self.load_dataset()
        self.cleaning_bp_dataset()

    def load_dataset(self):
        """
        load dataset to dataframe from the source path
        """
        self.df = pd.read_csv(self.source)

    def cleaning_bp_dataset(self):
        """
        perform dataset cleaning on self.df
        """
        # converting production and consommation to volume per year in bbl
        self.df.loc[:, 'oilcons_yearlybbl'] = self.df.oilcons_kbd.apply(
            lambda x: x * 361 / 1000000)
        self.df.loc[:, 'oilprod_yearlybbl'] = self.df.oilprod_kbd.apply(
            lambda x: x * 361 / 1000000)
        # restraining dataset only to useful feature for oil assertion
        self.df = self.df[['Country', 'Year', 'pop', 'ISO3166_alpha3', 'Region',
                           'SubRegion', 'OPEC', 'EU', 'OECD', 'CIS', 'oilprod_yearlybbl', 'oilcons_yearlybbl', 'oilreserves_bbl']]
        # remove all total counts from the data
        self.df = self.df[~self.df.Country.isin(['Total Africa',
                                                 'Total Asia Pacific', 'Total CIS', 'Total Central America',
                                                 'Total Eastern Africa', 'Total Europe', 'Total European Union',
                                                 'Total Middle Africa', 'Total Middle East', 'Total Non-OECD',
                                                 'Total North America', 'Total OECD', 'Total S. & Cent. America',
                                                 'Total Western Africa', 'Total World'])]
        # fill the missing reserves by backfill method
        self.df.oilreserves_bbl = self.df.oilreserves_bbl.fillna(
            method='bfill')
        # calculating the cumulated production of oil per country
        self.df.loc[:, 'cumulated_prod_bbl'] = self.df.groupby(
            ['Country'])['oilprod_yearlybbl'].cumsum()
        # calculating a corrected reserve by country by removing the production
        # which seems to have been eluded (ex. UAE)
        self.df['reserves_corr'] = self.df['oilreserves_bbl'] - \
            self.df['cumulated_prod_bbl']
        self.df['reserves_corr'] = self.df['reserves_corr'].apply(
            lambda x: x if x >= 0 else 0)


class reserves_model:

    """
    Create a model that makes prediction about oil reserves by country

    Attributes:

    - df:the historical data used for regression and fitting of the models
    - pred_end: the year until which to make the predictions
    - years_prediction: the array of all years for which the prediction will be made
    - lasso: a linear regression model
    - elNet: a linear regression model
    - linreg: a linear regression model
    - svr: a support vector regression model with a polynomial 2 kernel
    - rbf: an rbf regression model
    - model: the chosen model (lasso after initial tests)
    - preds: a dictionnary where keys are the model type and values are the dataframes containing predictions associated to this model
    - models: the list of model types to run for making predictions (this gives all the keys in preds)
    - end_of_reserves: likely to be deleted - test to be done
    - total_prod_2020: the overall production of crude oil in 2020
    - production_proportions_2019: the proportion in 2019 that represents each supplier country of crude oil within the total production
    - eor: a dataframe to indicates information about the end of reserves for each country or the limit reached at the end of the max prediction year (pred_end)
    - world_limits: a dictionary of world reserves limit by model type
    - shortage_world_reserves = a series used to plot by year the countries which comes to an end to their reserves and the associated world crude oil reserves remaining that year
    - pred_reserves_end: the intermediate dataframe with the list of country and the year of the end for their reserves

    Methods:
    - reg_country
    - end_of_reserves
    - plot_reg
    - world_reserves
    - reserve_limit
    - share_prod
    - find_country_eor
    - plot_world_reserves_vs_country_shortage

    """
    model = make_pipeline(StandardScaler(), Lasso(tol=.0001, alpha=.01))

    def __init__(self, data, param):
        """
        set the dataset and fit the models
        """

        # gather inputs and parameters
        self.df = data.df
        self.pred_start = param['year_start']
        self.pred_end = param['year_end']

        # set the prediction time range
        self.years_prediction = np.arange(
            self.pred_start, self.pred_end + 1, 1)

        # initiate dataframe to store predictions
        self.preds = pd.DataFrame(self.years_prediction, columns=['Year'])

        # make the predictions of reserves for all countries
        for i in self.df.Country.unique():
            self.reg_country(country=i)

        # correcting predictions to avoid negative values
        self.preds = self.preds.applymap(lambda x: x if x >= 0 else 0)

        # calculating at world level
        self.preds['World'] = self.preds[self.df.Country.unique()].sum(axis=1)

        # preparing information to compute how to share production when a
        # country does not have reserve to supply anymore

        # calculate the base production in 2019
        self.total_prod_2020 = self.df.query(
            'Year == 2019').oilprod_yearlybbl.sum()

        # compute for each country the proportion of the world production he
        # represents in 2019
        production_proportions_2019 = []
        for country in self.df.Country.unique():
            prod = dict()
            prod['Country'] = country
            prod['Proportion'] = self.df.query(
                'Country == @country & Year ==2019').oilprod_yearlybbl.sum() / self.total_prod_2020
            prod['Year'] = 2019
            production_proportions_2019.append(prod)
        self.production_proportions_2019 = pd.DataFrame(
            production_proportions_2019)

        # define when each country is supposed to come out of resources, it
        # will help to know when to share the prod
        self.compute_end_of_reserves()

        # apply funciton to share prod each year in order to correct the
        # reserves based on the extra production that will be needed to
        # compensate countries without reserves
        self.share_prod()

        # readjust world prediction and end of reserves year for countries
        # after correction of sharng prod has been made
        self.preds['World'] = self.preds[self.df.Country.unique()].sum(
            axis=1)
        self.compute_end_of_reserves()

    def reg_country(self, country):
        """
        perform the regression for the crude oil reserves of a country given a model type.
        it also takes some arguments to control the hypothesis taken to filter out historical values to get a consistent trend for oil reserves predictions
        """
        df = self.df
        model = self.model
        dfc = df.query("Country == @country")
    #     temporary correction for south sudan data - positive regression
        if country == 'South Sudan':
            dfc = dfc.query('Year >= 2015')
        else:
            # hypothesis: use only data after max reserves is reached (reserves
            # have been found and trend is the one of the reserves consumption
            # except south sudan where data are too poor to be able to do it)
            start_year = df.query('Country == @country')[df.query('Country == @country').reserves_corr == df.query(
                'Country == @country').reserves_corr.max()].Year.values[0]
            dfc = dfc.query('Year >= @start_year')

        # define training data
        X_train = dfc.Year.values.reshape(-1, 1)
        Y_train = dfc.reserves_corr.values

        # define data on which to predict
        X = df.loc[df['Country'] == country].Year.unique().reshape(-1, 1)

        # train model
        model.fit(X_train, Y_train)

        # make and store predictions
        self.preds[country] = model.predict(
            self.years_prediction.reshape(-1, 1))

    def compute_end_of_reserves(self):
        '''
        compute the end of reserves by country
        '''
        self.eor = self.preds.set_index('Year').idxmin().reset_index()
        self.eor.columns = ['country', 'end_year']

        # set all point to remain at the local min or 0 when the year with
        # minimum reserves is reached
        for c in self.df.Country.unique():
            self.preds.loc[self.preds['Year'] >=
                           self.eor[self.eor['country'] == c]['end_year'].values[0], c] = 0
            self.preds.loc[self.preds['Year'] >= self.eor[self.eor['country']
                                                          == c]['end_year'].values[0], c] = self.preds[c].min()

        # compute point on the world reserve where shortage of countries occur
        self.shortage_world_reserves = self.eor.end_year.apply(
            lambda x: self.preds[self.preds['Year'] == x].World.values[0])

    def world_reserves(self):
        """
        Plot the crude world oil reserve for a model type with the detail by country
        """
        sns.lineplot(x='Year', y='World', data=self.preds)
        plt.title('Crude oil world reserves predictions')
        plt.ylabel('World crude oil reserves (bbl)')
        plt.show()
        # other plot of the reserves by country
        plt.figure(figsize=(10, 10))
        sns.lineplot(data=self.preds.set_index('Year'), dashes=False)
        # Put the legend out of the figure
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.ylabel('Crude Oil Reserves (bbl)')
        plt.title('Crude oil reserves prediction by country')
        plt.show()

    def share_prod(self):
        """
        function to redistribute consumption to remaining suppliers when the reserves of a country or more come to an end
        """
        # number of countries who still have predicted reserves in 2019 for the
        # selected model (svr to test)
        nb_producers_2019 = self.eor.query('end_year > 2019').shape[0]
        add = 0
        # redistribute in the first year all the remaining reserves from
        # country bending before 2019
        corr = 0
        corr_i = 0
        producers = self.eor.query(
            'end_year > 2019').country.unique().tolist()
        for i in range(self.pred_start, self.pred_end + 1, 1):
            if self.eor.query('end_year == @i').shape[0] > 0:
                #         remove these countries from the list
                nb_producers_2019 -= len(self.eor.query(
                    'end_year == @i').country.values)
                for j in self.eor.query('end_year == @i').country.values:
                    producers.remove(j)
          # total of oil production to reassign to remaining countries
                add = add + self.production_proportions_2019[self.production_proportions_2019['Country'].isin(
                    self.eor.query('end_year == @i').country.values)].Proportion.sum() * self.total_prod_2020
           # new correction
                if nb_producers_2019 == 0:
                    continue
                corr_i = add / nb_producers_2019
         # add here the loop to change all preds in df or add a correction or
         # copy existing value to uncorrected
            self.preds.loc[self.preds['Year'] == i, producers] = self.preds.loc[self.preds['Year']
                                                                                == i, producers].applymap(lambda x: x - corr)
            self.preds.loc[self.preds['Year'] == i, producers] = self.preds.loc[self.preds['Year']
                                                                                == i, producers].applymap(lambda x: x if x >= 0 else 0)
            corr += corr_i

    def plot_world_reserves_vs_country_shortage(self):
        """
        plot the world reserves prediction for a model type with a point on the curve for each country where their reserves ends
        """
        sns.lineplot(x='Year', y='World', data=self.preds)
        plt.scatter(x=self.eor.end_year,
                    y=self.shortage_world_reserves, color='red')
        # plot country name next to each point
        for i, txt in enumerate(self.eor.country):
            plt.annotate(txt, (self.eor.end_year[i] + 1, self.shortage_world_reserves[i] + 1))
        plt.hlines(xmin=self.pred_start,
                   xmax=self.pred_end + 1, y=0, color='red')
        plt.ylabel('Worl crude oil reserves (bbl)')
        plt.title('Crude oil supplier country shortage along world reserves predictions')
        plt.show()


class price_model:
    """
    Create a price model to predict crude oil prices based on a trendline out of historical data and a price correction at each period when a country reserves come to an end.
    The oil correction factor is based on an historical correlationin 1980's Iran production fall with an oil price increase.

    Attributes:
    - pr: historical crude oil prices dataframe
    - df: crude oil reserves data
    - drop: dataframe of production drops around country reserves end
    - pr_increase = daatframe with price increase predicted for the associated production drop
    - dr: dataframe with the synthesis of drop and price_increase
    - pr_reg: price regression model to fit for correction based on 1980's Iran production drop price impact
    - price: historical crude oil prices dataframe
    - pipe: model for baseline regression of the crude oil price historical trend
    - shortage_dates: series of the year when shortage occur (reserves of a country come to an end)
    - pred: oil reserves predictions for the retained lasso model type
    - years_prediction: the array of years on which to make the predictions
    - pred_end: the max year until the preditcions are made

    Methods:
    - plot_price_historics
    - plot_price_vs_prod_change
    - price_correction
    - baseline_reg
    - plot_baseline
    - shortage_impact
    - predict_price
    - plot_price_preds

    """

    pr = pd.read_csv(os.path.join(os.path.dirname(__file__),'data','bp_stats_review_2020_all_Oil_Crude_prices_since_1861.csv'))
    pr_reg = SVR(kernel='poly', C=100, gamma='auto',
                 degree=2, epsilon=.1, coef0=1)
    pipe = make_pipeline(StandardScaler(), SVR(
        kernel='linear', C=100, gamma='auto'))

    def __init__(self, reserves_data, reserves_preds, param):
        """
        load data and fit the price model
        """
        # loading data and parameters
        self.pred_start = param['year_start']
        self.pred_end = param['year_end']
        self.years_prediction = np.arange(self.pred_start, self.pred_end + 1, 1)
        self.df = reserves_data.df
        self.shortage_dates = reserves_preds.eor
        self.preds = reserves_preds.preds
        

        # calculate the baseline based on regression of the historical price trend
        self.baseline_reg_fct()

        # compute the function to calculte the price increase based on production drop
        self.fit_price_vs_prod_change()

        # self.eor = self.preds.set_index('Year').idxmin().reset_index()
        # self.pred = reserves_preds.preds
        self.predict_price()
        
        

    def fit_price_vs_prod_change(self):
        """
        Correction shape -  function calculates the impact of the price increase function of the distance to the shortage in year
        """

        # gather the information about Iran produciton impact on crude oil price
        self.drop = self.df[(self.df['Country'] == 'Iran') & (self.df['Year'] >= 1978) & (
            self.df['Year'] <= 1982)].set_index('Year').oilprod_yearlybbl
        self.pr_increase = self.pr[(self.pr['Year'] >= 1978) & (
            self.pr['Year'] <= 1982)]['$ money of the day']

        # recreate training data based on this historics
        self.dr = pd.DataFrame()
        self.dr.loc[:, 'year'] = self.drop.index
        self.dr.loc[:, 'production'] = self.drop.values
        self.dr.loc[:, 'price'] = self.pr_increase.values

        # compute the change of price and production that happened in that years to have training data for the regression
        self.dr.loc[self.dr['year'] > self.dr.year.min(), 'production_change'] = self.dr[self.dr['year'] > self.dr.year.min()].year.apply(
            lambda x: (self.dr[self.dr['year'] == (x)].production.values[0] - self.dr[self.dr['year'] == x - 1].production.values[0]))
        self.dr.loc[self.dr['year'] > self.dr.year.min(), 'price_change'] = self.dr[self.dr['year'] > self.dr.year.min()].year.apply(
            lambda x: (self.dr[self.dr['year'] == (x)].price.values[0] - self.dr[self.dr['year'] == x - 1].price.values[0]))
        self.dr.loc[:, 'price_on_prod'] = self.dr['price_change'] / \
            (-1 * self.dr['production_change'])

        # fit the price correction model
        self.pr_reg.fit(np.array(self.dr['production_change'].dropna(
        )).reshape(-1, 1), self.dr['price_change'].dropna())

    

    def plot_price_vs_prod_change(self):
        """
        plot the defined model for production impact on price
        """
        # plot the historical data
        sns.lineplot(x='production_change', y='price_change', data=self.dr)

        # plot the prediction
        plt.scatter(x=np.arange(-1, 0.5, .1), y=self.pr_reg.predict(np.arange(-1, 0.5, .1).reshape(-1, 1)), color='orange')
        plt.xlabel('Production chhange (bbl)')
        plt.ylabel('Price change (USD/bl)')
        plt.title('Function for crude oil price correction vs production change')
        plt.show()

    def price_correction(self, prod):
        """
        compute correction of oil price for a given production drop based on fitted model
        """
        corr = self.pr_reg.predict(prod.reshape(-1, 1))
        return corr

    def baseline_reg_fct(self):
        """
        Fit price baseline
        """
        # truncate price historics to avoid past effects prior to today trend
        self.price = self.pr[self.pr['Year'] >= 1960]

        # define data for training
        self.baseline_reg = pd.DataFrame()
        self.baseline_reg['year_training'] = self.price['Year']
        self.baseline_reg['price_training'] = self.price['$ money of the day']

        # fit the model
        self.pipe.fit(np.array(self.baseline_reg['year_training'].values).reshape(-1, 1), self.baseline_reg['price_training'].values)

        # define data for post processing    
        self.baseline_reg['price_trained'] =  self.pipe.predict(np.array(self.baseline_reg['year_training']).reshape(-1, 1))
        self.baseline_reg_pred = pd.DataFrame()
        self.baseline_reg_pred['year_pred'] = self.years_prediction
        self.baseline_reg_pred['price_pred'] = self.pipe.predict(np.array(self.baseline_reg_pred['year_pred'].values).reshape(-1, 1))


    def plot_baseline(self):
        """
        plot curde oil price prediction baseline from historical data trend
        """

        # plot the predictions
        plt.plot(self.baseline_reg_pred['year_pred'], self.baseline_reg_pred['price_pred'], color='blue',label='Predicted')

        # plot the trained part
        plt.plot(self.baseline_reg['year_training'], self.baseline_reg['price_trained'], color='orange',label='Trained')

        # plot the data points used for training
        plt.scatter(self.baseline_reg['year_training'],  self.baseline_reg['price_training'], facecolor="none",edgecolor='blue', s=50,label='Training points')
       
        # plot set up and description
        plt.legend()
        plt.xlabel('Year')
        plt.ylabel('Baseline price (USD/bl)')
        plt.title("Regression for crude oil baseline price")
        plt.show()

    # we decide to impact shortage overs 5 years as Iran crisis and it seems
    # consistent. year 1 to 3 we consider the drop in production, while year 4
    # and 5 are equal in drop to 1 and 2 - logistic recovery time rest is 0
    # price impact
    def shortage_impact(self, country):
        """
        intermediate funciton to compute the cumulated price impact for all shortage and each year
        """
        #  get shortage year for the current country
        shortage_year = self.shortage_dates.loc[self.shortage_dates['country'] == country, 'end_year'].values[0]
        
        # set up range of the price impact in years before and after
        rg = 3

        # if it impacts the price in more than 3 years (eg we are not yet feeling its impact in 2020 when the historical data ends)
        if shortage_year > 2020 + rg:
            # gather prediction data for this country
            hist = self.preds[['Year', country]]

            # calculate production drop each year around the shortage year
            hist.loc[:, 'production_drop'] = hist[(hist['Year'] >= shortage_year - rg + 1) & (hist['Year'] <= shortage_year)].Year.apply(
                lambda x: hist.loc[hist['Year'] == x][country].values[0] - hist.loc[hist['Year'] == x - 1][country].values[0])
            
            # cumulate production drop
            hist.loc[:, 'production_drop_cum'] = hist.production_drop.cumsum()

            # calculate price change associated
            hist.loc[hist['Year'].isin(range(shortage_year - rg + 1, shortage_year + 1, 1)), 'price_change'] = self.price_correction(
                hist.loc[hist['Year'].isin(range(shortage_year - rg + 1, shortage_year + 1, 1)), 'production_drop_cum'].values)
            
            hist.loc[(hist['Year'] > shortage_year) & (hist['Year'] < shortage_year + rg), 'price_change'] = hist[(hist['Year'] > shortage_year) & (hist['Year'] < shortage_year + rg)
                                                                                                                  ].Year.apply(lambda x: hist[hist['Year'] == shortage_year - (hist.loc[hist['Year'] == x, 'Year'].values[0] - shortage_year)]['price_change'].values[0])
            # set price correction to 0 out of the range where the end of reserves has an impact
            hist = hist.fillna(0)

        else:
            # if the country reserves were already or never out then apply no price change
            hist = self.preds[['Year', country]]
            hist.loc[:, 'price_change'] = 0

        return hist.price_change

    def predict_price(self):
        """
        make the complete prediction associated baseline and corrections from shortage occurence when country reserves end
        """
        # compute the baseline price
        base_price = self.pipe.predict(self.years_prediction.reshape(-1, 1))

        # initialize the correction serie
        price_corrections = np.zeros(len(self.years_prediction))

        # calculate the sum of price corrrection for all countries
        for c in self.df.Country.unique():
            price_corrections += self.shortage_impact(c)

        # calculate the final price
        final_price = base_price + price_corrections

        # gather price predictions in a proper dataframe
        self.prices_pred = pd.DataFrame([self.years_prediction, final_price]).T
        self.prices_pred.columns = ['year', 'price']

    def plot_price_preds(self):
        """
        plot the final predictions for crude oil price
        """
        # plot the price prediction
        sns.lineplot(x='year', y='price', data = self.prices_pred)

        # add descriptions and plot setup
        plt.ylim(0, self.prices_pred.price.max() + 5)
        plt.xlim(self.pred_start, self.pred_end)
        plt.xlabel('Year')
        plt.ylabel('Crude oil price (USD/bl)')
        plt.title('Crude oil price predictions')
        plt.show()
