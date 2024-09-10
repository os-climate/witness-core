## Macroeconomics model

The goal of the macroeconomics model is to gather and sum the outputs from the economics sectors models and compute the total investment. The list of economics sectors is: 
- agriculture
- industry
- services

### Inputs 
- Sector capital_df: Dataframe with capital and usable capital quantity from each sector in T\$.
-  Sector production df: Dataframe with output and output net of damage in T\$ from each sector. 
- Total investement share of gdp: Dataframe with the percentage of the output that is invested.
- Sectors investment share of gdp: Dataframe with the percentage of the output invested in each sector.
  
### Outputs 
- Economics df: Dataframe with total net output per year in T\$.
- Investment df: Dataframe with total investment per year in T\$.
- Sectors investment df: Dataframe with investment per sector per year in T\$.
- Economics detail df: Dataframe with capital, usable capital, output and net output per year in T\$. 
           
### Time Step 
The time step $t$ in each equation represents the period we are looking at. In the inputs we initialize the data with 2020 information. 

### Output
The total output and output net of damage are the sum of the sectors output and output net of damage: 
$$Total\_net\_output = \sum_{s=i}^{j} net\_output_s$$
with s the list of sectors. And: 
$$Total\_output = \sum_{s=i}^{j} output_s$$

### Capital
The total capital and usable capital are the sum of the sectors capital and usable capital: 
$$Total\_usable\_capital = \sum_{s=i}^{j} usable\_capital_s$$
with s the list of sectors. And: 
$$Total\_capital = \sum_{s=i}^{j} capital_s$$

### Investment
A portion defined by the input $total\_investment\_share\_of\_gdp$ of the total net output is used for investment. The total investment is then:  
$$Investment = total\_net\_output * total\_investment\_share\_of\_gdp$$
The remaining output is used for consumption. 

#### Sectors investment
For each sector a defined portion of the net ouptut is use for investment:   
$$Sector\_investment = total\_net\_output * sector\_invest\_share$$. 
