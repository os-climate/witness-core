## Non-use capital objective

This model gathers all capitals non-used in all sectors for various reasons : 
1) non use of an energy industry because of lack of raw materials (needed energy or resources)
  
$$ non\_use\_capital = production *CAPEX *(1.0-use\_ratio) $$

with the ratio $use\_ratio$ defined as :

$$ use\_ratio = min(\frac{production(energy)}{consumption(energy)} for\ energy\ in\ used\_energy\_list ) $$

with $used\_energy\_list$ the list of energy used by the industry. If the consumption is higher than the production for a industry needed energy at a specific year, the following industry cannot be used full-time. Some capital is consequently not used at this year.

2) a wrong combination of opponent technologies (i.e. reforestation and deforestation, see Forest documentation) 
3) later non-use of any sectorized industry because lack of workforce or energy

The model computes then a non-use capital objective to minimize the sum of all non-use capitals : 


$$ non\_use\_capital\_objective = \frac{\sum(non\_use\_capital)}{ref \Delta t} $$

with a reference defined in input parameters. 

