## Project
Imbalance pricing insights<br><br>

## Info
The wholesale electricity market is set up so that supply should always meet demand on the transmission system, thorough ensuring the contracts between parties including organizations that require electricity for their customers (suppliers) and organizations that produce electricity (generators) are such that balancing actions given to a party, are in accordance with agreed rules, to either increase or decrease generation, or increase or decrease demand, depending on the appropriate balancing action, and the dataset utilized pertains to this balancing act, more specifically referred to as **imbalance pricing**.<br><br>

#### Imbalance pricing analysis:<br><br>
<img src="reports/figures/analysis.png" alt="price scatter" width="650"/><br><br>


Above, are key important aspects of the data acquired on energy trading under energy balancing. The **buy/sell price** fluctuates based on market conditions (supply & demand), the **net imbalance volume determines** the frequency and magnitude at which the system experiences imbalance, and the **total accepted bid and offers volumes** reflect the amount of energy accepted for generation (offer volumes) or consumption reduction (bid volumes) by the market operator to optimally balance supply and demand, so offers (generation increases or demand decreases) increase, as bids (generation decreases or demand increases) decrease, which achieves balance.<br>
For unlocking value from this data, certain strategic insights should be determined. A way to do this, could be through machine learning models to derive valuable insights based on acquired data through volume / price forecasting. A large benefit of this type of prediction / forecasting, helps to inform trading strategies, with:
* *Price* - Where bids can be higher during anticipated price spikes and lower during anticipated price drops, to improve the chance of successful bids, to optimizing market participation.<br><br>
* *Accepted offer volume* - By knowing when there is likely to be higher acceptance, they can submit more competitive bids that align with market conditions to increase the likelihood of acceptance.<br><br>
* *Net imbalance volume* - In a short system, prices are typically higher, and knowing this in advance can help producers maximize revenue, therefore knowing when the system is likely to be short (positive NIV) or long (negative NIV) allows market participants to adjust their trading strategies accordingly.<br><br>


### Models
* Long Short-term Memory (LSTM):
    * System Price
    * Volume
    * Net Imbalance Volume (NIV)<br><br>

### Performance
#### <u>System Buying / Selling Price</u>
* With a MSE of 935.03, its relatively high, indicating model's predictions have significant errors, with RMSE being 30.58 it suggests the time-series data isnt being captured effectively 
* A relatively low R2 Score of 0.183, indicates poor capture of the variability in the data and shows model is underfitting 
* Most points in predicted vs actual prices are not very close to the ideal $x = y$ line, supporting the idea of poor model performance
<br><br>

#### LSTM
##### Actual vs Prediction:
<img src="reports/figures/price_scatter.png" alt="price scatter" width="470" height="400"/><br><br>

#### <u>Total Accepted Offer Volume</u>
* With an MSE of 27879.55, and RMSE of 166.97 it's also a poorly performing model that's off by a rather large magnitude
* However, with an R2 score of 0.788, is a relatively high R2 score, indicating that the model fits the data reasonably well and captures most of the variability
* This varied performance in the model's evaluation is well reflected by the predicted vs. actual plot, with some values being concentrated close to the ideal line, but some concentrated away from it
<br><br>

#### LSTM
##### Actual vs Prediction:
<img src="reports/figures/accepted_offer_volume_scatter.png" alt="accepted offer volume scatter" width="470" height="400"/><br><br>


#### <u>Net Imbalance Volume</u>
* An MSE of 45417.135 is even way more off than other models, and a large RMSE of 213.112, and an R2 Score of 0.211, this model shows an even more chaotic performance in prediction than the others
* The scatter-plot reflects this chaos as well
<br><br>

#### LSTM
##### Actual vs Prediction:
<img src="reports/figures/niv_scatter.png" alt="heating load prediction" width="470" height="400"/><br><br>


### Application
##### System buy / sell price model:<br><br>
<img src="reports/figures/price_pred.png" alt="heating load prediction" width="650"/><br><br>

##### Accepted offer volume:<br><br>
<img src="reports/figures/accepted_offer_volume_pred.png" alt="heating load prediction" width="650"/><br><br>

##### Net imbalance volume:<br><br>
<img src="reports/figures/niv_pred.png" alt="heating load prediction" width="650"/><br><br>

Each model had different evaluation performances, with the first (price) being reasonable, with worsening performance with each subsequent one (offer volume -> NIV), which may be explained by complexity of curves, as its much more difficult to predict a curve with a lot of varience than one with fewer, so that's likely the reason, but further analysis and model tuning will be necessary to see if this, or other factors are in effect here.<br><br>

#### Practical:
* Generally, inform trading strategies based on forecast<br><br><br>

## Notebook
https://colab.research.google.com/drive/1So3vq08OLjLI7fcQpDyyJb2-iNQ7bnpa#scrollTo=A4ADKW3urL7y<br><br>

## References
Imbalance pricing guidance.<br>
by Elexon<br>

Link: https://www.elexon.co.uk/documents/training-guidance/bscguidance-notes/imbalance-pricing/#:~:text=After%20the%20end%20of%20the,be%20subject%20to%20imbalance%20charges<br><br>

System sell and buy prices.<br>
by Elexon<br>

Link: https://bmrs.elexon.co.uk/system-prices<br><br>