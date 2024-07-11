## Project
Building energy efficiency<br><br>

## Info
Using building heating and cooling data, the idea's to predict the heating and cooling needs of a building, to gauge efficiency.<br><br>

### Models
* Classification
* Regression<br><br>

### Performance
* Both algorithms perform well, given the confusion matrices and graphs showing good accuracy in prediction
    * The confusion matrix (classification) shows that it correctly classifies each label most of the time, with a few outliers
    * It also shows good predictive capabilities with the actual vs predicted graphs (regression), as most points are close to $x=y$ curve that shows the ideal scenario
<br><br>

### Classification
#### Confusion matrices:
<img src="reports/figures/heat_load_confusion_matrix.png" alt="heat load confusion matrix" width="600"/><br><br>
<img src="reports/figures/cooling_load_confusion_matrix.png" alt="cooling load confusion matrix" width="600"/><br><br>

### Regression
#### Actual vs Prediction:
<img src="reports/figures/heating_load_prediction.png" alt="heating load prediction" width="600"/><br><br>
<img src="reports/figures/cooling_load_prediction.png" alt="cooling load prediction" width="600"/><br><br>

### Application
#### Practical:
* Assessment of heating and cooling needs of building designs
* Plan renovations and improvements for existing buildings to move them to a lower energy load category
* Optimize building features such as insulation, orientation, and glazing area to achieve better energy efficiency<br><br><br>

## Notebook
https://colab.research.google.com/drive/1wD2YNJqh0A_YSL2IvcBuuTxrqtdLeyrE#scrollTo=y0Sa7Ikcbe0F<br>

## References
Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools.<br>
By A. Tsanas, Angeliki Xifara. 2012

Published in Energy and Buildings, vol. 49<br>

Link: https://archive.ics.uci.edu/dataset/242/energy+efficiency