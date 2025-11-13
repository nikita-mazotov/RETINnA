# RETINnA
Rebound Trained Intruder Neural network Approximator (RETINnA) is a project about simulating ISO flybys on a solar system with Rebound and training a Machine Learning algorithms to study the long-term effects thereof. 

## The Rebound simulation

## Ejection Classification
Planet ejection is defined as an orbit with a negative semimajor axis, and is the failure mode chosen for this study. A set of 10k datapoints generated in Rebound was processed with *data_processor.py* to create polar features from the input cartesian ones. The data is formatted into features comprehensible to the three methods:
- BDT
- MLP
- Symbolic Regressor
These methods were then used to classify categories "ejected" and "not ejected", and all achieved a ROC AUC of over 0.9. Feature importances were also computed for the BDT and the MLP (with the SHAP package). The methods are compared in *intruder_comparator.ipynb*. With the *\ejection_classification* direcrtory downloaded, all the models should be able to run "out of the box" using the dataset provided. The comparison will run only after all models have been saved. 
