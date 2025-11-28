# RETINnA

Rebound Trained Intruder Neural network Approximator (RETINnA) is a project about simulating ISO flybys on a solar system with Rebound and training Machine Learning algorithms to study the long-term effects thereof. 

## The Rebound simulation
REBOUND is an N-body simulation software package which we used to generate a Sun-Earth system where an intruder comes in. We set the initial mass, starting position, and initial velocity of the intruder. Then we integrate the simulation for 100 years - which should be enough time for the intruder to pass by. At the end, we check whether Earth is orbiting the Sun or not. This is done by checking the eccentricity - if it is more than 1, the orbit is a hyperbola and therefore the Earth has been ejected. 

We ran this 10000 times, using a software called Sobol to sample the intruder's mass, speed, initial distance from the Sun, and target distance from the Sun. Sobol was chosen over random sampling because it picks out combinations more evenly. 

All of this was done on ReboundGeneration.ipynb

## Ejection Classification

Planet ejection is defined as an orbit with a negative semimajor axis, and is the failure mode chosen for this study. A set of 10k datapoints generated in Rebound was processed with *data_processor.py* to create polar features from the input cartesian ones. The data is formatted into features comprehensible to the three methods:
- BDT
- MLP
- Symbolic Regressor
These methods were then used to classify categories "ejected" and "not ejected", and all achieved a ROC AUC of over 0.9. Feature importances were also computed for the BDT and the MLP (with the SHAP package). The methods are compared in *intruder_comparator.ipynb*. With the *\ejection_classification* direcrtory downloaded, all the models should be able to run "out of the box" using the dataset provided. The comparison will run only after all models have been saved. 

## Semimajor Axis Regression

This directory contains the incomplete attempt of using a BDT regressor to predict the value of the semimajor axis a long time after the ISO interaction. The error of this predictor was high, and it was aborted with no clear way of making it operational in the timeline of the project.
