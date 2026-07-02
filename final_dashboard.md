# Project Summary

IoT Contextual Predictive Maintenance

Goal:
Predict machine failures using
Random Forest
LightGBM
SMOTE

Dataset:
AI4I Predictive Maintenance Dataset


# Dataset Statistics

Rows : 10000

Features : 16 Base

Context Features : 18

Target :
Machine Failure

# Mark Down table

Model                  Macro F1
-------------          -------------
Random Forest          0.82
Context RF             0.84
LightGBM+SMOTE         0.88


# SHAP Insights

Top important features

• Torque

• Rotational Speed

• Tool Wear

• Air Temperature

• Process Temperature

# Robustness Analysis

Noise robustness testing will be
added in Week 4.


# Conclusion

LightGBM + SMOTE gives the
best performance.

Macro F1 > 0.85 achieved.