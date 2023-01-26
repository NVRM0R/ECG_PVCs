# Premature ventricular contraction detection & preprocessing

Aim of this course project is to create preprocessing the pipeline for ECG classification: normal or premature  ventricular contraction (PVC).

Raw data represented as sequence of samples saved in a CSV file. Figure below shows diffirent parts of that data:
<p align="center">
<img src="images/original.png"></img>
</p>

## Preprocessing pipeline
Methods in **preprocessor** file used to slices QRS complexes, align them by isoline and smooth high frequency components. After preprocessing step, data looks like shown below.

<p align="center">
<img src="images/normalized.png"></img>
</p>

Two approaches have been tested: filtration with Butterworth and Savitzky–Golay filter. 

<p align="center">
<img src="images/prePipeline.png"></img>
</p>

Signal for both ways looks pretty the same, so it was decided to leave only one filter.

## Implemeted features

* QS Width <p align="center"><img src="images/QRSWidth.png"></img></p>
* RS Slope <p align="center"><img src="images/RSSlope.png"></img></p>
* QRS "Fragmentation" <p align="center"><img src="images/QRSFrag.png"></img></p>
* R Direction <p align="center"><img src="images/RDirection.png"></img></p>
* QRS "Integral"<p align="center"><img src="images/Energy.png"></img></p>

## Models
Following models have been tested:
* Logistic regression
* KNN
* XGBoost

Best results have been obtained in KNN model. Diagrams shows below.
## Achieved model results
Features heatmap:
<p align="center"><img src="images/heatmap.png"></img></p>

Confusion Matrix:
<p align="center"><img src="images/confusionMatrix.png"></img></p>
