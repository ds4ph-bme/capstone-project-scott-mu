## Diabetes Deep Learning Prediction Shiny App


https://user-images.githubusercontent.com/78364878/118744405-d4f3d000-b843-11eb-9b05-53ed00e0886d.mp4


### Introduction

This app uses biometric and demographic data from a representative sample of the United States to predict the outcome of self-reported diabetes. It uses an artificial neural network deep learning model to capture non-linearities and avoid a priori assumptions about the relationships between the predictor variables. 

The Shiny app is hosted here: https://scott-mu.shinyapps.io/diabetes/

### Scientific Rationale/Motivation
Diabetes mellitus is a common condition affecting millions of Americans. Diagnostic testing is widely available and consists of biochemically testing a blood sample. However, biometric features can be obtained noninvasively and may be correlated with the presence of diabetes. Some predictors, such as age or weight, could even be considered causal factors. The goal of this app is to test the validity of using only demographic and anthropometric predictors to ascertain the probability of diabetes. Though only 10 predictors were used for this project, this methodology is easily extended to include hundreds or thousands of predictors, and can be adapted to a wide variety of clinical outcomes.

### Data Source
The data is from the 2017-2018 National Health and Nutrition Examination Survey (NHANES), which is a publicly available dataset from the Center for Disease Control and Prevention (CDC). The `nhanesA` package was used to automate and simplify data retrieval.

### Model Architecture
First, the data is preprocessed by removing missing and ambiguous values. Only nonpregnant adults with complete biometric and self-reported diabetes data are included, leaving a sample size of 4,897 individuals. The length and weight measurements are left as continuous numeric predictors, but the categorical variables of sex and race/ethnicity are one-hot encoded. Next, all predictors were standardized. 80% of the data was used to build the model, and 20% of the data was left aside to test the model and construct a receiver operating characteristics (ROC) plot. Approximately 15% of the individuals in the dataset had an outcome of diabetes, so the diabetes class weight was adjusted to be 6 times that of the no diabetes class.
The model consists of an input layer, and 2 densely connected hidden layers with 16 nodes each. A 10% dropout layer is also included after each hidden layer to combat possible overfitting. The hidden layers used a rectified linear unit (ReLU) activation function, and the final output layer used a sigmoid function to classify individuals as having diabetes or not.  

### Model Interpretability
Two plots are included to aid interpretability. The correlation plot displays the global correlation between the positive outcome and presence or magnitude of each predictor. Additionally, for the user-input data, the `lime` package provides local interpretable model-agnostic explanations for which predictors contribute or contradict the proposed prediction.

### Limitations and Future Objectives
One weakness of the present model is that it does not take into account the complex survey sampling strategy used in NHANES. Additionally, missingness was not appropriately accounted for. Ultimately, if the predictor variables simply do not numerically correlate with the outcome in a meaningful manner, this type of prediction model will fail regardless of the volume of training data. A natural extension of this project is to predict a different clinical outcome using hundreds of predictor variables whose real-world relationships and causal associations are unclear.




### References:
Dancho (2018, Jan. 11). RStudio AI Blog: Deep Learning With Keras To Predict Customer Churn. Retrieved from https://blogs.rstudio.com/tensorflow/posts/2018-01-11-keras-customer-churn/

Guillemot, Vincent (2019, July 2). MLR ROC curves with CI. Retrieved from https://rpubs.com/vguillem/465086

### Key Packages:
Christopher J. Endres (2021). nhanesA: NHANES Data Retrieval. R package version 0.6.5.3. https://CRAN.R-project.org/package=nhanesA

JJ Allaire and François Chollet (2021). keras: R Interface to 'Keras'. R package version 2.4.0. https://CRAN.R-project.org/package=keras

Thomas Lin Pedersen and Michaël Benesty (2021). lime: Local Interpretable Model-Agnostic Explanations. R package version 0.5.2. https://CRAN.R-project.org/package=lime

Xavier Robin, Natacha Turck, Alexandre Hainard, Natalia Tiberti, Frédérique Lisacek, Jean-Charles Sanchez and Markus Müller (2011). pROC: an open-source package for R and S+ to analyze and compare ROC curves. BMC Bioinformatics, 12, p. 77.  DOI: 10.1186/1471-2105-12-77 <http://www.biomedcentral.com/1471-2105/12/77/>

Max Kuhn, Simon Jackson and Jorge Cimentada (2020). corrr: Correlations in R. R package version 0.4.3. https://CRAN.R-project.org/package=corrr

Wickham et al., (2019). Welcome to the tidyverse. Journal of Open Source Software, 4(43), 1686, https://doi.org/10.21105/joss.01686
