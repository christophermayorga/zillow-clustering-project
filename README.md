# Predicting Home Values for Zillow
## About the Project
### Goals
When someone wants to know the value of a property, our website is often one of the top recommendations. However, there is always room for improvement, and I want to accomplish this improvement with clustering methodologies to determine where error may be coming from. How can we improve? By creating a model to predict log error in our current Zestimate, we can determine what is driving this error.
### Background
According to the Zillow Kaggle competition,
> "By continually improving the median margin of error (from 14% at the onset to 5% today), Zillow has since become established as one of the largest, most 
> trusted marketplaces for real estate information in the U.S. and a leading example of impactful machine learning."  

````
                    The log error is defined as logerror=log(Zestimate)−log(SalePrice)
````
 
### Outline
The organization of this project is visualized below. Only Final_Model.ipynb and the .py files are necessary to reproduce. However, the module notebooks go more in-depth than the final notebook. Both the module notebooks and pdfs are additional resources about the project.


<p align="center">
  <img src="https://raw.githubusercontent.com/christophermayorga/zillow-clustering-project/master/images/Screen%20Shot%202021-06-24%20at%2011.21.16%20AM.png" width="800" height="200" >
</p> 


### Data Dictionary
The linked Data Quality [Report](https://drive.google.com/file/d/1wh3iKkAX7o-PZ46EcsHzoZbxtD-BKB6-/view)
reviews the raw aquired data from the zillow database. It includes next steps for each feature, such as dropping the feature or preparing for modeling. The data dictionary below will include the columns used or created after prepping this data.  

<p align="center">
  <img src="https://i.pinimg.com/originals/90/9f/6e/909f6e6a63918d591f56079228fc8b3a.png" width="800" height="500" >
</p> 

## Hypothesis
Month of transaction / Property square footage / Year built 
                                                      ...effects log error.  
> Null hypothesis: There is no significant effect of _______ and log error.   
> Alternative hypothesis: There is a significant effect of ______ on log error.   

Tax Cluster / Sqft Cluster / Room and Age Cluster
                                                      ...effects log error.  
> Null hypothesis: There is no significant difference of average log error in ______ cluster groupings.  
> Alternative hypothesis: There is a significant difference of average log error in at least 2 ______ cluster groupings.  
# Project Steps
## Acquire
Within the acquire.py module, there are functions to:
- connect to the SQL company database (login credentials required)
- read a query to select the data and save to a csv file
- assign the data to a variable
- count the nulls of each columns and calculate the percentage of this
## Prepare
The Prepare.py has two functions:  
To prepare
- contains the Acquire module to obtain data
- removes nulls in columns/rows
- removes outliers
- creates new columns
- rearranges/renames columns
- splits into train, validate, test  

To scale  
- drops log error and unscaleable columns (census/transaction date)
- creates standard scaler fit on train
- transforms on train, validate, and test
- returns the dataframes
## Explore
Once the data is split, explore on train to leave the rest of the data as *unseen*. Visualized independent features versus log error and clusters created in Tableau. Features and clusters were also checked for significance using hypothesis testing with T-test, Correlation test, and Anova test.  

Clusters created:  
- Tax using tax_value, tax_amount, tax_rate
- Square feet using property_sqft and lot_sqft
- Room/Age using full_bathrooms, bed_plus_bath, room_count, property_age
## Model
Baseline model was based on the average log error and had a RMSE score of 0.01421  
Final model chosen was a Polynomial Linear Regression with degree=3 on the top 9 features from SelectKBest.
- Train RMSE: 0.01416
- Validate RMSE: 0.21491
- Test RMSE: 0.31837
## Conclusion
Clusters and features explored did not have a significant difference with respect to log error. More exploration is needed to determine if other clusters can be created. Could these improve the model that predicts log error? Right now, the model's root mean squared error was about .1 higher than the baseline.
# How to Reproduce
- [x] Read this README.md
- [ ] Download Acquire.py, Prepare.py, Model.py, and Final_Model.ipynb in your working directory.
- [ ] Run the Final_Modeling.ipynb Jupyter Notebook.
- [ ] Do your own exploring, modeling, etc.
