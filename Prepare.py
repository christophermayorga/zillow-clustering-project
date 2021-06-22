############################################ Imports #############################################

import Acquire

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import sklearn.preprocessing

######################################## Prep Zillow Data ########################################

def prepare_zillow():
    '''
    Acquire and prepare the zillow data obtained from the SQL database.
    Nulls are removed/replaced, outliers are removed, new features are created,
    and columns are renamed/rearranged. Returns the prepped df split into train, validate, test.
    '''
    # acquire the data from module
    df = Acquire.get_home_data()
    
    # Removing Nulls from Columns
    # sets thresh hold to 75 percent nulls, if more than %25 nulls it will be removed
    threshold = df.shape[0] * .75

    # remove columns with specified threshold
    df = df.dropna(axis=1, thresh=threshold)
    
    # Removing Nulls from Rows
    # sets thresh hold to 75 percent nulls, if more than %25 nulls it will be removed
    thresh_hold = df.shape[1] * .75

    # remove rows with specified threshold
    df = df.dropna(axis=0,thresh=thresh_hold)
    
    # Removing Columns with Repeated Data/Unecessary Data
    # don't need additional sqft, county id/city, assessment year, and census columns
    df = df.drop(columns=['finishedsquarefeet12', 'regionidcounty', 'rawcensustractandblock',
                          'regionidcity','assessmentyear','id','propertycountylandusecode'], axis=1)
    
    # Removing Outliers from Continuous Variables
    # assigning columns to remove outliers
    columns = ['calculatedfinishedsquarefeet','lotsizesquarefeet','structuretaxvaluedollarcnt',
           'landtaxvaluedollarcnt','taxamount']
    
    # looping through continuous variables to remove outliers
    for x in columns:
    
        # calculate IQR
        Q1 = df[x].quantile(0.25)
        Q3 = df[x].quantile(0.75)
        IQR = (Q3 - Q1) * 1.5
        
        # calculate upper and lower bounds, outlier if above or below these
        upper = Q3 + (1.5 * IQR)
        lower = Q1 - (1.5 * IQR)
    
        # creates df of values that are within the outlier bounds
        df = df[(df[x] > (lower)) | (df[x] < (upper))]
        
    # Filling Leftover Nulls by Columns
    # Full Bathroom Count Nulls
    # mode of bathroomcnt
    fullbath_mode = df.fullbathcnt.mode()[0]
    # filling nulls with the mode
    df['fullbathcnt'] = df.fullbathcnt.fillna(fullbath_mode)
    
    # Region Zip Code Nulls
    # filling with 90000 to represent no known zipcode (0 would skew the data)
    df['regionidzip'] = df.regionidzip.fillna(90_000)
    
    # Year Built Nulls
    # average of property year built
    year_avg = round(df.yearbuilt.mean())
    # filling nulls with average year built
    df['yearbuilt'] = df.yearbuilt.fillna(year_avg)
    
    # Census Tract and Block Nulls
    # mode of census tract and block
    census_mode = df.censustractandblock.mode()[0]
    # filling nulls with mode
    df['censustractandblock'] = df.censustractandblock.fillna(census_mode)
    
    # Feature Engineering - creating columns
    # calculating bed+bath from 0 null columns of bedroom/bathroom count
    df['bed_plus_bath'] = df.bathroomcnt + df.bedroomcnt
    # droping original calculated field that had nulls
    df = df.drop('calculatedbathnbr',axis=1)
    
    # Property Age
    # current year minus year built
    df['age'] = 2020 - df.yearbuilt
    
    # Transaction Month
    # converting date to string to use split method
    df['transactiondate'] = df.transactiondate.astype('str')
    # creating new feature as the second index (month) of the transaction date split
    df['transaction_month'] = df.transactiondate.str.split('-',expand=True)[1]
    
    # Calculating Tax Rate for Property
    # Tax paid / tax value * 100 = tax rate %
    df['tax_rate'] = (df.taxamount / df.taxvaluedollarcnt) * 100

    # Renaming Columns
    df.columns = ['parcel_id', 'bathrooms', 'bedrooms', 'property_sqft', 'county_id', 'full_bathrooms',
                  'latitude', 'longitude', 'lot_sqft', 'land_use_type', 'zip_code', 'room_count',
                  'year_built', 'structure_tax_value', 'tax_value', 'land_tax_value', 'tax_amount', 'census_id',
                  'log_error', 'transaction_date', 'bed_plus_bath', 'property_age', 'transaction_month','tax_rate'
             ]
    
    # Reordering Columns
    df = df[['parcel_id',
        'log_error', 'tax_value', 'structure_tax_value', 'land_tax_value', 'tax_amount', 'tax_rate',
        'county_id', 'zip_code', 'latitude', 'longitude', 'census_id',
        'bathrooms', 'bedrooms', 'full_bathrooms', 'bed_plus_bath', 'room_count',
        'property_sqft', 'lot_sqft', 'land_use_type',
        'year_built', 'property_age', 'transaction_date', 'transaction_month'
       ]]

    # Converting Unecessary Floats to Integers - such as county id 6011.0 to 6011
    df['county_id'] = df.county_id.astype('int')

    df['zip_code'] = df.zip_code.astype('int')

    df['bathrooms'] = df.bathrooms.astype('int')

    df['bedrooms'] = df.bedrooms.astype('int')

    df['full_bathrooms'] = df.full_bathrooms.astype('int')

    df['bed_plus_bath'] = df.bed_plus_bath.astype('int')

    df['room_count'] = df.room_count.astype('int')

    df['land_use_type'] = df.land_use_type.astype('int')

    df['transaction_month'] = df.transaction_month.astype('int')

    df['property_age'] = df.property_age.astype('int')

    df['year_built'] = df.year_built.astype('int')

    # split into train, validate, and test sets
    train_and_validate, test = train_test_split(df, test_size = .10, random_state=123)
    train, validate = train_test_split(train_and_validate, test_size = .22, random_state=123)

    # These two print functions allow us to ensure the date is properly split
    # Will print the shape of each variable when running the function
    print("train shape: ", train.shape, ", validate shape: ", validate.shape, ", test shape: ", test.shape)

    # Will print the shape of eachvariable as a percentage of the total data set
    # Varialbe to hold the sum of all rows (total observations in the data)
    total = df.count()[0]
    print("\ntrain percent: ", round(((train.shape[0])/total),2) * 100, 
            ", validate percent: ", round(((validate.shape[0])/total),2) * 100, 
            ", test percent: ", round(((test.shape[0])/total),2) * 100)
    
    return train, validate, test

######################################## Scale Zillow Data ########################################

def scale_data(train, validate, test):

    train = train.drop(['log_error','census_id','transaction_date'], axis=1)
    validate = validate.drop(['log_error','census_id','transaction_date'], axis=1)
    test = test.drop(['log_error','census_id','transaction_date'], axis=1)

    # 1. Create the Scaling Object
    scaler = sklearn.preprocessing.StandardScaler()

    # 2. Fit to the train data only
    scaler.fit(train)

    # 3. use the object on the whole df
    # this returns an array, so we convert to df in the same line
    train_scaled = pd.DataFrame(scaler.transform(train))
    validate_scaled = pd.DataFrame(scaler.transform(validate))
    test_scaled = pd.DataFrame(scaler.transform(test))

    # the result of changing an array to a df resets the index and columns
    # for each train, validate, and test, we change the index and columns back to original values

    # Train
    train_scaled.index = train.index
    train_scaled.columns = train.columns

    # Validate
    validate_scaled.index = validate.index
    validate_scaled.columns = validate.columns

    # Test
    test_scaled.index = test.index
    test_scaled.columns = test.columns

    return train_scaled, validate_scaled, test_scaled