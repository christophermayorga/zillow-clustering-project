from sklearn.cluster import KMeans

######################################## Create Clusters for Independent Features ########################################

def create_clusters(train_scaled, validate_scaled, test_scaled):
    '''
    Function creates three clusters from scaled train - Tax, SQFT, Rooms
    Fits KMeans to train, predicts on train, validate, test to create clusters for each.
    Appends clusters to scaled data for modeling.
    '''

    # Tax Cluster
    # Selecting Features
    X_1 = train_scaled[['tax_value', 'tax_amount','tax_rate']]
    X_2 = validate_scaled[['tax_value', 'tax_amount','tax_rate']]
    X_3 = test_scaled[['tax_value', 'tax_amount','tax_rate']]
    # Creating Object
    kmeans = KMeans(n_clusters=4)
    # Fitting to Train Only
    kmeans.fit(X_1)
    # Predicting to add column to train
    train_scaled['cluster_tax'] = kmeans.predict(X_1)
    # Predicting to add column to validate
    validate_scaled['cluster_tax'] = kmeans.predict(X_2)
    # Predicting to add column to test
    test_scaled['cluster_tax'] = kmeans.predict(X_3)

    # SQFT Cluster
    # Selecting Features
    X_4 = train_scaled[['property_sqft', 'lot_sqft']]
    X_5 = validate_scaled[['property_sqft', 'lot_sqft']]
    X_6 = test_scaled[['property_sqft', 'lot_sqft']]
    # Creating Object
    kmeans = KMeans(n_clusters=5)
    # Fitting to Train Only
    kmeans.fit(X_4)
    # Predicting to add column to train
    train_scaled['cluster_sqft'] = kmeans.predict(X_4)
    # Predicting to add column to validate
    validate_scaled['cluster_sqft'] = kmeans.predict(X_5)
    # Predicting to add column to test
    test_scaled['cluster_sqft'] = kmeans.predict(X_6)

    # Rooms Cluster
    # Selecting Features
    X_7 = train_scaled[['full_bathrooms', 'bed_plus_bath','room_count','property_age']]
    X_8 = validate_scaled[['full_bathrooms', 'bed_plus_bath','room_count','property_age']]
    X_9 = test_scaled[['full_bathrooms', 'bed_plus_bath','room_count','property_age']]
    # Creating Object
    kmeans = KMeans(n_clusters=5)
    # Fitting to Train Only
    kmeans.fit(X_7)
    # Predicting to add column to train
    train_scaled['cluster_rooms'] = kmeans.predict(X_7)
    # Predicting to add column to validate
    validate_scaled['cluster_rooms'] = kmeans.predict(X_8)
    # Predicting to add column to test
    test_scaled['cluster_rooms'] = kmeans.predict(X_9)

    return train_scaled, validate_scaled, test_scaled