# Import relevant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
pi = math.pi # for calculating mass of Jupiter

# Import the relevant model from sklearn, for the linear regression model
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
# Create an instance of the model
model = linear_model.LinearRegression(fit_intercept=True)

class Moons:
    def __init__(self, database_service, file, query):
        self.database_service = database_service
        self.file = file
        self.query = query
        self.connectable = f"{self.database_service}:///{self.file}"
        
        print(f"Our connectable for our database is {self.connectable}")
        
        self.data = pd.read_sql(self.query,self.connectable)
        
    # Upload Required Data, using SQL, set self.data as attribute
    # containing required data
        
    def select_moon(self, moon_name = None):
        self.moon = moon_name
        print(f"Moon selected: {self.moon}")
        self.data.moon_index = self.data.set_index("moon")
        self.moon_data = self.data.moon_index.loc[self.moon]
        print(f"Moon Info: {self.moon_data}")
        
    # Select specific moon and output information on said moon
    # from dataset (self.data)
        
    def summary(self):
        print(f"Data Head: \n{self.data.head()}\n")
        print(f"Data Describe: \n{self.data.describe()}\n")
        print(f"Data Info: \n{self.data.info()}")
        
    # Print basic statistical summary of data, and head to see the columns
    
    def print_moons(self):
        self.moon_names = self.data["moon"]
        print(f"The moons in the dataset are \n{self.moon_names}.")
        
    # Quick way to print all the moons/names of the ones in the dataset
        
    def grouped_summary(self, group=None):
        # Following summary only performed if a group name is provided
        
        self.group = group
        if self.group != None:
            self.data_grouped = self.data.groupby(self.group)
            print(f"Data of group: Max: {self.data_grouped.max()}, \nMean: {self.data_grouped.mean()}, \nStandard Deviation: {self.data_grouped.std()}")
        
    # Print a quick overview/summary of the dataset after grouping, 
    # e.g. general statistics (min/max etc.)
    
    def correlation(self, clmn1 = None, clmn2 = None, method = None):
        if clmn1 == None or clmn2 == None or method == None:
            print("Please provide two valid columns to calculate correlation, and method.")
            print("Possible method: pearson, kendall, spearman")
       # To make sure needed information is provided
    
        else:
            self.clmn1 = clmn1
            self.clmn2 = clmn2
            self.method = method
            self.data_to_correlate = self.data[[self.clmn1,self.clmn2]]
            print(f"Correlation between {self.clmn1} and {self.clmn2}: \n{self.data_to_correlate.corr(method = self.method)}")
            
    # Select two columns/variables to calculate correlation (coefficient)
    # between them. Print statement if incorrect/insufficient information provided
    
    def simple_plot(self, variable1=None, variable2=None):
        self.variable1 = variable1
        self.variable2 = variable2
        if variable1 != None and variable2 != None:
            plt.plot(self.data[self.variable1], self.data[self.variable2], "ro", ms = 3, mec = "k")
            plt.xlabel(self.variable1)
            plt.ylabel(self.variable2)
        else:
            print(f"Please provide two variables (columns in the dataframe) to plot. They should be of numerical type.")
            
    # A simple plot to visualize one variable against another
    # Useful for intuitive understanding/prediction of correlation etc.
    
    def return_max(self, max_variable = None):
        self.max_variable = max_variable
        if max_variable != None:
            return self.data.loc[self.data[self.max_variable].idxmax()]
        else:
            print(f"Please provide variable/column.")
        
    # Return the moon with maximum value of the variable/column provided
    
    def return_min(self, min_variable = None):
        self.min_variable = min_variable
        if min_variable != None:
            return self.data.loc[self.data[self.min_variable].idxmin()]
        else:
            print(f"Please provide variable/column.")
            
    # Return the moon with minimum value of the variable/column provided
    
    ### Following functions/code is for Task 2 (linear regression/mass of Jupiter) ###
    
    def convert_units_and_split(self):
        self.data["period_days_in_seconds"] = (self.data["period_days"]*86400)**2 # convert to seconds
        self.data["distance_km_in_metres"] = (self.data["distance_km"]*1000)**3   # convert to metres
        print("'period_days' column has been converted to seconds, and 'distance_km' column has been converted to metres.")
        print("Both have been concatenated to dataframe as additional columns.")
        self.converted_data = self.data[["period_days_in_seconds","distance_km_in_metres"]].copy()

        self.x = self.converted_data[["distance_km_in_metres"]]
        self.y = self.converted_data["period_days_in_seconds"]
            
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=42)
        
    # Convert the two required variables (a - distance_km, and T - period_days) into metres and seconds, 
    # as required for Kepler's Third Law, and concatenate to the existing dataset as new columns;
    # split the two variables' data into training and test sets
        
        
    def train_model(self):
        self.jupiter_model = model.fit(self.x_train,self.y_train)
        self.pred = model.predict(self.x_test)
        
    # Train the model with the training data, 
    # and produce set of predictions (basically of variable T in Kepler's Third Law)
        
    def validate_model(self):
        print("Visually assess two plots.")
        print("Scatter plot of the test data agaisnt the predictions we produced from training (the model),")
        plt.scatter(self.x_test,self.pred)
        print("and the Residual plot.")
        fig,ax = plt.subplots()
        ax.plot(self.x_test, self.y_test-self.pred,".")
        ax.axhline(0, color="k", linestyle="dashed")
        ax.set_xlabel("Distance in metres")
        ax.set_ylabel("Residuals")
        
    # Print a scatter plot of the test data set against the 'predictions' data set
    # to visually assess fit, and a residual plot
    
    def model_info(self):
        self.model_coef = model.coef_[0]
        self.model_intercept = model.intercept_
        print(f"The model coefficient is {self.model_coef} and the model intercept is {self.model_intercept}")
        self.r2_score = r2_score(self.y_test,self.pred)
        self.mse = mean_squared_error(self.y_test,self.pred, squared=False)
        print(f"The model's R2 score is {self.r2_score}, and the Mean Squared Error is {self.mse}")
        
    # Print out some information about the model (such as its coefficient)
    # which we will use to calculate Jupiter's Mass
    
    def calculate_Jupiters_mass(self):
        self.mass_of_jupiter = (4*(pi**2))/(self.model_coef*6.67e-11)
        print(f"The Mass of Jupiter is {self.mass_of_jupiter} in kg.")
        
    # Calculate/estimate the mass of Jupiter using the model's coefficient,
    # equating it to the constant of proportionality (i.e. the (4pi^2/GM) in Kepler's Third Law)

