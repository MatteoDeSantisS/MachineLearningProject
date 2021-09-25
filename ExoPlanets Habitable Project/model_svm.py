import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import svm
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


# Planetary and Stellar parameters    
planetary_stellar_parameter_indexes = (2,   # kepoi_name:      KOI Name
                                       15,  # koi period,      Orbital Period [days]
                                       42,  # koi_ror:         Planet-Star Radius Ratio
                                       45,  # koi_srho:        Fitted Stellar Density [g/cm**3] -
                                       49,  # koi_prad:        Planetary Radius [Earth radii]
                                       52,  # koi_sma:         Orbit Semi-Major Axis [AU]
                                       58,  # koi_teq:         Equilibrium Temperature [K]
                                       61,  # koi_insol:       Insolation Flux [Earth flux]
                                       64,  # koi_dor:         Planet-Star Distance over Star Radius
                                       76,  # koi_count:       Number of Planet 
                                       87,  # koi_steff:       Stellar Effective Temperature [K] 
                                       90,  # koi_slogg:       Stellar Surface Gravity [log10(cm/s**2)]
                                       93,  # koi_smet:        Stellar Metallicity [dex]
                                       96,  # koi_srad:        Stellar Radius [Solar radii]
                                       99   # koi_smass:       Stellar Mass [Solar mass]
                                       );
#Names of columns from kepler data
planetary_stellar_parameter_cols = (   "koi_period",    # koi_period       Orbital Period [days]
                                       "koi_ror",       # koi_ror:         Planet-Star Radius Ratio
                                       "koi_srho",      # koi_srho:        Fitted Stellar Density [g/cm**3] -
                                       "koi_prad",      # koi_prad:        Planetary Radius [Earth radii]
                                       "koi_sma",       # koi_sma:         Orbit Semi-Major Axis [AU]
                                       "koi_teq",       # koi_teq:         Equilibrium Temperature [K]
                                       "koi_insol",     # koi_insol:       Insolation Flux [Earth flux]
                                       "koi_dor",       # koi_dor:         Planet-Star Distance over Star Radius
                                       "koi_count",     # koi_count:       Number of Planet 
                                       "koi_steff",     # koi_steff:       Stellar Effective Temperature [K] 
                                       "koi_slogg",     # koi_slogg:       Stellar Surface Gravity [log10(cm/s**2)]
                                       "koi_smet",      # koi_smet:        Stellar Metallicity [dex]
                                       "koi_srad",      # koi_srad:        Stellar Radius [Solar radii]
                                       "koi_smass"      # koi_smass:       Stellar Mass [Solar mass]
                                       );
planetary_stellar_parameter_cols_dict = {   "koi_period":   "Orbital Period",
                                       "koi_ror":     "Planet-Star Radius Ratio",
                                       "koi_srho":      "Fitted Stellar Density",
                                       "koi_prad":     "Planetary Radius",
                                       "koi_sma":      "Orbit Semi-Major Axis",
                                       "koi_teq":       "Equilibrium Temperature",
                                       "koi_insol":     "Insolation Flux",
                                       "koi_dor":       "Planet-Star Distance over Star Radius",
                                       "koi_count":     "Number of Planet" ,
                                       "koi_steff":     "Stellar Effective Temperature" ,
                                       "koi_slogg":     "Stellar Surface Gravity",
                                       "koi_smet":      "Stellar Metallicity",
                                       "koi_srad":      "Stellar Radius",
                                       "koi_smass":      "Stellar Mass"
                                       };
def load_dataframe():
    habitable_not_habitable_planet = pd.read_csv("data/habitable_not_habitable_planet.csv")
    return habitable_not_habitable_planet

def data_visualizzation(habitable_not_habitable_planet):
    
    habitable_not_habitable_planet=habitable_not_habitable_planet.fillna(0)
    prediction=habitable_not_habitable_planet["Habitable"]
    habitable_not_habitable_planet=habitable_not_habitable_planet.drop(["Habitable"],axis=1)
    
    total_temperature_habitable_planet = 0
    minimum_temperature_habitable_planet = 1000
    maximum_temperature_habitable_planet= 0
    number_of_habitable_planets = 0
    
    total_radius_habitable_planet = 0
    minimum_radius_habitable_planet = 1000
    maximum_radius_habitable_planet= 0
   #------------------------------------------------------------------------------------------------ 
    total_temperature_not_habitable_planet = 0
    minimum_temperature_not_habitable_planet = 1000
    maximum_temperature_not_habitable_planet= 0
    number_of_not_habitable_planets = 0
    
    total_radius_not_habitable_planet = 0
    minimum_radius_not_habitable_planet = 1000
    maximum_radius_not_habitable_planet= 0
    
    for i in range(len(prediction)):
        if(prediction[i]==1):
            number_of_habitable_planets +=1
            
            planet_temperature = habitable_not_habitable_planet.iloc[i, 58] - 273.15
            radius_planet = habitable_not_habitable_planet.iloc[i, 49]
            
            total_temperature_habitable_planet += planet_temperature
            total_radius_habitable_planet += radius_planet  
            
            if planet_temperature > maximum_temperature_habitable_planet:
                maximum_temperature_habitable_planet = planet_temperature
            elif planet_temperature < minimum_temperature_habitable_planet:
                minimum_temperature_habitable_planet = planet_temperature
            
            if radius_planet > maximum_radius_habitable_planet:
                maximum_radius_habitable_planet = radius_planet
            elif radius_planet < minimum_radius_habitable_planet:
                minimum_radius_habitable_planet = radius_planet
        else:
            number_of_not_habitable_planets +=1
            planet_temperature = habitable_not_habitable_planet.iloc[i, 58] - 273.15
            radius_planet = habitable_not_habitable_planet.iloc[i, 49]
            
            total_temperature_not_habitable_planet += planet_temperature
            total_radius_not_habitable_planet += radius_planet   
            
            if planet_temperature > maximum_temperature_not_habitable_planet:
                maximum_temperature_not_habitable_planet = planet_temperature
            elif planet_temperature < minimum_temperature_not_habitable_planet:
                minimum_temperature_not_habitable_planet = planet_temperature
            
            if radius_planet > maximum_radius_not_habitable_planet:
                maximum_radius_not_habitable_planet = radius_planet
            elif radius_planet < minimum_radius_not_habitable_planet:
                minimum_radius_not_habitable_planet = radius_planet
            
    print("___________________________________________________________________________________________\n")    
    print("Number of habitable planets detected:", number_of_habitable_planets,'\n')
    print("Avg temperature of habitable planets detected",total_temperature_habitable_planet/number_of_habitable_planets)
    print("Min temperature of habitable planet ",minimum_temperature_habitable_planet)
    print("Max temperature of habitable planet",maximum_temperature_habitable_planet,'\n')
            
            
    print("Avg radius of habitable planets detected",total_radius_habitable_planet/number_of_habitable_planets)
    print("Min radius of habitable planet ",minimum_radius_habitable_planet)
    print("Max radius of habitable planet",maximum_radius_habitable_planet,'\n')
#-----------------------------------------------------------------------------------------------------------------------------
    print("Number of not habitable planets detected:" , number_of_not_habitable_planets)
    print("Avg temperature of not habitable planets detected",total_temperature_not_habitable_planet/number_of_not_habitable_planets)
    print("Min temperature of not habitable planet",minimum_temperature_not_habitable_planet)
    print("Max temperature of not habitable planet",maximum_temperature_not_habitable_planet,'\n')
            
            
    print("Avg radius of not habitable planets detected",total_radius_not_habitable_planet/number_of_not_habitable_planets)
    print("Min radius of not habitable planet ",minimum_radius_not_habitable_planet)
    print("Max radius of not habitable planet",maximum_radius_not_habitable_planet)
    print("\n___________________________________________________________________________________________\n\n")
    
def dataset_preprocessing(habitable_not_habitable_planet):
    columns_to_keep=['Habitable', 'koi_period', 'koi_time0bk', 'koi_time0', 'koi_impact','koi_duration', 'koi_depth', 'koi_ror', 'koi_srho', 'koi_srho_err1',
                     'koi_srho_err2','koi_prad','koi_sma', 'koi_incl','koi_teq', 'koi_insol','koi_dor','koi_max_sngle_ev', 'koi_max_mult_ev', 'koi_model_snr', 
                     'koi_count','koi_num_transits', 'koi_tce_plnt_num', 'koi_bin_oedp_sig', 'koi_steff','koi_slogg','koi_smet','koi_srad','koi_smass','ra', 
                     'dec', 'koi_kepmag', 'koi_gmag','koi_fwm_stat_sig', 'koi_fwm_sra','koi_fwm_sdec','koi_fwm_srao','koi_fwm_sdeco','koi_fwm_prao',  
                     'koi_fwm_pdeco','koi_dicco_mra','koi_dicco_mdec', 'koi_dicco_msky','koi_dikco_mra','koi_dikco_mdec','koi_dikco_msky']
   
    
    habitable_not_habitable_planet=habitable_not_habitable_planet[columns_to_keep]
    habitable_not_habitable_planet=habitable_not_habitable_planet.fillna(0)
    
    prediction= habitable_not_habitable_planet["Habitable"]
    habitable_not_habitable_planet=habitable_not_habitable_planet.drop(["Habitable"],axis=1)
    
    scaler = StandardScaler()
    normalizzation=pd.DataFrame(scaler.fit_transform(habitable_not_habitable_planet))
    normalizzation.columns= habitable_not_habitable_planet.columns
   
    normalizzation["Habitable"]=prediction
    correlation = normalizzation.corr()
    correlation_target = abs(correlation["Habitable"])
    
    print(correlation_target)
    sns.heatmap(correlation, 
            xticklabels=correlation.columns.values,
            yticklabels=correlation.columns.values)
    print("\n___________________________________________________________________________________________\n\n")
    plt.show()
    print("\n___________________________________________________________________________________________\n")
    print("Model Training\n")
  
    data=normalizzation.drop(["Habitable"],axis=1)
    
    X_train,X_test,Y_train,Y_test=train_test_split(data,prediction)
    
    return  X_train,X_test,Y_train,Y_test

def get_PCA(X_train,X_test):
    
    PCATransformer = PCA(n_components = 14, whiten = True, svd_solver = 'auto')
    X_train_pca = PCATransformer.fit_transform(X_train)
    X_test_pca=PCATransformer.transform(X_test)
    
    return X_train_pca,X_test_pca,PCATransformer


def get_SVM(x_train, y_train):
    parameters = {'kernel': ('linear', 'rbf','poly','sigmoid'), 'C': [0.1, 1, 5,10], 'gamma': [0.01,0.1, 0.5, 1, 10]}
    params_estimator = GridSearchCV(svm.SVC(), parameters, cv = StratifiedKFold(10), refit=True, 
                                        verbose=1, scoring = 'accuracy')
    params_estimator.fit(x_train,y_train)

    print("\n___________________________________________________________________________________________\n\n")
    print("Best parameters:", params_estimator.best_params_, "\nAccuracy score with estimated hyperparameters: ", params_estimator.best_score_)
    sv=svm.SVC(kernel=params_estimator.best_params_['kernel'],C=params_estimator.best_params_['C'],gamma=params_estimator.best_params_['gamma'])
    sv.fit(x_train,y_train)
    return sv

def find_habitable_planets(pca_model,model):
    columns_to_keep=['koi_period', 'koi_time0bk', 'koi_time0', 'koi_impact', 'koi_duration', 'koi_depth', 'koi_ror', 'koi_srho', 'koi_srho_err1',
                     'koi_srho_err2','koi_prad','koi_sma', 'koi_incl','koi_teq', 'koi_insol','koi_dor','koi_max_sngle_ev', 'koi_max_mult_ev', 'koi_model_snr', 
                     'koi_count', 'koi_num_transits', 'koi_tce_plnt_num','koi_bin_oedp_sig', 'koi_steff','koi_slogg','koi_smet','koi_srad','koi_smass','ra', 
                     'dec', 'koi_kepmag', 'koi_gmag','koi_fwm_stat_sig', 'koi_fwm_sra','koi_fwm_sdec','koi_fwm_srao','koi_fwm_sdeco','koi_fwm_prao',  
                     'koi_fwm_pdeco','koi_dicco_mra','koi_dicco_mdec', 'koi_dicco_msky','koi_dikco_mra','koi_dikco_mdec','koi_dikco_msky']
    
    planets=pd.read_csv('data/all_planet_without_habitable_not_habitable.csv')
    planets=planets[columns_to_keep]
    planets=planets.fillna(0)
    scaler = StandardScaler()
    normalizzation=pd.DataFrame(scaler.fit_transform(planets))
    normalizzation.columns= planets.columns
    
    planets=pca_model.transform(normalizzation)
    
    new_habitable_planets=model.predict(planets)
    
    new_planet=0
    for i in new_habitable_planets:
        if i==1:
            new_planet+=1
        
    print("New habitable planets discovered:", new_planet)



def pipeline():
    
    dataset=load_dataframe()
    data_visualizzation(dataset)
    
    x_train,x_test,y_train,y_test=dataset_preprocessing(dataset)
    
    x_train,x_test,pca_model=get_PCA(x_train,x_test)
    model=get_SVM(x_train,y_train)
    print("\n___________________________________________________________________________________________\n\n")
    print("Accuracy test:")
    
    predict=model.predict(x_test)
    accuracy=accuracy_score(predict,y_test)
    print(accuracy)
    
    find_habitable_planets(pca_model,model)
    

if __name__ == "__main__":
    pipeline()
