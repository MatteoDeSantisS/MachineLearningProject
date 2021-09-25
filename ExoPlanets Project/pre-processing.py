import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import KNNImputer

#Caricamento del dataset
def load_dataset():
    koi_data = pd.read_csv('data/koi_data.csv')

#Pulizia dataset: colonne selezionate per una prima operazione di pulizia. Sono state tolte colonne utili solo ai fini di identificazione delle ennuple (rowid), 
#descrittive (kepoi_name, kepler_name, kepid), quelle che tengono traccia degli errori di misurazione, che riguardano la Stella, e altre hanno alta percentuale di valori mancanti > 5%
    col_todrop = [
             'Unnamed: 0', 
             'rowid', #Identificazione
             'kepid', #Identificazione
             'kepoi_name', #Descrizione
             'kepler_name', #Descrizione
             'koi_vet_stat', #Zero varianza
             'koi_vet_date', #Zero varianza
             'koi_pdisposition', #Descrizione
             'koi_score', #NaN = 19.3%
             'koi_disp_prov', #Zero varianza
             'koi_comment',  #Testuale
             'koi_period_err1', #Errore
             'koi_period_err2', #Errore
             'koi_time0bk', #Duplicato
             'koi_time0bk_err1', #Errore
             'koi_time0bk_err2', #Errore
             'koi_time0_err1', #Errore
             'koi_time0_err2', #Errore
             'koi_eccen', #Zero varianza
             'koi_eccen_err1', #Errore
             'koi_eccen_err2', #Errore
             'koi_longp', #Tutti NaN
             'koi_longp_err1', #Errore
             'koi_longp_err2', #Errore
             'koi_impact_err1', #Errore
             'koi_impact_err2', #Errore
             'koi_duration_err1', #Errore
             'koi_duration_err2', #Errore
             'koi_ingress', #Tutti NaN
             'koi_ingress_err1', #Tutti NaN
             'koi_ingress_err2', #Tutti NaN
             'koi_depth_err1', #Errore
             'koi_depth_err2', #Errore
             'koi_ror_err1', #Errore
             'koi_ror_err2', #Errore
             'koi_srho_err1', #Errore
             'koi_srho_err2', #Errore
             'koi_prad_err1', #Errore
             'koi_prad_err2', #Errore
             'koi_sma_err1', #Errore
             'koi_sma_err2', #Errore
             'koi_incl_err1', #Errore
             'koi_incl_err2', #Errore
             'koi_teq_err1', #Errore
             'koi_teq_err2', #Errore
             'koi_insol_err1', #Errore
             'koi_insol_err2', #Errore
             'koi_dor_err1', #Errore
             'koi_dor_err2', #Errore
             'koi_limbdark_mod', #Descrittiva
             'koi_ldm_coeff4', #Zero varianza
             'koi_ldm_coeff3', #Zero varianza
             'koi_parm_prov', #Descrittiva
             'koi_max_sngle_ev', #NaN = 14.5% 
             'koi_max_mult_ev', #NaN = 14.5%
             'koi_num_transits', #NaN = 14.5%
             'koi_tce_delivname', #Testuale
             'koi_quarters', #NaN = 14.5%
             'koi_bin_oedp_sig', #NaN = 19.3%
             'koi_trans_mod', #Testuale
             'koi_model_dof', #Tutti NaN
             'koi_model_chisq', #Tutti NaN
             'koi_datalink_dvr', #Testuale
             'koi_datalink_dvs', #Testuale
             'koi_steff_err1', #Errore
             'koi_steff_err2', #Errore
             'koi_slogg_err1', #Errore
             'koi_slogg_err2', #Errore
             'koi_smet_err1', #Errore
             'koi_smet_err2', #Errore
             'koi_srad_err1', #Errore
             'koi_srad_err2', #Errore
             'koi_smass_err1', #Errore
             'koi_smass_err2', #Errore
             'koi_sage', #Tutti NaN
             'koi_sage_err1', #Errore
             'koi_sage_err2', #Errore
             'koi_zmag', #NaN = 6.9%
             'koi_fwm_stat_sig', #NaN = 12.1%
             'koi_fwm_sra_err', #Errore
             'koi_fwm_sdec_err', #Errore
             'koi_fwm_srao_err', #Errore
             'koi_fwm_sdeco_err', #Errore #Errore
             'koi_fwm_prao', #NaN = 9.3%
             'koi_fwm_prao_err', #Errore
             'koi_fwm_pdeco', #NaN = 9.2%
             'koi_fwm_pdeco_err', #Errore
             'koi_dicco_mra', #NaN = 6.7%
             'koi_dicco_mra_err', #Errore
             'koi_dicco_mdec', #NaN = 6.7%
             'koi_dicco_mdec_err', #Errore
             'koi_dicco_msky',  #NaN = 6.7%
             'koi_dicco_msky_err', #Errore
             'koi_dikco_mra',  #NaN = 6.3%
             'koi_dikco_mra_err', #Errore
             'koi_dikco_mdec', #NaN = 6.3%
             'koi_dikco_mdec_err', #Errore
             'koi_dikco_msky', #NaN = 6.3%
             'koi_dikco_msky_err', #Errore
             'koi_sparprov', #Testuale
             'koi_fpflag_nt',
             'koi_fpflag_ss',
             'koi_fpflag_ec',
             'koi_steff', #Caratteristica della stella
             'koi_slogg', #Caratteristica della stella
             'koi_srad', #Caratteristica della stella
             'koi_smass', #Caratteristica della stella
             'koi_kepmag' #Caratteristica della stella
                ]

    koi_data = koi_data.drop(columns=col_todrop, axis=1)
    #One-Hot Encoding delle colonne categoriche
    koi_fittype = np.array(koi_data['koi_fittype'])
    encoder = preprocessing.OneHotEncoder(handle_unknown='ignore').fit(koi_fittype.reshape(-1,1))
    ftype = encoder.transform(koi_fittype.reshape(-1,1)).toarray()
    ftype = pd.DataFrame(ftype, columns=['LS','LS+MCMC','MCMC','none'])
    ftype = ftype.drop('none', axis=1)

    koi_data = koi_data.drop('koi_fittype', axis=1)
    koi_data['koi_fittype_ls'] = ftype['LS']
    koi_data['koi_fittype_lsmcmc'] = ftype['LS+MCMC']
    koi_data['koi_fittype_mcmc'] = ftype['MCMC']
    #Imputting dei valori mancanti attraverso l'utilizzo del KNN-Imputer
    koi_data, koi_y = koi_data.drop(columns='koi_disposition', axis=1), koi_data['koi_disposition']
    koi_columns = koi_data.columns

    koy_X = np.array(koi_data)

    n_points = 7232
    k_parameter = math.floor( math.sqrt(n_points)) #
    imputer = KNNImputer(n_neighbors=k_parameter, weights='uniform', metric='nan_euclidean')
    imputer.fit(koy_X)
    koy_Xtrans = imputer.transform(koy_X)
    #Matrice di correlazione
    koi_imputted = pd.DataFrame(koy_Xtrans, columns=koi_columns)
    correlation_def = koi_imputted.corr(method='pearson')
    correlation_def.style.background_gradient(cmap='coolwarm')
    col_todrop = ['koi_gmag','koi_rmag', 'koi_imag', 'koi_jmag', 'koi_hmag', 'koi_kmag', 'koi_fwm_sra', 'koi_fwm_sdeco', 'koi_fwm_sdec', 'koi_period', 'koi_ldm_coeff1', 'koi_ror']
    koi_imputted = koi_imputted.drop(columns=col_todrop, axis=1)
    #La colonna target è identificata con 3 tipi di label:
    # FALSE POSITIVE: usata per identificare tutti i tipi di oggetti Koi che sono confermati NON ESSERE ESOPIANETI
    # CONFIRMED: usata per identificare tutti i tipi di oggetti Koi che sono confermati ESSERE ESOPIANETI
    # CANDIDATE: usata per identificare gli oggetti che ancora sono in fase di analisi
    koi_imputted['koi_disposition'] = koi_y

    koi_train = koi_imputted.loc[(koi_imputted['koi_disposition'] == 'FALSE POSITIVE') | (koi_imputted['koi_disposition'] == 'CONFIRMED')]
    koi_test = koi_imputted.loc[koi_imputted['koi_disposition'] == 'CANDIDATE']
    koi_test = koi_test.drop('koi_disposition', axis=1)

    #Label encoding del target - 1 = FALSE POSITIVE, 0 = CONFIRMED
    koi_y = koi_train['koi_disposition']
    label_encoder = preprocessing.LabelEncoder().fit(koi_y)
    koi_ylabelled = label_encoder.transform(koi_y)
    koi_train = koi_train.drop(columns='koi_disposition', axis=1)
    koi_train['koi_disposition'] = koi_ylabelled
    label_encoder.classes_

    #Creazione datasets processati
    koi_train.to_csv('processed-data/koi_train.csv', index=False)
    koi_test.to_csv('processed-data/koi_test.csv', index=False)
    
if __name__ == "__main__":
    load_dataset()



