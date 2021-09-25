import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle


habitable_planet=pd.read_csv('data/habitable_planets_detailed_list.csv')
not_habitable_planet=pd.read_csv('data/non_habitable_planets_confirmed_detailed_list.csv')
habitable_planet.insert(1, "Habitable", 1, True)
not_habitable_planet.insert(1, "Habitable", 0, True)

frames=[habitable_planet,not_habitable_planet]
habitable_not_habitable_planet=pd.concat(frames)
habitable_not_habitable_planet=shuffle(habitable_not_habitable_planet)
habitable_not_habitable_planet.to_csv("data/habitable_not_habitable_planet.csv",index=False)
rowid=habitable_not_habitable_planet["rowid"]
all_planet=pd.read_csv("data/cumulative_new_data.csv")
all_planet=all_planet.drop(rowid,axis=0)
all_planet.to_csv("data/all_planet_without_habitable_not_habitable.csv",index=False)