#!/usr/bin/env python3
import pandas as pd

dataset = {
    "toothed": ["True", "True", "True", "False", "True", "True", "True", "True", "True", "False"],
    "hair": ["True", "True", "False", "True", "True", "True", "False", "False", "True", "False"],
    "breathes": ["True", "True", "True", "True", "True", "True", "False", "True", "True", "True"],
    "legs": ["True", "True", "False", "True", "True", 'True', "False", "False", "True", "True"],
    "species": ["Mammal", "Mammal", "Reptile", "Mammal", "Mammal", "Mammal", "Reptile", "Reptile", "Mammal", "Reptile"]
}

columns = ["toothed", "hair", "breathes", "legs", "species"]
featuresList = ["toothed", "hair", "breathes", "legs"]

dataFrame = pd.DataFrame(data=dataset, columns=columns)
features = dataFrame[featuresList]
target = dataFrame['species']
print(dataFrame)
