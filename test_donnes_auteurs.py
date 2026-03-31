
import pandas as pd
from multe import multe
from multe import print_multe
import pyreadr

# Charger la base 
res_r = pyreadr.read_r(r"fl.rda")
df = res_r["fl"]

# Restreindre aux 3 races utilisées dans l’exemple R
df = df[df["race"].isin(["White", "Black", "Hispanic"])].copy()

# Restreindre le DataFrame aux colonnes EXACTES du modèle R
keep_cols = ["std_iq_24", "race", "age_24", "female", "W2C0"]
df = df[keep_cols].copy()


df["race"]   = pd.Categorical(df["race"], categories=["White", "Black", "Hispanic"])
df["age_24"] = pd.Categorical(df["age_24"])
df["female"] = df["female"].astype(float)   # numérique
# (W2C0 restera numérique pour servir de poids)

# Appel de ta fonction multe 
from multe import multe
res = multe(
    r=df,                    
    treatment_name="race",   
    y_col="std_iq_24",       
    weights_col="W2C0",      
    cluster=None,            
    tol=1e-7,
    cw_uniform=False
)

# Affichage
print_multe(res, digits=6)

print( res["cb_f"])

