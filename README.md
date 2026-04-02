# multe-python-implementation
Python implementation for analyzing contamination bias in multiple treatment regressions, inspired by the R package multe.



## Description

Ce dépôt propose une implémentation Python de méthodes récentes permettant d’analyser le **biais de contamination** dans les régressions comportant plusieurs traitements mutuellement exclusifs.

Dans ce type de modèle, les coefficients estimés peuvent refléter non seulement l’effet du traitement d’intérêt, mais également une combinaison des effets des autres traitements. Ce phénomène complique l’interprétation causale des résultats.

Ce projet a été développé dans le cadre d’un mémoire de Master et s’inspire du package R *multe* ainsi que des travaux récents en économétrie (Goldsmith-Pinkham et al., 2024).

---

## Objectifs

* Implémenter les méthodes de décomposition du biais de contamination
* Comparer différents estimateurs d’effets de traitement
* Reproduire les résultats du package R *multe* en Python
* Fournir un outil réutilisable pour l’analyse empirique

---

##  Méthodologie

On considère un modèle de régression du type :

Y = Xβ + Zγ + u

où :

* **X** représente les indicatrices de traitement
* **Z** les variables de contrôle

L’estimateur partiellement linéaire (PL) peut être décomposé en :

* un **effet propre (OWN)**
* un **biais de contamination (CB)**

Cette décomposition permet d’identifier si les coefficients estimés reflètent réellement un effet causal ou s’ils sont biaisés par la présence d’autres traitements.

---

## Estimateurs implémentés

Le projet inclut les estimateurs suivants :

* **PL** : Partially Linear estimator
* **OWN** : effet propre du traitement
* **ATE** : Average Treatment Effect
* **EW** : Efficient weighting
* **CW** : Common weighting

Fonctionnalités supplémentaires :

* Tests de Wald et LM
* Analyse de la variation des scores de propension
* Gestion des poids (WLS)
* Diagnostics d’overlap

---

## Structure du projet

```
multe-python/
│
├── multe.py                  # Implémentation principale
├── test_donnes_auteurs.py    # Script de reproduction des résultats
├── fl.rda                    # Données utilisées
├── README.md                 # Documentation
```

---

## Technologies utilisées

* Python 3
* NumPy
* pandas
* statsmodels
* pyreadr

---

## Exemple d’utilisation

```python
import pyreadr
from multe import multe, print_multe

# Charger les données
df = pyreadr.read_r("fl.rda")["fl"]

# Estimation
res = multe(
    r=df,
    treatment_name="race",
    y_col="std_iq_24",
    weights_col="W2C0"
)

# Affichage des résultats
print_multe(res)
```

Pour une reproduction complète, voir le fichier :
`test_donnes_auteurs.py`

---

## Données

Le fichier `fl.rda` correspond aux données utilisées dans l’article :

Fryer, R. G. & Levitt, S. D. (2013)
*Testing for Racial Differences in the Mental Ability of Young Children*

Ces données sont utilisées pour illustrer et valider les méthodes implémentées.

---

##  Validation

L’implémentation Python a été validée par :

* comparaison avec les résultats du package R *multe*
* reproduction d’applications empiriques
* tests de robustesse sur différentes spécifications

Les résultats montrent une forte cohérence entre les implémentations Python et R.

---

##  Auteur

**Abdoulaye Sagna**
Master en MIASHS (Parcours Buisness et data analyst)

---

##  Contexte académique

Mémoire de Master :

**Développement d’une bibliothèque Python pour l’analyse du biais de contamination dans les régressions à traitements multiples**

---

##  Perspectives d’amélioration
* Optimisation des performances
* Transformation en package Python installable (`pip install`)
* Ajout de notebooks explicatifs


---

##  Licence

Ce projet est destiné à un usage académique et de recherche.

---


