# -*- coding: utf-8 -*-
# Encodage UTF-8 afin d’éviter les problèmes de caractères spéciaux.

# ===========================================================
# Import des bibliothèques nécessaires
# ===========================================================

import numpy as np
import pandas as pd
from numpy.linalg import eig
from scipy.linalg import qr, solve_triangular
from scipy.linalg import qr as scipy_qr
import scipy.linalg as la
import statsmodels.api as sm



# ===========================================================
# Fonctions exportées par le module
# ===========================================================

__all__ = [
    "multe",
    "multe_from_dataframe",
    "print_multe",
    "decomposition",
    "build_matrix",
]

# Cette liste indique quelles fonctions sont accessibles
# lorsque le module est importé avec :
# from module import *


# ===========================================================
# Utilitaires
# Ces fonctions servent principalement à réaliser
# des opérations numériques utilisées dans l'algorithme.
# ===========================================================


def ginv(A, tol=np.finfo(float).eps ** (3/5)):
    """
    Calcul d'une pseudo-inverse généralisée d'une matrice.

    Cette fonction est utile lorsque la matrice à inverser
    n'est pas de rang plein.

    La pseudo-inverse est obtenue à partir de la décomposition
    en valeurs propres.
    """

    A = np.asarray(A)  # conversion en tableau numpy

    # calcul des valeurs propres et vecteurs propres
    values, vectors = np.linalg.eig(A)

    # sélection des valeurs propres suffisamment grandes
    pos = values >= max(tol * np.max(values), 0)

    # si aucune valeur propre n'est retenue, la matrice
    # est considérée comme de rang nul
    if not np.any(pos):
        return {"inverse": np.zeros_like(A), "rank": 0}

    # matrice des vecteurs propres associés
    V = vectors[:, pos]

    # inversion des valeurs propres retenues
    Dinv = np.diag(1.0 / values[pos])

    # reconstruction de la pseudo-inverse
    inverse = V @ Dinv @ V.T

    return {"inverse": inverse, "rank": np.sum(pos)}


def qfp(A, b, tol=None):
    """
    Calcul d'une forme quadratique du type :

        b' A^{-1} b

    Cette quantité apparaît dans les statistiques
    de test de type Wald ou LM.
    """

    Ap = ginv(A, tol)

    # calcul de la forme quadratique
    qf = np.sum(b * (Ap["inverse"] @ b))

    return {"qf": qf, "rank": Ap["rank"]}


def scale_range(x):
    """
    Mise à l’échelle d’un vecteur dans l’intervalle [0,1].

    Cette transformation permet d'améliorer la stabilité
    numérique lors de l'estimation des modèles.
    """

    x = np.asarray(x)

    # amplitude de la variable
    d = np.max(x) - np.min(x)

    # si la variable n'est pas constante
    if d > 0:
        x = (x - np.min(x)) / d

    return x


def qr_pivoted(X, tol=1e-12):
    """
    Décomposition QR avec pivotement des colonnes.

    Cette décomposition permet d'identifier le rang
    numérique d'une matrice et de détecter les
    problèmes de colinéarité.
    """

    X = np.asarray(X, dtype=float)

    # vérification que X est bien une matrice
    if X.ndim != 2:
        raise ValueError("qr_pivoted attend une matrice 2D")

    # cas particulier : matrice vide
    if X.size == 0:
        return {"Q": np.empty((X.shape[0], 0)),
                "R": np.empty((0, 0)),
                "pivot": np.array([], dtype=int),
                "rank": 0}

    # décomposition QR avec pivotement
    Qm, Rm, pivot = scipy_qr(X, pivoting=True, mode='economic')

    # diagonale de la matrice R
    diag = np.abs(np.diag(Rm))

    # calcul du rang numérique
    if diag.size == 0:
        r = 0
    else:
        ref = diag[0] if diag[0] != 0 else diag.max(initial=0.0)
        thr = tol * (ref if ref != 0 else 1.0)
        r = int(np.sum(diag > thr))

    return {"Q": Qm, "R": Rm, "pivot": np.asarray(pivot, dtype=int), "rank": r}


def reg_if(res, X):
    """
    Calcul de la fonction d'influence associée
    à une estimation par moindres carrés.

    Cette fonction est utilisée pour construire
    les erreurs standards basées sur les fonctions
    d'influence.
    """

    res = np.asarray(res)

    # décomposition QR
    Qm, Rm, pivot = qr(X, pivoting=True, mode='economic')

    # rang de la matrice
    rank = np.linalg.matrix_rank(Rm)

    # matrices réduites
    Qr = Qm[:, :rank]
    Rr = Rm[:rank, :rank]

    # matrice résultat
    ret = np.zeros_like(X, dtype=float)

    # résolution du système triangulaire
    tmp = solve_triangular(Rr, (res[:, None] * Qr).T, lower=False)

    ret[:, pivot[:rank]] = tmp.T

    return ret

def wls_fit(X, y, wgt=None):
    """
    Estimation par moindres carrés pondérés (Weighted Least Squares)
    Cette fonction estime les coefficients du modèle :
    
          y = Xβ + u
    
    en présence de poids d’observation.
    La méthode consiste à transformer le modèle en multipliant
    chaque observation par la racine carrée du poids.
    """

    X = np.asarray(X, dtype=float)  # conversion en matrice numpy
    y = np.asarray(y, dtype=float)  # conversion en vecteur numpy

    # si y est un vecteur simple on le transforme en matrice colonne
    if y.ndim == 1:
        y = y[:, None]

    n = X.shape[0]  # nombre d'observations

    # construction des racines carrées des poids
    # si aucun poids n'est fourni, on utilise des poids égaux à 1
    ws = np.ones(n) if wgt is None else np.sqrt(np.asarray(wgt, dtype=float).reshape(-1))

    # transformation pondérée des variables explicatives
    Xw = ws[:, None] * X

    # transformation pondérée de la variable dépendante
    yw = ws[:, None] * y

    # calcul des matrices normales des MCO
    XtX = Xw.T @ Xw
    Xty = Xw.T @ yw

    # estimation des coefficients par résolution du système
    coef = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

    # calcul des valeurs ajustées
    fitted = X @ coef

    # calcul des résidus
    resid = y - fitted

    # décomposition QR de la matrice transformée
    # utile pour diagnostiquer les problèmes de rang
    qrx = qr_pivoted(Xw)

    # simplification si y est un scalaire
    if coef.shape[1] == 1:
        coef = coef[:, 0]
        resid = resid[:, 0]

    return {"coef": coef, "resid": resid, "qr": qrx}


def weighted_mean_cols(X, w):
    """
    Calcul de la moyenne pondérée colonne par colonne
    
    Cette fonction est utilisée pour calculer les moyennes
    pondérées des covariables dans l'échantillon.
    """
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)

    # formule de la moyenne pondérée
    return (w[:, None] * X).sum(axis=0) / w.sum()


def weighted_cov(X, w):
    """
    Calcul d'une matrice de covariance pondérée
    
    La normalisation correspond à celle utilisée dans
    la fonction cov.wt de R avec l'option method="ML".
    """

    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)

    # calcul de la moyenne pondérée
    m = weighted_mean_cols(X, w)

    # centrage des données
    Xc = X - m

    # matrice de covariance pondérée
    return (w[:, None] * Xc).T @ Xc / w.sum()


def Vhat(psi, cluster=None):
    """
    Estimation d'une matrice de variance basée
    sur les fonctions d'influence.
    
    Cette matrice correspond à :
    
            V = Σ ψ_i ψ_i'
    
    Elle est utilisée pour calculer les erreurs standards
    robustes.
    """

    psi = np.asarray(psi)

    # transformation en matrice si nécessaire
    if psi.ndim == 1:
        psi = psi.reshape(-1, 1)

    # cas avec clustering
    if cluster is not None:

        cluster = np.asarray(cluster)

        # niveaux de cluster
        levels = np.unique(cluster)

        nS = len(levels)

        # agrégation des scores par cluster
        psi = np.sqrt(nS / (nS - 1)) * np.apply_along_axis(
            lambda col: np.array([np.sum(col[cluster == g]) for g in levels]),
            0,
            np.asarray(psi)
        )

    # matrice de variance
    return psi.T @ psi


def sehat(psi, cluster=None):
    # -----------------------------------------------------------
    # Calcul des erreurs standards associées
    # à la matrice de variance estimée.
    # -----------------------------------------------------------

    V = Vhat(psi, cluster)

    # racine carrée des éléments diagonaux
    return np.sqrt(np.diag(V))


def multHessian(pis, Z, wgt=None):
    """
    Construction de la matrice Hessienne associée
    au modèle multinomial.
    
    Cette matrice intervient notamment dans les
    tests de spécification (tests LM et Wald).
    """

    pis = np.asarray(pis)
    Z = np.asarray(Z)

    n, L = Z.shape

    # gestion du cas binaire
    if pis.ndim == 1 or pis.shape[1] == 1:
        pis = np.column_stack((1 - pis, pis))

    # gestion des poids
    if wgt is None:
        wgt = np.ones(n)
    else:
        wgt = np.asarray(wgt).reshape(-1)
        if wgt.shape[0] != n:
            wgt = np.ones(n)

    # nombre d'équations du modèle multinomial
    K = pis.shape[1] - 1

    # initialisation de la matrice Hessienne
    He = np.zeros((K * L, K * L))

    # construction bloc par bloc
    for k in range(K):
        for j in range(K):

            # facteur issu de la dérivée seconde
            factor = pis[:, k + 1] * ((k == j) - pis[:, j + 1])

            # pondération des covariables
            WZ = Z * (wgt * factor)[:, None]

            block = Z.T @ WZ

            # insertion dans la matrice Hessienne globale
            He[L * k: L * (k + 1), L * j: L * (j + 1)] = block

    return He

# ===========================================================
# Multinomiale « pondérée » (approx. via réplication)
# ===========================================================

def multinom_fit_probs_and_coef(X_codes, Zm, wgt, replicate_weights=True, rep_scale=50):
    """
    Estimation d'un modèle logit multinomial.

    L'objectif est d'estimer les probabilités conditionnelles :

            P(X = k | Z)

    appelées scores de propension multinomiaux.

    Ces probabilités sont utilisées ensuite dans la décomposition
    pour construire les différents estimateurs (ATE, CW, etc.).
    """

    # conversion en tableaux numpy
    X_codes = np.asarray(X_codes)
    Zm = np.asarray(Zm, dtype=float)

    # transformation de la variable traitement en variable catégorielle
    y_cat = pd.Categorical(X_codes)

    # ---------------------------------------------------------
    # Gestion des poids
    #
    # statsmodels ne gère pas directement les poids de fréquence
    # pour MNLogit. On utilise donc une approximation qui consiste
    # à répliquer les observations proportionnellement à leur poids.
    # ---------------------------------------------------------

    if replicate_weights and (wgt is not None):

        w = np.asarray(wgt, float).reshape(-1)

        # on évite les poids nuls
        w = np.clip(w, 1e-12, None)

        # normalisation des poids autour de leur médiane
        wf = np.maximum(np.round(w / (np.median(w) + 1e-12) * rep_scale), 1).astype(int)

        # réplication des observations
        idx = np.repeat(np.arange(len(wf)), wf)

        # construction du nouvel échantillon
        y = pd.Categorical(X_codes[idx], categories=y_cat.categories)
        Zm_fit = Zm[idx, :]

    else:
        # estimation standard sans réplication
        y = y_cat
        Zm_fit = Zm

    # ---------------------------------------------------------
    # Estimation du modèle logit multinomial
    # ---------------------------------------------------------

    model = sm.MNLogit(y, Zm_fit)

    result = model.fit(method="newton", maxiter=2000, tol=1e-12, disp=False)

    # extraction des coefficients estimés
    th1 = result.params

    # conversion en tableau numpy si nécessaire
    if isinstance(th1, pd.DataFrame):
        th1 = th1.values

    # réorganisation des dimensions
    th1 = th1.T  # (L, K) -> (K, L)

    # ---------------------------------------------------------
    # Calcul des probabilités estimées sur l'échantillon
    # original (non répliqué)
    # ---------------------------------------------------------

    eta = Zm @ th1.T

    # ajout de la catégorie de base
    eta = np.hstack([np.zeros((eta.shape[0], 1)), eta])

    # stabilisation numérique
    eta -= eta.max(axis=1, keepdims=True)

    # transformation logit -> probabilités
    exp_eta = np.exp(eta)
    pis = exp_eta / exp_eta.sum(axis=1, keepdims=True)

    return pis, th1

# ==========================
# Décomposition principale 
# ==========================

def decomposition(Y, X, Zm, wgt=None, cluster=None, tol=1e-7, cw_uniform=False):

    # import local pour éviter dépendance globale inutile
    from scipy.stats import chi2
    import warnings

    # ---------------------------------------------------------
    # Mise en forme des données
    # ---------------------------------------------------------

    # conversion de la variable dépendante en vecteur numpy
    Y = np.asarray(Y, dtype=float).reshape(-1)

    n = Y.shape[0]  # nombre d'observations

    # ---------------------------------------------------------
    # Encodage de la variable traitement
    # ---------------------------------------------------------

    # si X est déjà une variable catégorielle pandas
    if isinstance(X, pd.Series) and pd.api.types.is_categorical_dtype(X):

        levels = list(X.dtype.categories)
        X_vals = X.to_numpy()

    else:

        # sinon on identifie les niveaux distincts
        X_vals = np.asarray(X)
        levels = pd.unique(X_vals)

    # construction du dictionnaire niveau -> code numérique
    level_to_code = {lev: i for i, lev in enumerate(levels)}

    # transformation de X en codes numériques
    X_codes = np.array([level_to_code[v] for v in X_vals], dtype=int)

    # nombre total de catégories
    C = len(levels)

    if C < 2:
        raise ValueError("X doit avoir au moins 2 niveaux.")

    # nombre de groupes hors baseline
    K = C - 1

    # conversion de la matrice des contrôles
    Zm = np.asarray(Zm, dtype=float)

    # ---------------------------------------------------------
    # Mise à l'échelle des variables de contrôle
    # ---------------------------------------------------------

    Zm_scaled = Zm.copy()

    # chaque colonne est normalisée dans l'intervalle [0,1]
    for j in range(Zm.shape[1]):
        Zm_scaled[:, j] = scale_range(Zm[:, j])

    # ---------------------------------------------------------
    # Suppression des variables sans variation
    # (sauf l'intercept)
    # ---------------------------------------------------------

    if Zm.shape[1] > 1:

        vars_ = np.var(Zm[:, 1:], axis=0, ddof=1)

        keep = np.r_[True, vars_ > 0]

        Zm = Zm[:, keep]
        Zm_scaled = Zm_scaled[:, keep]

    # matrice finale des contrôles utilisés
    Zs = Zm_scaled

    L = Zs.shape[1]

    # ---------------------------------------------------------
    # Initialisation des matrices de résultats
    # ---------------------------------------------------------

    # estimations principales
    estA = np.zeros((K, 5), dtype=float)  # PL, OWN, ATE, EW, CW

    # décomposition PL - autres estimateurs
    estB = np.zeros((K, 5), dtype=float)

    # erreurs standards
    seO  = np.zeros((K, 5), dtype=float)
    seP  = np.zeros((K, 5), dtype=float)
    seB  = np.zeros((K, 5), dtype=float)

    # ---------------------------------------------------------
    # Gestion des poids d'observation
    # ---------------------------------------------------------

    # si aucun poids n'est fourni on utilise des poids égaux à 1
    if wgt is None:
        wgt = np.ones(n, dtype=float)
    else:
        wgt = np.asarray(wgt, dtype=float).reshape(-1)

        # vérification de la cohérence des dimensions
        if wgt.shape[0] != n:
            raise ValueError("wgt doit avoir la même longueur que Y.")

    # racine carrée des poids (utilisée dans les régressions pondérées)
    ws = np.sqrt(wgt)

    # ---------------------------------------------------------
    # Construction de la matrice des indicatrices de traitement
    # ---------------------------------------------------------

    # matrice (n x C) contenant les variables indicatrices
    # pour chaque catégorie de traitement
    Xf = np.zeros((n, C), dtype=float)

    # remplissage de la matrice
    Xf[np.arange(n), X_codes] = 1.0

    # suppression de la catégorie de base
    Xf_no_base = Xf[:, 1:]


    # ======================================================
    # PL (Partially Linear)
    # ======================================================
    """
    Estimation d'un modèle partiellement linéaire :
    
            Y = Xβ + Zγ + u
    
    où X représente les indicatrices de traitement et Z
    les variables de contrôle.
    
    Les coefficients associés aux indicatrices de traitement
    correspondent aux estimations PL.
    """
    # construction de la matrice de régression
    X_rl = np.column_stack([Xf_no_base, Zs])

    # estimation du modèle par moindres carrés pondérés
    rl = wls_fit(X_rl, Y, wgt=wgt)

    # extraction des coefficients associés au traitement
    estA[:, 0] = rl["coef"][:K]

    # -----------------------------------------------------
    # Construction de Xtilde
    #
    # Cette étape correspond à la projection des variables
    # de traitement sur l'espace orthogonal aux contrôles Z.
    # -----------------------------------------------------

    tX = wls_fit(Zs, Xf_no_base, wgt=wgt)["resid"]

    # fonction d'influence associée à l'estimateur PL
    psi_beta = reg_if(ws * rl["resid"], ws[:, None] * tX)

    # calcul des erreurs standards
    seP[:, 0] = sehat(psi_beta, cluster=cluster)

    # ======================================================
    # ATE (Average Treatment Effect)
    # ======================================================
    """
    Cet estimateur repose sur un modèle avec interactions
    complètes entre traitement et covariables :
    
           Y = Σ_k 1(X=k) * Z * γ_k + u
    
    Les effets moyens sont ensuite calculés en prenant
    la moyenne pondérée des effets marginaux.
    """

    # construction des interactions traitement × contrôles
    X_ri = np.hstack([(Xf[:, c:c + 1] * Zs) for c in range(C)])

    # estimation du modèle
    ri = wls_fit(X_ri, Y, wgt=wgt)

    # coefficients associés à la catégorie de base
    b0 = ri["coef"][:L]

    # coefficients pour les autres catégories
    bk = ri["coef"][L:].reshape(K, L).T

    # différences de coefficients
    gam = bk - b0[:, None]

    # moyenne pondérée des covariables
    Zb = weighted_mean_cols(Zs, ws**2)

    # calcul de l'ATE
    estA[:, 2] = (Zb @ gam).reshape(-1)

        # fonctions d'influence des coefficients
    psi_al = reg_if(ws * ri["resid"], ws[:, None] * X_ri)

    # contribution de chaque groupe
    psi_ate_blocks = [psi_al[:, c * L:(c + 1) * L] @ Zb for c in range(C)]

    psi_ate_blocks = np.column_stack(psi_ate_blocks)

    # différence avec la catégorie de base
    psi_ate = psi_ate_blocks[:, 1:] - psi_ate_blocks[:, [0]]

    # erreurs standards oracle
    seO[:, 2] = sehat(psi_ate, cluster=cluster)

    # correction pour estimation des scores
    psi_po = psi_ate + (ws**2)[:, None] * ((Zs - Zb[None, :]) @ gam) / np.sum(ws**2)

    seP[:, 2] = sehat(psi_po, cluster=cluster)

    # variance de la différence
    seB[:, 2] = sehat(psi_beta - psi_po, cluster=cluster)

    # ======================================================
    # OWN et EW
    # ======================================================
    #
    # OWN : mesure la contribution propre des covariables
    #       à la différence entre groupes.
    #
    # EW : approche "one-at-a-time reweighting" qui compare
    #      chaque groupe au groupe de référence en utilisant
    #      un échantillon restreint.
    # ======================================================


    for k in range(K):

        # construction de la variable dépendante spécifique au groupe k
        Yk = Xf[:, [k + 1]] * Zs

        # matrice de régression incluant indicateurs de groupe et contrôles
        X_rk = np.column_stack([Xf_no_base, Zs])

        # estimation WLS du modèle auxiliaire
        rk = wls_fit(X_rk, Yk, wgt=wgt)

        # coefficients associés au groupe k
        deltak = rk["coef"][k, :]

        # sélection des coefficients non nuls
        mask = np.abs(deltak) > 0

        # calcul de l'estimateur OWN
        estA[k, 1] = float(np.sum(gam[mask, k] * deltak[mask])) if np.any(mask) else 0.0

        # ===========================================================
        # construction de la variable résiduelle orthogonale
        # ===========================================================

        # cas où il n'y a qu'un seul groupe comparé
        if K == 1:
            dtX = tX[:, 0]
        else:
            # régression des résidus du groupe k sur ceux des autres groupes
            X_dt = np.column_stack([tX[:, j] for j in range(K) if j != k])

            dt = wls_fit(X_dt, tX[:, k], wgt=wgt)

            # résidu orthogonal utilisé dans la décomposition
            dtX = dt["resid"]

        # ===========================================================
        # fonction d'influence associée à OWN
        # ===========================================================

        # différence des contributions entre groupe k et catégorie de base
        pp = psi_al[:, (k + 1) * L:(k + 2) * L] - psi_al[:, 0:L]

        # résidus du modèle auxiliaire
        rk_resid = rk["resid"]

        # dénominateur pour la normalisation
        denom = np.sum((ws**2) * (dtX**2))

        # terme d'ajustement
        adj = ((ws**2 * dtX)[:, None] * rk_resid) / denom if denom != 0 else np.zeros_like(rk_resid)

        # fonction d'influence finale pour OWN
        psi_ownk = (pp[:, mask] @ deltak[mask]) + (adj[:, mask] @ gam[mask, k]) if np.any(mask) else np.zeros(n)

        # erreur standard associée à OWN
        seP[k, 1] = sehat(psi_ownk, cluster=cluster)[0]

        # variance de la différence avec PL
        seB[k, 1] = sehat(psi_beta[:, k] - psi_ownk, cluster=cluster)[0]

        # ===========================================================
        # Estimateur EW 
        # ===========================================================

        # sélection des observations appartenant à la base ou au groupe k
        s = (X_codes == 0) | (X_codes == (k + 1))

        # indicateur du groupe k dans l'échantillon restreint
        indk = Xf[s, k + 1]

        # résidualisation de l'indicateur de traitement
        Xhat = wls_fit(Zs[s, :], indk, wgt=wgt[s])["resid"]

        # matrice de régression pour l'estimateur EW
        X_ew = np.column_stack([Xhat, Zs[s, :]])

        # estimation WLS
        rk2 = wls_fit(X_ew, Y[s], wgt=wgt[s])

        # coefficient associé au traitement
        estA[k, 3] = float(rk2["coef"][0])

        # poids utilisés dans la fonction d'influence
        psi_k = (ws[s]**2) * Xhat / np.sum((ws[s]**2) * (Xhat**2))

        # cluster restreint si clustering utilisé
        cl_s = None if cluster is None else np.asarray(cluster)[s]

        # erreur standard population
        seP[k, 3] = sehat(rk2["resid"] * psi_k, cluster=cl_s)[0]

        # erreur standard oracle
        seO[k, 3] = sehat(ri["resid"][s] * psi_k, cluster=cl_s)[0]

        # fonction d'influence pour comparer EW et PL
        psi_1 = psi_beta[:, k].copy()

        psi_1[s] -= rk2["resid"] * psi_k

        # erreur standard de la différence
        seB[k, 3] = sehat(psi_1, cluster=cluster)[0]


    # ===========================================================
    # Estimation des scores de propension multinomiaux
    # ===========================================================

    # estimation du modèle logit multinomial
    # renvoie les probabilités estimées et les coefficients
    pis, th1 = multinom_fit_probs_and_coef(
        X_codes, Zs, wgt,
        replicate_weights=True,
        rep_scale=50
    )

    # sauvegarde des probabilités avant modification
    pis_raw = pis.copy()

    # suppression des probabilités extrêmement petites
    # permet d'éviter les problèmes numériques
    pis[pis < np.max(pis) * 1e-6] = 0.0

    # ===========================================================
    # Construction du score utilisé dans les tests
    # ===========================================================

    def score(pis_):

        # contribution de chaque groupe au score
        blocks = [(Xf[:, kk] - pis_[:, kk])[:, None] * Zs for kk in range(1, C)]

        # pondération par les poids d'observation
        return (ws**2)[:, None] * np.hstack(blocks)

    # matrice des scores
    Sc = score(pis)

    # ===========================================================
    # Calcul de la matrice Hessienne
    # ===========================================================

    # Hessienne du log-vraisemblance du modèle multinomial
    He = multHessian(pis, Zs, wgt)

    # ===========================================================
    # Séparation des paramètres intercepts et pentes
    # ===========================================================

    # indices correspondant aux intercepts
    idx1 = np.arange(K) * L

    # indices de tous les paramètres
    all_idx = np.arange(K * L)

    # indices correspondant aux coefficients des contrôles
    idx2 = np.array([i for i in all_idx if i not in set(idx1)], dtype=int)

    # sous-matrice associée aux intercepts
    H11 = He[np.ix_(idx1, idx1)]

    # interaction intercepts / autres paramètres
    H12 = He[np.ix_(idx1, idx2)]

    # calcul du terme d'ajustement
    He1112 = np.linalg.solve(H11, H12)

    # score associé aux intercepts
    Sc1 = Sc[:, idx1]

    # score associé aux autres paramètres
    Sc2 = Sc[:, idx2]

    # ===========================================================
    # Estimation de la matrice de variance
    # ===========================================================


    # matrice de variance robuste du score ajusté
    Vu = Vhat(Sc2 - Sc1 @ He1112, cluster=cluster)

        # si les coefficients du modèle multinomial sont sous forme de vecteur,
    # on les transforme en matrice ligne
    if th1.ndim == 1:
        th1 = th1[None, :]

    # suppression des intercepts
    th1_no_int = th1[:, 1:]

    # extraction des sous-matrices de la Hessienne
    H21 = He[np.ix_(idx2, idx1)]
    H22 = He[np.ix_(idx2, idx2)]

    # calcul du vecteur utilisé dans le test de Wald
    th = (H22 - H21 @ He1112) @ th1_no_int.ravel(order="C")

    # =======================
    # LM: pis0 = moyennes pondérées de Xf
    # =======================

    # calcul des probabilités moyennes pondérées
    Xf_means = weighted_mean_cols(Xf, ws**2)

    # création d'une matrice constante de probabilités
    pis0 = np.tile(Xf_means[None, :], (n, 1))

    # calcul du score sous l'hypothèse nulle
    Scr = score(pis0)

    # séparation des composantes du score
    Scr1 = Scr[:, idx1]
    Scr2 = Scr[:, idx2]

    # calcul de la Hessienne sous l'hypothèse nulle
    Her = multHessian(pis0, Zs, wgt)

    Her11 = Her[np.ix_(idx1, idx1)]
    Her12 = Her[np.ix_(idx1, idx2)]

    # résolution du système linéaire
    Her1112 = np.linalg.solve(Her11, Her12)

    # matrice de variance du score
    Vr = Vhat(Scr2 - Scr1 @ Her1112, cluster=cluster)

    # symétrisation numérique des matrices de variance
    Vr = 0.5 * (Vr + Vr.T)
    Vu = 0.5 * (Vu + Vu.T)

    # fonction de calcul des statistiques de test
    def testcov(tol_):

        # statistique LM
        LM = qfp(Vr, np.sum(Scr2, axis=0), tol=tol_)

        # statistique de Wald
        Wa = qfp(Vu, th, tol=tol_)

        # extraction des statistiques
        LM_qf = float(np.real_if_close(LM["qf"]))
        LM_df = float(np.real_if_close(LM["rank"]))

        W_qf  = float(np.real_if_close(Wa["qf"]))
        W_df  = float(np.real_if_close(Wa["rank"]))

        from scipy.stats import chi2

        # calcul des p-values
        return {
            "W": W_qf, "W_df": W_df, "p_W": 1 - chi2.cdf(W_qf, df=W_df),
            "LM": LM_qf, "LM_df": LM_df, "p_LM": 1 - chi2.cdf(LM_qf, df=LM_df),
            "tol": tol_
        }

    # calcul des tests avec la tolérance choisie
    tests  = testcov(tol)

    # calcul des tests avec une tolérance plus stricte
    tests2 = testcov(tol * 1e-3)

    # vérification de la sensibilité numérique du test LM
    if max(abs(tests["LM"] - tests2["LM"]),
           abs(tests["LM_df"] - tests2["LM_df"]),
           abs(tests["p_LM"] - tests2["p_LM"])) > 0:

        warnings.warn(
            f"LM statistic depends on numerical tolerance.\n"
            f"At tol={tol}, LM={tests['LM']:.2f}, df={tests['LM_df']}\n"
            f"At tol={tol*1e-3}, LM={tests2['LM']:.2f}, df={tests2['LM_df']}"
        )
    # =======================
    # CW (common weights)
    # =======================

    # calcul du terme vpi utilisé dans la construction des poids efficaces
    # si cw_uniform est True on utilise des poids uniformes
    vpi = pis0 * (1 - pis0) if not cw_uniform else np.ones_like(pis0)

    # inverse des scores de propension estimés (avec borne inférieure pour éviter division par zéro)
    ipi = 1.0 / np.maximum(pis, 1e-10)

    # calcul du facteur lambda utilisé dans la construction des poids communs
    lam = 1.0 / np.sum(vpi / np.maximum(pis, 1e-10), axis=1)

    # suppression des valeurs numériquement trop petites
    lam[lam < np.max(lam) * 1e-6] = 0.0

    # construction des poids communs (common weights)
    cw = lam / np.sum(Xf * pis, axis=1)

    # vérification que l'échantillon pondéré n'est pas vide
    if np.sum(lam) > 0:

        # poids finaux utilisés dans la régression CW
        w_cw = cw * (ws**2)

        # Régression Y ~ Intercept + Xdummies ; ne garder que les K dummies pour les effets

        # construction de la matrice de régression avec intercept et indicateurs de groupe
        X_cw = np.column_stack([np.ones((n, 1)), Xf_no_base])

        # estimation WLS avec les poids CW
        ro = wls_fit(X_cw, Y, wgt=w_cw)

        # coefficients associés aux groupes (effets CW)
        estA[:, 4] = ro["coef"][1:]

        # terme utilisé dans la fonction d'influence oracle
        term = Xf[:, 1:] * ipi[:, 1:] - Xf[:, [0]] * ipi[:, [0]]

        # fonction d'influence oracle associée à CW
        psi_or = ((ws**2) * lam / np.sum((ws**2) * lam))[:, None] * term

        # erreur standard oracle
        seO[:, 4] = sehat(ri["resid"][:, None] * psi_or, cluster=cluster)

        # f1 + M/M0 sur Zs

        # fonction auxiliaire utilisée dans le calcul des matrices M et M0
        def f1(x_col):
            return np.sum((x_col * (ws**2) * cw * ro["resid"])[:, None] * Zs, axis=0)

        # construction de la matrice M0
        M0_cols = []
        for j in range(1, C):

            # variable auxiliaire utilisée dans le calcul de M0
            x_col = lam * (vpi[:, j] * ipi[:, j]) * Xf[:, 0]

            # contribution correspondante
            M0_cols.append(f1(x_col))

        # concaténation des blocs
        M0 = np.concatenate(M0_cols, axis=0)

        # initialisation de la matrice M
        M = np.zeros((K * L, K))

        # matrice identité de dimension K
        eyeK = np.eye(K, dtype=float)

        # boucle sur les groupes
        for kk in range(K):

            cols = []

            # boucle sur les catégories du traitement
            for j in range(1, C):

                # construction du terme utilisé dans M
                x_col = lam * (vpi[:, j] * ipi[:, j]) - eyeK[kk, j - 1]

                # interaction avec l'indicateur de groupe
                x_col = x_col * Xf[:, kk + 1]

                # contribution correspondante
                cols.append(f1(x_col))

            # assemblage du bloc correspondant
            M[:, kk] = np.concatenate(cols, axis=0)

        # Projection via He

        # décomposition QR de la Hessienne
        Qh, Rh, piv = la.qr(He, mode="economic", pivoting=True)

        # calcul du rang de la matrice
        rank = np.linalg.matrix_rank(He)

        # indices correspondant aux colonnes indépendantes
        idx = np.array(piv[:rank], dtype=int)

        # sous-matrice de la Hessienne
        He_ii = He[np.ix_(idx, idx)]

        # score restreint aux colonnes indépendantes
        Sc_iT = Sc[:, idx].T

        # résolution du système linéaire
        sol, *_ = la.lstsq(He_ii, Sc_iT, lapack_driver='gelsy')

        # projection de la matrice M
        MM = (M - M0[:, None])[idx, :]

        # terme d'ajustement pour la variance
        a = (MM.T @ sol).T / np.sum((ws**2) * lam)

        # erreur standard population pour CW
        seP[:, 4] = sehat(ro["resid"][:, None] * psi_or + a, cluster=cluster)

        # erreur standard de la différence avec PL
        seB[:, 4] = sehat(psi_beta - ro["resid"][:, None] * psi_or - a, cluster=cluster)

    else:

        # avertissement si l'échantillon efficace est vide
        warnings.warn("Sample for efficient common weights is empty.")
    
    # =======================
    # Table B (PL - autres)
    # =======================

    # calcul des différences entre l'estimateur PL et les autres estimateurs
    # chaque colonne correspond à PL - (OWN, ATE, EW, CW)
    estB[:, 1:] = estA[:, [0]] - estA[:, 1:]


    # =======================
    # DataFrames de sortie
    # =======================

    # noms des lignes correspondant aux groupes comparés à la catégorie de base
    rownames = [str(levels[i]) for i in range(1, C)]

    # noms des colonnes correspondant aux différents estimateurs
    colnames = ["PL", "OWN", "ATE", "EW", "CW"]


    # construction du tableau A contenant les estimations et leurs erreurs standards
    A_rows, A_index = [], []

    # boucle sur les groupes
    for k in range(K):

        # ajout des estimations des effets
        A_rows.append(estA[k, :])

        # ajout des erreurs standards population
        A_rows.append(seP[k, :])

        # ajout des erreurs standards oracle
        A_rows.append(seO[k, :])

        # index correspondant aux lignes ajoutées
        A_index += [rownames[k], "pop_se", "oracle_se"]

    # création du DataFrame final pour les estimations
    A_df = pd.DataFrame(np.vstack(A_rows), index=A_index, columns=colnames)


    # construction du tableau B contenant les différences entre estimateurs
    B_rows, B_index = [], []

    # boucle sur les groupes
    for k in range(K):

        # ajout des différences d'estimation
        B_rows.append(estB[k, :])

        # ajout des erreurs standards associées
        B_rows.append(seB[k, :])

        # index correspondant aux lignes
        B_index += [rownames[k], "pop_se"]

    # création du DataFrame final pour les différences
    B_df = pd.DataFrame(np.vstack(B_rows), index=B_index, columns=colnames)


    # =======================
    # SD des scores de propension (pondéré ML)
    # =======================

    # matrice de covariance pondérée des probabilités estimées
    Cw = weighted_cov(pis_raw, ws**2)     # ML

    # écart-types des scores de propension
    pscore_sd = np.sqrt(np.diag(Cw))


    # objet final retourné par la fonction decomposition
    return {"A": A_df, "B": B_df, "tests": tests, "pscore_sd": pscore_sd}



# ===========================================================
# build_matrix & helpers 
# ===========================================================

def build_matrix(Cm, S):

    # si Cm est un DataFrame pandas, conversion en tableau numpy
    if isinstance(Cm, pd.DataFrame):
        Cm_mat = Cm.to_numpy()
    else:
        # sinon conversion directe en tableau numpy
        Cm_mat = np.asarray(Cm, dtype=float)

    # nombre d'observations
    n = Cm_mat.shape[0]

    # s'assurer que S est une Series pandas
    if not isinstance(S, pd.Series):
        S = pd.Series(S)

    # conversion en variable catégorielle si nécessaire
    if not pd.api.types.is_categorical_dtype(S):
        S = S.astype("category")

    # cas où la variable catégorielle possède plusieurs niveaux
    if S.cat.categories.size > 1:

        # récupération des niveaux de la variable
        levels = list(S.cat.categories)

        # création de la constante (intercept)
        intercept = np.ones((n, 1), dtype=float)

        # création des variables indicatrices pour chaque niveau (sauf la base)
        dummies = [((S.values == lev).astype(float)).reshape(-1, 1)
                   for lev in levels[1:]]

        # matrice contenant intercept + indicateurs de groupe
        SZ = np.hstack([intercept] + dummies) if dummies else intercept

        # matrice finale contenant indicateurs + variables de contrôle
        Zm = np.hstack([SZ, Cm_mat])

    else:

        # cas où la variable n'a qu'un seul niveau
        intercept = np.ones((n, 1), dtype=float)

        # matrice finale avec intercept et contrôles
        Zm = np.hstack([intercept, Cm_mat])

    return Zm

     
# ===========================================================
# Wrappers multe
# ===========================================================

def multe_from_dataframe(model_df, y_col, treatment_name, weights_col=None,
                         cluster=None, tol=1e-7, cw_uniform=False):

    # copie du DataFrame afin d'éviter toute modification de l'objet original
    df = model_df.copy()

    # extraction de la variable dépendante et conversion en vecteur numpy
    Y = np.asarray(df[y_col], dtype=float).reshape(-1)

    # extraction de la variable de traitement
    X = df[treatment_name]

    # conversion en variable catégorielle si nécessaire
    if not pd.api.types.is_categorical_dtype(X):
        X = X.astype("category")

    # =======================
    # Gestion des poids
    # =======================

    # initialisation des poids
    wgt = None

    # si une colonne de poids est fournie, on l'extrait
    if weights_col is not None and weights_col in df.columns:
        wgt = np.asarray(df[weights_col], dtype=float).reshape(-1)

    # suppression des observations ayant un poids nul
    if wgt is not None and np.any(wgt == 0):

        # sélection des observations avec poids non nul
        ok = (wgt != 0)

        # filtrage des variables
        Y = Y[ok]
        X = X[ok].cat.remove_unused_categories()
        df = df.loc[ok].reset_index(drop=True)

        # filtrage de la variable de cluster si elle existe
        if cluster is not None:
            cluster = np.asarray(cluster)[ok]

        # filtrage des poids
        wgt = wgt[ok]

    # =======================
    # Construction de la matrice des contrôles Z
    # =======================

    # colonnes à exclure de la matrice des contrôles
    cols_drop = [y_col, treatment_name]

    # exclusion de la colonne de poids si elle existe
    if weights_col is not None and weights_col in df.columns:
        cols_drop.append(weights_col)

    # DataFrame contenant uniquement les variables de contrôle
    Z = df.drop(columns=cols_drop)

    # transformation des variables catégorielles en indicatrices (dummy variables)
    Z = pd.get_dummies(Z, drop_first=True)

    # ajout d'une constante (intercept) en première colonne
    Z.insert(0, "Intercept", 1.0)

    # conversion finale en matrice numpy
    Zm = Z.values.astype(float)

    # vérification qu'il existe au moins un contrôle en plus de l'intercept
    if Zm.shape[1] == 1:
        raise ValueError("There are no controls beyond the intercept")

    # appel de la fonction principale de décomposition
    r1 = decomposition(Y, X, Zm, wgt=wgt, cluster=cluster, tol=tol, cw_uniform=cw_uniform)

    # nombre d'observations
    n1 = Y.shape[0]

    # nombre de variables de contrôle (hors intercept)
    k1 = Zm.shape[1] - 1

    # construction du dictionnaire de sortie
    out = dict(

        # estimations des effets
        est_f=r1["A"],

        # estimations oracle (non utilisées ici)
        est_o=None,

        # tableau des comparaisons entre estimateurs
        cb_f=r1["B"],

        # comparaisons oracle
        cb_o=None,

        # nombre d'observations utilisées
        n_f=n1,

        # nombre d'observations oracle
        n_o=None,

        # nombre de contrôles
        k_f=k1,

        # nombre de contrôles oracle
        k_o=None,

        # résultats des tests statistiques
        t_f=r1["tests"],

        # tests oracle
        t_o=None,

        # écart-types des scores de propension
        pscore_sd_f=r1["pscore_sd"],

        # écart-types oracle
        pscore_sd_o=None,

        # variables utilisées dans l'estimation
        Y=Y,
        X=X,
        Zm=Zm,
        wgt=wgt
    )

    # ajout d'un attribut indiquant la classe de l'objet retourné
    out["__class__"] = "multe"

    # objet final retourné
    return out


def multe(r, treatment_name, cluster=None, tol=1e-7, cw_uniform=False,
          y_col=None, weights_col=None):
    """
    Wrapper pratique :
    - Si r est un objet statsmodels (avec .model.data.frame), on extrait le DataFrame
      (et l'endogène si y_col n'est pas fourni).
    - Si r est un DataFrame pandas, y_col doit être fourni.
    - Puis on appelle multe_from_dataframe(...).
    """

    # initialisation du DataFrame
    df = None

    # Cas 1 : objet statsmodels (ex: sm.OLS(...).fit())
    if hasattr(r, "model") and hasattr(r.model, "data") and hasattr(r.model.data, "frame"):

        # extraction du DataFrame utilisé dans l'estimation
        df = r.model.data.frame.copy()

        # si y_col n'est pas fourni, on tente de l'inférer automatiquement
        if y_col is None:
            try:
                y_col = r.model.endog_names
            except Exception:
                raise ValueError("Impossible d'inférer y_col depuis l'objet 'r'. Spécifie y_col='...'.")

    # Cas 2 : DataFrame pandas
    elif isinstance(r, pd.DataFrame):

        # copie du DataFrame
        df = r.copy()

        # la variable dépendante doit être spécifiée
        if y_col is None:
            raise ValueError("Quand 'r' est un DataFrame, fournir y_col='...'.")

    # Cas non reconnu
    else:
        raise ValueError("Argument 'r' non reconnu. Utilise un objet statsmodels ou un DataFrame pandas.")

    # appel final à la fonction principale utilisant le DataFrame
    return multe_from_dataframe(
        df, y_col=y_col, treatment_name=treatment_name,
        weights_col=weights_col, cluster=cluster, tol=tol, cw_uniform=cw_uniform
    )

# ===========================================================
# Affichage des résultats 
# ===========================================================

def print_multe(res, digits=6):
    """
    Fonction d'affichage des résultats produits par l'estimateur multe.
    
    Paramètres
    ----------
    res : dict
        Objet résultat retourné par la fonction multe().
        Il contient notamment :
        - est_f : estimations sur l'échantillon complet
        - est_o : estimations sur l'échantillon overlap (si présent)
        - t_f   : résultats des tests statistiques
        - pscore_sd_f : écart-type des scores de propension
    
    digits : int
        Nombre de décimales utilisées pour l'arrondi lors de l'affichage.
    """

    # -------------------------------------------------------
    # Fonction interne pour formater les DataFrames comme dans R
    # -------------------------------------------------------
    def _fmt_df_like_R(A_df):
        """
        Reformate les résultats pour reproduire la présentation du package R.
        
        Dans l'objet original, chaque traitement possède 3 lignes :
        - estimation
        - erreur standard population (pop_se)
        - erreur standard oracle (oracle_se)
        
        Ici on ne conserve que :
        - estimation
        - erreur standard (SE)
        """

        # Copie du DataFrame pour éviter toute modification de l'original
        A = A_df.copy()

        # Liste qui contiendra les nouvelles lignes formatées
        rows = []

        # Liste pour stocker les index associés
        idx = []

        # Nombre de traitements
        # Chaque traitement correspond à un bloc de 3 lignes
        K = A.shape[0] // 3

        # Boucle sur chaque bloc de traitement
        for k in range(K):

            # Extraction du bloc correspondant au traitement k
            block = A.iloc[k*3:(k+1)*3].copy()

            # Première ligne : estimation
            est_row = block.iloc[0:1]

            # Deuxième ligne : erreur standard population
            se_row  = block.iloc[1:2]

            # La ligne oracle_se (troisième ligne) est ignorée
            # car l'affichage "façon R" ne la montre pas

            # On conserve l'index original pour la ligne d'estimation
            est_row.index = [est_row.index[0]]

            # On remplace l'index de l'erreur standard par "SE"
            se_row.index  = ['SE']

            # Ajout des deux lignes à la liste finale
            rows.append(est_row)
            rows.append(se_row)

            # Mise à jour de la liste des index
            idx += [est_row.index[0], 'SE']

        # Concaténation verticale de toutes les lignes
        out = pd.concat(rows, axis=0)

        # Arrondi des valeurs numériques
        # On applique une fonction colonne par colonne
        out = out.apply(lambda c: c.map(lambda x: round(float(x), digits)))

        # Retour du DataFrame formaté
        return out


    # -------------------------------------------------------
    # Affichage des estimations sur l'échantillon complet
    # -------------------------------------------------------

    print("Estimates on full sample:")

    # Formatage des estimations
    A_f = _fmt_df_like_R(res['est_f'])

    # Affichage sous forme de tableau texte
    print(A_f.to_string())


    # -------------------------------------------------------
    # Affichage des estimations sur l'échantillon overlap
    # -------------------------------------------------------

    # Si l'objet contient des résultats sur l'overlap sample
    if res.get('est_o', None) is not None:

        print("\nEstimates on overlap sample:")

        # Formatage identique
        A_o = _fmt_df_like_R(res['est_o'])

        # Affichage
        print(A_o.to_string())


    # -------------------------------------------------------
    # Affichage des tests statistiques
    # -------------------------------------------------------

    # Extraction des tests réalisés sur l'échantillon complet
    tf = res['t_f']

    # Fonction interne pour formater les p-values
    def _fmt_p(p):

        # Si la p-value est exactement zéro
        if p == 0:
            return "0"

        # Sinon on utilise un format scientifique compact
        return f"{p:.{digits}g}"


    # Impression des résultats des tests
    print("\nP-values for null hypothesis of no propensity score variation:")

    # Deux tests sont affichés :
    # - Wald test
    # - LM test (Lagrange Multiplier)
    print(f"Wald test: {_fmt_p(tf['p_W'])}, LM test: {_fmt_p(tf['p_LM'])}")


    # -------------------------------------------------------
    # Affichage de la dispersion des scores de propension
    # -------------------------------------------------------

    # Extraction des écarts-types des scores de propension
    sd_f = res.get('pscore_sd_f', None)

    # Vérification que ces valeurs existent
    if sd_f is not None:

        print("\nSD(estimated propensity score), maximum over treatment arms:")

        # On affiche le maximum des écarts-types entre traitements
        print(f"Full sample: {round(float(np.max(sd_f)), digits)}")

        # Si l'échantillon overlap existe
        if res.get('pscore_sd_o', None) is not None:

            print(f", Overlap sample: {round(float(np.max(res['pscore_sd_o'])), digits)}")


# ===========================================================
# Point d'entrée du script
# ===========================================================

if __name__ == "__main__":

    # Ce bloc ne s'exécute que si le fichier est lancé directement
    # (et pas lorsqu'il est importé comme module)
    
    pass