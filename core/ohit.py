import numpy as np
import pandas as pd
import statsmodels.api as sm


def oga_hdic(X, y, Kn=None, c1=5, HDIC_Type="HDBIC", c2=2, c3=2.01, intercept=True):
    """
    Python translation of the R function for OGA + HDIC + trimming.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray, shape (n, p)
        Feature matrix.
    y : pd.Series or np.ndarray, shape (n,)
        Response vector.
    Kn : int or None
        Max number of OGA steps. If None, uses floor(c1*sqrt(n/log(p))) clipped to [1, p].
    c1, c2, c3 : floats
        Tuning constants (same meaning as in R code).
    HDIC_Type : {"HDAIC", "HDBIC", "HDHQ"}
        Which HD information criterion to use.
    intercept : bool
        Whether to include intercept in the *final* OLS fits (HDIC model and Trim model).
        (Trimming用的是 centered 資料且不含截距，與原 R 程式相同)

    Returns
    -------
    result : dict
        {
          "n", "p", "Kn",
          "J_OGA"        : list of selected column indices (0-based) in OGA order,
          "HDIC"         : np.ndarray of HDIC values for k=1..K,
          "J_HDIC"       : sorted list of indices at k = argmin(HDIC),
          "J_Trim"       : sorted list of indices after trimming,
          "J_OGA_names"  : list of column names in OGA order,
          "J_HDIC_names" : list of column names for HDIC set,
          "J_Trim_names" : list of column names for Trim set,
          "betahat_HDIC" : statsmodels RegressionResultsWrapper,
          "betahat_Trim" : statsmodels RegressionResultsWrapper
        }
    """
    # --- Input checking & normalization of types ---
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=[f"x{j+1}" for j in range(X.shape[1])])
    elif isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        raise TypeError("X should be a numpy array or pandas DataFrame")

    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_vec = np.asarray(y).reshape(-1)
    elif isinstance(y, np.ndarray):
        if y.ndim == 1:
            y_vec = y
        elif y.ndim == 2 and y.shape[1] == 1:
            y_vec = y.reshape(-1)
        else:
            raise ValueError("y should be a 1-D vector")
    else:
        raise TypeError("y should be a numpy array or pandas Series")

    n, p = X_df.shape
    if y_vec.shape[0] != n:
        raise ValueError("the number of observations in y is not equal to the number of rows of X")
    if n == 1:
        raise ValueError("the sample size should be greater than 1")

    # --- Determine K (max steps) ---
    if Kn is None:
        # floor(c1 * sqrt(n/log(p))) clipped to [1, p]
        if p <= 1:
            K = 1
        else:
            K = int(np.floor(c1 * np.sqrt(n / np.log(p))))
            K = max(1, min(K, p))
    else:
        if (Kn < 1) or (Kn > p) or (int(Kn) != Kn):
            raise ValueError(f"Kn should be a positive integer between 1 and {p}")
        K = int(Kn)

    # --- Center y and X (demean) ---
    dy = y_vec - np.mean(y_vec)
    dX = X_df - X_df.mean(axis=0)
    dX_np = dX.to_numpy()
    u = dy.reshape(-1, 1)  # residual vector as column

    # --- Precompute norms of centered columns ---
    xnorms = np.sqrt(np.sum(dX_np**2, axis=0))  # shape (p,)
    # Avoid division by zero for constant columns
    zero_norm = xnorms == 0

    # --- OGA selection ---
    Jhat = np.zeros(K, dtype=int)  # selected indices in order
    sigma2hat = np.zeros(K, dtype=float)  # residual variance after each step
    XJhat_orth = np.zeros((n, K))  # orthonormalized selected regressors (columns)

    # step 1
    # aSSE = |u' dX| / ||dX||_2
    # dX^T u -> shape (p,1)
    aSSE = np.abs((dX_np.T @ u).reshape(-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        aSSE = np.where(zero_norm, -np.inf, aSSE / xnorms)
    Jhat[0] = int(np.argmax(aSSE))

    col = dX_np[:, Jhat[0]]
    denom = np.linalg.norm(col)
    if denom == 0:
        # Should not happen if handled above, but keep safeguard
        XJhat_orth[:, 0] = 0.0
    else:
        XJhat_orth[:, 0] = col / denom

    # Update residual u = (I - e1 e1^T) u
    e = XJhat_orth[:, [0]]
    u = u - e @ (e.T @ u)
    sigma2hat[0] = float(np.mean(u**2))

    # steps 2..K
    for k in range(1, K):
        aSSE = np.abs((dX_np.T @ u).reshape(-1))
        with np.errstate(divide="ignore", invalid="ignore"):
            aSSE = np.where(zero_norm, -np.inf, aSSE / xnorms)
        # zero out already-selected
        aSSE[Jhat[:k]] = -np.inf
        Jhat[k] = int(np.argmax(aSSE))

        # regress new column on existing orthonormal basis & take residual (Gram-Schmidt)
        v = dX_np[:, [Jhat[k]]]  # shape (n,1)
        if k > 0:
            E = XJhat_orth[:, :k]  # (n,k)
            proj = E @ (E.T @ v)  # projection onto span(E)
            rq = v - proj
        else:
            rq = v
        denom = float(np.linalg.norm(rq))
        if denom == 0:
            XJhat_orth[:, k] = 0.0
        else:
            XJhat_orth[:, k] = rq.reshape(-1) / denom

        # update residual
        e = XJhat_orth[:, [k]]
        u = u - e @ (e.T @ u)
        sigma2hat[k] = float(np.mean(u**2))

    # --- HDIC choice ---
    HDIC_Type = HDIC_Type.upper()
    if HDIC_Type not in {"HDAIC", "HDBIC", "HDHQ"}:
        raise ValueError('HDIC_Type should be "HDAIC", "HDBIC" or "HDHQ"')
    if HDIC_Type == "HDAIC":
        omega_n = c2
    elif HDIC_Type == "HDBIC":
        omega_n = np.log(n)
    else:  # "HDHQ"
        omega_n = c3 * np.log(np.log(n))

    # hdic[k-1] = n*log(sigma2hat[k-1]) + k * omega_n * log(p)
    hdic = n * np.log(sigma2hat) + (np.arange(1, K + 1)) * omega_n * np.log(p)
    kn_hat = int(np.argmin(hdic)) + 1  # number of selected variables at optimum (1..K)
    benchmark = hdic[kn_hat - 1]

    J_HDIC_unsorted = Jhat[:kn_hat].copy()
    J_HDIC = sorted(J_HDIC_unsorted.tolist())

    # --- Trimming step (on centered data, no intercept), same as R ---
    J_Trim = J_HDIC_unsorted.copy()
    if kn_hat > 1:
        trim_pos = np.zeros(kn_hat, dtype=int)
        for l in range(kn_hat - 1):  # try dropping each in order except last
            JDrop1 = np.delete(J_Trim, l)
            # dy ~ dX[:, JDrop1] without intercept
            X_trim = dX_np[:, JDrop1]
            # Use OLS via statsmodels without constant
            # If X_trim is empty (shouldn't happen since kn_hat>1 and we drop one), skip
            model = sm.OLS(dy, X_trim, hasconst=False)
            res = model.fit()
            uDrop1 = res.resid
            HDICDrop1 = n * np.log(np.mean(uDrop1**2)) + (kn_hat - 1) * omega_n * np.log(p)
            if HDICDrop1 > benchmark:
                trim_pos[l] = 1
        trim_pos[kn_hat - 1] = 1  # always keep the last (as in R code)
        J_Trim = J_Trim[np.where(trim_pos == 1)[0]]

    J_Trim_sorted = sorted(J_Trim.tolist())

    # --- Final OLS fits on original (un-centered) X with/without intercept ---
    # Build design matrices with original columns
    X_HDIC_df = X_df.iloc[:, J_HDIC].copy()
    X_Trim_df = X_df.iloc[:, J_Trim_sorted].copy()

    if intercept:
        X_HDIC_design = sm.add_constant(X_HDIC_df, has_constant="add")
        X_Trim_design = sm.add_constant(X_Trim_df, has_constant="add")
    else:
        X_HDIC_design = X_HDIC_df
        X_Trim_design = X_Trim_df

    model_HDIC = sm.OLS(y_vec, X_HDIC_design)
    fit_HDIC = model_HDIC.fit()

    model_Trim = sm.OLS(y_vec, X_Trim_design)
    fit_Trim = model_Trim.fit()

    # --- Package results ---
    result = {
        "n": n,
        "p": p,
        "Kn": K,
        "J_OGA": Jhat.tolist(),
        "HDIC": hdic.copy(),
        "J_HDIC": J_HDIC,
        "J_Trim": J_Trim_sorted,
        "J_OGA_names": [X_df.columns[j] for j in Jhat],
        "J_HDIC_names": [X_df.columns[j] for j in J_HDIC],
        "J_Trim_names": [X_df.columns[j] for j in J_Trim_sorted],
        "betahat_HDIC": fit_HDIC,  # statsmodels RegressionResultsWrapper
        "betahat_Trim": fit_Trim,
    }
    return result
