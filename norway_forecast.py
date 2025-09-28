# ----------------------------
# Imports & config
# ----------------------------
from pathlib import Path
import sys, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

print("Running file:", __file__)
print("CWD:", os.getcwd())
print("Python:", sys.executable)

TARGET_NAME = "norway_new_car_sales_by_make.csv"
REQUIRED_COLS = {"Year", "Month", "Quantity", "Make"}

# ----------------------------
# Data functions
# ----------------------------
def import_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Read CSV, validate, build Period, and pivot."""
    if csv_path is None:
        csv_path = Path(__file__).with_name(TARGET_NAME)
    csv_path = Path(csv_path)

    print(f"\n➡️ Trying to read CSV from: {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at {csv_path}. Put '{TARGET_NAME}' next to this script or pass a path."
        )
    df_raw = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df_raw.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df_raw.columns)}")

    df_raw["Period"] = df_raw["Year"].astype(str) + "-" + df_raw["Month"].astype(str).str.zfill(2)
    df_pvt = pd.pivot_table(
        data=df_raw, values="Quantity", index="Make", columns="Period", aggfunc="sum", fill_value=0
    )
    return df_pvt

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Sort columns by month, add per-make totals and a TOTAL row."""
    df = df.reindex(sorted(df.columns, key=lambda s: pd.Period(s, "M")), axis=1)
    df["Total_Units"] = df.sum(axis=1)
    total_row = df.sum(numeric_only=True)
    total_row.name = "TOTAL"
    df = pd.concat([df, total_row.to_frame().T])
    return df

def datasets(df: pd.DataFrame, x_len=12, y_len=1, test_loops=12):
    """Slice wide time-series (rows=series, cols=time) into supervised X/Y windows."""
    D = df.values
    rows, periods = D.shape
    window = x_len + y_len
    loops = periods - window + 1
    if loops <= 0:
        raise ValueError(f"Need ≥ {window} columns (x_len+y_len); have {periods}.")

    blocks = [D[:, c:c+window] for c in range(loops)]
    train = np.vstack(blocks)                 # (rows*loops, window)
    X, Y = np.split(train, [-y_len], axis=1)  # (.., x_len), (.., y_len)

    total = X.shape[0]
    test_rows = min(rows * max(0, int(test_loops)), total)

    if test_rows > 0:
        X_train, X_test = np.split(X, [-test_rows], axis=0)
        Y_train, Y_test = np.split(Y, [-test_rows], axis=0)
    else:
        X_train, Y_train = X, Y
        X_test = D[:, -x_len:]
        Y_test = np.full((rows, y_len), np.nan)

    if y_len == 1:
        Y_train = Y_train.ravel()
        Y_test  = Y_test.ravel()
    return X_train, Y_train, X_test, Y_test

# ----------------------------
# KPI helpers (define BEFORE calling)
# ----------------------------
def _to_1d_float(y):
    a = np.asarray(y, dtype=object)
    if a.ndim > 1:
        a = a.reshape(-1)
    def _to_f(v):
        if v is None: return np.nan
        if isinstance(v, (list, tuple, np.ndarray)):
            if len(v) == 0: return np.nan
            v = v[0]
        try: return float(v)
        except: return np.nan
    return np.array([_to_f(v) for v in a], dtype=float)

def _pct(numer, denom):
    m = np.nanmean(denom)
    return np.nan if (not np.isfinite(m) or m == 0) else 100.0 * np.nanmean(numer) / m

def kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='Regression'):
    ytr  = _to_1d_float(Y_train)
    ytrp = _to_1d_float(Y_train_pred)
    yte  = _to_1d_float(Y_test)
    ytep = _to_1d_float(Y_test_pred)

    n_tr = min(len(ytr), len(ytrp))
    n_te = min(len(yte), len(ytep))
    ytr, ytrp = ytr[:n_tr], ytrp[:n_tr]
    yte, ytep = yte[:n_te], ytep[:n_te]

    df_kpi = pd.DataFrame(index=['Train','Test'], columns=['MAE','RMSE','Bias'], dtype=float)
    df_kpi.index.name = name

    err_tr = ytr - ytrp
    df_kpi.loc['Train','MAE']  = _pct(abs(err_tr), ytr)
    df_kpi.loc['Train','RMSE'] = (_pct(err_tr**2, ytr))**0.5
    df_kpi.loc['Train','Bias'] = _pct(err_tr, ytr)

    if n_te > 0:
        err_te = yte - ytep
        df_kpi.loc['Test','MAE']  = _pct(abs(err_te), yte)
        df_kpi.loc['Test','RMSE'] = (_pct(err_te**2, yte))**0.5
        df_kpi.loc['Test','Bias'] = _pct(err_te, yte)
    else:
        df_kpi.loc['Test'] = [np.nan, np.nan, np.nan]

    df_kpi = df_kpi.round(2)
    print("\n📊 KPI Results\n")
    print(df_kpi.to_string())
    return df_kpi

# ----------------------------
# MAIN (everything runs here)
# ----------------------------
if __name__ == "__main__":
    # 1) Import & preview CSV
    df = import_data()
    csv_path = Path(__file__).with_name(TARGET_NAME)
    print("\n📄 CSV preview (first 5 rows):")
    print(pd.read_csv(csv_path).head())

    # 2) Transform & preview
    df = transform(df)
    print("\n📊 Transformed preview (first 10 rows):")
    print(df.head(10))

    # 3) Write Excel
    out_path = Path(__file__).with_name("demand.xlsx")
    df.reset_index(names="Make").to_excel(out_path, index=False)
    print(f"\n✅ Wrote Excel: {out_path}")

    # 4) Prepare for ML (drop leakage)
    df_ml = df.copy()
    if "TOTAL" in df_ml.index:
        df_ml = df_ml.drop(index="TOTAL")
    if "Total_Units" in df_ml.columns:
        df_ml = df_ml.drop(columns=["Total_Units"])

    # 5) Build datasets
    X_train, Y_train, X_test, Y_test = datasets(df_ml, x_len=12, y_len=1, test_loops=12)
    print("\nData shapes ->",
          "X_train:", X_train.shape,
          "| Y_train:", Y_train.shape,
          "| X_test:", X_test.shape,
          "| Y_test:", Y_test.shape)

    # 6) Train & predict
    reg = LinearRegression()
    reg.fit(X_train, Y_train)
    Y_train_pred = reg.predict(X_train)
    Y_test_pred  = reg.predict(X_test)

    # 7) KPIs
    kpi_df = kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name="Regression")

# Future forecast 
    X_train, Y_train, X_test, Y_test = datasets(df, x_len=12, y_len=1, test_loops=0)
    reg=LinearRegression()
    reg=reg.fit(X_train,Y_train)
    forecast = pd.DataFrame(data=reg.predict(X_test), index=df.index)
    print(forecast.head())





                                         




    