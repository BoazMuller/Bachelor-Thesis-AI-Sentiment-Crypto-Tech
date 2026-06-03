import os
import pandas as pd
from arch.unitroot import DFGLS, PhillipsPerron
from statsmodels.tsa.stattools import adfuller, kpss

def main():
    full_path = os.path.join("results", "tables", "armax_egarchx", "armax_egarchx_full_dataset.csv")
    restricted_path = os.path.join("results", "tables", "armax_egarchx", "armax_egarchx_restricted_dataset.csv")
    
    if not os.path.exists(full_path) or not os.path.exists(restricted_path):
        legacy_path = os.path.join("results", "tables", "armax_egarchx", "armax_egarchx_dataset.csv")
        if not os.path.exists(legacy_path):
            print(f"Error: Datasets not found.")
            return
        df_full = pd.read_csv(legacy_path)
        df_restricted = df_full
    else:
        df_full = pd.read_csv(full_path)
        df_restricted = pd.read_csv(restricted_path)

    print(f"Loaded full dataset with {len(df_full)} rows and restricted dataset with {len(df_restricted)} rows.")

    variables_mapping = {
        "raw_ais": df_full,
        "expectation_adjusted_ais": df_restricted,
        "robust_expectation_adjusted_ais": df_restricted
    }

    for var, df in variables_mapping.items():
        if var not in df.columns:
            print(f"Warning: {var} not found in columns.")
            continue

        series = df[var].dropna()
        print("\n" + "="*50)
        print(f" Diagnostics for: {var} (N={len(series)})")
        print("="*50)

        # 1. ADF Test
        adf_res = adfuller(series, autolag="BIC")
        print(f"ADF Test (Null: Unit Root):")
        print(f"  Statistic: {adf_res[0]:.4f}")
        print(f"  p-value:   {adf_res[1]:.4f}")
        print(f"  Lags:      {adf_res[2]}")
        print(f"  Decision:  {'Stationary (Reject Null)' if adf_res[1] < 0.05 else 'Non-Stationary (Fail to Reject)'}")

        # 2. Phillips-Perron
        pp = PhillipsPerron(series)
        print(f"\nPhillips-Perron Test (Null: Unit Root):")
        print(f"  Statistic: {pp.stat:.4f}")
        print(f"  p-value:   {pp.pvalue:.4f}")
        print(f"  Lags:      {pp.lags}")
        print(f"  Decision:  {'Stationary (Reject Null)' if pp.pvalue < 0.05 else 'Non-Stationary (Fail to Reject)'}")

        # 3. DF-GLS Test
        dfgls = DFGLS(series)
        # Note: arch.unitroot.DFGLS outputs a regression object, we can pull the stat and p-value.
        # It typically tests with/without trend. By default, it includes constant.
        print(f"\nDF-GLS Test (Null: Unit Root):")
        print(f"  Statistic: {dfgls.stat:.4f}")
        print(f"  p-value:   {dfgls.pvalue:.4f}")
        print(f"  Lags:      {dfgls.lags}")
        print(f"  Decision:  {'Stationary (Reject Null)' if dfgls.pvalue < 0.05 else 'Non-Stationary (Fail to Reject)'}")

        # 4. KPSS Test
        kpss_res = kpss(series, regression="c", nlags="auto")
        print(f"\nKPSS Test (Null: Stationary):")
        print(f"  Statistic: {kpss_res[0]:.4f}")
        print(f"  p-value:   {kpss_res[1]:.4f}")
        print(f"  Lags:      {kpss_res[2]}")
        print(f"  Decision:  {'Non-Stationary (Reject Null)' if kpss_res[1] < 0.05 else 'Stationary (Fail to Reject)'}")

if __name__ == "__main__":
    main()
