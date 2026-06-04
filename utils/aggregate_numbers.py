import os
import pandas as pd


dataset = 'eye'

result_dir = f"results/pred/{dataset}"

rows = []
for fname in sorted(os.listdir(result_dir)):
    if not fname.endswith(".csv"):
        continue
    method = fname[:-4]
    df = pd.read_csv(os.path.join(result_dir, fname), index_col=0)
    tstr_mean = df.loc["mean", "tstr_mean_mae"]
    tstr_std  = df.loc["std",  "tstr_mean_mae"]
    trtr_mean = df.loc["mean", "trtr_mean_mae"]
    trtr_std  = df.loc["std",  "trtr_mean_mae"]
    cross_corr = df.loc['mean', 'cross_corr_error']
    acf = df.loc['mean', 'acf_error']
    rows.append((method, tstr_mean, tstr_std, trtr_mean, trtr_std,acf,cross_corr))

col_w = max(len(r[0]) for r in rows)
print(f"{'Method':<{col_w}}  {'TSTR':>18}  {'TRTR':>18}  {'Deviation':>18} {'ACF':>18} {'Cross':>18} ")
print("-" * (col_w + 42))
for method, tstr_mean, tstr_std, trtr_mean, trtr_std,acf,cross_corr in rows:
    print(
        f"{method:<{col_w}}"
        f"  {tstr_mean:.4f} ± {tstr_std:.4f}"
        f"  {trtr_mean:.4f} ± {trtr_std:.4f}"
        f"  {(1-trtr_mean/tstr_mean)*100}"
        f"  {acf:.4f}"
        f"  {cross_corr:.4f}"
    )


print("================================")
print("DISC")
print("================================")

result_dir = f"results/disc/{dataset}"

rows = []
for fname in sorted(os.listdir(result_dir)):
    if not fname.endswith(".csv"):
        continue
    method = fname[:-4]
    df = pd.read_csv(os.path.join(result_dir, fname), index_col=0)
    disc_mean = df.loc["mean", "DS_acc_GRU_TCN"]
    disc_std  = df.loc["std",  "DS_acc_GRU_TCN"]
    lf_mean = df.loc["mean", "DS_acc_LR_RF"]
    lf_std  = df.loc["std",  "DS_acc_LR_RF"]
    
    rows.append((method, disc_mean, disc_std, lf_mean, lf_std))

col_w = max(len(r[0]) for r in rows)
print(f"{'Method':<{col_w}}  {'DISC':>18}  {'LF':>18}")
print("-" * (col_w + 42))
for method, disc_mean, disc_std, lf_mean, lf_std in rows:
    print(
        f"{method:<{col_w}}"
        f"  {disc_mean:.4f} ± {disc_std:.4f}"
        f"  {lf_mean:.4f} ± {lf_std:.4f}"

    )



print("================================")
print("PRIVACY")
print("================================")

result_dir = f"results/privacy/{dataset}"


rows = []
try:
    for fname in sorted(os.listdir(result_dir)):
        if not fname.endswith(".csv"):
            continue
        method = fname[:-4]
        df = pd.read_csv(os.path.join(result_dir, fname), index_col=0)
        ratio_mean = df.loc["mean", "nn_ratio_mean"]
        ratio_std  = df.loc["std",  "nn_ratio_mean"]
        mia_mean = df.loc["mean", "mia_auc"]
        mia_std  = df.loc["std",  "mia_auc"]
        mia_adv_mean = df.loc["mean", "mia_advantage"]
        mia_adv_std  = df.loc["std",  "mia_advantage"]

        
        rows.append((method, ratio_mean, ratio_std, mia_mean, mia_std, mia_adv_mean, mia_adv_std))

    col_w = max(len(r[0]) for r in rows)
    print(f"{'Method':<{col_w}}  {'r_nn':>18}  {'AUC_MIA':>18}  {'Adv_MIA':>18}")
    print("-" * (col_w + 42))
    for method, ratio_mean, ratio_std, mia_mean, mia_std, mia_adv_mean, mia_adv_std in rows:
        print(
            f"{method:<{col_w}}"
            f"  {ratio_mean:.4f} ± {ratio_std:.4f}"
            f"  {mia_mean:.4f} ± {mia_std:.4f}"
            f"  {mia_adv_mean:.4f} ± {mia_adv_std:.4f}"

        )
except:
    pass