import scanpy as sc
import os
import glob
ad_dir="/home/gangcai/projects/human_aging_clock/scageclock/datasets_cross_validation/train_val_1M_Age_Balanced/train_val/Fold1/"

i = 0
ad_files = glob.glob(f"{ad_dir}/*.h5ad")
for ad_file in ad_files:
    i += 1
    if i > 1:
        break
    filename = ad_file.split("/")[-1]
    print(filename)
    ad1 = sc.read_h5ad(ad_file)
    ad1_s = ad1[:500]
    ad1_s.write_h5ad(f"Pytest_{filename}")

