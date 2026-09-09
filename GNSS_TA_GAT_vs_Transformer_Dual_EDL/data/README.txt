Place the preprocessed UrbanNav-HK CSV files here:

medium.csv
harsh.csv
deep.csv

Required features:
numSV,hDOP,vDOP,hAcc,vAcc,gSpeed,CN0_mean,CN0_std,CN0_gap,low_elev_ratio,PR_RMS,Fault_SVID_count

Label:
pos_error_2d  (recommended; fault if > 3 m)
or fault_label

Optional:
soft_fault_prob

Accepted column aliases (normalized by load_gnss_csv):
  CNO_mean / CNO_std / CNO_gap  -> CN0_mean / CN0_std / CN0_gap   (letter O vs zero)
  meanCN0 / stdCN0              -> CN0_mean / CN0_std
  fault_SVID_count              -> Fault_SVID_count
  2D_error / posError2D         -> pos_error_2d
  hard_label                    -> fault_label
  soft_fault_prob_surrogate     -> soft_fault_prob
                                  (only when cfg.useSurrogateSoftLabel = true)

Weak (soft) labels:
  cfg.useSurrogateSoftLabel = false (default)
      ySoft = sigmoid((pos_error_2d - 3.0) / cfg.softLabelTemperature)
  cfg.useSurrogateSoftLabel = true
      ySoft = soft_fault_prob_surrogate as shipped in the CSV. Note this column
      is saturated near 1.0 for ~68% of rows while the true fault rate is
      31-68%, and correlates only ~0.25 with pos_error_2d.
