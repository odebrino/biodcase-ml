# CV-Focused KNN Search Summary

- evaluated candidates: `30`
- best feature_set: `lowfreq_all_plus_waveform_spectral`
- ranking mode: `cv_focused_stratified_first`
- best random CV accuracy: `0.7583`
- best random CV accuracy std: `0.0133`
- best random CV macro-F1: `0.7583`
- best random CV macro-F1 std: `0.0132`
- best domain-aware CV accuracy: `0.1612` ± `0.0780`
- best domain-aware CV macro-F1: `0.1089` ± `0.0339`
- worst-domain accuracy: `0.0509`
- worst-domain macro-F1: `0.0621`
- primary CV scenario: `leave_one_domain_out`
- final ranking scenario: `stratified_kfold`
- split strategy: `LeaveOneGroupOut`
- group column: `dataset`
- domain-aware primary: `True`

Top 10 candidates:

- `lowfreq_all_plus_waveform_spectral` | scaler=`standard` | reducer=`{'type': 'select_k_best', 'k': 1024}` | k=`7` | weights=`uniform` | metric=`minkowski` | random_acc=`0.7583` | random_macro_f1=`0.7583` | domain_acc=`0.1612` | domain_macro_f1=`0.1089`
- `lowfreq_all_plus_waveform_spectral` | scaler=`none` | reducer=`{'type': 'select_k_best', 'k': 64}` | k=`21` | weights=`uniform` | metric=`euclidean` | random_acc=`0.7427` | random_macro_f1=`0.7431` | domain_acc=`0.1114` | domain_macro_f1=`0.0849`
- `hybrid_spectral_band16` | scaler=`robust` | reducer=`{'type': 'variance_threshold', 'threshold': 0.0}` | k=`3` | weights=`uniform` | metric=`euclidean` | random_acc=`0.7409` | random_macro_f1=`0.7410` | domain_acc=`0.1588` | domain_macro_f1=`0.1100`
- `spectral_stats` | scaler=`standard` | reducer=`{'type': 'variance_threshold', 'threshold': 0.0}` | k=`21` | weights=`distance` | metric=`cosine` | random_acc=`0.7409` | random_macro_f1=`0.7404` | domain_acc=`0.1698` | domain_macro_f1=`0.1164`
- `lowfreq_all_plus_waveform_spectral` | scaler=`standard_l2` | reducer=`{'type': 'variance_threshold', 'threshold': 0.0}` | k=`3` | weights=`uniform` | metric=`cosine` | random_acc=`0.7377` | random_macro_f1=`0.7377` | domain_acc=`0.1790` | domain_macro_f1=`0.1267`
- `handcrafted_stats` | scaler=`robust_l2` | reducer=`{'type': 'none'}` | k=`1` | weights=`uniform` | metric=`cosine` | random_acc=`0.7276` | random_macro_f1=`0.7277` | domain_acc=`0.1952` | domain_macro_f1=`0.1393`
- `lowfreq_all` | scaler=`none` | reducer=`{'type': 'none'}` | k=`21` | weights=`uniform` | metric=`manhattan` | random_acc=`0.7270` | random_macro_f1=`0.7270` | domain_acc=`0.1363` | domain_macro_f1=`0.0922`
- `waveform_spectral_stats` | scaler=`robust_l2` | reducer=`{'type': 'none'}` | k=`21` | weights=`uniform` | metric=`euclidean` | random_acc=`0.7270` | random_macro_f1=`0.7262` | domain_acc=`0.1506` | domain_macro_f1=`0.1000`
- `logmel_stats_32` | scaler=`robust` | reducer=`{'type': 'none'}` | k=`21` | weights=`distance` | metric=`cosine` | random_acc=`0.7270` | random_macro_f1=`0.7236` | domain_acc=`0.1282` | domain_macro_f1=`0.1059`
- `lowfreq_all_plus_waveform_spectral` | scaler=`robust_l2` | reducer=`{'type': 'none'}` | k=`1` | weights=`uniform` | metric=`cosine` | random_acc=`0.7006` | random_macro_f1=`0.7004` | domain_acc=`0.2076` | domain_macro_f1=`0.1450`
