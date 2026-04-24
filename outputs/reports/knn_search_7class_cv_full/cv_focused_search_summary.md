# CV-Focused KNN Search Summary

- evaluated candidates: `213`
- best feature_set: `lowfreq_all_plus_logmel`
- task mode: `7class_strict_cv`
- dataset mode: `balanced_train_cv`
- ranking mode: `cv_focused_stratified_first`
- best random CV weighted-F1: `0.7910`
- best random CV weighted-F1 std: `0.0042`
- best random CV accuracy: `0.7911`
- best random CV accuracy std: `0.0041`
- best random CV macro precision: `0.7960`
- best random CV macro-F1: `0.7910`
- best random CV macro-F1 std: `0.0042`
- best domain-aware CV accuracy: `0.1852` ± `0.0888`
- best domain-aware CV macro-F1: `0.1232` ± `0.0406`
- worst-domain accuracy: `0.0599`
- worst-domain macro-F1: `0.0701`
- primary CV scenario: `stratified_kfold`
- final ranking scenario: `stratified_kfold`
- split strategy: `StratifiedKFold`
- group column: `None`
- domain-aware primary: `False`

Top 10 candidates:

- `lowfreq_all_plus_logmel` | scaler=`standard` | reducer=`{'type': 'variance_threshold', 'threshold': 0.0}` | k=`5` | weights=`distance` | metric=`manhattan` | random_acc=`0.7911` | random_weighted_f1=`0.7910` | random_macro_f1=`0.7910` | domain_acc=`0.1852` | domain_macro_f1=`0.1232`
- `lowfreq_all_plus_waveform_spectral` | scaler=`robust` | reducer=`{'type': 'select_k_best', 'k': 64}` | k=`7` | weights=`uniform` | metric=`manhattan` | random_acc=`0.7893` | random_weighted_f1=`0.7894` | random_macro_f1=`0.7894` | domain_acc=`0.1494` | domain_macro_f1=`0.1118`
- `lowfreq_all_plus_mfcc` | scaler=`standard_l2` | reducer=`{'type': 'none'}` | k=`7` | weights=`distance` | metric=`minkowski` | random_acc=`0.7743` | random_weighted_f1=`0.7730` | random_macro_f1=`0.7730` | domain_acc=`0.1772` | domain_macro_f1=`0.1201`
- `lowfreq_all_plus_logmel` | scaler=`standard_l2` | reducer=`{'type': 'none'}` | k=`41` | weights=`distance` | metric=`manhattan` | random_acc=`0.7696` | random_weighted_f1=`0.7678` | random_macro_f1=`0.7678` | domain_acc=`0.1345` | domain_macro_f1=`0.0994`
- `lowfreq_all_plus_mfcc` | scaler=`power` | reducer=`{'type': 'pca_components', 'value': 44}` | k=`5` | weights=`distance` | metric=`minkowski` | random_acc=`0.7669` | random_weighted_f1=`0.7664` | random_macro_f1=`0.7664` | domain_acc=`0.1742` | domain_macro_f1=`0.1190`
- `lowfreq_all_plus_waveform_spectral` | scaler=`standard_l2` | reducer=`{'type': 'select_k_best', 'k': 44}` | k=`9` | weights=`uniform` | metric=`cosine` | random_acc=`0.7634` | random_weighted_f1=`0.7636` | random_macro_f1=`0.7636` | domain_acc=`0.1537` | domain_macro_f1=`0.1105`
- `lowfreq_all_plus_mfcc` | scaler=`minmax` | reducer=`{'type': 'pca_components', 'value': 44}` | k=`31` | weights=`uniform` | metric=`minkowski` | random_acc=`0.7647` | random_weighted_f1=`0.7632` | random_macro_f1=`0.7632` | domain_acc=`0.1462` | domain_macro_f1=`0.0954`
- `lowfreq_all_plus_logmel` | scaler=`standard` | reducer=`{'type': 'pca_components', 'value': 44}` | k=`31` | weights=`uniform` | metric=`minkowski` | random_acc=`0.7631` | random_weighted_f1=`0.7628` | random_macro_f1=`0.7628` | domain_acc=`0.1331` | domain_macro_f1=`0.0924`
- `lowfreq_all_plus_waveform_spectral` | scaler=`robust_l2` | reducer=`{'type': 'select_k_best', 'k': 64}` | k=`11` | weights=`distance` | metric=`euclidean` | random_acc=`0.7617` | random_weighted_f1=`0.7614` | random_macro_f1=`0.7614` | domain_acc=`0.1553` | domain_macro_f1=`0.1074`
- `notebook_lowfreq_meanmax_64` | scaler=`robust` | reducer=`{'type': 'pca_components', 'value': 128}` | k=`11` | weights=`uniform` | metric=`minkowski` | random_acc=`0.7586` | random_weighted_f1=`0.7590` | random_macro_f1=`0.7590` | domain_acc=`0.1521` | domain_macro_f1=`0.1011`
