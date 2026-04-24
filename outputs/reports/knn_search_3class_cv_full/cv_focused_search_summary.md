# CV-Focused KNN Search Summary

- evaluated candidates: `213`
- best feature_set: `lowfreq_all_plus_waveform_spectral`
- task mode: `3class_notebook_cv`
- dataset mode: `balanced_train_cv`
- ranking mode: `cv_focused_stratified_first`
- best random CV weighted-F1: `0.8321`
- best random CV weighted-F1 std: `0.0082`
- best random CV accuracy: `0.8317`
- best random CV accuracy std: `0.0084`
- best random CV macro precision: `0.8330`
- best random CV macro-F1: `0.8321`
- best random CV macro-F1 std: `0.0082`
- best domain-aware CV accuracy: `0.4202` ± `0.4094`
- best domain-aware CV macro-F1: `0.1985` ± `0.1759`
- worst-domain accuracy: `0.0589`
- worst-domain macro-F1: `0.0371`
- primary CV scenario: `stratified_kfold`
- final ranking scenario: `stratified_kfold`
- split strategy: `StratifiedKFold`
- group column: `None`
- domain-aware primary: `False`

Top 10 candidates:

- `lowfreq_all_plus_waveform_spectral` | scaler=`robust` | reducer=`{'type': 'select_k_best', 'k': 64}` | k=`7` | weights=`uniform` | metric=`manhattan` | random_acc=`0.8317` | random_weighted_f1=`0.8321` | random_macro_f1=`0.8321` | domain_acc=`0.4202` | domain_macro_f1=`0.1985`
- `waveform_spectral_stats` | scaler=`minmax` | reducer=`{'type': 'pca_components', 'value': 44}` | k=`41` | weights=`distance` | metric=`minkowski` | random_acc=`0.8200` | random_weighted_f1=`0.8212` | random_macro_f1=`0.8212` | domain_acc=`0.2424` | domain_macro_f1=`0.1223`
- `waveform_spectral_stats` | scaler=`minmax` | reducer=`{'type': 'pca_components', 'value': 256}` | k=`11` | weights=`uniform` | metric=`chebyshev` | random_acc=`0.8137` | random_weighted_f1=`0.8151` | random_macro_f1=`0.8151` | domain_acc=`0.3835` | domain_macro_f1=`0.2015`
- `waveform_spectral_stats` | scaler=`minmax` | reducer=`{'type': 'select_k_best', 'k': 256}` | k=`11` | weights=`distance` | metric=`cosine` | random_acc=`0.8137` | random_weighted_f1=`0.8148` | random_macro_f1=`0.8148` | domain_acc=`0.4025` | domain_macro_f1=`0.2490`
- `lowfreq_all_plus_waveform_spectral` | scaler=`standard_l2` | reducer=`{'type': 'select_k_best', 'k': 44}` | k=`9` | weights=`uniform` | metric=`cosine` | random_acc=`0.8113` | random_weighted_f1=`0.8122` | random_macro_f1=`0.8122` | domain_acc=`0.3996` | domain_macro_f1=`0.2411`
- `waveform_spectral_stats` | scaler=`minmax` | reducer=`{'type': 'select_k_best', 'k': 64}` | k=`15` | weights=`uniform` | metric=`chebyshev` | random_acc=`0.8083` | random_weighted_f1=`0.8096` | random_macro_f1=`0.8096` | domain_acc=`0.3592` | domain_macro_f1=`0.1524`
- `waveform_spectral_stats` | scaler=`robust_l2` | reducer=`{'type': 'select_k_best', 'k': 44}` | k=`15` | weights=`uniform` | metric=`manhattan` | random_acc=`0.8080` | random_weighted_f1=`0.8083` | random_macro_f1=`0.8083` | domain_acc=`0.4061` | domain_macro_f1=`0.1994`
- `lowfreq_all_plus_waveform_spectral` | scaler=`standard` | reducer=`{'type': 'none'}` | k=`5` | weights=`uniform` | metric=`minkowski` | random_acc=`0.8040` | random_weighted_f1=`0.8055` | random_macro_f1=`0.8055` | domain_acc=`0.5322` | domain_macro_f1=`0.2232`
- `lowfreq_all_plus_waveform_spectral` | scaler=`standard` | reducer=`{'type': 'pca_components', 'value': 44}` | k=`9` | weights=`uniform` | metric=`manhattan` | random_acc=`0.8023` | random_weighted_f1=`0.8038` | random_macro_f1=`0.8038` | domain_acc=`0.5027` | domain_macro_f1=`0.2094`
- `waveform_spectral_stats` | scaler=`standard_l2` | reducer=`{'type': 'none'}` | k=`3` | weights=`distance` | metric=`minkowski` | random_acc=`0.8020` | random_weighted_f1=`0.8023` | random_macro_f1=`0.8023` | domain_acc=`0.4974` | domain_macro_f1=`0.2850`
