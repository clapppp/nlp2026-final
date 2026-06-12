# Paraphrase Decision Policy Report

## Selection

- Selection metric: `accuracy_band_cost`
- Accuracy tolerance: `0.001`
- Selected policy: `uncertain_weighted(base_threshold=0.4, base_weight=0.5, lower=0.4, threshold=0.55, upper=0.8)`
- Policy JSON: `{"kind": "uncertain_weighted", "params": {"base_threshold": 0.4, "base_weight": 0.5, "lower": 0.4, "threshold": 0.55, "upper": 0.8}}`

## Key Metrics

| Policy | Acc | Macro F1 | Precision | Recall | TN | FP | FN | TP | Improved Call Rate | Expected Cost / Ex |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_threshold(threshold=0.53) | 0.879567 | 0.871977 | 0.819346 | 0.863417 | 22702 | 2835 | 2034 | 12858 | 0.0000 | 0.120433 |
| improved_threshold(threshold=0.52) | 0.875980 | 0.867377 | 0.824081 | 0.843339 | 22856 | 2681 | 2333 | 12559 | 1.0000 | 0.124020 |
| uncertain_weighted(base_threshold=0.4, base_weight=0.5, lower=0.4, threshold=0.55, upper=0.8) | 0.881768 | 0.873374 | 0.834215 | 0.847435 | 23029 | 2508 | 2272 | 12620 | 0.1306 | 0.118232 |

## Top Policies By Accuracy

| Policy | Acc | Macro F1 | Precision | Recall | TN | FP | FN | TP | Improved Call Rate | Expected Cost / Ex |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| weighted_ensemble(base_weight=0.5, threshold=0.53) | 0.882510 | 0.874684 | 0.828560 | 0.858716 | 22891 | 2646 | 2104 | 12788 | 1.0000 | 0.117490 |
| weighted_ensemble(base_weight=0.5, threshold=0.52) | 0.882114 | 0.874496 | 0.825051 | 0.862947 | 22812 | 2725 | 2041 | 12851 | 1.0000 | 0.117886 |
| weighted_ensemble(base_weight=0.5, threshold=0.54) | 0.881991 | 0.873871 | 0.831206 | 0.852807 | 22958 | 2579 | 2192 | 12700 | 1.0000 | 0.118009 |
| uncertain_weighted(base_threshold=0.4, base_weight=0.5, lower=0.3, threshold=0.55, upper=0.9) | 0.881966 | 0.873600 | 0.834302 | 0.847972 | 23029 | 2508 | 2264 | 12628 | 0.2335 | 0.118034 |
| uncertain_weighted(base_threshold=0.5, base_weight=0.5, lower=0.3, threshold=0.55, upper=0.9) | 0.881966 | 0.873600 | 0.834302 | 0.847972 | 23029 | 2508 | 2264 | 12628 | 0.2335 | 0.118034 |
| uncertain_weighted(base_threshold=0.6, base_weight=0.5, lower=0.3, threshold=0.55, upper=0.9) | 0.881966 | 0.873600 | 0.834302 | 0.847972 | 23029 | 2508 | 2264 | 12628 | 0.2335 | 0.118034 |
| uncertain_weighted(base_threshold=0.4, base_weight=0.5, lower=0.2, threshold=0.55, upper=0.9) | 0.881941 | 0.873586 | 0.834115 | 0.848174 | 23025 | 2512 | 2261 | 12631 | 0.2627 | 0.118059 |
| uncertain_weighted(base_threshold=0.5, base_weight=0.5, lower=0.2, threshold=0.55, upper=0.9) | 0.881941 | 0.873586 | 0.834115 | 0.848174 | 23025 | 2512 | 2261 | 12631 | 0.2627 | 0.118059 |
| uncertain_weighted(base_threshold=0.6, base_weight=0.5, lower=0.2, threshold=0.55, upper=0.9) | 0.881941 | 0.873586 | 0.834115 | 0.848174 | 23025 | 2512 | 2261 | 12631 | 0.2627 | 0.118059 |
| uncertain_weighted(base_threshold=0.4, base_weight=0.5, lower=0.4, threshold=0.55, upper=0.9) | 0.881941 | 0.873503 | 0.835177 | 0.846562 | 23049 | 2488 | 2285 | 12607 | 0.2060 | 0.118059 |
| uncertain_weighted(base_threshold=0.5, base_weight=0.5, lower=0.4, threshold=0.55, upper=0.9) | 0.881941 | 0.873503 | 0.835177 | 0.846562 | 23049 | 2488 | 2285 | 12607 | 0.2060 | 0.118059 |
| uncertain_weighted(base_threshold=0.6, base_weight=0.5, lower=0.4, threshold=0.55, upper=0.9) | 0.881941 | 0.873503 | 0.835177 | 0.846562 | 23049 | 2488 | 2285 | 12607 | 0.2060 | 0.118059 |
| weighted_ensemble(base_weight=0.5, threshold=0.51) | 0.881916 | 0.874499 | 0.822065 | 0.867110 | 22742 | 2795 | 1979 | 12913 | 1.0000 | 0.118084 |
| weighted_ensemble(base_weight=0.5, threshold=0.55) | 0.881916 | 0.873551 | 0.834192 | 0.847972 | 23027 | 2510 | 2264 | 12628 | 1.0000 | 0.118084 |
| uncertain_weighted(base_threshold=0.4, base_weight=0.5, lower=0.3, threshold=0.55, upper=0.8) | 0.881793 | 0.873471 | 0.833344 | 0.848845 | 23009 | 2528 | 2251 | 12641 | 0.1581 | 0.118207 |

## Lowest Improved-Call Policies

| Policy | Acc | Macro F1 | Precision | Recall | TN | FP | FN | TP | Improved Call Rate | Expected Cost / Ex |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_threshold(threshold=0.53) | 0.879567 | 0.871977 | 0.819346 | 0.863417 | 22702 | 2835 | 2034 | 12858 | 0.0000 | 0.120433 |
| base_threshold(threshold=0.55) | 0.879567 | 0.871628 | 0.823678 | 0.856366 | 22807 | 2730 | 2139 | 12753 | 0.0000 | 0.120433 |
| base_threshold(threshold=0.56) | 0.879492 | 0.871381 | 0.825663 | 0.852941 | 22855 | 2682 | 2190 | 12702 | 0.0000 | 0.120508 |
| base_threshold(threshold=0.57) | 0.879369 | 0.871058 | 0.827866 | 0.849046 | 22908 | 2629 | 2248 | 12644 | 0.0000 | 0.120631 |
| base_threshold(threshold=0.58) | 0.879369 | 0.870852 | 0.830419 | 0.845085 | 22967 | 2570 | 2307 | 12585 | 0.0000 | 0.120631 |
| base_threshold(threshold=0.52) | 0.879295 | 0.871852 | 0.816956 | 0.866438 | 22646 | 2891 | 1989 | 12903 | 0.0000 | 0.120705 |
| base_threshold(threshold=0.54) | 0.879171 | 0.871387 | 0.820922 | 0.859455 | 22745 | 2792 | 2093 | 12799 | 0.0000 | 0.120829 |
| base_threshold(threshold=0.59) | 0.879146 | 0.870417 | 0.832558 | 0.841056 | 23018 | 2519 | 2367 | 12525 | 0.0000 | 0.120854 |
| base_threshold(threshold=0.51) | 0.878973 | 0.871706 | 0.814138 | 0.870064 | 22579 | 2958 | 1935 | 12957 | 0.0000 | 0.121027 |
| base_threshold(threshold=0.6) | 0.878973 | 0.870028 | 0.834840 | 0.837027 | 23071 | 2466 | 2427 | 12465 | 0.0000 | 0.121027 |
| base_threshold(threshold=0.61) | 0.878874 | 0.869703 | 0.837418 | 0.832863 | 23129 | 2408 | 2489 | 12403 | 0.0000 | 0.121126 |
| base_threshold(threshold=0.5) | 0.878849 | 0.871748 | 0.811845 | 0.873556 | 22522 | 3015 | 1883 | 13009 | 0.0000 | 0.121151 |
| base_threshold(threshold=0.48) | 0.878825 | 0.872036 | 0.807951 | 0.880271 | 22421 | 3116 | 1783 | 13109 | 0.0000 | 0.121175 |
| base_threshold(threshold=0.49) | 0.878701 | 0.871761 | 0.809571 | 0.876981 | 22465 | 3072 | 1832 | 13060 | 0.0000 | 0.121299 |
| base_threshold(threshold=0.62) | 0.878478 | 0.869062 | 0.839537 | 0.828431 | 23179 | 2358 | 2555 | 12337 | 0.0000 | 0.121522 |

## Selected Policy Segment Metrics

| Segment | N | Acc | Macro F1 | FP | FN |
|---|---:|---:|---:|---:|---:|
| jaccard < 0.20 | 8770 | 0.976967 | 0.856761 | 142 | 60 |
| 0.20 <= jaccard < 0.45 | 16529 | 0.870652 | 0.863971 | 1158 | 980 |
| 0.45 <= jaccard < 0.65 | 8245 | 0.817829 | 0.812726 | 782 | 720 |
| jaccard >= 0.65 | 6885 | 0.863762 | 0.863746 | 426 | 512 |
| max_len <= 8 | 5627 | 0.858895 | 0.858077 | 403 | 391 |
| 8 < max_len <= 16 | 24026 | 0.865063 | 0.861361 | 1736 | 1506 |
| max_len > 16 | 10776 | 0.930958 | 0.899086 | 369 | 375 |
| lexical rule suspicious | 4343 | 0.907207 | 0.862309 | 168 | 235 |
