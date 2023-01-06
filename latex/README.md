## powerful-benchmarker/latex

The following command reads the dataframes created by `../validator_tests/create_tables.py`, and creates latex tables that summarize the results:

```
python latex/create_tables.py --exp_group_excludes domainnet126 --exp_group_excludes_select_best mnist --exp_group_includes_select_best fl3fl6 --nlargest 5
python latex/create_tables.py --exp_group_includes domainnet126 --exp_group_excludes_select_best mnist --exp_group_includes_select_best fl3fl6 --nlargest 5
```

Make the `best_accuracy_per_adapter_ranked_by_score` tables use the same color tags as the `best_accuracy_per_adapter` tables.
```
python latex/replace_color_map_tags.py
```

Replace the header for domainnet:
```
python latex/replace_header_acronyms.py
```