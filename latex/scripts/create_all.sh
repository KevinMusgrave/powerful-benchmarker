python latex/create_tables.py --exp_group_excludes domainnet126 --exp_group_excludes_select_best mnist --nlargest 5
python latex/create_tables.py --exp_group_prefix domainnet126 --exp_group_excludes_select_best mnist --nlargest 5
python latex/create_tables.py --exp_group_prefix mnist --exp_group_excludes_select_best mnist --nlargest 5
python latex/create_tables.py --exp_group_prefix office --exp_group_excludes_select_best mnist --nlargest 5
python latex/create_tables.py --exp_group_prefix office31 --exp_group_excludes_select_best mnist --nlargest 5
python latex/create_tables.py --exp_group_prefix officehome --exp_group_excludes_select_best mnist --nlargest 5
python latex/create_tables.py --exp_group_excludes mnist --exp_group_excludes_select_best mnist --nlargest 5
python latex/replace_color_map_tags.py
python latex/replace_header_acronyms.py