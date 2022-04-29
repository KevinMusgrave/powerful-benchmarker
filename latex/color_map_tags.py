import numpy as np


def format_tag(prefix, column_name):
    return f"\\{prefix}{column_name.replace(' ', '')}"


def format_tag_and_float(prefix, column_name):
    def return_fn(x):
        return f"{format_tag(prefix, column_name)}{{{x:.1f}}}"

    return return_fn


def create_color_map_tags(
    df, tag_prefix, min_value=None, max_value=None, ascending=True
):
    output_strs = []
    for column_name in df.columns.values:
        curr_df = df[column_name]
        if min_value is None:
            min_value = 0
        if max_value is None:
            max_value = curr_df.max()

        intervals = np.linspace(min_value, max_value, 11)

        # if large values should be green
        if ascending:
            intervals = intervals[::-1]

        curr_str = f"\\def{format_tag(tag_prefix, column_name)}" + "#1{"

        for i, lower_bound in enumerate(intervals):
            if i == 0:
                continue
            greenness = (10 - i + 1) * 10
            operation = ">" if ascending else "<"
            curr_str += (
                f"\\ifdim#1pt{operation}{lower_bound:.1f}"
                + f"pt\\cellcolor{{lime!{greenness}}}\\else"
            )
        curr_str += f"\\cellcolor{{lime!0}}"
        curr_str += "\\fi" * (len(intervals) - 1)
        curr_str += "#1}"
        output_strs.append(curr_str)

    return "\n\n".join(output_strs)


def get_tags_dict(tag_prefix, task_columns):
    return {
        column_name: format_tag_and_float(tag_prefix, column_name)
        for column_name in task_columns
    }
