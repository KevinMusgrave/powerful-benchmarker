import numpy as np


def format_tag(prefix, column_name):
    return f"\\{prefix}{column_name.replace(' ', '')}"


def format_tag_and_float(prefix, column_name):
    def return_fn(x):
        return f"{format_tag(prefix, column_name)}{{{x:.1f}}}"

    return return_fn


def default_operation_fn(*_):
    return ">"


# operation fn
def absolute_value_greater_than(lower_bound, *_):
    if lower_bound >= 0:
        return ">"
    return "<"


def default_interval_fn(min_value, max_value, num_steps, *_):
    intervals = np.linspace(min_value, max_value, num_steps)
    return [intervals[::-1]]


def absolute_value_interval_fn(min_value, max_value, num_steps, *_):
    intervals = np.linspace(np.abs(min_value), max_value, num_steps)
    intervals = intervals[::-1]
    return [intervals, intervals * -1]


def reverse_interval_fn(*args):
    return [default_interval_fn(*args)[0][::-1]]


def create_color_map_tags(
    df,
    tag_prefix,
    min_value_fn=None,
    max_value_fn=None,
    operation_fn=None,
    interval_fn=None,
    num_steps=11,
):
    output_strs = []
    if operation_fn is None:
        operation_fn = default_operation_fn
    if interval_fn is None:
        interval_fn = default_interval_fn
    for column_name in df.columns.values:
        curr_df = df[column_name]
        min_value = 0 if min_value_fn is None else min_value_fn(curr_df, column_name)
        max_value = (
            curr_df.max()
            if max_value_fn is None
            else max_value_fn(curr_df, column_name)
        )
        intervals_list = interval_fn(min_value, max_value, num_steps, column_name)

        curr_str = f"\\def{format_tag(tag_prefix, column_name)}" + "#1{"
        ifdim_count = 0
        for intervals in intervals_list:
            for i, lower_bound in enumerate(intervals):
                if i == 0:
                    continue
                greenness = (10 - i + 1) * 10
                operation = operation_fn(lower_bound, column_name)
                curr_str += (
                    f"\\ifdim#1pt{operation}{lower_bound:.1f}"
                    + f"pt\\cellcolor{{lime!{greenness}}}\\else"
                )
                ifdim_count += 1
        curr_str += "\\cellcolor{{lime!0}}"
        curr_str += "\\fi" * ifdim_count
        curr_str += "#1}"
        output_strs.append(curr_str)

    return "\n\n".join(output_strs)


def get_tags_dict(tag_prefix, task_columns):
    return {
        column_name: format_tag_and_float(tag_prefix, column_name)
        for column_name in task_columns
    }
