from .df_utils import unify_validator_columns


def plot_heat_map(df):
    df = df[df["validator"] != "Accuracy"]
    df = df[df["src_threshold"] == -0.01]
    df = unify_validator_columns(df)
    print(df)
