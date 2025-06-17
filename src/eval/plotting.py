import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import pandas as pd
import numpy as np
import random
import re
import os


def legend_model_names(model_size, ao = 0):
    if ao:
        model_sizes = {"standard": "GPT-2 AO",
                   "eightlayer": "8-layer AO",
                   "fourlayer": "4-layer AO",
                   "twolayer": "2-layer AO",
                   }
    else: model_sizes = {"standard": "GPT-2",
                   "eightlayer": "8-layer transformer",
                   "fourlayer": "4-layer transformer",
                   "twolayer": "2-layer transformer",
                   "lstm": "LSTM",
                   "gru": "GRU",
                   "rnn": "Elman RNN",
                   "mlponly": "MLP-Only",
                   }
    return model_sizes[model_size]


def plot_mse(models: list, n_points=50, savefig_path=None, train_steps=200000):
    all_data = []
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    rcParams['font.weight'] = 'bold'

    for model_size in models:
        if ("mlponly" not in models) or (model_size == "mlponly"):
            if train_steps != 200000 and model_size == "standard":
                filename = f"mse/context_length_{model_size}_{train_steps}.csv"
            else:
                filename = f"mse/context_length_{model_size}.csv"
            df = pd.read_csv(filename)
            mse_values = df.iloc[0, 3:].values.astype(float)

            for i, val in enumerate(mse_values[:n_points]):
                if "_" in model_size:
                    model_size = model_size.replace(model_size, (re.findall(r"([a-z]+)", model_size)[0]))
                model_name = legend_model_names(model_size)
                all_data.append({
                    "Number of in-context examples": i + 1,
                    "In-Context MSE (log)": np.log(val),
                    "Model": model_name
                })

        if ("mlponly" in models) and model_size in ["onelayer", "twolayer", "threelayer", "fourlayer", "eightlayer", "standard"]:
            if train_steps != 200000 and model_size == "standard":
                filename_ao = f"mse/context_length_{model_size}_ao_{train_steps}.csv"
            else:
                filename_ao = f"mse/context_length_{model_size}_ao.csv"
            df_ao = pd.read_csv(filename_ao)
            mse_values_ao = df_ao.iloc[0, 3:].values.astype(float)

            for i, val in enumerate(mse_values_ao[:n_points]):
                model_name = legend_model_names(model_size, ao=1)
                all_data.append({
                    "Number of in-context examples": i + 1,
                    "In-Context MSE (log)": np.log(val),
                    "Model": model_name
                })

    plot_df = pd.DataFrame(all_data)
    if "mlponly" not in models:
        custom_palette = ["#0073E6", "#B51963", "#5FAD56", "#F2C14E"]
    else:
        custom_palette = ["#0073E6", "#B51963", "#011638", "#5FAD56", "#F2C14E"]
    palette = custom_palette + sns.color_palette("husl", max(0, len(plot_df["Model"].unique()) - len(custom_palette)))

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax = sns.lineplot(
        data=plot_df,
        x="Number of in-context examples",
        y="In-Context MSE (log)",
        hue="Model",
        palette=palette,
        linewidth=4
    )

    ax.tick_params(axis='both', labelsize=20)

    ax.set_xlabel("Number of in-context examples", fontsize=22)
    ax.set_ylabel("In-Context MSE (log)", fontsize=22)

    if "mlponly" in models: ax.set_ylim([0,5.4])

    ax.legend(fontsize=22, loc="upper right")
    
    plt.tight_layout()

    if savefig_path is not None:
        plt.savefig(f"../visualizations/mse_{savefig_path}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_mse_with_ci(models: list, n_points=50, savefig_path=None):
    all_data = []

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    rcParams['font.weight'] = 'bold'

    for model_size in models:
        if model_size == "mlponly": filename = f"mse/context_length_{model_size}.csv"
        else: filename = f"mse/context_length_{model_size}_ao.csv"
        df = pd.read_csv(filename)

        for i in range(1, n_points + 1):
            i_str = str(i)
            if "mlponly" in models and model_size != "mlponly":
                df_mean = df[(df["model_size"] == model_size) & (df["attention_only"] == True) & (df["stat"] == "mean")]
                df_q025 = df[(df["model_size"] == model_size) & (df["attention_only"] == True) & (df["stat"] == "q025")]
                df_q975 = df[(df["model_size"] == model_size) & (df["attention_only"] == True) & (df["stat"] == "q975")]
            
            else:
                df_mean = df[(df["model_size"] == model_size) & (df["attention_only"] == False) & (df["stat"] == "mean")]
                df_q025 = df[(df["model_size"] == model_size) & (df["attention_only"] == False) & (df["stat"] == "q025")]
                df_q975 = df[(df["model_size"] == model_size) & (df["attention_only"] == False) & (df["stat"] == "q975")]

            model_name = re.sub(r'[_\d]', '', model_size)
            
            if "_" in model_size:
                model_name = model_size.replace(model_size, (re.findall(r"([a-z]+)", model_size)[0]))

            if "mlponly" in models and model_size != "mlponly": model_name = legend_model_names(model_name, ao = 1)
            else: model_name = legend_model_names(model_name, ao = 0)

            all_data.append({
                "Number of in-context examples": int(i_str),
                "Model": model_name,
                "In-Context MSE (log)": np.log(df_mean[i_str].values[0]),
                "q025": np.log(df_q025[i_str].values[0]),
                "q975": np.log(df_q975[i_str].values[0])
            })

    plot_df = pd.DataFrame(all_data)

    if len(models) == 5: custom_palette = ["#0073E6", "#B51963", "#011638", "#5FAD56", "#F2C14E"]
    elif len(models) == 4: custom_palette = ["#0073E6", "#B51963", "#5FAD56", "#F2C14E"]
    else: raise NotImplementedError
    palette = custom_palette + sns.color_palette("husl", max(0, len(plot_df["Model"].unique()) - len(custom_palette)))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax = sns.lineplot(
        data=plot_df,
        x="Number of in-context examples",
        y="In-Context MSE (log)",
        hue="Model",
        palette=palette,
        linewidth=4,
        ax=ax
    )

    handles, labels = ax.get_legend_handles_labels()
    color_dict = {label: handle.get_color() for label, handle in zip(labels, handles)}

    for model_name, group in plot_df.groupby("Model"):
        ax.fill_between(
            group["Number of in-context examples"],
            group["q025"],
            group["q975"],
            alpha=0.2,
            color=color_dict.get(model_name, "#999999")
        )

    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel("Number of in-context examples", fontsize=22)
    ax.set_ylabel("In-Context MSE (log)", fontsize=22)
    if "mlponly" in models: ax.legend(fontsize=22, loc="right")
    else: ax.legend(fontsize=22, loc="upper right")

    plt.tight_layout()

    if savefig_path is not None:
        plt.savefig(f"../visualizations/mse_ci_{savefig_path}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_mse_vs_ess(ess_file='wandb/ess.csv', context_length=50, savefig_path=None):
    # require to import ess.csv file from weights and biases
    # anonymized version included in supplementary material
    sns.set_style("whitegrid")
    rcParams['font.weight'] = 'bold'
    ess_df = pd.read_csv(ess_file)
    ess_df["suffix"] = ess_df["Name"]
    ess_df["effective_support_size"] = ess_df["effective_support_size"].astype(float)

    records = []

    for _, row in ess_df.iterrows():
        suffix = row["suffix"]
        ess = row["effective_support_size"]

        if "_u_" in suffix:
            train_dist = "Uniform"
            eval_dists = [("Uniform", f"mse/context_length_{suffix}.csv"),
                          ("Normal",  f"mse/context_length_{suffix}_eval_on_n.csv")]
            # [("Uniform", f"mse/context_length_{suffix}.csv")]
            # [("Normal",  f"mse/context_length_{suffix}_eval_on_n.csv")]
        elif "_n_" in suffix:
            train_dist = "Normal"
            eval_dists = [("Uniform",  f"mse/context_length_{suffix}.csv"),
                          ("Normal", f"mse/context_length_{suffix}_eval_on_n.csv")]
            # [("Uniform",  f"mse/context_length_{suffix}.csv")]
            # [("Normal", f"mse/context_length_{suffix}_eval_on_n.csv")]
        else:
            print(f"Unknown distribution in suffix: {suffix}")
            continue

        for eval_dist, filename in eval_dists:
            if not os.path.exists(filename):
                print(f"File not found: {filename}")
                continue

            try:
                df = pd.read_csv(filename)
                mean = df.loc[(df["stat"] == "mean") & (df["attention_only"] == False), str(context_length)].values[0]
                q025 = df.loc[(df["stat"] == "q025") & (df["attention_only"] == False), str(context_length)].values[0]
                q975 = df.loc[(df["stat"] == "q975") & (df["attention_only"] == False), str(context_length)].values[0]

                if q025 == 0: q025 = 1e-15
                records.append({
                    "ESS": np.log(ess),
                    "MSE": np.log(mean),
                    "q025": max(0, np.log(q025)),
                    "q975": np.log(q975),
                    "TrainDist": train_dist,
                    "EvalDist": eval_dist,
                    "Label": suffix
                })

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    if not records:
        raise ValueError("No data was loaded successfully — check file names and contents.")

    plot_df = pd.DataFrame(records).sort_values("ESS")
    plot_df["LineStyle"] = np.where(plot_df["TrainDist"] == plot_df["EvalDist"], "in-dist", "OOD")

    color_map = {
        ("Uniform", "in-dist"): "#0073E6",  # deep navy
        ("Uniform", "OOD"): "#0073E6",      # "#66A9FF",  # light navy
        ("Normal", "in-dist"): "#E05D91",   # deep magenta
        ("Normal", "OOD"): "#E05D91",       # "#B51963",  # light magenta
    }

    plt.figure(figsize=(9, 6))
    sns.set_context("paper", font_scale=2)

    for (train_dist, linestyle), group in plot_df.groupby(["TrainDist", "LineStyle"]):
        style = '-'
        if (linestyle == "OOD") and (train_dist == "Uniform"): style = '--'
        if (linestyle == "in-dist") and (train_dist == "Normal"): style = '--'
        color = color_map[(train_dist, linestyle)]

        ax = plt.gca()
        ax.plot(group["ESS"], group["MSE"], linestyle=style, color=color, linewidth=2.5, label=f"{train_dist} ({linestyle})")

        for _, row in group.iterrows():
            ax.errorbar(
                x=row["ESS"],
                y=row["MSE"],
                yerr=[[row["MSE"] - row["q025"]], [row["q975"] - row["MSE"]]],
                fmt='none',
                ecolor=color,
                elinewidth=1.5,
                capsize=4
            )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    print(by_label)
    legend_properties = {"weight": "bold"}
    plt.legend(by_label.values(), by_label.keys(), title="Training distribution", fontsize = 22, prop = legend_properties)

    ax.set_ylim([0, 6.2])
    plt.xlabel("Effective support size (log)")
    plt.ylabel(f"In-context MSE (log)")
    plt.tight_layout()

    if savefig_path is not None:
        plt.savefig(f"../visualizations/{savefig_path}_{context_length}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_bar_mse_context35(model_sizes, context_length=35, savefig_path=None):
    sns.set_style("whitegrid")
    rcParams['font.weight'] = 'bold'
    colors = ["#0073E6", "#B51963", "#F2C14E", "#5FAD56"]
    alpha_standard = 0.5
    width = 0.35
    x = np.arange(len(model_sizes))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_context("paper", font_scale=1.5)

    for i, model in enumerate(model_sizes):
        base_file = f"mse/context_length_{model}.csv"
        ao_file = f"mse/context_length_{model}_ao.csv"

        df_base = pd.read_csv(base_file)
        df_ao = pd.read_csv(ao_file)

        mse_b = df_base.loc[(df_base["stat"] == "mean") & (~df_base["attention_only"]), str(context_length)].values[0]
        q025_b = df_base.loc[(df_base["stat"] == "q025") & (~df_base["attention_only"]), str(context_length)].values[0]
        q975_b = df_base.loc[(df_base["stat"] == "q975") & (~df_base["attention_only"]), str(context_length)].values[0]

        log_mse_b = np.log(mse_b)
        err_b = [[log_mse_b - np.log(q025_b)], [np.log(q975_b) - log_mse_b]]

        ax.bar(x[i] - width/2, log_mse_b, width, label="Full" if i == 0 else "", color=colors[i], alpha=1.0,
               yerr=err_b, capsize=5, ecolor='black')

        mse_ao = df_ao.loc[(df_ao["stat"] == "mean") & (df_ao["attention_only"]), str(context_length)].values[0]
        q025_ao = df_ao.loc[(df_ao["stat"] == "q025") & (df_ao["attention_only"]), str(context_length)].values[0]
        q975_ao = df_ao.loc[(df_ao["stat"] == "q975") & (df_ao["attention_only"]), str(context_length)].values[0]

        log_mse_ao = np.log(mse_ao)
        err_ao = [[log_mse_ao - np.log(q025_ao)], [np.log(q975_ao) - log_mse_ao]]

        ax.bar(x[i] + width/2, log_mse_ao, width, label="AO" if i == 0 else "", color=colors[i], alpha=alpha_standard,
               yerr=err_ao, capsize=5, ecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(["1-layer, 8-head", "2-layer, 4-head", "4-layer, 2-head", "8-layer, 1-head"], fontsize=20)
    ax.set_yticklabels([0, 1, 2, 3, 4, 5], size = 20)
    ax.set_ylabel("In-context MSE (log)", fontsize=22)
    ax.set_xlabel("Model", fontsize=22)
    ax.set_ylim([0,5.4])
    ax.legend(fontsize=22)
    plt.tight_layout()

    if savefig_path:
        plt.savefig(f"../visualizations/mse_bar_{context_length}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_sde_mse(model_sizes, window_lwr = 10, window_upr = 31, savefig_path=None):
    """
    Plots MSE vs. Number of in-context examples for multiple models from CSV files.
    
    Parameters:
        file_paths (list of str): Paths to CSV files.
        labels (list of str): Corresponding labels for each model.
        colors (list of str): Colors to use for each model.
        savefig_path (str): Optional path to save the figure.
    """
    sns.set_style("whitegrid")
    colors = ["#0073E6", "#B51963", "#011638", "#5FAD56", "#F2C14E"]

    plt.figure(figsize=(10, 6))
    sns.set_context("paper", font_scale=2)

    for i, model in enumerate(model_sizes):
        base_file = f"mse/context_length_sde_{model}.csv"
        df = pd.read_csv(base_file, header=None)

        mse_strings = np.array(df.iloc[1, (2):])
        mse_values = []
        for s in mse_strings:
            d = re.findall("\d+(\.\d+)?", s)[0]
            mse_values.append(float(d))

        context_lengths = np.array(df.iloc[2, 2:]).astype(int)
        mse_values = np.array(mse_values)

        data = pd.DataFrame({"Number of in-context examples": context_lengths, "MSE": mse_values})
        data = data[data["Number of in-context examples"] >= window_lwr]
        data = data[data["Number of in-context examples"] <= window_upr]
        mean_mse_by_length = data.groupby("Number of in-context examples").mean().reset_index()

        plt.plot(mean_mse_by_length["Number of in-context examples"],
                 mean_mse_by_length["MSE"],
                 color=colors[i], linewidth=4, label=str(model))

    plt.xlabel("Number of in-context examples")
    plt.ylabel("In-Context MSE")
    legend_properties = {"weight": "bold"}
    plt.legend(title="Model", fontsize=19, prop = legend_properties, loc = "upper right")

    plt.tight_layout()


    if savefig_path:
        plt.savefig(f"../visualizations/mse_sde_{savefig_path}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_sde_bar_chart(model_sizes, examples, savefig_path = None):
    random.seed(19459)
    sns.set_style("whitegrid")
    rcParams['font.weight'] = "bold"
    colors = ["#0073E6", "#B51963", "#5FAD56", "#F2C14E"]

    sns.set_context("paper", font_scale=2)
    width = 0.5
    x = np.arange(len(model_sizes))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_context("paper", font_scale=1.5)

    def compute_internal_mse(data, examples):
        sub_data = data[data["Number of in-context examples"] == examples].reset_index()
        mse_sub = sub_data["MSE"]

        total_number_examples = len(sub_data)

        quantile_indices = random.choices(range(total_number_examples), k = total_number_examples)
        print(total_number_examples)
        quantiles = np.quantile(mse_sub[quantile_indices], q=[0.025, 0.975])

        mean_mse = np.mean(mse_sub)
        lower_q = max(2 * mean_mse - quantiles[1], 0)
        upper_q = max(2 * mean_mse - quantiles[0], 0)

        err_ao = [[mean_mse - lower_q], [upper_q - mean_mse]]

        return mean_mse, err_ao

    for i, model in enumerate(model_sizes):
        base_file = f"mse/context_length_sde_{model}.csv"
        df = pd.read_csv(base_file, header=None)

        mse_strings = np.array(df.iloc[1, (2):])
        mse_values = []
        for s in mse_strings:
            d = re.findall("\d+(\.\d+)?", s)[0]
            mse_values.append(float(d))
        
        context_lengths = np.array(df.iloc[2, 2:]).astype(int)
        mse_values = np.array(mse_values)

        data = pd.DataFrame({"Number of in-context examples": context_lengths, "MSE": mse_values})

        for ex_idx, ex in enumerate(examples):
            mean_mse, err_ao = compute_internal_mse(data, ex)

            if ex == 15:
                ax.bar(x[i] - width/2, mean_mse, width/2, color=colors[i], alpha=1,
                    yerr=err_ao, capsize=5, ecolor='black', label = "15 time steps" if model == "rnn" else '')
            elif ex == 20:
                ax.bar(x[i], mean_mse, width/2, color=colors[i], alpha=0.7,
                    yerr=err_ao, capsize=5, ecolor='black', label = "20 time steps" if model == "rnn" else '')
            elif ex == 25:
                ax.bar(x[i] + width/2, mean_mse, width/2, color=colors[i], alpha=0.4,
                    yerr=err_ao, capsize=5, ecolor='black', label = "25 time steps" if model == "rnn" else '')
        
        
    ax.set_xticks(x)
    ax.set_ylim([0, 0.012])
    ax.set_xticklabels(["Elman RNN", "LSTM", "GRU", "GPT-2"], fontsize=20)
    ax.set_yticklabels(ax.get_yticks(), size = 20)
    plt.xlabel("Model", fontsize = 22)
    plt.ylabel("In-Context MSE", fontsize = 22)
    plt.legend(title="Evaluated at", fontsize=16, loc = "upper center", ncol = 3)

    plt.tight_layout()

    if savefig_path:
        plt.savefig(f"../visualizations/mse_sde_bar_{savefig_path}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_sde_chart(model_size, ao = False, savefig_path = None):
    filename = f"mse/prediction_chart_sde_{model_size}{'_ao' if ao else ''}.csv"

    sns.set_style("whitegrid")
    rcParams['font.weight'] = "bold"
    colors = ["#0073E6", "#B51963", "#5FAD56", "#F2C14E"]

    plt.figure(figsize=(10, 6))
    sns.set_context("paper", font_scale=1.5)

    df = pd.read_csv(filename, index_col = 0).transpose()

    print(df)

    plt.plot(df["event_times"], df["obs_x"], color = colors[0], linewidth = 4, label = "Observational x")
    plt.plot(df["event_times"], df["obs_y"], color = colors[1], linewidth = 4, label = "Observational y")
    plt.plot(df["event_times"], df["gt_x"], color = colors[2], linewidth = 4, label = "Counterfactual x")
    plt.plot(df["event_times"], df["pred_x"], color = colors[2], alpha = 0.5, linewidth = 4, label = "Predicted x")
    plt.plot(df["event_times"], df["gt_y"], color = colors[3], linewidth = 4, label = "Counterfactual y")
    plt.plot(df["event_times"], df["pred_y"], color = colors[3], alpha = 0.5, linewidth = 4, label = "Predicted y")

    ax = plt.gca()
    ax.set_ylim([0, 4])
    ax.set_xticklabels(labels = [0.0, 0.0, 0.02, 0.04, 0.06, 0.08], size = 20)
    ax.set_yticklabels(labels = [0, "", 1, "", 2, "", 3, "", 4], size = 20)
    ax.set_ylabel("Lotka-Volterra concentration", fontsize=22)
    ax.set_xlabel("Time", fontsize=22)
    
    plt.legend(title = "Sequence", fontsize = 16, loc = "upper center", ncol = 3)
    plt.tight_layout()
    if savefig_path:
        plt.savefig(f"../visualizations/chart_sde_{savefig_path}.pdf", format="pdf", bbox_inches="tight")
    else: plt.show()


if __name__ == "__main__":
    models = ["standard","eightlayer","fourlayer","threelayer","twolayer"]
    models_attention = ["mlponly", "twolayer", "fourlayer", "eightlayer", "standard"]
    recursive_sweep = ["lstm_128_2", "lstm_256_2", "lstm_128_3", "lstm_256_3", "gru_128_2", "gru_256_2", "gru_128_3", "gru_256_3", "rnn_128_2", "rnn_256_2", "rnn_128_3", "rnn_256_3"]
    data_diversity_norm= ["cf_n_1", "cf_n_3", "cf_n_5", "cf_n_10", "cf_n_15", "cf_n_20", "cf_n_30", "cf_n_50", "cf_n_75", "cf_n_100"]
    data_diversity_uniform = ["cf_u_1", "cf_u_3", "cf_u_5", "cf_u_10", "cf_u_15", "cf_u_20", "cf_u_30", "cf_u_50", "cf_u_75", "cf_u_100"]
    depth = ["eighthead", "4h_2l", "2h_4l", "eightlayer"]
    rnn_type = ["rnn", 'lstm', 'gru', "standard"]
    models_transformer = ["rnn_256_3", "lstm_256_3", "gru_256_3", "standard"]

    savefig_path="attention"
    if not os.path.exists('../visualizations'):
        os.makedirs('../visualizations')
    plot_mse_with_ci(models_attention, savefig_path=savefig_path)
    # plot_mse(models_attention, savefig_path=savefig_path)
    # for i in [1, 35, 45]:
    #     plot_mse_vs_ess(context_length=i, savefig_path="ess_both_dist")
