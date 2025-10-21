import matplotlib.pyplot as plt
from decimal import Decimal
from matplotlib import rcParams
import seaborn as sns
import pandas as pd
import numpy as np
import random
import re
import os


def legend_model_names(model_size, ao = 0, conference = "iclr"):
    if conference == "neurips":
        if ao:
            model_sizes = {"standard": "GPT-2 AO",
                    "eightlayer": "8-layer AO",
                    "fourlayer": "4-layer AO",
                    "twolayer": "2-layer AO",
                    }
        else: model_sizes = {"standard": "GPT-2",
                    "eightlayer": "8-layer Transformer",
                    "fourlayer": "4-layer Transformer",
                    "twolayer": "2-layer Transformer",
                    "lstm": "LSTM",
                    "gru": "GRU",
                    "rnn": "Elman RNN",
                    "mlponly": "MLP-Only",
                    "prediction": "Prediction on Y",
                    "continuation": "8-layer continuation",
                    "fiveembeds": "hand-written embeddings, size 5",
                    "sixembeds": "hand-written embeddings, size 6",
                    "twentyembeds": "hand-written embeddings, size 20"
                    }
    elif conference == "iclr":
        model_sizes =  {"12l_8h": "GPT-2",
                        "one_head": "8-layer, 1-head Transformer",
                        "4l_2h": "4-layer, 2-head Transformer",
                        "2l_4h": "2-layer, 4-head Transformer",
                        "1l_8h": "1-layer, 8-head Transformer",
                        "lstm_3l_8h": "LSTM",
                        "gru_3l_8h": "GRU",
                        "rnn_3l_8h": "Elman RNN",
                        "lstm_2l_8h": "LSTM",
                        "gru_2l_8h": "GRU",
                        "rnn_2l_8h": "Elman RNN",
                        "gpt2_mlp": "MLP-Only",
                        "gpt2_ao": "8-layer AO",
                        "gpt2_ao_4l_2h": "4-layer, 2-head AO",
                        "gpt2_ao_2l_4h": "2-layer, 4-head AO",
                        "gpt2_ao_1l_8h": "1-layer, 8-head AO",
                        "gpt2_ao_12l_8h": "GPT-2 AO",
                        "gpt2_ao_4l": "4-layer AO",
                        "gpt2_ao_2l": "2-layer AO",
                        "4l": "4-layer Transformer",
                        "2l": "2-layer Transformer",
                        "12l_8h_1_minex30": "GPT-2 with over 30 examples",
                        "12l_8h_1_constz14_minex30": "GPT-2 with fixed " + r'$z$' + " and over 30 examples",
                        "12l_8h_1_1000000steps": "GPT-2 trained for 1m steps",
                        "12l_8h_1000000steps": "GPT-2 counterfactual (1m)",
                        "gpt2_cont_12l_8h_cont_1000000steps": "GPT-2 continuation (1m)",
                        "gpt2_ao_elossmae": "8-layer AO evaluated on MAE",
                        "gpt2_ao_tlossmae": "8-layer AO trained on MAE",
                        "gpt2_ao_tlossmae_elossmae": "8-layer AO trained, evaluated on MAE",
                        "gpt2_ao_cont_cont": "8-layer AO continuation",
                        "gpt2_ao_1000000steps": "8-layer AO (1m)",
                        "gpt2_ao_cont_cont_1000000steps": "8-layer AO continuation (1m)",
                        "optimal": "Optimal MSE",
                        }
    return model_sizes[model_size]


def plot_mse(models: list, n_points=50, savefig_path=None, train_steps=200000, conference = "iclr"):
    all_data = []
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    rcParams['font.weight'] = 'bold'

    if conference == "neurips":
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
                    model_name = legend_model_names(model_size, conference="neurips")
                    all_data.append({
                        "Number of in-context examples": i + 1,
                        "In-context MSE (log)": np.log(val),
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
                    model_name = legend_model_names(model_size, ao=1, conference="neurips")
                    all_data.append({
                        "Number of in-context examples": i + 1,
                        "In-context MSE (log)": np.log(val),
                        "Model": model_name
                    })

    elif conference == "iclr":
        for suffix in models:
            if suffix == "one_head": filename = f"iclr/mse/context_length.csv"
            elif suffix == "optimal": filename = f"iclr/mse/estimated_loss.csv"
            else: filename = f"iclr/mse/context_length_{suffix}.csv"

            df = pd.read_csv(filename)
            
            if suffix == "optimal": mse_values = df.iloc[0, 1:].values.astype(float)
            else: mse_values = df.iloc[0, 3:].values.astype(float)

            for i, val in enumerate(mse_values[:n_points]):
                model_name = legend_model_names(suffix, conference="iclr")
                all_data.append({
                    "Number of in-context examples": i + 1,
                    "In-context MSE (log)": np.log(val),
                    "Model": model_name
                })


    plot_df = pd.DataFrame(all_data)
    custom_palette = color_palette(models)
    palette = custom_palette + sns.color_palette("husl", max(0, len(plot_df["Model"].unique()) - len(custom_palette)))

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax = sns.lineplot(
        data=plot_df,
        x="Number of in-context examples",
        y="In-context MSE (log)",
        hue="Model",
        palette=palette,
        linewidth=4
    )

    ax.set_xlim([0,50])
    ax.set_ylim([-2,3])

    ax.tick_params(axis='both', labelsize=20)

    ax.set_xlabel("Number of in-context examples", fontsize=22)
    ax.set_ylabel("In-context MSE (log)", fontsize=22)

    if "mlponly" in models or "gpt2_mlp" in models: ax.set_ylim([0,5.4])

    ax.legend(fontsize=22, loc="upper right")

    plt.tight_layout()

    if savefig_path is not None:
        if conference == "iclr": plt.savefig(f"../visualizations/iclr/mse_{savefig_path}.pdf", format="pdf", bbox_inches="tight")
        else: plt.savefig(f"../visualizations/mse_{savefig_path}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def color_palette(models = None, number: int = 5):
    if models == None:
        assert number <= 5
        custom_palette = ["#0073E6", "#B51963", "#011638", "#5FAD56", "#F2C14E"]
        if number in [3, 4]: custom_palette.pop(2)
        if number == 3: custom_palette.pop(2)
        return custom_palette[:number]
    else:
        if len(models) == 5:    custom_palette = ["#0073E6", "#B51963", "#011638", "#5FAD56", "#F2C14E"]
        elif len(models) == 4:  custom_palette = ["#0073E6", "#B51963", "#5FAD56", "#F2C14E"]
        elif len(models) == 3:  custom_palette = ["#0073E6", "#B51963", "#F2C14E"]
        elif len(models) == 2:  custom_palette = ["#F2C14E", "#B51963"]
        elif len(models) == 1:  custom_palette = ["#0073E6"]
        else: raise NotImplementedError
    return custom_palette


def plot_mse_with_ci(models: list, n_points=50, savefig_path=None, conference = "iclr"):
    all_data = []

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    rcParams['font.weight'] = 'bold'

    if conference == "neurips":
        for model_size in models:
            if (model_size == "mlponly") or (model_size == "eightlayer_ao_cont") or (model_size == "predict_y_ao_y"): filename = f"mse/context_length_{model_size}.csv"
            else: filename = f"mse/context_length_{model_size}_ao.csv"
            if "embeds" in model_size:
                filename = f"mse/context_length_{model_size}_ao.csv"
                cnt_file = f"mse/context_length_{model_size}_ao_cont.csv"
                df_cnt = pd.read_csv(cnt_file)
            df = pd.read_csv(filename)
            if model_size == "eightlayer_ao_cont": 
                model_size = "continuation"
                df["model_size"] = model_size
            if model_size == "predict_y_ao_y": 
                model_size = "prediction"
                df["model_size"] = model_size

            for i in range(1, n_points + 1):
                i_str = str(i)
                if ("mlponly" in models and model_size != "mlponly") or model_size in ["eightlayer_new_theta", "continuation", "prediction"]:
                    df_mean = df[(df["model_size"] == model_size) & (df["attention_only"] == True) & (df["stat"] == "mean")]
                    df_q025 = df[(df["model_size"] == model_size) & (df["attention_only"] == True) & (df["stat"] == "q025")]
                    df_q975 = df[(df["model_size"] == model_size) & (df["attention_only"] == True) & (df["stat"] == "q975")]
                
                elif "embeds" in model_size:
                    df_mean = df[(df["model_size"] == model_size) & (df["attention_only"] == True) & (df["stat"] == "mean")]
                    df_q025 = df[(df["model_size"] == model_size) & (df["attention_only"] == True) & (df["stat"] == "q025")]
                    df_q975 = df[(df["model_size"] == model_size) & (df["attention_only"] == True) & (df["stat"] == "q975")]
                
                    df_mean_cnt = df_cnt[(df_cnt["model_size"] == model_size) & (df_cnt["attention_only"] == True) & (df_cnt["stat"] == "mean")]
                    df_q025_cnt = df_cnt[(df_cnt["model_size"] == model_size) & (df_cnt["attention_only"] == True) & (df_cnt["stat"] == "q025")]
                    df_q975_cnt = df_cnt[(df_cnt["model_size"] == model_size) & (df_cnt["attention_only"] == True) & (df_cnt["stat"] == "q975")]

                    all_data.append({
                        "Number of in-context examples": int(i_str),
                        "Model": "continuation",
                        "In-context MSE (log)": np.log(df_mean_cnt[i_str].values[0]),
                        "q025": np.log(df_q025_cnt[i_str].values[0]),
                        "q975": np.log(df_q975_cnt[i_str].values[0])
                    })

                else:
                    df_mean = df[(df["model_size"] == model_size) & (df["attention_only"] == False) & (df["stat"] == "mean")]
                    df_q025 = df[(df["model_size"] == model_size) & (df["attention_only"] == False) & (df["stat"] == "q025")]
                    df_q975 = df[(df["model_size"] == model_size) & (df["attention_only"] == False) & (df["stat"] == "q975")]

                model_name = re.sub(r'[_\d]', '', model_size)

                if ("_" in model_size) and "embeds" not in model_size:
                    model_name = model_size.replace(model_size, (re.findall(r"([a-z]+)", model_size)[0]))

                if "mlponly" in models and model_size != "mlponly": model_name = legend_model_names(model_name, ao = 1, conference = "neurips")
                else: model_name = legend_model_names(model_name, ao = 0, conference = "neurips")

                all_data.append({
                    "Number of in-context examples": int(i_str),
                    "Model": model_name,
                    "In-context MSE (log)": np.log(df_mean[i_str].values[0]),
                    "q025": np.log(df_q025[i_str].values[0]),
                    "q975": np.log(df_q975[i_str].values[0])
                })

    elif conference == "iclr":
        for suffix in models:
            if suffix == "one_head": filename = f"iclr/mse/context_length.csv"
            elif suffix == "optimal": filename = f"iclr/mse/estimated_loss.csv"
            else: filename = f"iclr/mse/context_length_{suffix}.csv"

            df = pd.read_csv(filename)
            
            if suffix == "optimal": df["stat"] = df["statistic"]
            
            for i in range(2, n_points + 1):
                i_str = str(i)
                df_mean = df[df["stat"] == "mean"]
                df_q025 = df[df["stat"] == "q025"]
                df_q975 = df[df["stat"] == "q975"]

                model_name = legend_model_names(suffix, conference="iclr")

                all_data.append({
                    "Number of in-context examples": int(i_str),
                    "Model": model_name,
                    "In-context MSE (log)": np.log(df_mean[i_str].values[0]),
                    "q025": np.log(df_q025[i_str].values[0]),
                    "q975": np.log(df_q975[i_str].values[0])
                })

    plot_df = pd.DataFrame(all_data)

    custom_palette = color_palette(models)
    palette = custom_palette + sns.color_palette("husl", max(0, len(plot_df["Model"].unique()) - len(custom_palette)))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax = sns.lineplot(
        data=plot_df,
        x="Number of in-context examples",
        y="In-context MSE (log)",
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
    ax.set_ylabel("In-context MSE (log)", fontsize=22)
    if "mlponly" in models: ax.legend(fontsize=22, loc="right")
    else: ax.legend(fontsize=22, loc="upper right", ncol = 2)

    ax.set_xlim([0,50])
    ax.set_ylim([-0.2,2.4])
    ax.set_xticklabels(["0", "10", "20", "30", "40", "50"])
    ax.set_yticklabels(["", "0", "", "1", "", "2"])

    plt.tight_layout()

    if savefig_path is not None:
        if conference == "iclr": plt.savefig(f"../visualizations/iclr/mse_ci_{savefig_path}.pdf", format="pdf", bbox_inches="tight")
        else: plt.savefig(f"../visualizations/mse_ci_{savefig_path}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_estimated_mse(file = "iclr/mse/estimated_loss.csv", savefig_path = None):
    all_data = []

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    rcParams['font.weight'] = 'bold'
    
    df = pd.read_csv(file)

    low, high = int(df.columns[1]), int(df.columns[-1])

    for i in range(low, high + 1):
        i_str = str(i)
        df_mean = df[df["statistic"] == "mean"]
        df_q025 = df[df["statistic"] == "q025"]
        df_q975 = df[df["statistic"] == "q975"]

        all_data.append({
            "Number of in-context examples": int(i_str),
            "In-context MSE (log)": np.log(df_mean[i_str].values[0]),
            "q025": np.log(df_q025[i_str].values[0]),
            "q975": np.log(df_q975[i_str].values[0])
        })

    plot_df = pd.DataFrame(all_data)

    color = color_palette("h")[0]
    
    fig, ax = plt.subplots(figsize=(10, 6))

    ax = sns.lineplot(
        data=plot_df,
        x="Number of in-context examples",
        y="In-context MSE (log)",
        color=color,
        linewidth=4,
        ax=ax
    )

    ax.fill_between(
        plot_df["Number of in-context examples"],
        plot_df["q025"],
        plot_df["q975"],
        alpha=0.2,
        color=color
    )

    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel("Number of in-context examples", fontsize=22)
    ax.set_ylabel("In-context MSE (log)", fontsize=22)

    ax.set_xlim([0, 50])
    ax.set_ylim([-1.2, 4.2])

    plt.tight_layout()

    if savefig_path is not None:
        plt.savefig(f"../visualizations/iclr/estimated_mse.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_probing_statistcs(
    probing_types=["theta", "beta"],
    statistic="adj_r_squared",
    arguments = {
        "probe_diff": False,
        "constant_z": -1,
        "min_examples": 2,
        "n_embd": 256,
        "n_layer": 8,
    }, # we suppose that o_dims = 1 and train_steps = 50000
    savefig_path=None,
    conference = "iclr"
):
    all_data = []

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    rcParams['font.weight'] = 'bold'

    if len(probing_types) == 3: custom_palette = ["#0073E6", "#B51963", "#F2C14E"]
    if len(probing_types) == 2: custom_palette = ["#0073E6", "#B51963"]
    if len(probing_types) == 1: custom_palette = ["#0073E6"]

    assert conference == "iclr"

    for i, p_type in enumerate(probing_types):
        parts = ["iclr/probing/statistics"]
        if arguments["probe_diff"]:
            parts.append("diff")
        parts.append(p_type)

        if arguments["constant_z"] != -1:
            parts.append("constz" + str(arguments["constant_z"]))
        if arguments["min_examples"] != 2:
            parts.append("minex" + str(arguments["min_examples"]))
        if arguments["n_embd"] != 256:
            parts.append(f"{arguments['n_embd']}embd")
        if arguments["n_layer"] != 8:
            parts.append(f"{arguments['n_layer']}l")

        filename = "_".join(parts) + ".csv"

        stats = pd.read_csv(filename)[statistic]
        if not arguments["probe_diff"]: stats = stats[1:]
        print(f"Loaded {filename}, n={len(stats)}")

        df = pd.DataFrame({
            "Statistic": stats.values,
            "Probing type": p_type,
            "Index": range(1, len(stats) + 1)
        })
        all_data.append(df)

    plot_df = pd.concat(all_data, ignore_index=True)
    print(plot_df.head())

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=plot_df,
        x="Index",
        y="Statistic",
        hue="Probing type",
        palette=custom_palette,
        linewidth=4,
        marker="o",
        markersize=10,
        ax=ax
    )

    ax.set_xlabel("Layer Index", fontsize=22)
    def statistic_label(statistic):
        if statistic == "adj_r_squared": return "Adjusted R²"
        if statistic == "r_squared": return "R²"
        if statistic == "eval_loss": return "Probe Loss"
        return statistic
    
    ax.set_ylabel(statistic_label(statistic), fontsize=22)
    if "r_squared" in statistic: ax.set_ylim([-0.1, 1.1])
    else: ax.set_ylim([0, 2.4])
    ax.tick_params(axis="both", labelsize=20)
    handles, labels = ax.get_legend_handles_labels()
    mapping = {'theta': r'$\theta$', 'beta': r'$\beta$'}
    new_labels = [mapping.get(l, l) for l in labels]
    ax.legend(handles=handles, labels=new_labels, fontsize=22, loc="lower center", title="Probing type")
    plt.tight_layout()

    if savefig_path is not None:
        path = "_".join(parts[1:])
        full_path = f"../visualizations/iclr/probing_statistics_{path}.pdf"
        plt.savefig(full_path, format="pdf", bbox_inches="tight")
        print(full_path)
    plt.show()


def plot_mse_vs_ess(ess_file='wandb/ess.csv', context_length=50, savefig_path=None, conference = "iclr"):

    if conference == "iclr": ess_file = 'wandb/iclr_ess.csv'
    sns.set_style("whitegrid")
    rcParams['font.weight'] = 'bold'
    ess_df = pd.read_csv(ess_file)

    if conference == "neurips":
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


    elif conference == "iclr":
        records = []

        for _, row in ess_df.iterrows():
            dvrsty = row["diversity"]
            ess = row["ess"]
            if row["distribution"] == "uniform":
                train_dist = "Uniform"
                eval_dists = [("Normal", f"iclr/mse/context_length_gpt2_ao_normeval_dvrsty{dvrsty}.csv"),
                              ("Uniform", f"iclr/mse/context_length_gpt2_ao_dvrsty{dvrsty}.csv")]

            elif row["distribution"] == "norm":
                train_dist = "Normal"
                eval_dists = [("Normal", f"iclr/mse/context_length_gpt2_ao_normtrain_normeval_dvrsty{dvrsty}.csv"),
                              ("Uniform", f"iclr/mse/context_length_gpt2_ao_normtrain_dvrsty{dvrsty}.csv")]
            else: raise NotImplementedError

            for eval_dist, filename in [eval_dists[1]]:
                print(filename)
                assert os.path.exists(filename)

                df = pd.read_csv(filename)
                mean = df.loc[df["stat"] == "mean", str(context_length)].values[0]
                q025 = df.loc[df["stat"] == "q025", str(context_length)].values[0]
                q975 = df.loc[df["stat"] == "q975", str(context_length)].values[0]

                if q025 == 0: q025 = 1e-15
                records.append({
                    "ESS": np.log(ess),
                    "MSE": np.log(mean),
                    "q025": max(0, np.log(q025)),
                    "q975": np.log(q975),
                    "TrainDist": train_dist,
                    "EvalDist": eval_dist
                })


    if not records:
        raise ValueError("No data was loaded successfully — check file names and contents.")

    plot_df = pd.DataFrame(records).sort_values("ESS")
    plot_df["LineStyle"] = np.where(plot_df["TrainDist"] == plot_df["EvalDist"], "in-dist", "OOD")

    color_map = {
        ("Uniform", "in-dist"): "#0073E6",  # deep navy
        ("Uniform", "OOD"): "#0073E6",      # "#66A9FF",  # light navy
        ("Normal", "in-dist"): "#B51963",   # deep magenta
        ("Normal", "OOD"): "#B51963",       # "#E05D91",  # light magenta
    }

    # plt.figure(figsize=(9, 6))
    plt.figure(figsize=(10, 6))
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
    legend_properties = {"weight": "bold"}
    plt.legend(by_label.values(), by_label.keys(), title="Training distribution", fontsize = 22, prop = legend_properties)

    if conference == "neurips": ax.set_ylim([0, 6.2])
    if conference == "iclr": ax.set_ylim([0, 6.8])
    plt.xlabel("Effective support size (log)")
    plt.ylabel("In-context MSE (log)")
    plt.tight_layout()

    if savefig_path is not None:
        if conference == "iclr": plt.savefig(f"../visualizations/iclr/{savefig_path}_{context_length}.pdf", format="pdf", bbox_inches="tight")
        else: plt.savefig(f"../visualizations/{savefig_path}_{context_length}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_bar_mse_context35(models, context_length=35, savefig_path=None, conference="iclr"):
    sns.set_style("whitegrid")
    rcParams['font.weight'] = 'bold'
    colors = ["#0073E6", "#B51963", "#F2C14E", "#5FAD56"]
    alpha_standard = 0.5
    width = 0.35
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_context("paper", font_scale=1.5)

    if conference == "neurips":
        for i, model in enumerate(models):
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

    elif conference == "iclr":
        for i, suffix in enumerate(models):
            if suffix == "one_head": 
                base_file = f"iclr/mse/context_length.csv"
                ao_file = f"iclr/mse/context_length_gpt2_ao.csv"
            else: 
                base_file = f"iclr/mse/context_length_{suffix}.csv"
                ao_file = f"iclr/mse/context_length_gpt2_ao_{suffix}.csv"

            df_base = pd.read_csv(base_file)
            df_ao = pd.read_csv(ao_file)

            mse_b = df_base.loc[(df_base["stat"] == "mean"), str(context_length)].values[0]
            q025_b = df_base.loc[(df_base["stat"] == "q025"), str(context_length)].values[0]
            q975_b = df_base.loc[(df_base["stat"] == "q975"), str(context_length)].values[0]

            log_mse_b = np.log(mse_b)
            err_b = [[log_mse_b - np.log(q025_b)], [np.log(q975_b) - log_mse_b]]

            ax.bar(x[i] - width/2, log_mse_b, width, label="Full" if i == 0 else "", color=colors[i], alpha=1.0,
                yerr=err_b, capsize=5, ecolor='black')

            mse_ao = df_ao.loc[(df_ao["stat"] == "mean"), str(context_length)].values[0]
            q025_ao = df_ao.loc[(df_ao["stat"] == "q025"), str(context_length)].values[0]
            q975_ao = df_ao.loc[(df_ao["stat"] == "q975"), str(context_length)].values[0]

            log_mse_ao = np.log(mse_ao)
            err_ao = [[log_mse_ao - np.log(q025_ao)], [np.log(q975_ao) - log_mse_ao]]

            ax.bar(x[i] + width/2, log_mse_ao, width, label="AO" if i == 0 else "", color=colors[i], alpha=alpha_standard,
                yerr=err_ao, capsize=5, ecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(["1-layer, 8-head", "2-layer, 4-head", "4-layer, 2-head", "8-layer, 1-head"], fontsize=20)
    ax.set_ylabel("In-context MSE (log)", fontsize=22)
    ax.set_xlabel("Model", fontsize=22)
    if conference == "iclr" and context_length != 35:
        ax.set_ylim([0, 3.5])
        ax.set_yticklabels([0, "", 1, "", 2, "", 3], size = 20)
    else:
        ax.set_yticklabels([0, 1, 2, 3, 4, 5], size = 20)
        ax.set_ylim([0,5.4])
    ax.legend(fontsize=22)
    plt.tight_layout()

    if savefig_path:
        if conference == "iclr": plt.savefig(f"../visualizations/iclr/mse_bar_{context_length}.pdf", format="pdf", bbox_inches="tight")
        else: plt.savefig(f"../visualizations/mse_bar_{context_length}.pdf", format="pdf", bbox_inches="tight")
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
    plt.ylabel("In-context MSE")
    legend_properties = {"weight": "bold"}
    plt.legend(title="Model", fontsize=19, prop = legend_properties, loc = "upper right")

    plt.tight_layout()


    if savefig_path:
        plt.savefig(f"../visualizations/mse_sde_{savefig_path}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_sde_bar_chart(models, examples = None, conference = "iclr", train_steps = 50000, number_events = None,
                       lamb = 40, poisson = 1, savefig_path = None):
    random.seed(19459)
    sns.set_style("whitegrid")
    rcParams['font.weight'] = "bold"
    colors = ["#0073E6", "#B51963", "#5FAD56", "#F2C14E"]

    sns.set_context("paper", font_scale=2)
    width = 0.5
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_context("paper", font_scale=1.5)

    def compute_internal_mse(data, examples = None, poisson = 1):
        if poisson and len(examples) > 1: 
            assert float(examples).is_integer()
            sub_data = data[data["Number of in-context examples"] == examples].reset_index()
            mse_sub = sub_data["MSE"]
            total_number_examples = len(sub_data)
        else: 
            mse_sub = data["MSE"]
            total_number_examples = len(mse_sub)


        quantile_indices = random.choices(range(total_number_examples), k = total_number_examples)
        # print(total_number_examples)
        quantiles = np.quantile(mse_sub[quantile_indices], q=[0.025, 0.975])

        mean_mse = np.mean(mse_sub)
        lower_q = max(2 * mean_mse - quantiles[1], 0)
        upper_q = max(2 * mean_mse - quantiles[0], 0)

        err_ao = [[np.maximum(mean_mse - lower_q, 0)], [np.maximum(upper_q - mean_mse, 0)]]

        return mean_mse, err_ao

    if conference == "neurips":
        for i, model in enumerate(models):
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
        
    elif conference == "iclr":
        for i, family in enumerate(models):
            parts = [""]
            parts.append(str(train_steps))
            parts.append(f"{lamb}lamb")
            parts.append(family)
            if number_events != None:
                parts.append(f"{number_events}events")


            filename = "_".join(parts) + ".csv"

            base_file = f"iclr/sde/" + filename
            df = pd.read_csv(base_file, header=None)
            
            if len(examples) > 1:
                context_lengths = np.array(df.iloc[2, 2:]).astype(int)
                mse_values = np.array(df.iloc[1, 2:]).astype(float)

                data = pd.DataFrame({"Number of in-context examples": context_lengths, "MSE": mse_values})

                for ex in examples:
                    mean_mse, err_ao = compute_internal_mse(data, ex)

                    if ex == 8:
                        ax.bar(x[i] - width/2, mean_mse, width/2, color=colors[i], alpha=1,
                            yerr=err_ao, capsize=5, ecolor='black', label = f"{ex} time steps" if "rnn" in family else '')
                    elif ex == 22:
                        ax.bar(x[i], mean_mse, width/2, color=colors[i], alpha=0.7,
                            yerr=err_ao, capsize=5, ecolor='black', label = f"{ex} time steps" if "rnn" in family else '')
                    elif ex == 26:
                        ax.bar(x[i] + width/2, mean_mse, width/2, color=colors[i], alpha=0.4,
                            yerr=err_ao, capsize=5, ecolor='black', label = f"{ex} time steps" if "rnn" in family else '')
                    
            else:
                mse_values = np.array(df.iloc[1, 2:]).astype(float)
                data = pd.DataFrame({"MSE": mse_values})

                if not poisson: mean_mse, err_ao = compute_internal_mse(data, poisson = poisson)
                else: mean_mse, err_ao = compute_internal_mse(data, examples=examples)
                # mean_mse, err_ao = np.power(mean_mse, 0.2), np.power(err_ao, 0.2)
                ax.bar(x[i], mean_mse, width, color=colors[i], alpha=1,
                       yerr=err_ao, capsize=5, ecolor='black')
                print(mean_mse.round(4))
        
        
    ax.set_xticks(x)

    if poisson: ax.set_xticklabels(["Elman RNN", "LSTM", "GRU", "GPT-2"], fontsize=20)
    else: ax.set_xticklabels(["GPT2-SDE", "GPT2-SDE\nwith constant " + r'$\Delta t$', "GPT2-ODE\nwith constant " + r'$\Delta t$'])

    # ax.set_yticklabels([0.00, "", 0.01, "", 0.02, "", 0.03], size = 20)
    ax.set_yticklabels([0.00, "", "", 0.01, "", "", 0.02], size = 20)
    ax.set_yticks([0, 0.01/3, 0.02/3, 0.01, 0.04/3, 0.05/3, 0.02])
    if poisson: ax.set_ylim([0, 0.02])
    plt.xlabel("Model", fontsize = 22)
    plt.ylabel("In-context MSE", fontsize = 22)
    if poisson and len(examples) > 1: plt.legend(title="Evaluated at", fontsize=16, loc = "upper center", ncol = 3)

    plt.tight_layout()

    if savefig_path:
        if conference == "neurips": plt.savefig(f"../visualizations/mse_sde_bar_{savefig_path}.pdf", format="pdf", bbox_inches="tight")
        else: 
            parts.pop(0)
            figname = "../visualizations/iclr/" + "_".join(parts) + ".pdf"
            plt.savefig(figname, format = "pdf", bbox_inches = "tight")
    plt.show()


def plot_sde_chart(family = "gpt2_sde", ao = False, savefig_path = None, conference = "iclr", train_steps = 50000, lamb = 5, ode = 0, o_dims = 5, poisson = 1, n_layer = 8, n_head = 1, position = 0, dimension = 0, diffusion = 20, number_events = None):
    if conference == "neurips": filename = f"mse/prediction_chart_sde_{family}{'_ao' if ao else ''}.csv"
    else: 
        parts = ["iclr/sde/prediction/chart", str(train_steps)]
        
        if lamb != 5:
            parts.append(f"{str(lamb)}lamb")
        if ode:
            parts.append("ode")
        if o_dims != 5:
            parts.append(f"{o_dims}dim")
        if family != "gpt2_sde":
            parts.append(family)
        if poisson != 1:
            parts.append(f"poisson{str(poisson)}")
        if n_layer != 8:
            parts.append(f"{n_layer}l")
        if n_head != 1:
            parts.append(f"{n_head}h")
        if diffusion != 20:
            parts.append(f"diffusion{diffusion}")
        if number_events != None:
            parts.append(f"{number_events}events")

        ### to be deleted
        if "gpt" in family and number_events == None:
            parts.append(f"theta{position}")
            parts.append(f"dimension{dimension}")
        ### to be deleted

        filename = "_".join(parts) + ".csv"

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
    if conference == "neurips":
        ax.set_ylim([0, 4])
        ax.set_xticklabels(labels = [0.0, 0.0, 0.02, 0.04, 0.06, 0.08], size = 20)
        ax.set_yticklabels(labels = [0, "", 1, "", 2, "", 3, "", 4], size = 20)
    elif conference == "iclr":
        if lamb == 20:
            ax.set_ylim([0, 2.5])
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticks([0, 0.5, 1, 1.5, 2])
            ax.set_xticklabels(labels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size = 20)
            ax.set_yticklabels(labels = [0, "", 1, "", 2], size = 20)
        elif lamb == 40:
            ax.set_ylim([0, 3])
            ax.set_xlim([0.0, 0.5])
            ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
            # ax.set_yticks([1, 1.5, 2])
            # ax.set_yticks([0, 1, 2, 3, 4])

            ax.set_xticklabels(labels = ["", 0.1, 0.2, 0.3, 0.4, 0.5], size = 20)
            ax.set_yticklabels(labels = [0, "", 1, "", 2, "", 3], size = 20)
            # ax.set_yticklabels(labels = [0, 1, 2, 3, 4], size = 20)

    # label = f"tht {position}\ndim {dimension}"
    # ax.text(0.1, 0.5, label, transform=ax.transAxes,
    #         ha='right', va='bottom', fontsize=22)

    ax.set_ylabel("Lotka-Volterra concentration", fontsize=22)
    ax.set_xlabel("Time", fontsize=22)
    
    plt.legend(title = "Sequence", fontsize = 16, loc = "upper center", ncol = 3)
    plt.tight_layout()
    if savefig_path:
        if conference == "neurips": plt.savefig(f"../visualizations/chart_sde_{savefig_path}.pdf", format="pdf", bbox_inches="tight")
        elif conference == "iclr":
            parts[0] = "../visualizations/iclr/chart_sde"
            figpath = "_".join(parts) + ".pdf"
            plt.savefig(figpath, format = "pdf", bbox_inches = "tight")
            
    else: plt.show()


def plot_phase_transitions(steps = 100000, with_theoretical = False, continuation = False, savefig_path = False):
    if continuation:
        assert steps == 1000000
        file = "wandb/training_phases_cont.csv"
    elif steps == 100000: file = "wandb/training_phases_100.csv"
    elif steps == 1000000: file = "wandb/training_phases_1mio.csv"
    else: file = "wandb/training_phases.csv"
    
    df = pd.read_csv(file)
    df = df.apply(pd.to_numeric)

    sns.set_style("whitegrid")
    rcParams['font.weight'] = "bold"

    if continuation:
        df = df.iloc[:, [0,1,4]]
        colors = color_palette(number = 5)
    elif steps == 100000: 
        df = df.iloc[:, [0,1,4,7,10,13]]
        colors = color_palette(number = 5)
    elif steps == 1000000: 
        df = df.iloc[:, [0,1]]
        colors = color_palette(number = 5)
    else: 
        df = df.iloc[:, [0,1,4,7,10]]
        colors = color_palette(number = 4)

    plt.figure(figsize=(10, 6))
    sns.set_context("paper", font_scale=2)

    if continuation:
        plt.plot(df["Step"], np.log(df.iloc[:, 2]), color = colors[4], linewidth = 2, label = "Counterfactual")
        plt.plot(df["Step"], np.log(df.iloc[:, 1]), color = colors[1], linewidth = 2, label = "Continuation")
    elif steps == 100000:
        plt.plot(df["Step"], np.log(df.iloc[:, 1]), color = colors[4], linewidth = 2, label = "5-dimensional")
        plt.plot(df["Step"], np.log(df.iloc[:, 2]), color = colors[3], linewidth = 2, label = "4-dimensional")
        plt.plot(df["Step"], np.log(df.iloc[:, 3]), color = colors[2], linewidth = 2, label = "3-dimensional")
        plt.plot(df["Step"], np.log(df.iloc[:, 4]), color = colors[1], linewidth = 2, label = "2-dimensional")
        plt.plot(df["Step"], np.log(df.iloc[:, 5]), color = colors[0], linewidth = 2, label = "1-dimensional")
    elif steps == 1000000:
        plt.plot(df["Step"], np.log(df.iloc[:, 1]), color = colors[0], linewidth = 2, label = "Training phases")
    else: raise NotImplementedError

    if with_theoretical:
        loss_file = "iclr/mse/estimated_loss.csv"
        df_loss = pd.read_csv(loss_file)
        upper, lower = np.log(df_loss["30"][0]), np.log(df_loss["50"][0])
        plt.fill_between(df["Step"], lower, upper, color=colors[2], alpha=0.3, label='Optimal MSE for 30 to 50 examples')

    ax = plt.gca()

    ax.set_ylabel("In-context training loss (log)", fontsize=22)
    ax.set_xlabel("Training iterations (×10³)", fontsize=22)

    ax.set_xlim([0,steps])
    if steps == 100000: 
        ax.set_ylim([-3, 4])
        ax.set_xticklabels(["0", "20", "40", "60", "80", "100"])
    elif steps == 1000000:
        ax.set_ylim([-2, 3])
        ax.set_xticks([0, 200000, 400000, 600000, 800000, 1000000])
        ax.set_xticklabels(["0", "200", "400", "600", "800", "1000"])
    
    plt.legend(title = "Sequence", fontsize = 16, loc = "upper center", ncol = 2)
    plt.tight_layout()
    if savefig_path:
        filedir = "../visualizations/iclr/"
        filename = "phase_transitions"
        if steps != 100000: filename += "_" + str(steps)
        if continuation: filename += "_cont"
        figpath = filedir + filename + ".pdf"
        plt.savefig(figpath, format = "pdf", bbox_inches = "tight")
            
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
    # plot_mse_with_ci(models_attention, savefig_path=savefig_path)
    # plot_mse_with_ci(["eightlayer_new_theta", "eightlayer_ao_cont"]) #, "predict_y_ao_y"])
    
    # plot_probing_statistcs(probe_diff=True, probing_types=["theta", "beta"], savefig_path="diff")
    # plot_probing_statistcs(probing_types=["theta", "beta"], savefig_path="cum")

    # savefig_path="handwritten_embeds_E_5"
    # plot_mse_with_ci(["five_embeds"], savefig_path=savefig_path)
    # savefig_path="handwritten_embeds_E_6"
    # plot_mse_with_ci(["six_embeds"], savefig_path=savefig_path)
    # savefig_path="handwritten_embeds_E_20"
    # plot_mse_with_ci(["twenty_embeds"], savefig_path=savefig_path)
    # plot_mse(models_attention, savefig_path=savefig_path)
    # for i in [1, 35, 45]:
    #     plot_mse_vs_ess(context_length=i, savefig_path="ess_both_dist")

    ### iclr
    rnn_type = ["rnn_2l_8h", "lstm_2l_8h", "gru_2l_8h", "12l_8h"]
    attention = ["gpt2_mlp", "gpt2_ao_2l", "gpt2_ao_4l", "gpt2_ao", "gpt2_ao_12l_8h"]
    plot_mse(rnn_type, conference='iclr')#, savefig_path="attention")
    plot_mse_with_ci(rnn_type, conference='iclr', savefig_path="rnn")
    heads = ["1l_8h", "2l_4h", "4l_2h", "one_head"]
    optimal = ["optimal", "12l_8h_1"]
    minex = ["optimal", "12l_8h_1_minex30", "12l_8h_1_constz14_minex30"]
    onemio = ["optimal", "12l_8h_1000000steps"]
    # plot_mse(onemio, savefig_path="onemio")
    # plot_mse_with_ci(onemio, savefig_path="onemio")

    mae = ["gpt2_ao_elossmae", "gpt2_ao_tlossmae_elossmae"] #"gpt2_ao", "gpt2_ao_tlossmae"]# ]
    # plot_mse(mae, savefig_path="mae")
    # plot_mse_with_ci(mae, savefig_path="mae")
    # plot_mse(minex, savefig_path="minex")
    # plot_mse_with_ci(minex, savefig_path="minex")
    # plot_bar_mse_context35(heads, 2, "anything")
    # plot_bar_mse_context35(heads, 35, "anything")
    # plot_bar_mse_context35(heads, 45, "anything")

    continuation = ["gpt2_ao_1000000steps", "gpt2_ao_cont_cont_1000000steps"]
    # plot_mse_with_ci(continuation, conference='iclr', savefig_path="ao_onemio")
    # plot_mse(continuation, conference='iclr', savefig_path="ao_onemio")
    
    continuation = ["12l_8h_1000000steps", "gpt2_cont_12l_8h_cont_1000000steps"]
    # plot_mse_with_ci(continuation, conference='iclr', savefig_path="cont_onemio")
    # plot_mse(continuation, conference='iclr', savefig_path="cont_onemio")
    # plot_phase_transitions(1000000, continuation = True, savefig_path=True)
    # plot_probing_statistcs(
    #     probing_types = ["theta", "beta"],
    #     arguments = {
    #         "probe_diff": False,
    #         "constant_z": 14,
    #         "min_examples": 30,
    #         "n_embd": 256,
    #         "n_layer": 8,
    # })
    # plot_probing_statistcs(["theta", "beta"], statistic="adj_r_squared", savefig_path="bla")
    
    # plot_estimated_mse(savefig_path="bla")
    # plot_phase_transitions(savefig_path=True, with_theoretical=True)
    # for t in range(8):
    #     for d in range(5):
    # plot_sde_chart(family = "gpt2_sde", lamb = 40, poisson = 1, n_layer = 12, n_head = 8, ode = 0, #diffusion = 10,
    #                 position=1, dimension=0, savefig_path="true")
    # plot_sde_chart(family = "gpt2_sde", lamb = 40, poisson = 0, n_layer = 12, n_head = 8, ode = 0, #diffusion = 10,
    #                 position=4, dimension=2, savefig_path="true")
    # plot_sde_chart(family = "gpt2_sde", lamb = 40, poisson = 1, n_layer = 12, n_head = 8, ode = 0, number_events = 20, savefig_path="true") #diffusion = 10,
    # plot_sde_chart(family = "gru_sde", lamb = 40, poisson = 1, n_layer = 2, n_head = 8, ode = 0, number_events = 20)#, savefig_path="true") #diffusion = 10,
    # plot_sde_chart(family = "rnn_sde", lamb = 40, poisson = 1, n_layer = 2, n_head = 8, ode = 0, number_events = 20)#, savefig_path="true") #diffusion = 10,
    # plot_sde_chart(family = "lstm_sde", lamb = 40, poisson = 1, n_layer = 2, n_head = 8, ode = 0, number_events = 20)#, savefig_path="true") #diffusion = 10,
                    # position=2, dimension=1, savefig_path="true")

    sde = ["rnn_sde_2l_8h", "lstm_sde_2l_8h", "gru_sde_2l_8h", "12l_8h"]
    
    ode = ["12l_8h", "poisson0_12l_8h", "ode_poisson0_12l_8h"]
    # plot_sde_bar_chart(sde, examples = [20], number_events = 20)
    # plot_sde_bar_chart(models = sde, examples = [8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], poisson=1) #, savefig_path="true")
    # plot_mse_vs_ess(context_length=2, conference='iclr', savefig_path='ess')
    # plot_mse_vs_ess(context_length=2, conference='iclr', savefig_path='ess_normal_eval')
    # plot_mse_vs_ess(context_length=35, conference='iclr', savefig_path='ess')#, savefig_path='ess_normal_eval')
