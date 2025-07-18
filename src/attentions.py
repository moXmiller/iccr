print("Started")
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import rcParams


def retrieve_attentions(layer, head, continuation = False, model_size = "small", ao = False, position = -1, itr = None, sde_used = False, only_counterfactual = False,
                        predict_y = False):
    # if continuation: raise NotImplementedError
    if not os.path.exists("eval/attentions"):
        os.makedirs('eval/attentions')
    if position == -1:
        attention_path = f"eval/attentions/attentions_{model_size}_{layer}_layer_{head}_head{'_ao' if ao else ''}_{itr}_m{'_cont' if continuation else ''}{'_y' if predict_y else ''}{'_sde' if sde_used else ''}.csv"
    else:
        attention_path = f"eval/attentions/attentions_{model_size}_{layer}_layer_{head}_head{'_ao' if ao else ''}_itr{itr}_pos{position}{'_cont' if continuation else ''}{'_y' if predict_y else ''}{'_sde' if sde_used else ''}.csv"
    df_att = pd.read_csv(attention_path)
    if predict_y: df_att = df_att.iloc[:-1,:-1]
    _, c = df_att.shape
    if sde_used:
        n = (c-1) / 4
        assert n.is_integer()
        n = int(n)
        labels = [[f"X{i}", f"Y{i}"] for i in range(n)] + [["Z"]] + [[f"X{i}CF", f"Y{i}CF"] for i in range(n)]
        labels = sum(labels, [])
    else:
        # n = (c-2) / 2
        if predict_y or "embeds" in model_size: n = (c-1-2) / 2
        assert n.is_integer()
        n = int(n)
        labels = [[f"X{i}", f"Y{i}"] for i in range(n)] + [["Z"]] + [["XzCF"]] + [["YzCF"]] # + [["YzCF"]] added only for six_embeds
        labels = sum(labels, [])

    df_att = df_att.set_axis(labels, axis = 0)
    df_att = df_att.set_axis(labels, axis = 1)
    return df_att


def colormap(continuous = True):
    colors = ["#0073E6", "#B51963", "#011638", "#5FAD56", "#F2C14E"]
    if continuous: 
        cmap = LinearSegmentedColormap.from_list("attn_heatmap", colors=["#FFFFFF", colors[2]])
        return cmap
    else: return colors


def colorbar():
    cmap = colormap()
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(10, 0.3))
    cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')

    plt.savefig("visualizations/custom_colorbar.pdf", bbox_inches='tight')
    plt.show()


def construct_heatmap(layer, head, continuation = False, model_size = "small", ao = False, only_counterfactual = False, savefig = False, position = -1, itr=None, sde_used = False,
                      predict_y = False):
    # if continuation: raise NotImplementedError
    rcParams['font.weight'] = "bold"
    data = retrieve_attentions(layer, head, continuation, model_size, ao, position = position, itr = itr, sde_used=sde_used, only_counterfactual=only_counterfactual, predict_y=predict_y)
    if only_counterfactual:
        r, c = data.shape
        assert r == c
        ct = (c-1) / 2 # ct: context
        assert ct.is_integer()
        data = data.iloc[int(ct)::,int(ct)::]
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.set_context("paper", font_scale=1.5)

    plot_data = data.iloc[:,:]
    cmap = colormap()
    sns.heatmap(plot_data.iloc[:,:], rasterized=True, cmap=cmap, vmin=0, vmax=1, cbar=False)

    plt.ylabel("Processed query token", fontsize = 22)
    plt.xlabel("Attended-to keys", fontsize = 22)

    if sde_used:
        color = colormap(continuous=False)[2]
        layer_label = f"{layer+1}{'st' if layer==0 else 'nd' if layer==1 else 'rd' if layer==2 else 'th'} layer"
        ax.text(0.9, 0.9, layer_label, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=22, color=color)

    if not savefig: plt.show()
    else:
        if not os.path.exists('visualizations'):
            os.makedirs('visualizations')
        figpath = f"visualizations/attentions_layer_{layer}_head_{head}_{model_size}{'_ao' if ao else ''}{'_sde' if sde_used else ''}.pdf"
        plt.savefig(figpath, format = "pdf", bbox_inches = "tight")
        plt.close()


def subplots_heatmap(continuation, model_size, layer, only_counterfactual = False, ao = False, savefig = False, position = -1, itr=None, sde_used = False, predict_y = False):
    # if continuation: raise NotImplementedError
    if layer == "all": head = 0
    fig = plt.figure(figsize=(9, 9))
    if model_size == "tiny": 
        n_heads = 2
        fig.subplots_adjust(hspace=0.4, wspace=1)
    elif model_size == "small": 
        n_heads = 4
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
    elif model_size == "standard": 
        n_heads = 8
        fig.subplots_adjust(hspace=0.2, wspace=0.4)
    elif "fourlayer" in model_size: 
        n_layers = 4
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
    elif "twolayer" in model_size: 
        n_layers = 2
        fig.subplots_adjust(hspace=0.4, wspace=1)
    elif "threelayer" in model_size: 
        n_layers = 3
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
    elif ("eightlayer" in model_size) or (model_size == "one_mlp") or ("predict_y" in model_size):
        n_layers = 8
        fig.subplots_adjust(hspace=0.2, wspace=0.4)
    cmap = colormap()
    
    # we distinguish between the models with multiple heads per layer and single-head layers
    if model_size in ["tiny", "small", "standard"]:
        for h in range(n_heads):
            df_att = retrieve_attentions(layer, h, continuation, model_size, ao, position=position, itr = itr, sde_used=sde_used, only_counterfactual=only_counterfactual, predict_y=predict_y)
            if only_counterfactual and not sde_used:
                r, c = df_att.shape
                assert r == c
                df_att = df_att.iloc[:,:]

            if only_counterfactual and sde_used:
                r, c = df_att.shape
                assert r == c
                ct = (c-1) / 2
                assert ct.is_integer()
                df_att = df_att.iloc[int(ct)::,int(ct)::]

            if n_heads == 2: ax = fig.add_subplot(2, 1, h + 1)
            elif n_heads == 4: ax = fig.add_subplot(2, 2, h + 1)
            elif n_heads == 8: ax = fig.add_subplot(4, 2, h + 1)

            sns.heatmap(df_att,ax=ax, rasterized=True, cmap=cmap, vmin=0, vmax=1, cbar=False)
    
    # all one-head-per-layer heads in one plot
    else:
        for l in range(n_layers):
            df_att = retrieve_attentions(l, head, continuation, model_size, ao, position=position, itr = itr, sde_used=sde_used, only_counterfactual=only_counterfactual, predict_y=predict_y)
            if only_counterfactual and not sde_used:
                r, c = df_att.shape
                assert r == c
                df_att = df_att.iloc[int(r/2)::,:]

            if only_counterfactual and sde_used:
                r, c = df_att.shape
                assert r == c
                ct = (c-1) / 2 # ct: context
                assert ct.is_integer()
                df_att = df_att.iloc[int(ct)::,int(ct)::]

            if n_layers == 2: ax = fig.add_subplot(2, 1, l + 1)
            elif n_layers == 4: ax = fig.add_subplot(2, 2, l + 1)
            elif n_layers == 8: ax = fig.add_subplot(4, 2, l + 1)

            sns.heatmap(df_att,ax=ax, rasterized=True, cmap=cmap, vmin=0, vmax=1, cbar=False)
    if not savefig: plt.show()
    else:
        # if continuation: raise NotImplementedError
        if not os.path.exists('visualizations'):
            os.makedirs('visualizations')
        if model_size in ["tiny", "small", "standard"]:
            figpath = f"visualizations/attentions_layer_{layer}_{model_size}{'_ao' if ao else ''}{'_cont' if continuation else ''}{'_y' if predict_y else ''}.pdf"
        else:
            figpath = f"visualizations/attentions_{model_size}{'_ao' if ao else ''}{'_cont' if continuation else ''}{'_y' if predict_y else ''}.pdf"
        plt.savefig(figpath, format = "pdf", bbox_inches = "tight")
        plt.close()


if __name__ == "__main__":
    # subplots_heatmap(continuation=False, model_size="predict_y", layer="all", ao = 1, position = 0, itr = 2, predict_y=1)
    # colorbar()
    for i in range(8):
        construct_heatmap(layer = i, head = 0, continuation= False, model_size = "five_embeds", ao = True, itr=0, only_counterfactual=False, predict_y = False, position = -1)