print("Started")
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import rcParams


def retrieve_attentions(layer, head, continuation = False, model_size = "small", ao = False, position = -1, itr = None, sde_used = False, only_counterfactual = False,
                        predict_y = False, conference = "iclr", n_layer = 8, n_head = 1, data = "gaussian", lamb = 5, ode = 0, o_dims = 5, family = "gpt2", poisson = 1,
                        n_embd = 256, transformation = "addlin", train_steps = 50000):
    # if continuation: raise NotImplementedError
    if conference == "neurips":
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

    elif conference == "iclr":
        dir_path = f"eval/iclr/attentions/{data}/{n_layer}layers{n_head}heads/"
        parts = [dir_path + "attn"]
            
        if n_layer != 8:
            parts.append(f"{n_layer}layers")
        if n_head != 1:
            parts.append(f"{n_head}heads")
        if ao:
            parts.append("ao")
        if position != -1:
            parts.append(f"pos{position}")


        if data == "sde":
            if lamb != 5:
                parts.append(f"{int(lamb)}lamb")
            if ode:
                parts.append("ode")
            if o_dims != 5:
                parts.append(f"{o_dims}dim")
            if family != "gpt2_sde":
                parts.append(family)
            if poisson != 1:
                parts.append(f"poisson{poisson}")
        else:
            if family != "gpt2":
                parts.append(family)
            if n_embd != 256:
                parts.append(f"{n_embd}embd")
            if continuation:
                parts.append("cont")
            if transformation != "addlin":
                parts.append(transformation)
            if o_dims != 5:
                parts.append(str(o_dims))
            if train_steps != 50000:
                parts.append(f"{train_steps}steps")


        parts.append(f"l{layer}")
        parts.append(f"h{head}")
        parts.append(f"itr{itr}")
        attention_path = "_".join(parts) + ".csv"

        df_att = pd.read_csv(attention_path)

        _, c = df_att.shape
        if sde_used:
            n = (c-1) / 4
            assert n.is_integer()
            n = int(n)
            labels = [[f"X{i}", f"Y{i}"] for i in range(n)] + [["Z"]] + [[f"X{i}CF", f"Y{i}CF"] for i in range(n)]
            labels = sum(labels, [])
            df_att = df_att.set_axis(labels, axis = 1)
        elif data == "gaussian":
            n = (c-1-2) / 2
            assert n.is_integer()
            n = int(n)
            xlabels = [[f"", f"Y{i}"] for i in range(n)] + [["Z"]] + [[""]] + [["YzCF"]] # + [["YzCF"]] added only for six_embeds
            ylabels = [[f"X{i}", f"Y{i}"] for i in range(n)] + [["Z"]] + [["XzCF"]] + [["YzCF"]] # + [["YzCF"]] added only for six_embeds
            xlabels = sum(xlabels, [])
            ylabels = sum(ylabels, [])
            df_att = df_att.set_axis(ylabels, axis = 0)
            df_att = df_att.set_axis(xlabels, axis = 1)

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


def construct_heatmap(layer, head, continuation = False, model_size = "small", ao = False, only_counterfactual = False, savefig = False, position = -1, itr=None,
                      predict_y = False, conference = "iclr", n_layer = 8, n_head = 1, data = "gaussian", lamb = 5, ode = 0, o_dims = 5, family = "gpt2", poisson = 1,
                      n_embd = 256, transformation = "addlin", train_steps = 50000):
    # if continuation: raise NotImplementedError
    rcParams['font.weight'] = "bold"
    sde_used = (data == "sde")
    df = retrieve_attentions(layer, head, continuation, model_size, ao, position = position, itr = itr, sde_used=sde_used, only_counterfactual=only_counterfactual, predict_y=predict_y,
                             conference = conference, n_layer = n_layer, n_head = n_head, data = data, lamb = lamb, ode = ode, o_dims = o_dims, family = family, poisson = poisson,
                             n_embd = n_embd, transformation = transformation, train_steps=train_steps)

    if only_counterfactual:
        r, c = df.shape
        assert r == c
        if conference == "neurips":
            ct = (c-1) / 2 # ct: context
            assert ct.is_integer()
            df = df.iloc[int(ct)::,int(ct)::]
        if conference == "iclr":
            if data == "sde":
                ct = 42
                df = df.iloc[int(ct)::,int(ct):]
            elif data == "gaussian":
                ct = (c-1) / 2 # ct: context
                assert ct.is_integer()
                df = df.iloc[int(ct)::, int(ct)::]
                print(1, df.iloc[:, 0::2].sum(1).max())
                print(2, df.iloc[:, 1::2].sum(1).max())

    fig, ax = plt.subplots(figsize=(20, 1))
    sns.set_context("paper", font_scale=1)

    plot_data = df.iloc[100:,:]
    cmap = colormap()
    if data == "sde" or conference == "neurips": sns.heatmap(plot_data.iloc[:,:], rasterized=True, cmap=cmap, vmin=0, vmax=1, cbar=False)
    else: sns.heatmap(plot_data.iloc[:,:], rasterized=True, cmap=cmap, cbar = False)

    plt.ylabel("Processed query token", fontsize = 22)
    plt.xlabel("Attended-to keys", fontsize = 22)

    if sde_used:# or n_layer == 12:
        color = colormap(continuous=False)[2]
        layer_label = f"{layer+1}{'st' if layer==0 else 'nd' if layer==1 else 'rd' if layer==2 else 'th'} layer"
        if n_head > 1:
            layer_label = layer_label + "\n" + f"{head+1}{'st' if head==0 else 'nd' if head==1 else 'rd' if head==2 else 'th'} head"
        ax.text(0.9, 0.9, layer_label, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=22, color=color)
            
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=8)
    # xticks = [[f"X{i}", f"Y{i}"] for i in range(50)] + ["Z"] + ["X_Z"] + ["Y_Z"]
    # yticks = ["Z"] + ["X_Z"] + ["Y_Z"]
    # ax.set_yticklabels(yticks)
    # ax.set_xticklabels(xticks)

    if not savefig: plt.show()
    else:
        if not os.path.exists('visualizations/iclr'):
            os.makedirs('visualizations/iclr')
        if conference == "neurips":
            figpath = f"visualizations/attentions_layer_{layer}_head_{head}_{model_size}{'_ao' if ao else ''}{'_sde' if sde_used else ''}.pdf"
        elif conference == "iclr":
            parts = ['visualizations/iclr/attn']
            if n_layer != 8:
                parts.append(f"{n_layer}layers")
            if n_head != 1:
                parts.append(f"{n_head}heads")
            if ao:
                parts.append("ao")
            if position != -1:
                parts.append(f"pos{position}")


            if data == "sde":
                if lamb != 5:
                    parts.append(f"{int(lamb)}lamb")
                if ode:
                    parts.append("ode")
                if o_dims != 5:
                    parts.append(f"{o_dims}dim")
                if family != "gpt2_sde":
                    parts.append(family)
                if poisson != 1:
                    parts.append(f"poisson{poisson}")
            else:
                if family != "gpt2":
                    parts.append(family)
                if n_embd != 256:
                    parts.append(f"{n_embd}embd")
                if continuation:
                    parts.append("cont")
                if transformation != "addlin":
                    parts.append(transformation)
                if o_dims != 5:
                    parts.append(str(o_dims))

            parts.append(f"l{layer}")
            parts.append(f"h{head}")
            parts.append(f"itr{itr}")

            figpath = "_".join(parts) + ".pdf"
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
    construct_heatmap(layer = 0, head = 0, itr = 0, data = "sde", family = "gpt2_ao_sde", ao = 1, only_counterfactual=True, lamb = 40, poisson = 0)    # hard-coded above
    n_layer = 12
    n_head = 8
    # construct_heatmap(6, 5, position = 0, n_layer = 12, n_head = 8, o_dims = 1, train_steps = 1000000, itr = 0)
    # for l in range(n_layer):
    #     for h in range(n_head):
    #         # construct_heatmap(layer = i, head = 0, continuation= False, model_size = "five_embeds", ao = True, itr=0, only_counterfactual=False, predict_y = False, position = -1)
    #         print(l, h)
    #         # construct_heatmap(layer = l, head = h, itr = 0, data = "sde", family = "gpt2_ao_sde", ao = 1, only_counterfactual=False, lamb = 40, poisson = 1, n_layer = n_layer, n_head = n_head)
    #         construct_heatmap(l, h, position = -1, n_layer = n_layer, n_head = n_head, o_dims = 1, train_steps = 1000000, itr = 2)

    for itr in range(15):
        print(itr)
        construct_heatmap(6, 5, position = -1, n_layer = n_layer, n_head = n_head, o_dims = 1, train_steps = 1000000, itr = itr, savefig=True)
