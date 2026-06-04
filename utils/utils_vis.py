import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import seaborn as sns



#### mainly from https://github.com/azencot-group/ImagenTime/blob/main/utils/utils_vis.py

def load_data(name,train=True):
    
    if train:
        try:
            data = np.load(f"./data/{name}/train.npy")
        except:
            data = torch.load(f"./data/{name}/train.pt").numpy()
    else:
        try:
            data = np.load(f"./data/{name}/test.npy")
        except:
            data = torch.load(f"./data/{name}/test.pt").numpy()

    return data

def reduce_channels(sig):
    """
    Converts [N, seq_len, channels] -> [N, seq_len]
    by averaging over channels.
    """
    sig = np.asarray(sig)

    if sig.ndim != 3:
        raise ValueError(f"Expected shape [N, seq_len, channels], got {sig.shape}")

    return np.mean(sig, axis=2)


def prepare_all_data(ori_sig, gen_dict, max_samples=1000):
    """
    ori_sig: real data, shape [N, T, C]
    gen_dict: dict method_name -> synthetic data, each [N, T, C]

    Returns:
        data_all: [total_samples, seq_len]
        labels: list of labels for plotting
    """
    ori_sig = np.asarray(ori_sig)

    sample_num = min(
        [max_samples, len(ori_sig)] + [len(v) for v in gen_dict.values()]
    )

    idx = np.random.permutation(sample_num)

    data_list = []
    labels = []

    prep_ori = reduce_channels(ori_sig[idx])
    data_list.append(prep_ori)
    labels.extend(["Real"] * sample_num)

    for method, gen_sig in gen_dict.items():
        gen_sig = np.asarray(gen_sig)
        prep_gen = reduce_channels(gen_sig[idx])

        data_list.append(prep_gen)
        labels.extend([method] * sample_num)

    data_all = np.concatenate(data_list, axis=0)

    return data_all, labels, sample_num


def PCA_plot_all(data_all, labels, save_path):
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(data_all)

    plt.figure(figsize=(7, 6))

    unique_labels = list(dict.fromkeys(labels))

    for label in unique_labels:
        mask = np.array(labels) == label
        plt.scatter(
            pca_results[mask, 0],
            pca_results[mask, 1],
            alpha=0.25,
            label=label,
            s=12,
        )

    plt.legend()
    plt.title("PCA plot")
    plt.xlabel("x-pca")
    plt.ylabel("y-pca")
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.show()


def TSNE_plot_all(data_all, labels, save_path):
    perplexity = min(40, data_all.shape[0] - 1)

    tsne = TSNE(
        n_components=2,
        verbose=1,
        perplexity=perplexity,
        max_iter=300,
        random_state=0,
    )

    tsne_results = tsne.fit_transform(data_all)

    plt.figure(figsize=(7, 6))

    unique_labels = list(dict.fromkeys(labels))

    for label in unique_labels:
        mask = np.array(labels) == label
        plt.scatter(
            tsne_results[mask, 0],
            tsne_results[mask, 1],
            alpha=0.25,
            label=label,
            s=12,
        )

    plt.legend()
    plt.title("t-SNE plot")
    plt.xlabel("x-tsne")
    plt.ylabel("y-tsne")
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.show()



def plot_pca_tsne_combined(data_all, labels, save_path):
    labels = np.asarray(labels)

    # PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(data_all)

    # t-SNE
    perplexity = min(40, data_all.shape[0] - 1)
    tsne = TSNE(
        n_components=2,
        verbose=1,
        perplexity=perplexity,
        max_iter=300,
        random_state=0,
    )
    tsne_results = tsne.fit_transform(data_all)

    unique_labels = list(dict.fromkeys(labels.tolist()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # PCA subplot
    ax = axes[0]
    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            pca_results[mask, 0],
            pca_results[mask, 1],
            alpha=0.25,
            label=label,
            s=12,
        )
    ax.set_title("PCA")
    #ax.set_xlabel("PC 1")
    #ax.set_ylabel("PC 2")

    # t-SNE subplot
    ax = axes[1]
    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            tsne_results[mask, 0],
            tsne_results[mask, 1],
            alpha=0.25,
            label=label,
            s=12,
        )
    ax.set_title("t-SNE")
    #ax.set_xlabel("t-SNE 1")
    #ax.set_ylabel("t-SNE 2")
    axes[0].legend()
    # Shared legend
    # handles, legend_labels = axes[0].get_legend_handles_labels()
    # fig.legend(
    #     handles,
    #     legend_labels,
    #     loc="upper center",
    #     ncol=min(len(unique_labels), 5),
    # )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
    plt.show()


def main(args):
    print("Dataset:", args.dataname)
    print("Methods:", args.methods)

    ori_data = load_data(args.dataname)

    gen_dict = {}
    for method in args.methods:
        if method == "imagentime":
            path = f"synthetic/{args.dataname}/{method}_tsne_plot.npy"
            gen_dict['ImagenTime'] = np.load(path)
            print(f"Loading {method} from {path}")
        
        elif method =="path":
            path = f"synthetic/{args.dataname}/{method}.npy"
            print(f"Loading {method} from {path}")
            gen_dict['PATH (Ours)'] = np.load(path)
        
        elif method == "path_ada_irr_0.3":
            path = f"synthetic/{args.dataname}/{method}.npy"
            print(f"Loading {method} from {path}")
            gen_dict['PATH 30%'] = np.load(path)

        elif method == "path_ada_irr_0.5":
            path = f"synthetic/{args.dataname}/{method}.npy"
            print(f"Loading {method} from {path}")
            gen_dict['PATH 50%'] = np.load(path)


        else:
            path = f"synthetic/{args.dataname}/{method}.npy"
            print(f"Loading {method} from {path}")
            gen_dict[method] = np.load(path)

    print("Original shape:", np.asarray(ori_data).shape)

    for method, data in gen_dict.items():
        print(f"{method} shape:", np.asarray(data).shape)

    data_all, labels, sample_num = prepare_all_data(ori_data, gen_dict)

    print("Combined plotting data shape:", data_all.shape)
    print("Samples per group:", sample_num)

    save_prefix = "_".join(args.methods)

    tsne_path = f"./results/img/{save_prefix}_{args.dataname}_tsne"
    TSNE_plot_all(data_all, labels, tsne_path)

    pca_path = f"./results/img/{save_prefix}_{args.dataname}_pca"
    PCA_plot_all(data_all, labels, pca_path)

    TSNE_PCA_path = f"./results/img/{save_prefix}_{args.dataname}_pca_tsne"
    plot_pca_tsne_combined(data_all, labels, TSNE_PCA_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA/t-SNE visualization")

    parser.add_argument("--dataname", type=str, default="sine", help="Name of dataset.")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        required=True,
        help="List of synthetic method names.",
    )

    args = parser.parse_args()
    main(args)