import matplotlib.pyplot as plt

def plot_variance(eigenvalues, title="Cumulative Explained Variance (Proxy)", save_path=None):
    plt.figure()
    component_range = range(1, len(eigenvalues) + 1)
    cumulative_variance = eigenvalues.cumsum() / eigenvalues.sum()
    plt.plot(component_range, cumulative_variance, marker='o')
    plt.title(title)
    plt.xlabel("Component")
    plt.ylabel("Cumulative variance ratio")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_latent_space(transformed_data, labels=None, title="Kernel PCA Latent Space", save_path=None):
    plt.figure()
    if transformed_data.shape[1] >= 2:
        plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap='coolwarm', s=20, alpha=0.7)
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.colorbar(label="Label")
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()