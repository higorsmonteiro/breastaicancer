import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Tuple

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

def roc_curve_with_ci(
    labels: List[int], 
    preds: List[float], 
    n_bootstraps: Optional[int] = 10, 
    random_state: Optional[int] = 42
) -> Dict[str, Any]:
    '''
    Calculate the ROC curve and its confidence interval using bootstrapping,
    including the AUROC and its 95% confidence interval.
    '''
    # -- compute base ROC curve
    y_true, y_scores = np.array(labels), np.array(preds) 
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auroc = roc_auc_score(y_true, y_scores)

    # -- fixed FPR points for interpolation
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aurocs = []  # <-- collect bootstrap AUROCs

    rng = np.random.RandomState(random_state)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue

        fpr_boot, tpr_boot, _ = roc_curve(y_true[indices], y_scores[indices])
        tpr_interp = np.interp(mean_fpr, fpr_boot, tpr_boot)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        aurocs.append(roc_auc_score(y_true[indices], y_scores[indices]))  # <--

    # -- compute mean and CI for TPR at each FPR point
    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    lower_tpr = np.percentile(tprs, 2.5, axis=0)
    upper_tpr = np.percentile(tprs, 97.5, axis=0)

    # -- compute AUROC CI from bootstrap distribution
    aurocs = np.array(aurocs)
    auroc_lower = np.percentile(aurocs, 2.5)
    auroc_upper = np.percentile(aurocs, 97.5)

    results = {
        "auroc": auroc,
        "auroc_lower": auroc_lower,  # <--
        "auroc_upper": auroc_upper,  # <--
        "fpr": fpr, "tpr": tpr, "tprs": tprs,
        "mean_fpr": mean_fpr, "mean_tpr": mean_tpr,
        "std_tpr": std_tpr, "lower_tpr": lower_tpr, "upper_tpr": upper_tpr
    }
    return results

def topk_calculation(
    labels: List[int], 
    preds: List[float], 
    K: Optional[int] = 100
) -> Tuple[float, float, float]:
    '''
        Calculate precision, recall and specificity at Top-K.
    '''
    df = pd.DataFrame({"prob": preds, "label": labels})
    df_sorted = df.sort_values("prob", ascending=False).reset_index(drop=True)

    df_sorted["predicted"] = 0
    df_sorted.loc[:K-1, "predicted"] = 1

    TP = ((df_sorted["predicted"] == 1) & (df_sorted["label"] == 1)).sum()
    FP = ((df_sorted["predicted"] == 1) & (df_sorted["label"] == 0)).sum()
    TN = ((df_sorted["predicted"] == 0) & (df_sorted["label"] == 0)).sum()
    FN = ((df_sorted["predicted"] == 0) & (df_sorted["label"] == 1)).sum()

    recall_at_k = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision_at_k = TP / (TP + FP) if (TP + FP) > 0 else 0
    specificity_at_k = TN / (TN + FP) if (TN + FP) > 0 else 0
    return precision_at_k, recall_at_k, specificity_at_k

def bootstrap_topk(
    labels: List[int], 
    preds: List[float], 
    K: Optional[int] = 100, 
    n_bootstraps: Optional[int] = 50, 
    random_state: Optional[int] = 42
):
    rng = np.random.RandomState(random_state)
    precision_scores, recall_scores, specificity_scores = [], [], []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(labels), len(labels))
        boot_preds = np.array(preds)[indices]
        boot_labels = np.array(labels)[indices]

        ppv, recall, specificity = topk_calculation(boot_labels, boot_preds, K=K)
        precision_scores.append(ppv)
        recall_scores.append(recall)
        specificity_scores.append(specificity)

    # -- 95% CIs
    precision_ci = np.percentile(precision_scores, [2.5, 97.5])
    recall_ci = np.percentile(recall_scores, [2.5, 97.5])
    specificity_ci = np.percentile(specificity_scores, [2.5, 97.5])

    return {
        "precision_mean": np.mean(precision_scores),
        "precision_ci": precision_ci,
        "recall_mean": np.mean(recall_scores),
        "recall_ci": recall_ci,
        "specificity_mean": np.mean(specificity_scores),
        "specificity_ci": specificity_ci
    }

def get_pr_values_per_k(
    labels: List[int], 
    preds: List[float],
    K: Optional[int] = 100, 
    n_bootstraps: Optional[int] = 10,
    upper_K_perc: Optional[float] = 0.30
):
    K_perc = int(upper_K_perc*len(labels))
    K_values = np.arange(K, K_perc+K+1, K)

    precision_means, recall_means, specificity_means = [], [], []
    precision_cis, recall_cis, specificity_cis = [], [], []

    for K in tqdm(K_values):
        results = bootstrap_topk(labels, preds, K=K, n_bootstraps=n_bootstraps)
    
        precision_means.append(results["precision_mean"])
        recall_means.append(results["recall_mean"])
        specificity_means.append(results["specificity_mean"])

        precision_cis.append(results["precision_ci"])
        recall_cis.append(results["recall_ci"])
        specificity_cis.append(results["specificity_ci"])

    return {
        "K_values": K_values, "K_perc": (K_values/len(preds))*100,
        "precision_means": precision_means, "recall_means": recall_means, "specificity_means": specificity_means,
        "precision_cis": precision_cis, "recall_cis": recall_cis, "specificity_cis": specificity_cis
    }

def bottomk_calculation(
    labels: List[int], 
    preds: List[float], 
    K: Optional[int] = 100
) -> Tuple[float, int]:
    '''
        Calculate NPV and FN count at Bottom-K (lowest risk scores).
    '''
    df = pd.DataFrame({"prob": preds, "label": labels})
    df_sorted = df.sort_values("prob", ascending=True).reset_index(drop=True)

    df_sorted["predicted"] = 1
    df_sorted.loc[:K-1, "predicted"] = 0

    TN = ((df_sorted["predicted"] == 0) & (df_sorted["label"] == 0)).sum()
    FN = ((df_sorted["predicted"] == 0) & (df_sorted["label"] == 1)).sum()

    npv_at_k = TN / (TN + FN) if (TN + FN) > 0 else 0
    return npv_at_k, int(FN)

def bootstrap_bottomk(
    labels: List[int], 
    preds: List[float], 
    K: Optional[int] = 100, 
    n_bootstraps: Optional[int] = 50, 
    random_state: Optional[int] = 42
):
    rng = np.random.RandomState(random_state)
    npv_scores, fn_counts = [], []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(labels), len(labels))
        boot_preds = np.array(preds)[indices]
        boot_labels = np.array(labels)[indices]

        npv, fn = bottomk_calculation(boot_labels, boot_preds, K=K)
        npv_scores.append(npv)
        fn_counts.append(fn)

    return {
        "npv_mean": np.mean(npv_scores),
        "npv_ci": np.percentile(npv_scores, [2.5, 97.5]),
        "fn_mean": np.mean(fn_counts),
        "fn_ci": np.percentile(fn_counts, [2.5, 97.5])
    }


def get_npv_values_per_k(
    labels: List[int], 
    preds: List[float],
    K: Optional[int] = 100, 
    n_bootstraps: Optional[int] = 10,
    upper_K_perc: Optional[float] = 0.30
):
    K_perc = int(upper_K_perc * len(labels))
    K_values = np.arange(K, K_perc + K + 1, K)

    npv_means, npv_cis = [], []
    fn_means, fn_cis = [], []

    for k in tqdm(K_values):
        results = bootstrap_bottomk(labels, preds, K=k, n_bootstraps=n_bootstraps)

        npv_means.append(results["npv_mean"])
        npv_cis.append(results["npv_ci"])
        fn_means.append(results["fn_mean"])
        fn_cis.append(results["fn_ci"])

    return {
        "K_values": K_values, "K_perc": (K_values / len(preds)) * 100,
        "npv_means": npv_means, "npv_cis": npv_cis,
        "fn_means": fn_means, "fn_cis": fn_cis
    }

class GenerateCurvesForPlotting:
    '''
        Generate the AUROC and AUPRC curves for plotting.

        Args:
        -----
            results: dict. 'validation' and 'test' are expected as first levels of the dictionary.
            For each key, we expected a dictionary of format {'preds': List, 'labels': List}. It will
            be used to generate the curves. 
    '''
    def __init__(self, results: dict):
        self.results = dict(results)

        self.roc_results_val = {}
        self.roc_results_test = {}

        self.pr_results_val = {}
        self.pr_results_test = {}

    def _calculate_roc_with_ci(
        self, 
        n_bootstraps: Optional[int] = 100
    ):
        self.roc_results_val = roc_curve_with_ci(self.results['validation']['labels'], self.results['validation']['preds'], n_bootstraps=n_bootstraps)
        self.roc_results_test = roc_curve_with_ci(self.results['test']['labels'], self.results['test']['preds'], n_bootstraps=n_bootstraps)

    def _calculate_pr_with_ci(
        self, 
        K: Optional[int] = 100,
        n_bootstraps: Optional[int] = 10,
        upper_K_perc: Optional[float] = 0.30
    ):
        '''
            Calculate recall, precision and specificity as functions over the Top-K% highest risk.

            Args:
            -----
                K: int. Interval for Top-K. Larger values will make the calculation faster, but at the cost of less
                resolution of the final plot.
                n_bootstraps: int. Number of bootstrap samples to use.
                upper_K_perc: float. highest K% to consider when calculating the metrics.  
        '''
        self.pr_results_val = get_pr_values_per_k(self.results['validation']['labels'], self.results['validation']['preds'], K=K, n_bootstraps=n_bootstraps, upper_K_perc=upper_K_perc)
        self.pr_results_test = get_pr_values_per_k(self.results['test']['labels'], self.results['test']['preds'], K=K, n_bootstraps=n_bootstraps, upper_K_perc=upper_K_perc)

    def _calculate_npv_with_ci(
        self, 
        K: Optional[int] = 100,
        n_bootstraps: Optional[int] = 10,
        upper_K_perc: Optional[float] = 0.30
    ):
        '''
            Calculate NPV as a function over the Bottom-K% lowest risk scores.

            Args:
            -----
                K: int. Interval for Bottom-K. Larger values = faster but lower resolution.
                n_bootstraps: int. Number of bootstrap samples.
                upper_K_perc: float. Highest K% to consider (e.g. 0.30 = bottom 30%).
        '''
        self.npv_results_val = get_npv_values_per_k(
            self.results['validation']['labels'], self.results['validation']['preds'],
            K=K, n_bootstraps=n_bootstraps, upper_K_perc=upper_K_perc
        )
        self.npv_results_test = get_npv_values_per_k(
            self.results['test']['labels'], self.results['test']['preds'],
            K=K, n_bootstraps=n_bootstraps, upper_K_perc=upper_K_perc
        )

# ------------------------------------------------------------- #
# ------------------------- PLOTTING -------------------------- #
# ------------------------------------------------------------- #
class Plotting(GenerateCurvesForPlotting):
    def __init__(self, results: dict):
        super().__init__(results)

    def plot_roc(self, **kwargs):
        # -- collect custom args
        validation_color = kwargs.get('validation_color', '#215F9A')
        test_color = kwargs.get('test_color', '#F28C7C')
        axis_label_size = kwargs.get('axis_label_size', 15)
        tick_label_size = kwargs.get('tick_label_size', 12)
        legend_size = kwargs.get('legend_size', 12)
        legend_alpha = kwargs.get('alpha', 0.2)
        legend_loc = kwargs.get('legend_loc', 0)
        figsize = kwargs.get('figsize', (6,5))

        fig, ax = plt.subplots(1, figsize=figsize)
        ax.plot(self.roc_results_val["fpr"], self.roc_results_val["tpr"], lw=0, label=f'Validation AUROC = {self.roc_results_val["auroc"]:.3f}')
        ax.plot(self.roc_results_test["fpr"], self.roc_results_test["tpr"], lw=0, label=f'Test AUROC = {self.roc_results_test["auroc"]:.3f}')
        ax.plot(self.roc_results_val["mean_fpr"], self.roc_results_val["mean_tpr"], color=validation_color, label='Mean Validation ROC')
        ax.plot(self.roc_results_test["mean_fpr"], self.roc_results_test["mean_tpr"], color=test_color, label='Mean Test ROC')
        ax.fill_between(self.roc_results_val["mean_fpr"], self.roc_results_val["lower_tpr"], self.roc_results_val["upper_tpr"], color=validation_color, alpha=0.2, label='Validation 95% CI')
        ax.fill_between(self.roc_results_test["mean_fpr"], self.roc_results_test["lower_tpr"], self.roc_results_test["upper_tpr"], color=test_color, alpha=0.2, label='Test 95% CI')
        ax.plot([0,1], [0,1], ls="--", color="#333333")
        
        for axis in [ax]:
            axis.grid(alpha=legend_alpha)
            axis.legend(loc=legend_loc, prop={'size': legend_size}, frameon=False)
            axis.set_xlim([0,1])
            axis.set_ylim([0,1])
            axis.spines['top'].set_linewidth(0)
            axis.spines['right'].set_linewidth(0)
            axis.set_xlabel("False positive rate", fontsize=axis_label_size)
            axis.set_ylabel("True positive rate", fontsize=axis_label_size)
            axis.tick_params(labelsize=tick_label_size)

        fig.tight_layout()
        return fig, ax

    def plot_topk(self, percent_viz, **kwargs):
        # -- collect custom args
        validation_color = kwargs.get('validation_color', '#215F9A')
        test_color = kwargs.get('test_color', '#F28C7C')
        axis_label_size = kwargs.get('axis_label_size', 15)
        tick_label_size = kwargs.get('tick_label_size', 12)
        legend_size = kwargs.get('legend_size', 12)
        legend_alpha = kwargs.get('alpha', 0.2)
        legend_loc = kwargs.get('legend_loc', 0)
        xlim = kwargs.get('xlim', [1,19])
        figsize = kwargs.get('figsize', (8,5))
        
        # -- extract results
        K_values_val = self.pr_results_val["K_values"]
        K_values_test = self.pr_results_test["K_values"]
        K_perc_val = self.pr_results_val["K_perc"]
        K_perc_test = self.pr_results_test["K_perc"]

        precision_means_val, recall_means_val, specificity_means_val  = self.pr_results_val["precision_means"], self.pr_results_val["recall_means"], self.pr_results_val["specificity_means"]
        precision_cis_val, recall_cis_val, specificity_cis_val = self.pr_results_val["precision_cis"], self.pr_results_val["recall_cis"], self.pr_results_val["specificity_cis"]

        precision_means_test, recall_means_test, specificity_means_test  = self.pr_results_test["precision_means"], self.pr_results_test["recall_means"], self.pr_results_test["specificity_means"]
        precision_cis_test, recall_cis_test, specificity_cis_test = self.pr_results_test["precision_cis"], self.pr_results_test["recall_cis"], self.pr_results_test["specificity_cis"]

        recall_cis_val_arr = np.array(recall_cis_val)
        specificity_cis_val_arr = np.array(specificity_cis_val)
        recall_cis_test_arr = np.array(recall_cis_test)
        specificity_cis_test_arr = np.array(specificity_cis_test)
        
        # -- define figure (values can be changed outside the function)
        fig, ax = plt.subplots(1, figsize=figsize)

        # -- plot
        # ---- validation
        ax.plot(K_perc_val, recall_means_val, label="Sensitivity (Validation)", color=test_color, ls="--", lw=1.5)
        ax.fill_between(K_perc_val, recall_cis_val_arr[:, 0], recall_cis_val_arr[:, 1], color=test_color, alpha=0.1)
        ax.plot(K_perc_val, specificity_means_val, label="Specificity (Validation)", color=validation_color, ls="--", lw=1.5)
        ax.fill_between(K_perc_val, specificity_cis_val_arr[:, 0], specificity_cis_val_arr[:, 1], color=validation_color, alpha=0.1)
        # ---- test
        ax.plot(K_perc_test, recall_means_test, label="Sensitivity (Test)", color=test_color, ls="-", lw=1.5)
        ax.fill_between(K_perc_test, recall_cis_test_arr[:, 0], recall_cis_test_arr[:, 1], color=test_color, alpha=0.2)
        ax.plot(K_perc_test, specificity_means_test, label="Specificity (Test)", color=validation_color, ls="-", lw=1.5)
        ax.fill_between(K_perc_test, specificity_cis_test_arr[:, 0], specificity_cis_test_arr[:, 1], color=validation_color, alpha=0.2)

        for current_percent_viz in percent_viz:
            ax.axvline(x=current_percent_viz, ls=":", color="#505050", alpha=0.8)
            
            perc_val_index = np.array(recall_means_val)[K_perc_val<=current_percent_viz].shape[0]
            perc_test_index = np.array(recall_means_test)[K_perc_test<=current_percent_viz].shape[0]
            recall_at_perc_val, recall_at_perc_test = recall_means_val[perc_val_index], recall_means_test[perc_test_index]
            specificity_at_perc_val, specificity_at_perc_test = specificity_means_val[perc_val_index], specificity_means_test[perc_test_index]

            ax.text(s=f"{recall_at_perc_val*100:.1f}%", x=current_percent_viz+0.05, y=recall_at_perc_val+0.02, fontsize=13.5, color="#1a1a1a")
            ax.text(s=f"{recall_at_perc_test*100:.1f}%", x=current_percent_viz+0.05, y=recall_at_perc_test-0.02, fontsize=13.5, color="#1a1a1a")
            ax.text(s=f"{specificity_at_perc_test*100:.1f}%", x=current_percent_viz+0.05, y=specificity_at_perc_test+0.02, fontsize=13.5, color="#1a1a1a")

        for axis in [ax]:
            axis.grid(alpha=legend_alpha)
            axis.legend(loc=legend_loc, prop={'size': legend_size}, frameon=False)
            axis.set_xlim(xlim)
            axis.set_ylim([0,1])
            axis.spines['top'].set_linewidth(0)
            axis.spines['right'].set_linewidth(0)
            axis.set_xlabel("Top-K%", fontsize=axis_label_size)
            axis.set_ylabel("Sensitivity & Specificity", fontsize=axis_label_size)
            axis.set_xticks(np.arange(xlim[-1]+1))
            axis.tick_params(labelsize=tick_label_size)
        fig.tight_layout()
        return fig, ax

    def plot_training_epochs(self, results_df: pd.DataFrame, **kwargs):
        training_color = kwargs.get('training_color', '#c4a484')
        validation_color = kwargs.get('validation_color', '#215F9A')
        test_color = kwargs.get('test_color', '#F28C7C')
        axis_label_size = kwargs.get('axis_label_size', 15)
        tick_label_size = kwargs.get('tick_label_size', 12)
        legend_size = kwargs.get('legend_size', 12)
        grid_alpha = kwargs.get('grid_alpha', 0.2)
        legend_loc = kwargs.get('legend_loc', 0)
        figsize = kwargs.get('figsize', (10,5))
        
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=figsize)

        min_loss = min(results_df["training loss"].min(), results_df["validation loss"].min())
        max_loss = min(results_df["training loss"].max(), results_df["validation loss"].max())
        min_auroc = min(results_df["training auroc"].min(), results_df["validation auroc"].min())

        min_ylim = max([ elem for elem in np.arange(-20, 500, 0.05) if elem < min_loss ])
        min_ylim_auroc = max([ elem for elem in np.arange(0.5, 1.0, 0.05) if elem < min_auroc ])
        max_ylim = min([ elem for elem in np.arange(-20, 500, 0.05) if elem > max_loss ])

        ax1.plot(results_df["epochs"], results_df["training loss"], color=training_color, label="Training", lw=1.5)
        ax1.plot(results_df["epochs"], results_df["validation loss"], color=validation_color, label="Validation", lw=1.5)

        ax2.plot(results_df["epochs"], results_df["training auroc"], color=training_color, lw=1.5)
        ax2.plot(results_df["epochs"], results_df["validation auroc"], color=validation_color, lw=1.5)

        for axis in [ax1, ax2]:
            axis.grid(alpha=grid_alpha)
            axis.legend(loc=legend_loc, prop={'size': legend_size}, frameon=False)
            axis.spines['top'].set_linewidth(0)
            axis.spines['right'].set_linewidth(0)
            axis.set_xlabel("Epoch", fontsize=axis_label_size)
            axis.tick_params(labelsize=tick_label_size)

        ax1.set_xlim([results_df['epochs'].min(),results_df['epochs'].max()+1])
        ax1.set_ylim([float(f"{min_ylim:.4f}"),float(f"{max_ylim:.4f}")])
        ax2.set_xlim([results_df['epochs'].min(),results_df['epochs'].max()+1])
        ax2.set_ylim([float(f"{min_ylim_auroc:.4f}"), 1.0])
        ax1.set_ylabel("Loss", fontsize=axis_label_size)
        ax2.set_ylabel("AUROC", fontsize=axis_label_size)

        fig.tight_layout()
        return fig, (ax1, ax2)







