# Step 3: 최적 피처셋으로 Sub-Clustering (QuantileTransformer)
from sklearn.preprocessing import QuantileTransformer

best_fs    = list(best_row["features"])
best_k_sub = int(best_row["k"])

data_sub   = sub0[best_fs].dropna()
idx_sub    = data_sub.index
qt         = QuantileTransformer(n_quantiles=min(len(data_sub), 1000),
                                  output_distribution="normal", random_state=42)
X_sub_best = qt.fit_transform(data_sub.values)

km_sub = KMeans(n_clusters=best_k_sub, n_init=20, random_state=42)
sub0.loc[idx_sub, "subcluster"] = km_sub.fit_predict(X_sub_best)

diag_cols2 = ["avg_monthly_delta_level", "delta_cp_winsor",
              "avg_monthly_delta_union_level", "stagnation_score",
              "arcane_stagnant", "level"]
diag_cols2 = [c for c in diag_cols2 if c in sub0.columns]

prof_sub = sub0.loc[idx_sub].groupby("subcluster")[
    best_fs + diag_cols2
].mean().round(2)
prof_sub["n"] = sub0.loc[idx_sub].groupby("subcluster").size()
prof_sub["delta_level_le_0.1_pct"] = (
    sub0.loc[idx_sub].groupby("subcluster")
    .apply(lambda g: (g["avg_monthly_delta_level"] <= 0.1).mean() * 100)
).round(1)
prof_sub["stag5_pct"] = (
    sub0.loc[idx_sub].groupby("subcluster")
    .apply(lambda g: (g["stagnation_score"] == 5).mean() * 100)
).round(1)

sil_sub = silhouette_score(X_sub_best, km_sub.labels_)
print("=== Sub-Clustering (QuantileTransformer) ===")
print("피처:", best_fs)
print("k=%d  Silhouette=%.4f" % (best_k_sub, sil_sub))
print()
display(prof_sub.T)

print()
print("레벨 밴드 분포 (행 비율 %):")
display(pd.crosstab(
    sub0.loc[idx_sub, "subcluster"],
    sub0.loc[idx_sub, "level_bin"],
    normalize="index"
).round(3) * 100)

if "class_group" in sub0.columns:
    print()
    print("계열 분포 (행 비율 %):")
    display(pd.crosstab(
        sub0.loc[idx_sub, "subcluster"],
        sub0.loc[idx_sub, "class_group"],
        normalize="index"
    ).round(3) * 100)

if len(best_fs) == 2:
    coords = X_sub_best
    xlabel = best_fs[0] + " (qt)"
    ylabel = best_fs[1] + " (qt)"
    title_suffix = "QuantileTransformed"
else:
    pca_sub = PCA(n_components=2, random_state=42)
    coords = pca_sub.fit_transform(X_sub_best)
    ev = pca_sub.explained_variance_ratio_
    xlabel = "PC1 (%.1f%%)" % (ev[0] * 100)
    ylabel = "PC2 (%.1f%%)" % (ev[1] * 100)
    title_suffix = "PCA 2D (%.1f%%)" % ((ev[0] + ev[1]) * 100)

palette = plt.cm.tab10.colors
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for cl in range(best_k_sub):
    mask = sub0.loc[idx_sub, "subcluster"] == cl
    axes[0].scatter(coords[mask, 0], coords[mask, 1],
                    s=10, alpha=0.5, color=palette[cl % 10],
                    label="Sub-%d (n=%d)" % (cl, mask.sum()))
axes[0].set_xlabel(xlabel)
axes[0].set_ylabel(ylabel)
axes[0].set_title("Sub-Cluster (%s)" % title_suffix)
axes[0].legend(markerscale=2, fontsize=9)

sc2 = axes[1].scatter(coords[:, 0], coords[:, 1],
                      c=sub0.loc[idx_sub, "stagnation_score"],
                      cmap="Reds", s=10, alpha=0.6, vmin=0, vmax=5)
plt.colorbar(sc2, ax=axes[1], label="stagnation_score")
axes[1].set_xlabel(xlabel)
axes[1].set_ylabel(ylabel)
axes[1].set_title("stagnation_score")

plt.suptitle("Cluster 0 Sub-Clustering (QuantileTransformer)  k=%d  sil=%.4f" % (best_k_sub, sil_sub), y=1.02)
plt.tight_layout()
plt.savefig("figures/12_subclustering.png", dpi=150, bbox_inches="tight")
plt.show()
