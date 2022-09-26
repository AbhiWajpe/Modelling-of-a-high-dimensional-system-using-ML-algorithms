# %%
import numpy as np
from sklearn.decomposition import PCA
import pysindy as ps
from pydmd import DMD
import matplotlib.pyplot as plt

# %% load in data
data_folder = '0-10000/'
X_train = np.load(data_folder + 'p.npy')
# t_train = np.load(data_folder + 't.npy')
xc = np.load(data_folder + 'x.npy')[0]
yc_train = np.load(data_folder + 'y.npy')

# split into train and test data
idx_train = [500, 20000]
# idx_test = [15000, 20000]

# X_test = X_train[idx_test[0]:idx_test[1]]
X_train = X_train[idx_train[0]:idx_train[1]]

# t_test = t_train[idx_test[0]:idx_test[1]]
# t_train = t_train[idx_train[0]:idx_train[1]]
# t_train -= t_train[0]
# t_test -= t_test[0]
t_train = np.arange(X_train.shape[0])  # use time steps instead of simulation time

# yc_test = yc_train[idx_test[0]:idx_test[1]]
yc_train = yc_train[idx_train[0]:idx_train[1]]

# normalize
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train = (X_train - X_mean) / X_std

xc = (xc - xc.min()) / (xc.max() - xc.min())

# %% visualize the data
t_plot_idx = [0, 500]


def visualize_data(X, save_file, t_plot_idx=t_plot_idx):
    xgrid1, tgrid1 = np.meshgrid(xc[:100], t_train[t_plot_idx[0]:t_plot_idx[1]])
    xgrid2, tgrid2 = np.meshgrid(xc[100:], t_train[t_plot_idx[0]:t_plot_idx[1]])

    fig, axs = plt.subplots(ncols=2, figsize=(6.5, 5), sharey=True, sharex=True)
    vmin = -3
    vmax = 3
    axs[0].pcolormesh(xgrid1, tgrid1, X[t_plot_idx[0]:t_plot_idx[1], :100], vmin=vmin, vmax=vmax)
    cf = axs[1].pcolormesh(xgrid2, tgrid2, X[t_plot_idx[0]:t_plot_idx[1], 100:], vmin=vmin, vmax=vmax)

    fig.supxlabel('x Location On Body')
    fig.supylabel('Time')
    axs[0].set_title('Top Side of Body')
    axs[1].set_title('Bottom Side of Body')

    fig2, ax2 = plt.subplots(figsize=(1, 4.5))
    fig2.colorbar(cf, cax=ax2)

    fig.tight_layout()
    fig.savefig(save_file, dpi=1000)
    fig2.tight_layout()
    fig2.savefig('colorbar.png', dpi=1000)


def frob_norm(X, Xhat):
    return np.sqrt(np.sum((X - Xhat) ** 2))


visualize_data(X_train, 'X_train.png')

# %% compare with linear model w/ DMD
svd_rank = X_train.shape[1]
dmd = DMD(svd_rank=svd_rank)
dmd.fit(X_train.T)
# dmd.modes(X_test)

# %% get number of DMD modes with 95% energy
energy = 0.95
K_dmd = 0
ratio = 0
while ratio < energy:
    K_dmd += 1
    ratio = np.sum(dmd.eigs[:K_dmd]) / np.sum(dmd.eigs)
print("Number of Principal Components =", K_dmd)

# %% redo DMD with 95% energy
# K_dmd = 80
dmd = DMD(svd_rank=K_dmd)
dmd.fit(X_train.T)

# %% reconstruct using DMD
X_train_dmd_rc = dmd.reconstructed_data.T.real
# X_train_dmd_rc = (dmd.modes @ dmd.dynamics).T
print(frob_norm(X_train[:t_plot_idx[1]], X_train_dmd_rc[:t_plot_idx[1]]))

visualize_data(X_train_dmd_rc, 'X_train_dmd_rc.png')

# %% visualize DMD modes
for mode in dmd.modes.T:
    plt.plot(xc, mode.real)
    plt.title('Modes')
    plt.xlabel('x location')
plt.savefig('dmd_modes.png', dpi=1000)
plt.show()

for dynamic in dmd.dynamics:
    plt.plot(t_train[t_plot_idx[0]:t_plot_idx[1]], dynamic.real[t_plot_idx[0]:t_plot_idx[1]])
    plt.title('Dynamics')
    plt.xlabel('Time')
plt.savefig('dmd_dynamics.png', dpi=1000)
plt.show()

# %% dimensionality reduction w/ PCA
pca = PCA(n_components=None)
pca.fit(X_train)

# singular values and modes
sv = pca.singular_values_ ** 2

# %% find the number of principal components for 99% energy
K = 0
ratio = 0
while ratio < energy:
    K += 1
    ratio = np.sum(sv[:K]) / np.sum(sv)
print("Number of Principal Components =", K)

# %% get another PCA model using only K components
K = 7
pca = PCA(n_components=K)
X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)

# singular values and modes
sv = pca.singular_values_ ** 2
mod = pca.components_

# %% 
plt.figure()
plt.plot(X_train_pca[:500,:])
plt.title('Evolution of PCA Modes')
plt.xlabel('Time')
plt.savefig('pca_dynamics.png', dpi=1000)

plt.figure()
plt.plot(xc, mod.T)
plt.title('PCA Modes')
plt.xlabel('x location')
plt.savefig('pca_modes.png', dpi=1000)

# %% reconstruct X
X_train_pca_rc = pca.inverse_transform(X_train_pca)
# X_test_pca_rc = pca.inverse_transform(X_test_pca)

visualize_data(X_train_pca_rc, 'X_train_pca_rc.png')
# visualize_data(X_test_pca_rc, 'X_test_pca_rc.png')

# %% normalize PCA modes before using SINDy
norm = True
if norm:
    X_pca_mean = X_train_pca.mean(axis=0)
    X_pca_std = X_train_pca.std(axis=0)
    X_train_pca_norm = (X_train_pca - X_pca_mean) / X_pca_std
else:
    X_train_pca_norm = X_train_pca

# %% dynamical modeling w/ SINDy


def sindy(threshold):
    opt = ps.SR3(
        threshold=threshold,
        thresholder="l0",
        max_iter=10000,
        tol=1e-10)       # The new optimizer

    feature_library = ps.PolynomialLibrary(degree=1, include_bias=False)
    # feature_library = ps.FourierLibrary(n_frequencies=2) + ps.PolynomialLibrary(degree=2, include_bias=True)

    model = ps.SINDy(
        differentiation_method=ps.SmoothedFiniteDifference(order=2),
        feature_library=feature_library,
        optimizer=opt,
    )

    # weak form
    # X, T = np.meshgrid(np.arange(X_train_pca.shape[1]), t_train)
    # XT = np.array([X, T]).T
    # pde_lib = ps.WeakPDELibrary(
    #     library_functions=feature_library,
    #     derivative_order=3,
    #     spatiotemporal_grid=XT,
    #     is_uniform=True,
    # )

    # model = ps.SINDy(
    #     feature_library=pde_lib,
    #     optimizer=opt,
    # )

    model.fit(X_train_pca_norm, t=t_train, multiple_trajectories=False)
    model.print()

    # simulate the model
    t_f = t_plot_idx[1]
    X_train_pca_sindy_rc = model.simulate(X_train_pca_norm[0, :], t=t_train[:t_f])

    # convert from PCA modes to original coordinates
    if norm:  # undo normalzation first
        X_train_pca_sindy_rc = X_train_pca_sindy_rc * X_pca_std + X_pca_mean
    X_train_sindy_rc = pca.inverse_transform(X_train_pca_sindy_rc)

    # get reconstruction error
    err = frob_norm(X_train[:t_f], X_train_sindy_rc)
    print('Reconstruction error =', err)

    return [model, X_train_sindy_rc, err]


# %% hyperparameter tuning for SINDy
# higher threshold = less terms in eqn
# thresholds = [0.085, 0.0875, 0.09, 0.095, 0.1, 0.2]  # 4 PCA modes
# thresholds = np.linspace(0.042, 0.2, num=20)  # 4 PCA modes
# thresholds = np.linspace(0.08, 0.15, num=20)  # 7 PCA modes
# thresholds = np.linspace(0.04, 0.15, num=20)  # 7 PCA modes degree 2
thresholds = np.linspace(0.0001, 0.12, num=20)  # 7 PCA modes degree 1

models = [[]] * len(thresholds)
X_train_sindy_rcs = [[]] * len(thresholds)
errs = [[]] * len(thresholds)

for i in range(len(thresholds)):
    print('Threshold =', thresholds[i])
    models[i], X_train_sindy_rcs[i], errs[i] = sindy(thresholds[i])

# %% show best model
i_best = np.argmin(errs)
models[i_best].print()
print('Threshold =', thresholds[i_best])
print('Reconstruction error =', errs[i_best])
visualize_data(X_train_sindy_rcs[i_best], 'X_train_sindy_rc.png', t_plot_idx=[0, t_plot_idx[1]])

# %%
plt.plot(thresholds, errs)
plt.plot(thresholds[i_best], errs[i_best], 'ro', label='Minimum Error')
plt.xlabel('Thresholds for SR3')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.title('Threshold vs. Error Using {} PCA Modes'.format(K))
plt.savefig('sindy_hyperparams_K={}.png'.format(K), dpi=1000)

# %%
