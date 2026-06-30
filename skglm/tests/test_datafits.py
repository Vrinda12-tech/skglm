import numpy as np
import scipy.optimize
import pytest

from sklearn.linear_model import HuberRegressor
from numpy.testing import assert_allclose, assert_array_less

from skglm.datafits import (Huber, Logistic, Poisson, Gamma, Cox, WeightedQuadratic,
                            Quadratic, QuadraticHessian,WeightedPoisson)
from skglm.penalties import L1, WeightedL1
from skglm.solvers import AndersonCD, ProxNewton
from skglm import GeneralizedLinearEstimator
from skglm.utils.data import make_correlated_data
from skglm.utils.data import make_dummy_survival_data


@pytest.mark.parametrize('fit_intercept', [False, True])
def test_huber_datafit(fit_intercept):
    # test only datafit: there does not exist other implems with sparse penalty
    X, y, _ = make_correlated_data(n_samples=20, n_features=10, random_state=0)
    # disable L2^2 regularization (alpha=0)
    their = HuberRegressor(
        fit_intercept=fit_intercept, alpha=0, tol=1e-12, epsilon=1.35
    ).fit(X, y)

    # sklearn optimizes over a scale, we must match delta:
    delta = their.epsilon * their.scale_

    # TODO we should have an unpenalized solver
    ours = GeneralizedLinearEstimator(
        datafit=Huber(delta),
        penalty=WeightedL1(1, np.zeros(X.shape[1])),
        solver=AndersonCD(tol=1e-14, fit_intercept=fit_intercept),
    ).fit(X, y)

    assert_allclose(ours.coef_, their.coef_, rtol=1e-3)
    assert_allclose(ours.intercept_, their.intercept_, rtol=1e-4)
    assert_array_less(ours.stop_crit_, ours.solver.tol)


def test_log_datafit():
    n_samples, n_features = 10, 20

    w = np.ones(n_features)
    X, y, _ = make_correlated_data(n_samples, n_features)
    y = np.sign(y)
    Xw = X @ w

    log_datafit = Logistic()
    grad = log_datafit.raw_grad(y, Xw)
    hess = log_datafit.raw_hessian(y, Xw)

    np.testing.assert_equal(grad.shape, (n_samples,))
    np.testing.assert_equal(hess.shape, (n_samples,))

    exp_minus_yXw = np.exp(-y * Xw)
    np.testing.assert_almost_equal(
        exp_minus_yXw / (1 + exp_minus_yXw) ** 2 / len(y), hess)
    np.testing.assert_almost_equal(-grad * (y + n_samples * grad), hess)


def test_logistic_intercept_newton_step():
    # The intercept update is a 1D Newton step on
    #     beta_0 -> F(beta_0, beta) = mean log(1 + exp(-y(X beta + beta_0))).
    # Check (a) the closed-form matches sum(raw_grad)/sum(raw_hessian), and
    # (b) iterating the update converges quadratically to the analytic
    # optimizer log(p/(1-p)) when beta = 0.
    rng = np.random.RandomState(0)
    n_samples = 200
    y = np.where(rng.random(n_samples) < 0.95, 1.0, -1.0)  # mu_0 ~= 0.95
    p_hat = (y == 1).mean()

    df = Logistic()
    Xw = np.zeros(n_samples)  # beta = 0, beta_0 = 0

    step = df.intercept_update_step(y, Xw)
    expected = np.sum(df.raw_grad(y, Xw)) / np.sum(df.raw_hessian(y, Xw))
    np.testing.assert_allclose(step, expected)

    intercept = 0.0
    for _ in range(10):
        intercept -= df.intercept_update_step(y, Xw + intercept)
    np.testing.assert_allclose(intercept, np.log(p_hat / (1 - p_hat)), rtol=1e-8)


def test_logistic_intercept_imbalanced_converges():
    # Regression test for the Newton intercept update: on a strongly imbalanced
    # logistic problem the previous gradient-strategy update (step 1/(4n)
    # scaled gradient) stalled and failed to reach the tolerance within the
    # default iteration budget. The Newton step converges quickly.
    from skglm import SparseLogisticRegression

    rng = np.random.RandomState(0)
    n_samples, n_features = 300, 100
    X = rng.randn(n_samples, n_features)
    # mu_0 ~ 0.97
    y = np.where(rng.random(n_samples) < 0.97, 1.0, -1.0)

    tol = 1e-8
    est = SparseLogisticRegression(
        alpha=0.05, tol=tol, fit_intercept=True, max_iter=50,
    ).fit(X, y)
    assert est.stop_crit_ < tol


def test_poisson():
    try:
        from statsmodels.discrete.discrete_model import Poisson as PoissonRegressor  # noqa
    except ImportError:
        pytest.xfail("`statsmodels` not found. `Poisson` datafit can't be tested.")

    n_samples, n_features = 10, 22
    tol = 1e-14
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)
    y = np.abs(y)

    alpha_max = np.linalg.norm(X.T @ (np.ones(n_samples) - y), ord=np.inf) / n_samples
    alpha = alpha_max * 0.1

    df = Poisson()
    pen = L1(alpha)

    solver = ProxNewton(tol=tol, fit_intercept=False)
    model = GeneralizedLinearEstimator(df, pen, solver).fit(X, y)

    poisson_regressor = PoissonRegressor(y, X, offset=None)
    res = poisson_regressor.fit_regularized(
        method="l1", size_trim_tol=tol, alpha=alpha * n_samples, trim_mode="size")
    w_statsmodels = res.params

    assert_allclose(model.coef_, w_statsmodels, rtol=1e-4)


def test_gamma():
    try:
        import statsmodels.api as sm
    except ImportError:
        pytest.xfail("`statsmodels` not found. `Gamma` datafit can't be tested.")

    # When n_samples < n_features, the unregularized Gamma objective does not have a
    # unique minimizer.
    rho = 1e-2
    n_samples, n_features = 100, 10
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)
    y[y <= 0] = 0.1
    tol = 1e-14

    alpha_max = np.linalg.norm(X.T @ (1 - y), ord=np.inf) / n_samples
    alpha = rho * alpha_max

    gamma_model = sm.GLM(y, X, family=sm.families.Gamma(sm.families.links.Log()))
    gamma_results = gamma_model.fit_regularized(
        method="elastic_net", L1_wt=1, cnvrg_tol=tol, alpha=alpha)

    clf = GeneralizedLinearEstimator(
        datafit=Gamma(),
        penalty=L1(alpha),
        solver=ProxNewton(fit_intercept=False, tol=tol)
    ).fit(X, y)

    np.testing.assert_allclose(clf.coef_, gamma_results.params, rtol=1e-6)


@pytest.mark.parametrize("use_efron", [True, False])
def test_cox(use_efron):
    rng = np.random.RandomState(1265)
    n_samples, n_features = 10, 30

    # generate data
    X, y = make_dummy_survival_data(n_samples, n_features, normalize=True,
                                    with_ties=use_efron, random_state=0)

    # generate dummy w, Xw
    w = rng.randn(n_features)
    Xw = X @ w

    # check datafit
    cox_df = Cox(use_efron)

    cox_df.initialize(X, y)
    cox_df.value(y, w, Xw)

    # perform test 10 times to consider truncation errors
    # due to usage of finite differences to evaluate grad and Hessian
    for _ in range(10):

        # generate dummy w, Xw
        w = rng.randn(n_features)
        Xw = X @ w

        # check gradient
        np.testing.assert_allclose(
            scipy.optimize.check_grad(
                lambda x: cox_df.value(y, w, x),
                lambda x: cox_df.raw_grad(y, x),
                x0=Xw,
                seed=rng
            ),
            0., atol=1e-6
        )

        # check hessian upper bound
        # Hessian minus its upper bound must be negative semi definite
        hess_upper_bound = np.diag(cox_df.raw_hessian(y, Xw))
        hess = scipy.optimize.approx_fprime(
            xk=Xw,
            f=lambda x: cox_df.raw_grad(y, x),
        )

        positive_eig = np.linalg.eigh(hess - hess_upper_bound)[0]
        positive_eig = positive_eig[positive_eig >= 0.]

        np.testing.assert_allclose(positive_eig, 0., atol=1e-6)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_sample_weights(fit_intercept):
    """Test that integers sample weights give same result as duplicating rows."""

    rng = np.random.RandomState(0)

    n_samples = 20
    n_features = 100
    X, y, _ = make_correlated_data(
        n_samples=n_samples, n_features=n_features, random_state=0)

    indices = rng.choice(n_samples, 3 * n_samples)

    sample_weights = np.zeros(n_samples)
    for i in indices:
        sample_weights[i] += 1

    X_overs, y_overs = X[indices], y[indices]

    df_weight = WeightedQuadratic(sample_weights=sample_weights)
    df_overs = Quadratic()

    # same df value
    w = np.random.randn(n_features)
    val_overs = df_overs.value(y_overs, X_overs, X_overs @ w)
    val_weight = df_weight.value(y, X, X @ w)
    np.testing.assert_allclose(val_overs, val_weight)

    pen = L1(alpha=1)
    alpha_max = pen.alpha_max(df_weight.gradient(X, y, np.zeros(X.shape[0])))
    pen.alpha = alpha_max / 10
    solver = AndersonCD(tol=1e-12, verbose=10, fit_intercept=fit_intercept)

    model_weight = GeneralizedLinearEstimator(df_weight, pen, solver)
    model_weight.fit(X, y)
    print("#" * 80)
    res = model_weight.coef_
    model = GeneralizedLinearEstimator(df_overs, pen, solver)
    model.fit(X_overs, y_overs)
    res_overs = model.coef_

    np.testing.assert_allclose(res, res_overs)
    # n_iter = model.n_iter_
    # n_iter_overs = model.n_iter_
    # due to numerical errors the assert fails, but (inspecting the verbose output)
    # everything matches up to numerical precision errors in tol:
    # np.testing.assert_equal(n_iter, n_iter_overs)


def test_HessianQuadratic():
    n_samples = 20
    n_features = 10
    X, y, _ = make_correlated_data(
        n_samples=n_samples, n_features=n_features, random_state=0)
    A = X.T @ X / n_samples
    b = -X.T @ y / n_samples
    alpha = np.max(np.abs(b)) / 10

    pen = L1(alpha)
    solv = AndersonCD(warm_start=False, verbose=2, fit_intercept=False)
    lasso = GeneralizedLinearEstimator(Quadratic(), pen, solv).fit(X, y)
    qpl1 = GeneralizedLinearEstimator(QuadraticHessian(), pen, solv).fit(A, b)

    np.testing.assert_allclose(lasso.coef_, qpl1.coef_)
    # check that it's not just because we got alpha too high and thus 0 coef
    np.testing.assert_array_less(0.1, np.max(np.abs(qpl1.coef_)))
# =============================================================================
# Tests for WeightedPoisson
# =============================================================================


# =============================================================================
# Tests for WeightedPoisson
# =============================================================================


def test_weighted_poisson_equality():
    """Test that WeightedPoisson with equal weights matches Poisson."""
    import numpy as np
    from skglm.datafits import Poisson, WeightedPoisson

    np.random.seed(42)
    n_samples, n_features = 100, 10
    X = np.random.randn(n_samples, n_features)
    w_true = np.random.randn(n_features)
    y = np.random.poisson(lam=np.exp(X @ w_true))

    weights = np.ones(n_samples)
    weighted = WeightedPoisson(weights)
    weighted.initialize(X, y)
    unweighted = Poisson()

    w = np.random.randn(n_features)
    Xw = X @ w

    assert np.allclose(weighted.value(y, w, Xw), unweighted.value(y, w, Xw))
    assert np.allclose(weighted.raw_grad(y, Xw), unweighted.raw_grad(y, Xw))
    assert np.allclose(weighted.raw_hessian(y, Xw), unweighted.raw_hessian(y, Xw))

    for j in range(n_features):
        assert np.allclose(
            weighted.gradient_scalar(X, y, w, Xw, j),
            unweighted.gradient_scalar(X, y, w, Xw, j)
        )


def test_weighted_poisson_scaling():
    """Test that weights correctly scale the loss and gradients."""
    import numpy as np
    from skglm.datafits import WeightedPoisson

    np.random.seed(42)
    n_samples, n_features = 50, 5
    X = np.random.randn(n_samples, n_features)
    w = np.random.randn(n_features)
    y = np.random.poisson(lam=2, size=n_samples)
    Xw = X @ w

    weights = np.random.rand(n_samples)
    wp = WeightedPoisson(weights)
    wp.initialize(X, y)

    weights_norm = weights / np.sum(weights)
    expected_value = np.sum(weights_norm * (np.exp(Xw) - y * Xw))
    expected_grad = weights_norm * (np.exp(Xw) - y)

    assert np.allclose(wp.value(y, w, Xw), expected_value)
    assert np.allclose(wp.raw_grad(y, Xw), expected_grad)


def test_weighted_poisson_zero_weights():
    """Test that samples with zero weight are ignored."""
    import numpy as np
    from skglm.datafits import WeightedPoisson

    np.random.seed(42)
    n_samples, n_features = 10, 5
    X = np.random.randn(n_samples, n_features)
    w = np.random.randn(n_features)
    y = np.random.poisson(lam=2, size=n_samples)
    Xw = X @ w

    weights = np.ones(n_samples)
    weights[0] = 0

    wp = WeightedPoisson(weights)
    wp.initialize(X, y)

    raw_grad = wp.raw_grad(y, Xw)
    assert raw_grad[0] == 0


def test_weighted_poisson_validation():
    """Test input validation for WeightedPoisson."""
    import numpy as np
    import pytest
    from skglm.datafits import WeightedPoisson

    n_samples = 10
    X = np.random.randn(n_samples, 5)
    y = np.random.poisson(lam=2, size=n_samples)

    # Test 1: Negative y
    y_neg = y.copy()
    y_neg[0] = -1
    with pytest.raises(ValueError, match="y must be non-negative"):
        wp = WeightedPoisson(np.ones(n_samples))
        wp.initialize(X, y_neg)

    # Test 2: Negative weights
    weights_neg = np.ones(n_samples)
    weights_neg[0] = -1
    with pytest.raises(ValueError, match="Weights must be non-negative"):
        wp = WeightedPoisson(weights_neg)
        wp.initialize(X, y)

    # Test 3: Sum of weights = 0
    weights_zero = np.zeros(n_samples)
    with pytest.raises(ValueError, match="Sum of weights must be strictly greater than 0"):
        wp = WeightedPoisson(weights_zero)
        wp.initialize(X, y)

    # Test 4: Shape mismatch
    weights_wrong = np.ones(n_samples + 1)
    with pytest.raises(ValueError, match="Shape mismatch"):
        wp = WeightedPoisson(weights_wrong)
        wp.initialize(X, y)


def test_weighted_poisson_lipschitz():
    """Test Lipschitz constant computation with weights."""
    import numpy as np
    from skglm.datafits import WeightedPoisson

    np.random.seed(42)
    n_samples, n_features = 50, 5
    X = np.random.randn(n_samples, n_features)
    y = np.random.poisson(lam=2, size=n_samples)

    weights = np.random.rand(n_samples)
    wp = WeightedPoisson(weights)
    wp.initialize(X, y)

    lipschitz = wp.get_lipschitz(X, y)

    expected = np.array([np.sum(weights * X[:, j] ** 2) / np.sum(weights) for j in range(n_features)])

    assert np.allclose(lipschitz, expected)


def test_weighted_poisson_sparse():
    """Test WeightedPoisson with sparse matrices."""
    import numpy as np
    from scipy.sparse import csc_matrix
    from skglm.datafits import WeightedPoisson

    np.random.seed(42)
    n_samples, n_features = 50, 10

    X_dense = np.random.randn(n_samples, n_features)
    X_dense[np.random.rand(*X_dense.shape) > 0.2] = 0
    X_sparse = csc_matrix(X_dense)

    y = np.random.poisson(lam=2, size=n_samples)
    w = np.random.randn(n_features)
    Xw = X_dense @ w

    weights = np.random.rand(n_samples)
    wp = WeightedPoisson(weights)
    wp.initialize(X_dense, y)

    for j in range(n_features):
        grad_dense = wp.gradient_scalar(X_dense, y, w, Xw, j)
        grad_sparse = wp.gradient_scalar_sparse(
            X_sparse.data, X_sparse.indptr, X_sparse.indices, y, Xw, j
        )
        assert np.allclose(grad_dense, grad_sparse)


def test_weighted_poisson_integration():
    """Test WeightedPoisson with GeneralizedLinearEstimator."""
    import numpy as np
    from skglm import GeneralizedLinearEstimator
    from skglm.datafits import WeightedPoisson
    from skglm.penalties import L1
    from skglm.solvers import ProxNewton  # <-- EXPLICITLY USE ProxNewton

    np.random.seed(42)
    n_samples, n_features = 100, 10
    X = np.random.randn(n_samples, n_features)
    w_true = np.random.randn(n_features)
    y = np.random.poisson(lam=np.exp(X @ w_true))

    weights = np.random.rand(n_samples)

    model = GeneralizedLinearEstimator(
        datafit=WeightedPoisson(weights),
        penalty=L1(alpha=0.1),
        solver=ProxNewton(tol=1e-8, fit_intercept=False)  # <-- ProxNewton
    )
    model.fit(X, y)

    assert model.coef_ is not None
    assert model.n_iter_ > 0
    assert not np.any(np.isnan(model.coef_))
    assert not np.any(np.isinf(model.coef_))

    preds = model.predict(X)
    assert np.all(preds > 0)

if __name__ == '__main__':
    pass
