"""
Monte Carlo Credit Portfolio Simulation Engine
Implements a bottom-up horizon model:
1. Simulate correlated credit quality changes via multi-factor model
2. Determine default/migration outcomes per trial
3. Revalue instruments at horizon
4. Aggregate to portfolio loss distribution
"""
import numpy as np
from scipy.stats import norm
from .correlation import (
    build_factor_correlation_matrix, generate_correlated_factors,
    compute_obligor_asset_returns, COUNTRY_FACTORS, INDUSTRY_FACTORS, NUM_FACTORS
)


def build_factor_loadings(counterparties):
    """Build factor loading matrix from counterparty attributes."""
    n_obligors = len(counterparties)
    loadings = np.zeros((n_obligors, NUM_FACTORS))

    for i, cp in enumerate(counterparties):
        country = cp.get('country_code', 'US')
        sector = cp.get('sector_code', 'CapitalGoods')
        country_idx = COUNTRY_FACTORS.get(country, 0)
        industry_idx = INDUSTRY_FACTORS.get(sector, INDUSTRY_FACTORS.get('CapitalGoods', 22))

        # Split loading: 60% country, 40% industry (GCorr-style)
        loadings[i, country_idx] = 0.6
        loadings[i, industry_idx] = 0.4

    return loadings


def simulate_defaults(asset_returns, pd_values):
    """
    Determine default events using the Merton threshold model.
    Default occurs when asset return falls below the default threshold.

    Threshold = Phi^{-1}(PD) where Phi is the standard normal CDF.
    """
    default_thresholds = norm.ppf(np.clip(pd_values, 1e-10, 1.0 - 1e-10))
    defaults = asset_returns < default_thresholds[np.newaxis, :]
    return defaults


def simulate_migrations(asset_returns, migration_matrix, current_ratings):
    """
    Simulate rating migrations (vectorized).
    Uses CreditMetrics-style thresholds derived from the transition matrix.
    """
    n_sim, n_obligors = asset_returns.shape
    n_ratings = migration_matrix.shape[1]
    new_ratings = np.full((n_sim, n_obligors), n_ratings - 1, dtype=np.int32)

    for i, rating in enumerate(current_ratings):
        trans_probs = migration_matrix[rating]
        cumulative = np.cumsum(trans_probs)
        thresholds = norm.ppf(np.clip(cumulative[:-1], 1e-10, 1.0 - 1e-10))
        asset_vals = asset_returns[:, i]
        # Vectorized: use searchsorted on thresholds
        new_ratings[:, i] = np.searchsorted(thresholds, asset_vals)

    return new_ratings


def compute_lgd_with_correlation(defaults, base_lgd, systematic_factor,
                                  lgd_vol=0.15, pd_lgd_corr=0.3, seed=None):
    """
    Compute stochastic LGD with PD-LGD correlation (fully vectorized).
    In stress scenarios, LGD increases when systematic factor is negative.
    Uses a simplified normal-based approach clipped to [0,1] for efficiency.
    """
    rng = np.random.default_rng(seed + 100 if seed else None)
    n_sim, n_obligors = defaults.shape

    # Use first systematic factor as the common driver for LGD correlation
    sys_driver = systematic_factor[:, 0] if systematic_factor.shape[1] > 0 else np.zeros(n_sim)

    # Broadcast: adjusted mean LGD per obligor per simulation
    mean_lgd = np.clip(base_lgd, 0.05, 0.95)[np.newaxis, :]  # (1, n_obligors)
    adjustment = pd_lgd_corr * lgd_vol * sys_driver[:, np.newaxis]  # (n_sim, 1)
    adjusted_mean = np.clip(mean_lgd - adjustment, 0.05, 0.95)  # (n_sim, n_obligors)

    # Draw LGD from normal, clip to [0,1]
    lgd_noise = rng.standard_normal((n_sim, n_obligors)) * lgd_vol
    lgd_raw = adjusted_mean + lgd_noise
    lgd_draws = np.clip(lgd_raw, 0.01, 0.99)

    # Zero out non-defaults
    lgd_draws = lgd_draws * defaults.astype(np.float64)
    return lgd_draws


def compute_ead(instruments):
    """
    Compute Exposure at Default for each instrument.
    - Loans: outstanding balance
    - Revolvers: drawn + CCF * undrawn
    - Derivatives: MTM + add-on
    - CDS Sold: +notional (loss if reference entity defaults)
    - CDS Bought: -notional (gain if reference entity defaults — hedge)
    """
    eads = []
    for inst in instruments:
        itype = inst.get('instrument_type', 'TermLoan')
        if itype == 'TermLoan':
            eads.append(inst.get('drawn_amount', 0))
        elif itype == 'Revolver':
            drawn = inst.get('drawn_amount', 0)
            undrawn = inst.get('undrawn_amount', 0)
            ccf = inst.get('ccf', 0.75)
            eads.append(drawn + ccf * undrawn)
        elif itype == 'Derivative_IR':
            mtm = max(inst.get('mtm_value', 0), 0)
            notional = inst.get('notional', 0)
            addon = notional * inst.get('addon_factor', 0.005)
            eads.append(mtm + addon)
        elif itype == 'Derivative_FX':
            notional = inst.get('notional', 0)
            addon = notional * inst.get('addon_factor', 0.01)
            mtm = max(inst.get('mtm_value', 0), 0)
            eads.append(mtm + addon)
        elif itype == 'CDS':
            notional = inst.get('notional', 0)
            direction = inst.get('cds_direction', 'Protection_Sold')
            if direction == 'Protection_Bought':
                eads.append(-notional)  # Hedge: gain on reference entity default
            else:
                eads.append(notional)   # Sold: loss on reference entity default
        elif itype == 'CDS_CVA':
            # Counterparty risk on bought CDS protection seller
            mtm = max(inst.get('mtm_value', 0), 0)
            notional = inst.get('notional', 0)
            addon = notional * inst.get('addon_factor', 0.005)
            eads.append(mtm + addon)
        else:
            eads.append(inst.get('drawn_amount', inst.get('notional', 0)))
    return np.array(eads, dtype=np.float64)


def run_simulation(counterparties, instruments, pd_values, lgd_values,
                   migration_matrix=None, current_ratings=None,
                   n_simulations=100000, seed=42, scenario=None,
                   progress_callback=None):
    """
    Run full Monte Carlo credit portfolio simulation with chunked processing
    to manage memory for large portfolios (up to 1M+ simulations).

    Returns dict with:
        - portfolio_losses: (n_sim,) total portfolio loss per trial
        - instrument_el/var_contrib/es_contrib: per-instrument summary stats
        - default_flags_summary: per-obligor default rates
        - metrics: computed risk metrics
    """
    n_obligors = len(counterparties)

    # Inject CDS counterparty risk (CVA) instruments for bought protection
    cp_id_to_idx = {cp['counterparty_id']: i for i, cp in enumerate(counterparties)}
    cva_instruments = []
    for inst in instruments:
        if (inst.get('instrument_type') == 'CDS'
                and inst.get('cds_direction') == 'Protection_Bought'
                and inst.get('cds_seller_id')
                and inst['cds_seller_id'] in cp_id_to_idx):
            mtm = max(inst.get('mtm_value', 0), 0)
            notional = inst.get('notional', 0)
            # CVA exposure: replacement cost (positive MTM + add-on)
            cva_instruments.append({
                'instrument_id': inst['instrument_id'] + '_CVA',
                'instrument_type': 'CDS_CVA',
                'counterparty_id': inst['cds_seller_id'],
                'notional': notional,
                'mtm_value': mtm,
                'addon_factor': 0.005,
                'lgd': 0.40,  # Standard financial counterparty LGD
                'seniority': 'Senior Unsecured',
                'currency': inst.get('currency', 'USD'),
                'rating': '',
                'maturity_date': inst.get('maturity_date', '2028-01-01'),
            })

    all_instruments = list(instruments) + cva_instruments
    n_instruments = len(all_instruments)

    # Map instruments to obligors
    inst_to_obligor = np.array([
        cp_id_to_idx.get(inst['counterparty_id'], 0) for inst in all_instruments
    ])

    # Build factor model
    if progress_callback:
        progress_callback(0.05, "Building correlation structure...")
    factor_corr = build_factor_correlation_matrix()
    factor_loadings = build_factor_loadings(counterparties)
    obligor_rsq = np.array([cp.get('rsq', 0.25) for cp in counterparties])
    ead = compute_ead(all_instruments)

    # Determine chunk size based on available memory
    # Target ~500MB per chunk: n_sim_chunk * n_instruments * 8 bytes < 500MB
    max_bytes = 400 * 1024 * 1024  # 400MB
    chunk_size = max(1000, min(n_simulations, int(max_bytes / (n_instruments * 8))))

    n_chunks = (n_simulations + chunk_size - 1) // chunk_size
    portfolio_losses = np.zeros(n_simulations)
    inst_loss_sum = np.zeros(n_instruments)
    inst_loss_sq_sum = np.zeros(n_instruments)
    obligor_default_sum = np.zeros(n_obligors)

    # For tail statistics, we'll track the worst trials
    tail_threshold_pct = 99.9
    tail_n = max(10, int(n_simulations * (1 - tail_threshold_pct / 100)))
    # We'll accumulate all portfolio losses, then compute tail stats at the end

    if progress_callback:
        progress_callback(0.10, f"Running {n_chunks} chunks of {chunk_size:,} trials...")

    rng_master = np.random.default_rng(seed)

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, n_simulations)
        n_chunk = chunk_end - chunk_start
        chunk_seed = seed + chunk_idx * 1000

        pct = 0.10 + 0.70 * (chunk_idx / n_chunks)
        if progress_callback:
            progress_callback(pct, f"Chunk {chunk_idx+1}/{n_chunks}: simulating {n_chunk:,} trials...")

        # Generate correlated factors for this chunk
        factor_draws = generate_correlated_factors(n_chunk, factor_corr, seed=chunk_seed)
        if scenario is not None:
            factor_draws = apply_scenario_conditioning(factor_draws, scenario, factor_corr)

        # Asset returns
        asset_returns = compute_obligor_asset_returns(
            factor_draws, factor_loadings, obligor_rsq, seed=chunk_seed
        )

        # Defaults
        defaults = simulate_defaults(asset_returns, pd_values)
        obligor_default_sum += np.sum(defaults, axis=0)

        # Stochastic LGD
        systematic_component = factor_draws[:, :min(5, factor_draws.shape[1])]
        lgd_draws = compute_lgd_with_correlation(
            defaults, lgd_values, systematic_component, seed=chunk_seed
        )

        # Instrument losses (vectorized)
        inst_defaults = defaults[:, inst_to_obligor].astype(np.float64)
        inst_lgd = lgd_draws[:, inst_to_obligor]
        chunk_inst_losses = inst_defaults * inst_lgd * ead[np.newaxis, :]

        # Accumulate
        chunk_portfolio_losses = np.sum(chunk_inst_losses, axis=1)
        portfolio_losses[chunk_start:chunk_end] = chunk_portfolio_losses
        inst_loss_sum += np.sum(chunk_inst_losses, axis=0)
        inst_loss_sq_sum += np.sum(chunk_inst_losses ** 2, axis=0)

        # Free memory
        del factor_draws, asset_returns, defaults, lgd_draws
        del inst_defaults, inst_lgd, chunk_inst_losses

    # Compute metrics from accumulated statistics
    if progress_callback:
        progress_callback(0.85, "Computing risk metrics...")

    inst_el = inst_loss_sum / n_simulations
    default_rates = obligor_default_sum / n_simulations

    # For tail-based contributions, identify tail trials and recompute
    var_999 = np.percentile(portfolio_losses, 99.9)
    tail_mask = portfolio_losses >= var_999
    n_tail = np.sum(tail_mask)

    # Recompute tail contributions by re-running tail trials
    # Find which trials are in the tail
    tail_indices = np.where(tail_mask)[0]
    if len(tail_indices) > 0:
        var_contributions = np.zeros(n_instruments)
        # Re-simulate only tail trials to get per-instrument breakdown
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, n_simulations)
            chunk_seed = seed + chunk_idx * 1000

            local_tail = tail_indices[(tail_indices >= chunk_start) & (tail_indices < chunk_end)]
            if len(local_tail) == 0:
                continue
            local_idx = local_tail - chunk_start

            factor_draws = generate_correlated_factors(chunk_end - chunk_start, factor_corr, seed=chunk_seed)
            if scenario is not None:
                factor_draws = apply_scenario_conditioning(factor_draws, scenario, factor_corr)
            asset_returns = compute_obligor_asset_returns(
                factor_draws, factor_loadings, obligor_rsq, seed=chunk_seed
            )
            defaults = simulate_defaults(asset_returns, pd_values)
            systematic_component = factor_draws[:, :min(5, factor_draws.shape[1])]
            lgd_draws = compute_lgd_with_correlation(
                defaults, lgd_values, systematic_component, seed=chunk_seed
            )
            inst_defaults = defaults[:, inst_to_obligor].astype(np.float64)
            inst_lgd = lgd_draws[:, inst_to_obligor]
            chunk_inst_losses = inst_defaults * inst_lgd * ead[np.newaxis, :]
            var_contributions += np.sum(chunk_inst_losses[local_idx], axis=0)

            del factor_draws, asset_returns, defaults, lgd_draws, inst_defaults, inst_lgd, chunk_inst_losses

        var_contributions /= n_tail
    else:
        var_contributions = inst_el.copy()

    es_contributions = var_contributions.copy()

    # Build metrics dict
    el = np.mean(portfolio_losses)
    ul = np.std(portfolio_losses)
    var_95 = np.percentile(portfolio_losses, 95)
    var_99 = np.percentile(portfolio_losses, 99)
    var_9999 = np.percentile(portfolio_losses, 99.99)
    es_95 = np.mean(portfolio_losses[portfolio_losses >= var_95])
    es_99 = np.mean(portfolio_losses[portfolio_losses >= var_99])
    es_999 = np.mean(portfolio_losses[portfolio_losses >= var_999])
    econ_capital = var_999 - el
    total_ead = np.sum(np.abs(ead))  # Use absolute EAD for denominator
    standalone_vars = np.zeros(n_instruments)
    # Approximate standalone VaR from mean + scaled UL
    for j in range(n_instruments):
        standalone_vars[j] = inst_el[j] + 3.09 * np.sqrt(max(0, inst_loss_sq_sum[j]/n_simulations - inst_el[j]**2))
    sum_standalone = np.sum(standalone_vars)
    diversification_benefit = (sum_standalone - var_999) / sum_standalone if sum_standalone > 0 else 0

    abs_ead = np.abs(ead)
    hhi = np.sum((abs_ead / total_ead) ** 2) if total_ead > 0 else 0

    metrics = {
        'expected_loss': el,
        'unexpected_loss': ul,
        'var_95': var_95,
        'var_99': var_99,
        'var_999': var_999,
        'var_9999': var_9999,
        'es_95': es_95,
        'es_99': es_99,
        'es_999': es_999,
        'economic_capital': econ_capital,
        'total_ead': total_ead,
        'el_as_pct_ead': el / total_ead * 100 if total_ead > 0 else 0,
        'ec_as_pct_ead': econ_capital / total_ead * 100 if total_ead > 0 else 0,
        'diversification_benefit': diversification_benefit,
        'var_contributions': var_contributions,
        'es_contributions': es_contributions,
        'instrument_el': inst_el,
        'default_rates': default_rates,
        'portfolio_default_rate': np.mean(default_rates),
        'hhi': hhi,
        'max_loss': np.max(portfolio_losses),
        'min_loss': np.min(portfolio_losses),
        'skewness': float(np.mean(((portfolio_losses - el) / ul) ** 3)) if ul > 0 else 0,
        'kurtosis': float(np.mean(((portfolio_losses - el) / ul) ** 4)) if ul > 0 else 0,
    }

    if progress_callback:
        progress_callback(1.0, "Simulation complete.")

    return {
        'portfolio_losses': portfolio_losses,
        'instrument_losses': None,
        'default_flags': None,
        'lgd_draws': None,
        'ead': ead,
        'instruments': all_instruments,
        'n_original_instruments': len(instruments),
        'n_cva_instruments': len(cva_instruments),
        'metrics': metrics,
        'asset_returns': None,
        'n_simulations': n_simulations,
        'seed': seed
    }


def apply_scenario_conditioning(factor_draws, scenario, factor_corr):
    """
    Apply GCorr Macro-style scenario conditioning.
    Condition factor draws on specified macro factor levels.
    Uses conditional multivariate normal: X|Y = mu_X|Y + sigma_X|Y * Z
    """
    if scenario is None or 'factor_shocks' not in scenario:
        return factor_draws

    shocks = scenario['factor_shocks']
    conditioned = factor_draws.copy()

    for factor_idx, shock_value in shocks.items():
        if isinstance(factor_idx, str):
            from .correlation import COUNTRY_FACTORS, INDUSTRY_FACTORS
            idx = COUNTRY_FACTORS.get(factor_idx, INDUSTRY_FACTORS.get(factor_idx))
            if idx is None:
                continue
        else:
            idx = factor_idx

        # Shift the factor draws toward the scenario value
        severity = shock_value  # in standard deviations
        conditioned[:, idx] = conditioned[:, idx] + severity

        # Propagate through correlations
        for j in range(conditioned.shape[1]):
            if j != idx:
                conditioned[:, j] += factor_corr[idx, j] * severity * 0.5

    return conditioned

