"""
Multi-Factor Correlation Model
Implements GCorr-style factor decomposition with GICS 24 Industry Groups:
  r_i = sqrt(RSQ_i) * Z_systematic_i + sqrt(1 - RSQ_i) * epsilon_i
where Z_systematic_i = sum(w_ij * F_j) is the weighted systematic factor
and F_j are correlated country/industry factors.

Factor layout: 15 country factors [0-14] + 24 GICS industry group factors [15-38] = 39 total
"""
import numpy as np
from scipy.linalg import cholesky

# ── Factor Definitions ──
COUNTRY_FACTORS = {
    'US': 0, 'UK': 1, 'DE': 2, 'FR': 3, 'JP': 4,
    'CN': 5, 'BR': 6, 'IN': 7, 'CA': 8, 'AU': 9,
    'SG': 10, 'CH': 11, 'KR': 12, 'MX': 13, 'ZA': 14
}

# GICS 24 Industry Group factors (indices 15-38)
INDUSTRY_FACTORS = {
    'EnergyEquipSvc': 15, 'OilGasFuels': 16,
    'Chemicals': 17, 'ConstructionMaterials': 18, 'ContainersPkg': 19,
    'MetalsMining': 20, 'PaperForest': 21,
    'CapitalGoods': 22, 'CommercialProfSvc': 23, 'Transportation': 24,
    'AutosComponents': 25, 'ConsumerDurablesApparel': 26,
    'ConsumerServices': 27, 'Retailing': 28,
    'FoodStaplesRetail': 29, 'FoodBevTobacco': 30, 'HouseholdProducts': 31,
    'HealthCareEquipSvc': 32, 'PharmaBiotech': 33,
    'Banks': 34, 'DiversifiedFinancials': 35, 'Insurance': 36,
    'SoftwareServices': 37, 'TechHardware': 38, 'Semiconductors': 39,
    'MediaEntertainment': 40, 'TelecomServices': 41,
    'Utilities': 42,
    'EquityREITs': 43, 'REMgmtDev': 44,
}

NUM_FACTORS = len(COUNTRY_FACTORS) + len(INDUSTRY_FACTORS)  # 15 + 30 = 45

# Map GICS industry groups to their parent GICS sector for intra-sector correlation
_GICS_SECTOR_GROUPS = {
    'Energy': ['EnergyEquipSvc', 'OilGasFuels'],
    'Materials': ['Chemicals', 'ConstructionMaterials', 'ContainersPkg', 'MetalsMining', 'PaperForest'],
    'Industrials': ['CapitalGoods', 'CommercialProfSvc', 'Transportation'],
    'ConsumerDiscretionary': ['AutosComponents', 'ConsumerDurablesApparel', 'ConsumerServices', 'Retailing'],
    'ConsumerStaples': ['FoodStaplesRetail', 'FoodBevTobacco', 'HouseholdProducts'],
    'HealthCare': ['HealthCareEquipSvc', 'PharmaBiotech'],
    'Financials': ['Banks', 'DiversifiedFinancials', 'Insurance'],
    'InformationTechnology': ['SoftwareServices', 'TechHardware', 'Semiconductors'],
    'CommunicationServices': ['MediaEntertainment', 'TelecomServices'],
    'Utilities': ['Utilities'],
    'RealEstate': ['EquityREITs', 'REMgmtDev'],
}


def build_factor_correlation_matrix(country_intra=0.4, industry_intra_sector=0.50,
                                     industry_cross_sector=0.20, cross_corr=0.12,
                                     seed=42):
    """
    Build the factor-factor correlation matrix for country + GICS industry group factors.
    Industry groups within the same GICS sector get higher intra-correlation.
    """
    rng = np.random.default_rng(seed)
    n = NUM_FACTORS
    corr = np.eye(n)
    n_country = len(COUNTRY_FACTORS)

    # Country-country correlations
    for i in range(n_country):
        for j in range(i + 1, n_country):
            corr[i, j] = corr[j, i] = country_intra + rng.uniform(-0.1, 0.1)

    # Build a lookup: factor_index -> parent GICS sector
    idx_to_sector = {}
    for sector, groups in _GICS_SECTOR_GROUPS.items():
        for g in groups:
            idx_to_sector[INDUSTRY_FACTORS[g]] = sector

    # Industry-industry correlations (higher within same GICS sector)
    industry_indices = list(range(n_country, n))
    for i_pos, i_idx in enumerate(industry_indices):
        for j_pos in range(i_pos + 1, len(industry_indices)):
            j_idx = industry_indices[j_pos]
            same_sector = idx_to_sector.get(i_idx) == idx_to_sector.get(j_idx)
            base = industry_intra_sector if same_sector else industry_cross_sector
            corr[i_idx, j_idx] = corr[j_idx, i_idx] = base + rng.uniform(-0.08, 0.08)

    # Country-industry cross correlations
    for i in range(n_country):
        for j in range(n_country, n):
            corr[i, j] = corr[j, i] = cross_corr + rng.uniform(-0.05, 0.05)

    # Ensure positive semi-definite
    corr = np.clip(corr, -0.99, 0.99)
    np.fill_diagonal(corr, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def generate_correlated_factors(n_simulations, factor_corr_matrix, seed=None):
    """Generate correlated factor draws using Cholesky decomposition."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    n_factors = factor_corr_matrix.shape[0]
    L = cholesky(factor_corr_matrix, lower=True)
    Z_independent = rng.standard_normal((n_simulations, n_factors))
    Z_correlated = Z_independent @ L.T
    return Z_correlated


def compute_obligor_asset_returns(factor_draws, obligor_factor_loadings,
                                   obligor_rsq, seed=None):
    """
    Compute correlated asset returns for each obligor using the factor model.

    r_i = sqrt(RSQ_i) * (sum w_ij * F_j) + sqrt(1 - RSQ_i) * eps_i
    """
    if seed is not None:
        rng = np.random.default_rng(seed + 42)
    else:
        rng = np.random.default_rng()

    n_sim = factor_draws.shape[0]
    n_obligors = obligor_rsq.shape[0]

    # Systematic component: weighted sum of factor draws
    systematic = factor_draws @ obligor_factor_loadings.T

    # Normalize systematic component to unit variance per obligor
    loading_norm = np.sqrt(np.sum(obligor_factor_loadings ** 2, axis=1))
    loading_norm = np.where(loading_norm > 0, loading_norm, 1.0)
    systematic = systematic / loading_norm[np.newaxis, :]

    # Idiosyncratic component
    epsilon = rng.standard_normal((n_sim, n_obligors))

    # Combined asset return
    sqrt_rsq = np.sqrt(np.clip(obligor_rsq, 0.01, 0.99))
    sqrt_1_rsq = np.sqrt(1.0 - np.clip(obligor_rsq, 0.01, 0.99))

    asset_returns = sqrt_rsq[np.newaxis, :] * systematic + sqrt_1_rsq[np.newaxis, :] * epsilon
    return asset_returns
