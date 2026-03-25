"""
Dummy Data Generator for Corporate Bank Portfolio
Generates 1 year of realistic bank data across product types:
- Term Loans, Revolving Credit, Derivatives (IR/FX), CDS, Trade Finance, Guarantees

Industry classification follows GICS (Global Industry Classification Standard)
at the 24 Industry Group level.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── GICS 24 Industry Groups ──
GICS_INDUSTRY_GROUPS = [
    'EnergyEquipSvc', 'OilGasFuels',
    'Chemicals', 'ConstructionMaterials', 'ContainersPkg', 'MetalsMining', 'PaperForest',
    'CapitalGoods', 'CommercialProfSvc', 'Transportation',
    'AutosComponents', 'ConsumerDurablesApparel', 'ConsumerServices', 'Retailing',
    'FoodStaplesRetail', 'FoodBevTobacco', 'HouseholdProducts',
    'HealthCareEquipSvc', 'PharmaBiotech',
    'Banks', 'DiversifiedFinancials', 'Insurance',
    'SoftwareServices', 'TechHardware', 'Semiconductors',
    'MediaEntertainment', 'TelecomServices',
    'Utilities',
    'EquityREITs', 'REMgmtDev',
]

GICS_GROUP_TO_SECTOR = {
    'EnergyEquipSvc': 'Energy', 'OilGasFuels': 'Energy',
    'Chemicals': 'Materials', 'ConstructionMaterials': 'Materials',
    'ContainersPkg': 'Materials', 'MetalsMining': 'Materials', 'PaperForest': 'Materials',
    'CapitalGoods': 'Industrials', 'CommercialProfSvc': 'Industrials', 'Transportation': 'Industrials',
    'AutosComponents': 'ConsumerDiscretionary', 'ConsumerDurablesApparel': 'ConsumerDiscretionary',
    'ConsumerServices': 'ConsumerDiscretionary', 'Retailing': 'ConsumerDiscretionary',
    'FoodStaplesRetail': 'ConsumerStaples', 'FoodBevTobacco': 'ConsumerStaples',
    'HouseholdProducts': 'ConsumerStaples',
    'HealthCareEquipSvc': 'HealthCare', 'PharmaBiotech': 'HealthCare',
    'Banks': 'Financials', 'DiversifiedFinancials': 'Financials', 'Insurance': 'Financials',
    'SoftwareServices': 'InformationTechnology', 'TechHardware': 'InformationTechnology',
    'Semiconductors': 'InformationTechnology',
    'MediaEntertainment': 'CommunicationServices', 'TelecomServices': 'CommunicationServices',
    'Utilities': 'Utilities',
    'EquityREITs': 'RealEstate', 'REMgmtDev': 'RealEstate',
}

GICS_GROUP_DISPLAY_NAMES = {
    'EnergyEquipSvc': 'Energy Equipment & Services', 'OilGasFuels': 'Oil, Gas & Consumable Fuels',
    'Chemicals': 'Chemicals', 'ConstructionMaterials': 'Construction Materials',
    'ContainersPkg': 'Containers & Packaging', 'MetalsMining': 'Metals & Mining',
    'PaperForest': 'Paper & Forest Products',
    'CapitalGoods': 'Capital Goods', 'CommercialProfSvc': 'Commercial & Professional Services',
    'Transportation': 'Transportation',
    'AutosComponents': 'Automobiles & Components', 'ConsumerDurablesApparel': 'Consumer Durables & Apparel',
    'ConsumerServices': 'Consumer Services', 'Retailing': 'Retailing',
    'FoodStaplesRetail': 'Food & Staples Retailing', 'FoodBevTobacco': 'Food, Beverage & Tobacco',
    'HouseholdProducts': 'Household & Personal Products',
    'HealthCareEquipSvc': 'Health Care Equipment & Services', 'PharmaBiotech': 'Pharmaceuticals, Biotech & Life Sciences',
    'Banks': 'Banks', 'DiversifiedFinancials': 'Diversified Financials', 'Insurance': 'Insurance',
    'SoftwareServices': 'Software & Services', 'TechHardware': 'Technology Hardware & Equipment',
    'Semiconductors': 'Semiconductors & Semiconductor Equipment',
    'MediaEntertainment': 'Media & Entertainment', 'TelecomServices': 'Telecommunication Services',
    'Utilities': 'Utilities',
    'EquityREITs': 'Equity Real Estate Investment Trusts (REITs)',
    'REMgmtDev': 'Real Estate Management & Development',
}

GICS_SECTORS = sorted(set(GICS_GROUP_TO_SECTOR.values()))

# Backward compat alias
SECTORS = GICS_INDUSTRY_GROUPS

COUNTRIES = ['US', 'UK', 'DE', 'FR', 'JP', 'CN', 'BR', 'IN', 'CA', 'AU',
             'SG', 'CH', 'KR', 'MX', 'ZA']

COUNTRY_WEIGHTS = [0.25, 0.10, 0.08, 0.07, 0.08, 0.08, 0.05, 0.05, 0.06, 0.04,
                   0.03, 0.03, 0.03, 0.03, 0.02]

RATINGS = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']
RATING_PDS = {
    'AAA': 0.0001, 'AA': 0.0005, 'A': 0.0010, 'BBB': 0.0025,
    'BB': 0.0100, 'B': 0.0350, 'CCC': 0.1500, 'D': 1.0
}
RATING_WEIGHTS = [0.02, 0.05, 0.15, 0.35, 0.25, 0.12, 0.05, 0.01]

INSTRUMENT_TYPES = ['TermLoan', 'Revolver', 'Derivative_IR', 'Derivative_FX', 'CDS', 'TradeFinance', 'Guarantee']
INST_TYPE_WEIGHTS = [0.30, 0.20, 0.12, 0.10, 0.08, 0.12, 0.08]

SENIORITY = ['Senior Secured', 'Senior Unsecured', 'Subordinated', 'Junior Subordinated']
LGD_BY_SENIORITY = {
    'Senior Secured': 0.25, 'Senior Unsecured': 0.45,
    'Subordinated': 0.65, 'Junior Subordinated': 0.80
}

CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'SGD', 'CNY', 'BRL']

COMPANY_PREFIXES = [
    'Global', 'National', 'Pacific', 'Atlantic', 'Summit', 'Meridian', 'Apex',
    'Pinnacle', 'Vanguard', 'Sterling', 'Titan', 'Horizon', 'Crown', 'Delta',
    'Quantum', 'Phoenix', 'Nexus', 'Zenith', 'Crest', 'Prime', 'Atlas',
    'Falcon', 'Orion', 'Eclipse', 'Nova'
]

COMPANY_SUFFIXES = {
    'EnergyEquipSvc': ['Drilling', 'Oilfield Services', 'Energy Systems', 'Subsea', 'Well Services'],
    'OilGasFuels': ['Petroleum', 'Oil & Gas', 'Energy', 'Resources', 'Fuels'],
    'Chemicals': ['Chemicals', 'Polymers', 'Specialty Chem', 'Chemical Corp', 'Compounds'],
    'ConstructionMaterials': ['Cement', 'Aggregates', 'Building Materials', 'Construction Supply', 'Masonry'],
    'ContainersPkg': ['Packaging', 'Containers', 'Bottle Corp', 'Pack Solutions', 'Container Systems'],
    'MetalsMining': ['Mining', 'Metals', 'Resources', 'Minerals', 'Steel'],
    'PaperForest': ['Paper', 'Forest Products', 'Timber', 'Pulp & Paper', 'Wood Products'],
    'CapitalGoods': ['Industries', 'Manufacturing', 'Engineering', 'Heavy Equipment', 'Machinery'],
    'CommercialProfSvc': ['Services', 'Consulting', 'Solutions Group', 'Business Services', 'Professional Svc'],
    'Transportation': ['Logistics', 'Freight', 'Shipping', 'Airlines', 'Transport'],
    'AutosComponents': ['Automotive', 'Motors', 'Auto Parts', 'Vehicle Systems', 'Mobility'],
    'ConsumerDurablesApparel': ['Brands', 'Apparel', 'Home Products', 'Lifestyle', 'Fashion'],
    'ConsumerServices': ['Hospitality', 'Leisure', 'Entertainment', 'Travel', 'Services Group'],
    'Retailing': ['Retail', 'Stores', 'Commerce', 'Merchants', 'Retail Group'],
    'FoodStaplesRetail': ['Grocery', 'Food Mart', 'Supermarkets', 'Food Distribution', 'Fresh Markets'],
    'FoodBevTobacco': ['Foods', 'Beverages', 'Consumer Products', 'Nutrition', 'Brands'],
    'HouseholdProducts': ['Home Care', 'Personal Products', 'Consumer Goods', 'Essentials', 'Household'],
    'HealthCareEquipSvc': ['Medical', 'Health Systems', 'Medical Devices', 'Diagnostics', 'HealthTech'],
    'PharmaBiotech': ['Pharma', 'BioSciences', 'Therapeutics', 'Biotech', 'Life Sciences'],
    'Banks': ['Bank', 'Bancorp', 'Financial', 'Banking Group', 'Savings'],
    'DiversifiedFinancials': ['Capital', 'Asset Management', 'Financial Services', 'Investments', 'Holdings'],
    'Insurance': ['Insurance', 'Assurance', 'Underwriters', 'Risk Group', 'Reinsurance'],
    'SoftwareServices': ['Software', 'Digital', 'Cloud', 'Tech Solutions', 'Data Systems'],
    'TechHardware': ['Systems', 'Electronics', 'Hardware', 'Devices', 'Computing'],
    'Semiconductors': ['Semiconductor', 'Chip Corp', 'Micro Systems', 'Silicon', 'Integrated Circuits'],
    'MediaEntertainment': ['Media', 'Entertainment', 'Studios', 'Broadcasting', 'Digital Media'],
    'TelecomServices': ['Telecom', 'Communications', 'Networks', 'Wireless', 'Connect'],
    'Utilities': ['Utilities', 'Electric', 'Water', 'Gas', 'Power'],
    'EquityREITs': ['REIT', 'Properties', 'Real Estate Trust', 'Property Fund', 'Realty Income'],
    'REMgmtDev': ['Development', 'Real Estate', 'Property Group', 'Land Corp', 'Realty'],
}

# Weights for industry group selection in dummy data (roughly proportional to real market)
_raw_weights = [
    0.02, 0.04,           # Energy
    0.03, 0.01, 0.01, 0.02, 0.01,  # Materials
    0.06, 0.03, 0.04,     # Industrials
    0.02, 0.02, 0.02, 0.04,  # Consumer Discretionary
    0.02, 0.03, 0.02,     # Consumer Staples
    0.03, 0.05,           # Health Care
    0.08, 0.04, 0.04,     # Financials
    0.06, 0.03, 0.04,     # Information Technology
    0.03, 0.02,           # Communication Services
    0.04,                 # Utilities
    0.05, 0.02,           # Real Estate
]
GICS_GROUP_WEIGHTS = [w / sum(_raw_weights) for w in _raw_weights]

# 8-state migration matrix (AAA through D)
MIGRATION_MATRIX = np.array([
    [0.9081, 0.0769, 0.0100, 0.0045, 0.0003, 0.0001, 0.0000, 0.0001],
    [0.0070, 0.9065, 0.0720, 0.0100, 0.0030, 0.0010, 0.0003, 0.0002],
    [0.0009, 0.0227, 0.9105, 0.0552, 0.0074, 0.0020, 0.0006, 0.0007],
    [0.0002, 0.0033, 0.0595, 0.8693, 0.0530, 0.0102, 0.0020, 0.0025],
    [0.0003, 0.0014, 0.0067, 0.0773, 0.8053, 0.0884, 0.0100, 0.0106],
    [0.0001, 0.0011, 0.0024, 0.0043, 0.0648, 0.8346, 0.0407, 0.0520],
    [0.0022, 0.0000, 0.0022, 0.0130, 0.0238, 0.1124, 0.6486, 0.1978],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
])


def generate_counterparties(n_counterparties=500, seed=42):
    rng = np.random.default_rng(seed)
    counterparties = []
    used_names = set()

    for i in range(n_counterparties):
        sector = rng.choice(GICS_INDUSTRY_GROUPS, p=GICS_GROUP_WEIGHTS)
        country = rng.choice(COUNTRIES, p=COUNTRY_WEIGHTS)
        rating = rng.choice(RATINGS[:-1], p=np.array(RATING_WEIGHTS[:-1]) / sum(RATING_WEIGHTS[:-1]))

        prefix = rng.choice(COMPANY_PREFIXES)
        suffix = rng.choice(COMPANY_SUFFIXES[sector])
        name = f"{prefix} {suffix}"
        attempt = 0
        while name in used_names:
            name = f"{prefix} {suffix} {rng.choice(['Inc', 'Ltd', 'Corp', 'SA', 'AG', 'PLC', 'Group'])}"
            attempt += 1
            if attempt > 5:
                name = f"{prefix}{i} {suffix}"
        used_names.add(name)

        rsq = np.clip(rng.normal(0.25, 0.08), 0.05, 0.60)
        parent_id = f"CP{max(0,i-rng.integers(1,20)):04d}" if rng.random() < 0.15 else None

        counterparties.append({
            'counterparty_id': f'CP{i:04d}',
            'legal_name': name,
            'parent_id': parent_id,
            'sector_code': sector,
            'gics_sector': GICS_GROUP_TO_SECTOR[sector],
            'country_code': country,
            'region_code': _get_region(country),
            'rating': rating,
            'pd_1y': RATING_PDS[rating] * rng.uniform(0.7, 1.3),
            'rsq': rsq,
            'is_public': bool(rng.random() < 0.4),
            'revenue_mm': round(rng.lognormal(6, 1.5), 1),
            'total_assets_mm': round(rng.lognormal(7, 1.5), 1),
        })
    return counterparties


def _get_region(country):
    regions = {
        'US': 'NorthAmerica', 'CA': 'NorthAmerica', 'MX': 'LatAm', 'BR': 'LatAm',
        'UK': 'Europe', 'DE': 'Europe', 'FR': 'Europe', 'CH': 'Europe',
        'JP': 'AsiaPac', 'CN': 'AsiaPac', 'IN': 'AsiaPac', 'AU': 'AsiaPac',
        'SG': 'AsiaPac', 'KR': 'AsiaPac', 'ZA': 'Africa'
    }
    return regions.get(country, 'Other')


def generate_instruments(counterparties, n_instruments=2000, seed=42):
    rng = np.random.default_rng(seed)
    instruments = []

    for i in range(n_instruments):
        cp = rng.choice(counterparties)
        itype = rng.choice(INSTRUMENT_TYPES, p=INST_TYPE_WEIGHTS)
        seniority = rng.choice(SENIORITY, p=[0.40, 0.35, 0.15, 0.10])
        currency = rng.choice(CURRENCIES, p=[0.35, 0.20, 0.10, 0.08, 0.05, 0.05, 0.05, 0.04, 0.05, 0.03])
        maturity_years = rng.choice([1, 2, 3, 5, 7, 10], p=[0.10, 0.15, 0.20, 0.25, 0.15, 0.15])
        maturity_date = datetime(2025, 3, 1) + timedelta(days=int(maturity_years * 365))
        base_lgd = LGD_BY_SENIORITY[seniority] * rng.uniform(0.8, 1.2)

        inst = {
            'instrument_id': f'INS{i:05d}',
            'instrument_type': itype,
            'counterparty_id': cp['counterparty_id'],
            'currency': currency,
            'maturity_date': maturity_date.strftime('%Y-%m-%d'),
            'seniority': seniority,
            'lgd': np.clip(base_lgd, 0.05, 0.95),
            'rating': cp['rating'],
        }

        if itype == 'TermLoan':
            amount = rng.lognormal(3, 1.0) * 1e6
            inst.update({
                'drawn_amount': round(amount, 2),
                'undrawn_amount': 0,
                'notional': round(amount, 2),
                'mtm_value': round(amount * rng.uniform(0.95, 1.02), 2),
                'interest_rate': round(rng.uniform(0.02, 0.08), 4),
                'collateral_type': rng.choice(['RealEstate', 'Equipment', 'Receivables', 'Unsecured'],
                                               p=[0.30, 0.20, 0.20, 0.30]),
            })
        elif itype == 'Revolver':
            limit = rng.lognormal(3, 1.0) * 1e6
            utilization = rng.uniform(0.1, 0.8)
            inst.update({
                'drawn_amount': round(limit * utilization, 2),
                'undrawn_amount': round(limit * (1 - utilization), 2),
                'notional': round(limit, 2),
                'limit_amount': round(limit, 2),
                'ccf': round(rng.uniform(0.5, 0.9), 2),
                'mtm_value': round(limit * utilization * rng.uniform(0.98, 1.01), 2),
                'interest_rate': round(rng.uniform(0.025, 0.07), 4),
                'collateral_type': rng.choice(['Receivables', 'Inventory', 'Unsecured'],
                                               p=[0.35, 0.25, 0.40]),
            })
        elif itype == 'Derivative_IR':
            notional = rng.lognormal(4, 1.0) * 1e6
            inst.update({
                'notional': round(notional, 2),
                'drawn_amount': 0,
                'undrawn_amount': 0,
                'mtm_value': round(rng.normal(0, notional * 0.02), 2),
                'addon_factor': round(rng.uniform(0.003, 0.015), 4),
                'derivative_subtype': rng.choice(['IRS', 'Swaption', 'Cap', 'Floor']),
                'collateral_type': 'Cash',
            })
        elif itype == 'Derivative_FX':
            notional = rng.lognormal(3.5, 1.0) * 1e6
            inst.update({
                'notional': round(notional, 2),
                'drawn_amount': 0,
                'undrawn_amount': 0,
                'mtm_value': round(rng.normal(0, notional * 0.03), 2),
                'addon_factor': round(rng.uniform(0.005, 0.075), 4),
                'derivative_subtype': rng.choice(['FXForward', 'FXSwap', 'FXOption']),
                'collateral_type': rng.choice(['Cash', 'Unsecured'], p=[0.6, 0.4]),
            })
        elif itype == 'CDS':
            notional = rng.lognormal(3, 1.2) * 1e6
            direction = rng.choice(['Protection_Bought', 'Protection_Sold'], p=[0.45, 0.55])
            cds_data = {
                'notional': round(notional, 2),
                'drawn_amount': 0,
                'undrawn_amount': 0,
                'mtm_value': round(rng.normal(0, notional * 0.05), 2),
                'cds_direction': direction,
                'cds_spread_bps': round(rng.uniform(20, 500), 1),
                'collateral_type': 'Cash',
            }
            # For bought protection, assign a seller counterparty (financial institution)
            if direction == 'Protection_Bought':
                financial_cps = [c for c in counterparties
                                 if c.get('gics_sector') == 'Financials'
                                 and c['counterparty_id'] != cp['counterparty_id']]
                if financial_cps:
                    seller = rng.choice(financial_cps)
                else:
                    seller = rng.choice([c for c in counterparties if c['counterparty_id'] != cp['counterparty_id']])
                cds_data['cds_seller_id'] = seller['counterparty_id']
                cds_data['cds_seller_name'] = seller['legal_name']
            inst.update(cds_data)
        elif itype == 'TradeFinance':
            amount = rng.lognormal(2, 0.8) * 1e6
            inst.update({
                'drawn_amount': round(amount, 2),
                'undrawn_amount': round(amount * rng.uniform(0, 0.3), 2),
                'notional': round(amount, 2),
                'mtm_value': round(amount, 2),
                'trade_type': rng.choice(['LetterOfCredit', 'BankersAcceptance', 'DocumentaryCollection']),
                'collateral_type': 'Goods',
            })
        elif itype == 'Guarantee':
            amount = rng.lognormal(2.5, 1.0) * 1e6
            inst.update({
                'drawn_amount': 0,
                'undrawn_amount': round(amount, 2),
                'notional': round(amount, 2),
                'mtm_value': 0,
                'ccf': round(rng.uniform(0.3, 0.7), 2),
                'guarantee_type': rng.choice(['Financial', 'Performance', 'Standby']),
                'collateral_type': 'Unsecured',
            })

        instruments.append(inst)
    return instruments


def generate_monthly_snapshots(instruments, n_months=12, seed=42):
    """Generate 12 monthly snapshots with realistic evolution."""
    rng = np.random.default_rng(seed)
    snapshots = []
    base_date = datetime(2025, 3, 1)

    for month in range(n_months):
        snap_date = base_date + timedelta(days=month * 30)
        for inst in instruments:
            snap = {
                'snapshot_date': snap_date.strftime('%Y-%m-%d'),
                'instrument_id': inst['instrument_id'],
                'counterparty_id': inst['counterparty_id'],
                'instrument_type': inst['instrument_type'],
            }
            drift = 1.0 + rng.normal(0, 0.02)
            snap['drawn_amount'] = round(inst.get('drawn_amount', 0) * drift, 2)
            snap['undrawn_amount'] = round(inst.get('undrawn_amount', 0) * drift, 2)
            snap['notional'] = round(inst.get('notional', 0) * drift, 2)
            snap['mtm_value'] = round(inst.get('mtm_value', 0) + rng.normal(0, abs(inst.get('notional', 1e6)) * 0.005), 2)
            snap['ead'] = snap['drawn_amount'] + inst.get('ccf', 0.75) * snap['undrawn_amount']
            if inst['instrument_type'] in ('Derivative_IR', 'Derivative_FX'):
                snap['ead'] = max(snap['mtm_value'], 0) + snap['notional'] * inst.get('addon_factor', 0.01)
            elif inst['instrument_type'] == 'CDS':
                snap['ead'] = snap['notional']
            snapshots.append(snap)
    return snapshots


def generate_pd_curves(counterparties, seed=42):
    """Generate PD term structures for each counterparty."""
    rng = np.random.default_rng(seed)
    pd_curves = []
    tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10]

    for cp in counterparties:
        pd_1y = cp['pd_1y']
        for tenor in tenors:
            if tenor <= 1:
                pd_val = pd_1y * tenor
            else:
                pd_val = 1 - (1 - pd_1y) ** tenor
            pd_val *= rng.uniform(0.9, 1.1)
            pd_curves.append({
                'counterparty_id': cp['counterparty_id'],
                'curve_type': rng.choice(['PIT', 'TTC'], p=[0.6, 0.4]),
                'tenor': tenor,
                'pd': round(np.clip(pd_val, 0.0001, 0.9999), 6),
                'source': 'Internal Model',
                'model_version': 'v2.1'
            })
    return pd_curves


def generate_factor_loadings(counterparties, seed=42):
    """Generate factor loading data for each counterparty."""
    rng = np.random.default_rng(seed)
    from ..engine.correlation import COUNTRY_FACTORS, INDUSTRY_FACTORS
    loadings = []

    for cp in counterparties:
        country = cp['country_code']
        sector = cp['sector_code']
        rsq = cp['rsq']

        country_loading = 0.6 * np.sqrt(rsq) * rng.uniform(0.9, 1.1)
        industry_loading = 0.4 * np.sqrt(rsq) * rng.uniform(0.9, 1.1)

        loadings.append({
            'counterparty_id': cp['counterparty_id'],
            'factor_type': 'Country',
            'factor_name': country,
            'loading': round(country_loading, 4),
            'rsq': round(rsq, 4),
        })
        loadings.append({
            'counterparty_id': cp['counterparty_id'],
            'factor_type': 'Industry',
            'factor_name': sector,
            'loading': round(industry_loading, 4),
            'rsq': round(rsq, 4),
        })
    return loadings


def generate_scenarios():
    """Generate standard stress scenarios with GICS industry group factor shocks."""
    return [
        {
            'scenario_id': 'BASE',
            'scenario_name': 'Baseline',
            'scenario_type': 'baseline',
            'description': 'Current economic conditions continue',
            'gdp_shock': 0.0, 'rate_shock': 0.0, 'spread_shock': 0.0,
            'factor_shocks': {}
        },
        {
            'scenario_id': 'MILD_DOWN',
            'scenario_name': 'Mild Downturn',
            'scenario_type': 'stress',
            'description': 'Moderate economic slowdown',
            'gdp_shock': -1.0, 'rate_shock': -0.5, 'spread_shock': 1.0,
            'factor_shocks': {'US': -1.0, 'OilGasFuels': -1.5, 'EnergyEquipSvc': -1.2}
        },
        {
            'scenario_id': 'SEVERE_RECESS',
            'scenario_name': 'Severe Recession',
            'scenario_type': 'stress',
            'description': 'Deep recession with credit crisis',
            'gdp_shock': -3.0, 'rate_shock': -1.5, 'spread_shock': 3.0,
            'factor_shocks': {
                'US': -2.5, 'UK': -2.0, 'DE': -2.0,
                'Banks': -3.0, 'DiversifiedFinancials': -3.0, 'Insurance': -2.5,
                'EquityREITs': -3.5, 'REMgmtDev': -3.5,
                'CapitalGoods': -2.0, 'AutosComponents': -2.5,
            }
        },
        {
            'scenario_id': 'EM_CRISIS',
            'scenario_name': 'Emerging Market Crisis',
            'scenario_type': 'stress',
            'description': 'Emerging market contagion',
            'gdp_shock': -2.0, 'rate_shock': 1.5, 'spread_shock': 4.0,
            'factor_shocks': {'CN': -3.0, 'BR': -4.0, 'IN': -2.5, 'MX': -3.5, 'ZA': -3.0}
        },
        {
            'scenario_id': 'RATE_SPIKE',
            'scenario_name': 'Interest Rate Spike',
            'scenario_type': 'stress',
            'description': 'Sudden rate increase with real estate impact',
            'gdp_shock': -1.5, 'rate_shock': 3.0, 'spread_shock': 2.0,
            'factor_shocks': {
                'EquityREITs': -4.0, 'REMgmtDev': -3.5,
                'Utilities': -2.0,
                'AutosComponents': -2.0, 'ConsumerDurablesApparel': -2.5,
                'Retailing': -1.5,
            }
        },
    ]


def generate_all_data(n_counterparties=500, n_instruments=2000, seed=42):
    """Generate complete dataset."""
    counterparties = generate_counterparties(n_counterparties, seed)
    instruments = generate_instruments(counterparties, n_instruments, seed)
    snapshots = generate_monthly_snapshots(instruments, 12, seed)
    pd_curves = generate_pd_curves(counterparties, seed)
    scenarios = generate_scenarios()

    return {
        'counterparties': counterparties,
        'instruments': instruments,
        'snapshots': snapshots,
        'pd_curves': pd_curves,
        'migration_matrix': MIGRATION_MATRIX,
        'scenarios': scenarios,
        'ratings': RATINGS,
    }
