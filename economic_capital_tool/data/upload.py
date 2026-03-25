"""
Excel Upload Parser & Validator for Portfolio Data
Reads an uploaded .xlsx workbook and converts it to the internal data structure
expected by run_simulation(). Validates required fields, value ranges, and referential integrity.
"""
import numpy as np
import pandas as pd
from io import BytesIO
from .generator import (
    GICS_INDUSTRY_GROUPS, GICS_GROUP_TO_SECTOR, GICS_GROUP_DISPLAY_NAMES,
    COUNTRIES, RATINGS, RATING_PDS, INSTRUMENT_TYPES, MIGRATION_MATRIX,
    generate_scenarios, generate_monthly_snapshots, generate_pd_curves, _get_region,
)


def parse_uploaded_excel(file_bytes):
    """
    Parse an uploaded Excel workbook into the internal data structure.

    Required sheets: Counterparties, Instruments
    Optional sheets: Migration Matrix, Scenarios

    Returns: (data_dict, errors, warnings)
        data_dict: same structure as generate_all_data() or None if critical errors
        errors: list of critical error strings (block simulation)
        warnings: list of warning strings (non-blocking)
    """
    errors = []
    warnings = []

    try:
        xls = pd.ExcelFile(BytesIO(file_bytes) if isinstance(file_bytes, bytes) else file_bytes)
    except Exception as e:
        return None, [f"Cannot read Excel file: {e}"], []

    sheet_names = [s.lower().strip() for s in xls.sheet_names]
    sheet_map = {s.lower().strip(): s for s in xls.sheet_names}

    # ── Parse Counterparties (required) ──
    cp_sheet = _find_sheet(sheet_map, ['counterparties', 'counterparty', 'obligors', 'borrowers'])
    if cp_sheet is None:
        return None, ["Missing required sheet: 'Counterparties'"], warnings

    counterparties, cp_errors, cp_warnings = _parse_counterparties(xls, cp_sheet)
    errors.extend(cp_errors)
    warnings.extend(cp_warnings)

    if counterparties is None:
        return None, errors, warnings

    # ── Parse Instruments (required) ──
    inst_sheet = _find_sheet(sheet_map, ['instruments', 'instrument', 'facilities', 'exposures'])
    if inst_sheet is None:
        return None, errors + ["Missing required sheet: 'Instruments'"], warnings

    instruments, inst_errors, inst_warnings = _parse_instruments(
        xls, inst_sheet, {cp['counterparty_id'] for cp in counterparties}
    )
    errors.extend(inst_errors)
    warnings.extend(inst_warnings)

    if instruments is None:
        return None, errors, warnings

    # ── Parse Migration Matrix (optional) ──
    mig_sheet = _find_sheet(sheet_map, ['migration matrix', 'migration', 'transition matrix', 'transitions'])
    migration_matrix = MIGRATION_MATRIX
    if mig_sheet:
        parsed_mig, mig_errors = _parse_migration_matrix(xls, mig_sheet)
        if parsed_mig is not None:
            migration_matrix = parsed_mig
        warnings.extend(mig_errors)
    else:
        warnings.append("No 'Migration Matrix' sheet found; using default transition matrix.")

    # ── Parse Scenarios (optional) ──
    sc_sheet = _find_sheet(sheet_map, ['scenarios', 'scenario', 'stress scenarios', 'stress tests'])
    scenarios = generate_scenarios()
    if sc_sheet:
        parsed_sc, sc_errors = _parse_scenarios(xls, sc_sheet)
        if parsed_sc:
            scenarios = parsed_sc
        warnings.extend(sc_errors)
    else:
        warnings.append("No 'Scenarios' sheet found; using default stress scenarios.")

    if errors:
        return None, errors, warnings

    # ── Generate derived data ──
    snapshots = generate_monthly_snapshots(instruments, 12, seed=42)
    pd_curves = generate_pd_curves(counterparties, seed=42)

    data = {
        'counterparties': counterparties,
        'instruments': instruments,
        'snapshots': snapshots,
        'pd_curves': pd_curves,
        'migration_matrix': migration_matrix,
        'scenarios': scenarios,
        'ratings': RATINGS,
    }

    return data, errors, warnings


def _find_sheet(sheet_map, candidates):
    """Find a sheet by trying multiple name variations."""
    for c in candidates:
        if c in sheet_map:
            return sheet_map[c]
    return None


# Column name normalization: strip, lowercase, remove underscores/spaces
def _norm(col):
    return str(col).strip().lower().replace('_', '').replace(' ', '').replace('(', '').replace(')', '')


def _parse_counterparties(xls, sheet_name):
    """Parse and validate counterparties sheet."""
    errors, warnings = [], []

    try:
        df = pd.read_excel(xls, sheet_name=sheet_name)
    except Exception as e:
        return None, [f"Error reading '{sheet_name}': {e}"], warnings

    if len(df) == 0:
        return None, [f"'{sheet_name}' sheet is empty"], warnings

    # Normalize column names for flexible matching
    col_map = {_norm(c): c for c in df.columns}

    # Map to internal field names
    field_mapping = {
        'counterparty_id': ['counterpartyid', 'cpid', 'obligorid', 'borrowerid', 'id'],
        'legal_name': ['legalname', 'name', 'companyname', 'entityname', 'borrowername'],
        'sector_code': ['sectorcode', 'sector', 'gicsindustrygroup', 'industrygroup', 'industry'],
        'country_code': ['countrycode', 'country', 'domicile'],
        'rating': ['rating', 'creditrating', 'grade', 'ratinggrade'],
        'pd_1y': ['pd1y', 'pd', 'probabilityofdefault', 'defaultprobability', 'pd1year', 'annualpd'],
        'rsq': ['rsq', 'rsquared', 'assetcorrelation', 'r2'],
        'parent_id': ['parentid', 'parent', 'parententity', 'groupid'],
        'revenue_mm': ['revenuemm', 'revenue', 'revenues', 'annualrevenue'],
        'total_assets_mm': ['totalassetsmm', 'totalassets', 'assets'],
    }

    resolved = {}
    for internal_name, candidates in field_mapping.items():
        for c in candidates:
            if c in col_map:
                resolved[internal_name] = col_map[c]
                break

    # Check required fields
    required = ['counterparty_id', 'sector_code', 'country_code', 'pd_1y', 'rating']
    for f in required:
        if f not in resolved:
            errors.append(f"Counterparties: missing required column '{f}' (tried: {field_mapping[f]})")

    if errors:
        return None, errors, warnings

    counterparties = []
    valid_sectors = set(GICS_INDUSTRY_GROUPS)
    valid_countries = set(COUNTRIES)
    valid_ratings = set(RATINGS)

    for idx, row in df.iterrows():
        cp_id = str(row[resolved['counterparty_id']]).strip()
        if not cp_id or cp_id == 'nan':
            warnings.append(f"Counterparties row {idx+2}: empty counterparty_id, skipping")
            continue

        sector = str(row[resolved['sector_code']]).strip()
        if sector not in valid_sectors:
            # Try to match display name
            reverse_display = {v: k for k, v in GICS_GROUP_DISPLAY_NAMES.items()}
            if sector in reverse_display:
                sector = reverse_display[sector]
            else:
                warnings.append(f"Counterparty {cp_id}: unknown sector '{sector}', defaulting to 'CapitalGoods'")
                sector = 'CapitalGoods'

        country = str(row[resolved['country_code']]).strip().upper()
        if country not in valid_countries:
            warnings.append(f"Counterparty {cp_id}: unknown country '{country}', defaulting to 'US'")
            country = 'US'

        rating = str(row[resolved['rating']]).strip().upper()
        if rating not in valid_ratings:
            warnings.append(f"Counterparty {cp_id}: unknown rating '{rating}', defaulting to 'BBB'")
            rating = 'BBB'

        try:
            pd_1y = float(row[resolved['pd_1y']])
            if pd_1y > 1:
                pd_1y = pd_1y / 100  # Convert from percentage
            pd_1y = np.clip(pd_1y, 0.0001, 0.9999)
        except (ValueError, TypeError):
            pd_1y = RATING_PDS.get(rating, 0.0025)
            warnings.append(f"Counterparty {cp_id}: invalid PD, using rating default {pd_1y}")

        rsq = 0.25
        if 'rsq' in resolved:
            try:
                rsq = float(row[resolved['rsq']])
                rsq = np.clip(rsq, 0.05, 0.60)
            except (ValueError, TypeError):
                pass

        legal_name = ''
        if 'legal_name' in resolved:
            legal_name = str(row[resolved['legal_name']]).strip()
            if legal_name == 'nan':
                legal_name = cp_id

        parent_id = None
        if 'parent_id' in resolved:
            val = row[resolved['parent_id']]
            if pd.notna(val) and str(val).strip() and str(val).strip() != 'nan':
                parent_id = str(val).strip()

        revenue = 0
        if 'revenue_mm' in resolved:
            try:
                revenue = float(row[resolved['revenue_mm']])
            except (ValueError, TypeError):
                pass

        total_assets = 0
        if 'total_assets_mm' in resolved:
            try:
                total_assets = float(row[resolved['total_assets_mm']])
            except (ValueError, TypeError):
                pass

        counterparties.append({
            'counterparty_id': cp_id,
            'legal_name': legal_name or cp_id,
            'parent_id': parent_id,
            'sector_code': sector,
            'gics_sector': GICS_GROUP_TO_SECTOR.get(sector, 'Industrials'),
            'country_code': country,
            'region_code': _get_region(country),
            'rating': rating,
            'pd_1y': pd_1y,
            'rsq': rsq,
            'is_public': False,
            'revenue_mm': revenue,
            'total_assets_mm': total_assets,
        })

    if len(counterparties) == 0:
        errors.append("No valid counterparties found in uploaded data")
        return None, errors, warnings

    # Check for duplicate IDs
    ids = [cp['counterparty_id'] for cp in counterparties]
    dupes = set(x for x in ids if ids.count(x) > 1)
    if dupes:
        errors.append(f"Duplicate counterparty IDs: {dupes}")
        return None, errors, warnings

    return counterparties, errors, warnings


def _parse_instruments(xls, sheet_name, valid_cp_ids):
    """Parse and validate instruments sheet."""
    errors, warnings = [], []

    try:
        df = pd.read_excel(xls, sheet_name=sheet_name)
    except Exception as e:
        return None, [f"Error reading '{sheet_name}': {e}"], warnings

    if len(df) == 0:
        return None, [f"'{sheet_name}' sheet is empty"], warnings

    col_map = {_norm(c): c for c in df.columns}

    field_mapping = {
        'instrument_id': ['instrumentid', 'instid', 'facilityid', 'id', 'exposureid'],
        'instrument_type': ['instrumenttype', 'type', 'producttype', 'facilitytype', 'product'],
        'counterparty_id': ['counterpartyid', 'cpid', 'obligorid', 'borrowerid'],
        'lgd': ['lgd', 'lossgivendefault', 'lgdpct'],
        'currency': ['currency', 'ccy', 'curr'],
        'maturity_date': ['maturitydate', 'maturity', 'tenor', 'expiry'],
        'seniority': ['seniority', 'seniorityrank', 'rank'],
        'rating': ['rating', 'facilityrating', 'instrumentrating'],
        'drawn_amount': ['drawnamount', 'drawn', 'outstanding', 'outstandingbalance', 'balance'],
        'undrawn_amount': ['undrawnamount', 'undrawn', 'available', 'commitmentundrawn'],
        'notional': ['notional', 'notionalamount', 'commitment', 'facilityamount', 'limit'],
        'mtm_value': ['mtmvalue', 'mtm', 'marktomarket', 'fairvalue', 'marketvalue'],
        'interest_rate': ['interestrate', 'rate', 'coupon', 'spread'],
        'ccf': ['ccf', 'creditconversionfactor', 'conversionfactor'],
        'addon_factor': ['addonfactor', 'addon', 'pfe', 'potentialfutureexposure'],
        'collateral_type': ['collateraltype', 'collateral', 'security'],
        'cds_direction': ['cdsdirection', 'direction', 'protectiondirection', 'buysell'],
        'cds_spread_bps': ['cdsspreadbps', 'cdsspread', 'spread', 'spreadbps'],
        'cds_seller_id': ['cdssellerid', 'sellerid', 'protectionsellerid', 'seller'],
    }

    resolved = {}
    for internal_name, candidates in field_mapping.items():
        for c in candidates:
            if c in col_map:
                resolved[internal_name] = col_map[c]
                break

    required = ['instrument_id', 'instrument_type', 'counterparty_id', 'lgd']
    for f in required:
        if f not in resolved:
            errors.append(f"Instruments: missing required column '{f}' (tried: {field_mapping[f]})")

    if errors:
        return None, errors, warnings

    valid_types = set(INSTRUMENT_TYPES)
    instruments = []

    for idx, row in df.iterrows():
        inst_id = str(row[resolved['instrument_id']]).strip()
        if not inst_id or inst_id == 'nan':
            continue

        cp_id = str(row[resolved['counterparty_id']]).strip()
        if cp_id not in valid_cp_ids:
            warnings.append(f"Instrument {inst_id}: counterparty '{cp_id}' not in Counterparties, skipping")
            continue

        itype = str(row[resolved['instrument_type']]).strip()
        if itype not in valid_types:
            warnings.append(f"Instrument {inst_id}: unknown type '{itype}', defaulting to 'TermLoan'")
            itype = 'TermLoan'

        try:
            lgd = float(row[resolved['lgd']])
            if lgd > 1:
                lgd = lgd / 100
            lgd = np.clip(lgd, 0.05, 0.95)
        except (ValueError, TypeError):
            lgd = 0.45

        def _get_float(field, default=0.0):
            if field not in resolved:
                return default
            try:
                val = float(row[resolved[field]])
                return val if not np.isnan(val) else default
            except (ValueError, TypeError):
                return default

        def _get_str(field, default=''):
            if field not in resolved:
                return default
            val = str(row[resolved[field]]).strip()
            return default if val == 'nan' else val

        inst = {
            'instrument_id': inst_id,
            'instrument_type': itype,
            'counterparty_id': cp_id,
            'currency': _get_str('currency', 'USD'),
            'maturity_date': _get_str('maturity_date', '2028-01-01'),
            'seniority': _get_str('seniority', 'Senior Unsecured'),
            'lgd': lgd,
            'rating': _get_str('rating', ''),
            'drawn_amount': _get_float('drawn_amount'),
            'undrawn_amount': _get_float('undrawn_amount'),
            'notional': _get_float('notional'),
            'mtm_value': _get_float('mtm_value'),
            'interest_rate': _get_float('interest_rate'),
            'ccf': _get_float('ccf', 0.75),
            'addon_factor': _get_float('addon_factor', 0.01),
            'collateral_type': _get_str('collateral_type', 'Unsecured'),
        }

        # Ensure notional is set if drawn/undrawn are present
        if inst['notional'] == 0 and inst['drawn_amount'] > 0:
            inst['notional'] = inst['drawn_amount'] + inst['undrawn_amount']

        # CDS-specific fields
        if itype == 'CDS':
            direction = _get_str('cds_direction', 'Protection_Sold')
            if direction.lower() in ('bought', 'buy', 'long', 'protection_bought'):
                direction = 'Protection_Bought'
            else:
                direction = 'Protection_Sold'
            inst['cds_direction'] = direction
            inst['cds_spread_bps'] = _get_float('cds_spread_bps', 100.0)
            seller_id = _get_str('cds_seller_id', '')
            if seller_id and seller_id in valid_cp_ids:
                inst['cds_seller_id'] = seller_id

        instruments.append(inst)

    if len(instruments) == 0:
        errors.append("No valid instruments found in uploaded data")
        return None, errors, warnings

    inst_ids = [i['instrument_id'] for i in instruments]
    dupes = set(x for x in inst_ids if inst_ids.count(x) > 1)
    if dupes:
        warnings.append(f"Duplicate instrument IDs found (first occurrence kept): {dupes}")
        seen = set()
        deduped = []
        for i in instruments:
            if i['instrument_id'] not in seen:
                seen.add(i['instrument_id'])
                deduped.append(i)
        instruments = deduped

    return instruments, errors, warnings


def _parse_migration_matrix(xls, sheet_name):
    """Parse an 8x8 migration matrix."""
    errors = []
    try:
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        # Try to find the 8x8 numeric block
        for start_row in range(min(5, len(df))):
            for start_col in range(min(5, len(df.columns))):
                block = df.iloc[start_row:start_row+8, start_col:start_col+8]
                try:
                    matrix = block.values.astype(float)
                    if matrix.shape == (8, 8) and np.all(matrix >= 0) and np.all(matrix <= 1):
                        row_sums = np.sum(matrix, axis=1)
                        if np.allclose(row_sums, 1.0, atol=0.01):
                            return matrix, []
                except (ValueError, TypeError):
                    continue
        errors.append("Migration Matrix: could not find valid 8x8 probability matrix; using default")
    except Exception as e:
        errors.append(f"Migration Matrix: error reading sheet: {e}")
    return None, errors


def _parse_scenarios(xls, sheet_name):
    """Parse scenarios sheet."""
    errors = []
    try:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if len(df) == 0:
            return None, ["Scenarios sheet is empty; using defaults"]

        col_map = {_norm(c): c for c in df.columns}
        scenarios = []
        for idx, row in df.iterrows():
            name_col = None
            for c in ['scenarioname', 'scenario', 'name']:
                if c in col_map:
                    name_col = col_map[c]
                    break
            if name_col is None:
                continue

            sc = {
                'scenario_id': str(row.get(col_map.get('scenarioid', name_col), f'SC{idx}')).strip(),
                'scenario_name': str(row[name_col]).strip(),
                'scenario_type': 'baseline' if idx == 0 else 'stress',
                'description': '',
                'gdp_shock': 0.0, 'rate_shock': 0.0, 'spread_shock': 0.0,
                'factor_shocks': {},
            }
            for c in ['description', 'desc']:
                if c in col_map:
                    sc['description'] = str(row[col_map[c]]).strip()
                    break
            for field, keys in [('gdp_shock', ['gdpshock', 'gdp']),
                                ('rate_shock', ['rateshock', 'rate']),
                                ('spread_shock', ['spreadshock', 'spread'])]:
                for k in keys:
                    if k in col_map:
                        try:
                            sc[field] = float(row[col_map[k]])
                        except (ValueError, TypeError):
                            pass
                        break

            # Parse factor shocks from a column like "US: -2.5; Banks: -3.0"
            for k in ['factorshocks', 'shocks', 'factors']:
                if k in col_map:
                    val = str(row[col_map[k]]).strip()
                    if val and val != 'nan' and val.lower() != 'none':
                        for part in val.split(';'):
                            if ':' in part:
                                factor, shock = part.rsplit(':', 1)
                                try:
                                    sc['factor_shocks'][factor.strip()] = float(shock.strip().replace('σ', ''))
                                except ValueError:
                                    pass
                    break

            scenarios.append(sc)

        if scenarios:
            return scenarios, []
        return None, ["No valid scenarios parsed; using defaults"]
    except Exception as e:
        return None, [f"Scenarios: error reading sheet: {e}"]
