import pandas as pd
import numpy as np
import joblib
import yaml
import pyarrow as pa
from pathlib import Path
from typing import Union, List, Optional

def load_config_file(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_bmi_models(config):
    path_to_bmi_models = Path(config["load"]["bmi_model"]["path"])
    linreg_bmi_model_filename = config["load"]["bmi_model"]["linreg_model"]
    randfor_bmi_model_filename = config["load"]["bmi_model"]["randfor_model"]
    lin_reg = joblib.load(path_to_bmi_models.joinpath(linreg_bmi_model_filename))
    randfor = joblib.load(path_to_bmi_models.joinpath(randfor_bmi_model_filename))
    return lin_reg, randfor

def get_valid_birads_5_indices_v0(x, grace_period_days=7, interval_months=None):
    return x["event_birads5_indices"]

def get_valid_birads_5_indices_v1(x, grace_period_days=7, interval_months=6):
    '''
        Not all BI-RADS 5 should be considered as labels for positives.

        Here we provide one way of deciding which BI-RADS 5 should be used as
        labels for positives: If a BI-RADS 5 does not have either a benign BI-RADS 
        or benign biopsy within a given interval (birads_dt + grace_period_days, 
        birads_dt + interval_months), then it can be used as a positive label.
    '''
    # -- custom 'is_missing' function for fields such as the biopsy dates/results.
    def is_missing(x):
        if isinstance(x, (list, np.ndarray)):
            return False
        return pd.isna(x)
    
    # -- no BI-RADS 5 for validation.
    if len(x["event_birads5_indices"]) == 0:
        return []
    
    # if BI-RADS 5 is present, then collect its indices in the array of mammograms
    birads5_indices = x["event_birads5_indices"] 
    mammo_dates = pd.to_datetime(x["mammogram_complete_dates"])
    benign_birads_dates = mammo_dates[x["event_benign_birads_indices"]]
    
    # -- the field for biopsy's dates can be empty 
    if is_missing(x["biopsy_dates"]):
        benign_biopsy_dates = pd.Series([])
    else:
        biopsy_dates = pd.to_datetime(x["biopsy_dates"])   
        benign_biopsy_dates = biopsy_dates[x["event_benign_biopsy_indices"]] 

    benign_biopsy_dates = benign_biopsy_dates.sort_values()
    benign_birads_dates = benign_birads_dates.sort_values()

    res = []
    # -- check which BI-RADS 5 are valid according the current criterion
    for i, birads5_index in enumerate(birads5_indices):
        birads5_date = mammo_dates[birads5_index]
        start_interval = birads5_date + pd.DateOffset(days=grace_period_days)
        end_interval = birads5_date + pd.DateOffset(months=interval_months)

        # -- filtering using boolean masking
        has_benign_biopsy = ((benign_biopsy_dates >= start_interval) & (benign_biopsy_dates <= end_interval)).any()
        has_benign_birads = ((benign_birads_dates >= start_interval) & (benign_birads_dates <= end_interval)).any()

        if not has_benign_biopsy and not has_benign_birads:
            res.append(birads5_index)
    return res

def get_valid_birads_5_indices_v2(x, grace_period_days=7, interval_months=None):
    '''
        In this version, a BI-RADS 5 is valid only if after its date there are either another exam with
        BI-RADS 6 or a confirmatory biopsy for breast cancer. 
    '''
    if len(x["event_birads5_indices"])==0:
        return []
    
    dates_birads5 = [ x["mammogram_complete_dates"][birads5_index] for birads5_index in x["event_birads5_indices"] ]
    dates_birads6 = [ x["mammogram_complete_dates"][birads6_index] for birads6_index in x["event_birads6_indices"] ]
    dates_positive_biopsy = [ x["biopsy_dates"][biopsy_index] for biopsy_index in x["event_biopsy_indices"] ]
    # -- keep only the BIRADS-5 indices where there is BIRADS-6 or Malignant Biopsy after the current BIRADS-5.
    new_event_birads5_indices = [ x["event_birads5_indices"][index] for index, birads5_date in enumerate(dates_birads5) if any([ birads5_date <= birads6_date for birads6_date in dates_birads6 ]) or any([ birads5_date <= biopsy_date for biopsy_date in dates_positive_biopsy ]) ]
    return new_event_birads5_indices


def calculate_survival_time(
    mammogram_date_arr, 
    event_date_arr, 
    limit_time=60, 
    limit_date=pd.Timestamp("2024-12-01")
):
    """
        Calculates the survival time for each mammogram, capping follow-up at `limit_time` months.

        Parameters:
        - mammogram_date_arr: Array of pandas.Timestamp representing mammogram dates.
        - event_date_arr: Array of pandas.Timestamp (or NaT) representing cancer diagnosis dates.
        - limit_time: Maximum follow-up period in months (default = 60 months).
        - limit_date: Absolute last date of observation (default = Dec 1, 2024).

        Returns:
        - survival_time_arr: Array with survival time (in months).
        - event_indicator_arr: Array (1 if cancer occurs within `limit_time`, 0 if censored).
    """
    survival_time_arr = []
    event_indicator_arr = []

    for mammogram_date, event_date in zip(mammogram_date_arr, event_date_arr):
        # -- define max follow-up period (e. g. 60 months from mammogram date)
        max_followup_date = mammogram_date + pd.DateOffset(months=limit_time)
        # -- ensure we do not exceed the global `limit_date`
        max_observation_date = min(max_followup_date, limit_date)

        # -- convert to numpy.datetime64[M] for month difference calculation
        mammogram_np = np.datetime64(mammogram_date, "M")
        max_observation_np = np.datetime64(max_observation_date, "M")

        if pd.isna(event_date):  # -- no cancer event (censored)
            survival_time = (max_observation_np - mammogram_np).astype(int)
            event_indicator = 0
        else:  # -- cancer event occurred
            event_np = np.datetime64(event_date, "M")
            if event_np <= max_observation_np:
                survival_time = (event_np - mammogram_np).astype(int)
                event_indicator = 1  # -- event occurred
            else:  # -- event happens **after max observation period → censored**
                survival_time = (max_observation_np - mammogram_np).astype(int)
                event_indicator = 0 

        # -- append results
        survival_time_arr.append(min(survival_time, limit_time))  # Ensure it never exceeds `limit_time`
        event_indicator_arr.append(event_indicator)
    return np.array(survival_time_arr), np.array(event_indicator_arr)

def calculate_survival_time(
    mammogram_date_arr: Union[pd.Series, np.ndarray], 
    event_date_arr: Union[pd.Series, np.ndarray],
    lower_limit_time: int,
    upper_limit_time: int,  
    limit_date=pd.Timestamp("2024-12-01")
):
    """
        Calculates the survival time for each mammogram, capping follow-up at `limit_time` months.

        Parameters:
        - mammogram_date_arr: Array of pandas.Timestamp representing mammogram dates.
        - event_date_arr: Array of pandas.Timestamp (or NaT) representing cancer diagnosis dates.
        - limit_time: Maximum follow-up period in months (default = 60 months).
        - limit_date: Absolute last date of observation (default = Dec 1, 2024).

        Returns:
        - survival_time_arr: Array with survival time (in months).
        - event_indicator_arr: Array (1 if cancer occurs within `limit_time`, 0 if censored).
    """
    survival_time_arr = []
    event_indicator_arr = []

    min_followup_date_arr = [ np.datetime64(cur_mammogram_date + pd.DateOffset(months=lower_limit_time), "M") for cur_mammogram_date in mammogram_date_arr ]
    min_followup_date_arr = [ np.datetime64(elem, "M") if elem>=limit_date else np.nan for elem in min_followup_date_arr ] 
    max_followup_date_arr = [ np.datetime64(cur_mammogram_date + pd.DateOffset(months=lower_limit_time), "M") for cur_mammogram_date in mammogram_date_arr ]
    max_followup_date_arr = [ np.datetime64(min(elem, limit_date), "M") for elem in max_followup_date_arr ] 

    for mammogram_date, event_date in zip(mammogram_date_arr, event_date_arr):
        # -- define max follow-up period (e. g. 60 months from mammogram date)
        #max_followup_date = mammogram_date + pd.DateOffset(months=limit_time)
        min_followup_date = mammogram_date + pd.DateOffset(months=lower_limit_time)
        max_followup_date = mammogram_date + pd.DateOffset(months=upper_limit_time)
        # -- ensure we do not exceed the global `limit_date`
        max_observation_date = min(max_followup_date, limit_date)


        # -- convert to numpy.datetime64[M] for month difference calculation
        mammogram_np = np.datetime64(mammogram_date, "M")
        max_observation_np = np.datetime64(max_observation_date, "M")

        if pd.isna(event_date):  # -- no cancer event (censored)
            survival_time = (max_observation_np - mammogram_np).astype(int)
            event_indicator = 0
        else:  # -- cancer event occurred
            event_np = np.datetime64(event_date, "M")
            if event_np <= max_observation_np:
                survival_time = (event_np - mammogram_np).astype(int)
                event_indicator = 1  # -- event occurred
            else:  # -- event happens **after max observation period → censored**
                survival_time = (max_observation_np - mammogram_np).astype(int)
                event_indicator = 0 

        # -- append results
        survival_time_arr.append(min(survival_time, limit_time))  # Ensure it never exceeds `limit_time`
        event_indicator_arr.append(event_indicator)
    return np.array(survival_time_arr), np.array(event_indicator_arr)

def calculate_interval_label(
    mammogram_index_date_arr: Union[pd.Series, np.ndarray], 
    event_date_arr: Union[pd.Series, np.ndarray],
    lower_limit_time: int,
    upper_limit_time: int,  
    limit_date=pd.Timestamp("2024-12-01")
):
    """
        Calculate the interval label and survival time delta (absolute and in months).

        Args:
        -----
        - mammogram_index_date_arr: Array of pandas.Timestamp representing mammogram dates.
        - event_date_arr: Array of pandas.Timestamp (or NaT) representing cancer diagnosis dates (if exists).
        - lower_limit_time: Number of months after the index mammogram date to define the lower bound date of the interval. 
        - upper_limit_time: Number of months after the index mammogram date to define the upper bound date of the interval.
        - limit_date: Absolute last date of observation (end of the cohort).

        Returns:
        --------
        - interval_label: event indicator for the defined interval.
        - survival_timedelta: Array with survival time delta (absolute).
        - survival_days: Array with survival time delta (approximated to months).
    """
    mammogram_index_date_arr = np.asarray(mammogram_index_date_arr, dtype='datetime64[ns]')
    event_date_arr = np.asarray(event_date_arr, dtype='datetime64[ns]')
    min_followup_date_arr = np.asarray([ np.datetime64(cur_mammogram_date + pd.DateOffset(months=lower_limit_time, normalize=True)) for cur_mammogram_date in mammogram_index_date_arr ], dtype='datetime64[ns]')
    min_followup_date_arr = np.asarray([ np.datetime64(elem) if elem<limit_date else np.datetime64('NaT') for elem in min_followup_date_arr ])
    max_followup_date_arr = np.asarray([ np.datetime64(cur_mammogram_date + pd.DateOffset(months=upper_limit_time, normalize=True)) for cur_mammogram_date in mammogram_index_date_arr ], dtype='datetime64[ns]')
    max_followup_date_arr = np.asarray([ np.datetime64(min(elem, limit_date)) for elem in max_followup_date_arr ])
    
    #return min_followup_date_arr, max_followup_date_arr
    # -- create interval label
    bounds_ok = (~np.isnat(min_followup_date_arr)) & (~np.isnat(max_followup_date_arr))
    in_interval = bounds_ok & (event_date_arr >= min_followup_date_arr) & (event_date_arr <= max_followup_date_arr)
    event_missing = np.isnat(event_date_arr)
    interval_label = np.where(~event_missing & in_interval, 1, 0).astype(np.int8)

    # -- calculate survival time (time delta)
    target = np.where(~np.isnat(event_date_arr), event_date_arr, max_followup_date_arr)
    survival_timedelta = target - mammogram_index_date_arr

    # -- approximation to months
    td_ns = np.timedelta64(30 * 24 * 60 * 60 * 10**9, 'ns')
    survival_days = survival_timedelta / np.timedelta64(1, 'D')
    return interval_label, survival_timedelta, survival_days

def get_interval_label_for_benign_followup(
    lower_date_arr: Union[pd.Series, np.ndarray],
    upper_date_arr: Union[pd.Series, np.ndarray],
    results_arr: Union[pd.Series, np.ndarray],
    dates_arr: Union[pd.Series, np.ndarray],
    label_to_identify: Union[int, List] = [0],
    return_binary: Optional[bool] = True
):
    """
        Calculate the interval label for a benign follow-up exam.

        If there is any benign biopsies or mammograms within the intervals defined by 'lower_date_arr'
        and 'upper_date_arr', returns either a binary flag as indicator or the amount of
        biopsies/mammograms in this interval (if 'return_bionary'=False).

        Args:
        -----
        - lower_date_arr: Array of pandas.Timestamp representing lower bound dates for the intervals.
        - upper_date_arr: Array of pandas.Timestamp representing upper bound dates for the intervals.
        - results_arr: Array of lists representing biopsy or mammogram results (0 - benign or 1 - malignant).
        - dates_arr: Array of lists representing biopsy or mammograms dates respective to the results array.
        - label_to_identify: [0] for biopsies, [0,1,2,3,4] for mammograms
        - return_binary: Boolean. Return binary indicator if True, else return the counts.

        Returns:
        --------
        - counts: event indicator (or count) for the defined interval.
    """
    low = pd.to_datetime(pd.Series(lower_date_arr), errors="coerce")
    up  = pd.to_datetime(pd.Series(upper_date_arr), errors="coerce")
    res = pd.Series(results_arr, dtype=object)
    dts = pd.Series(dates_arr, dtype=object)

    # -- ensure empty/missing lists don’t break explode
    res = res.apply(lambda x: [] if x is None or (isinstance(x, float) and pd.isna(x)) else list(x))
    dts = dts.apply(lambda x: [] if x is None or (isinstance(x, float) and pd.isna(x)) else list(x))

    # -- build long table; explode both columns in parallel (same lengths per row)
    long = pd.DataFrame({"res": res, "dt": dts})
    if len(long):
        long = long.explode(["res", "dt"], ignore_index=False)
    else:
        long = pd.DataFrame(columns=["res", "dt"])

    # Coerce exploded datetimes and attach row-wise bounds
    long["dt"] = pd.to_datetime(long["dt"], errors="coerce")
    long["low"] = low.reindex(long.index)
    long["up"]  = up.reindex(long.index)

    # Valid only when both bounds exist
    bounds_ok = long["low"].notna() & long["up"].notna() & long["dt"].notna()

    # Count biopsies with result == 0 inside (low, up]
    hit = bounds_ok & (long["res"].isin(label_to_identify)) & (long["dt"] > long["low"]) & (long["dt"] <= long["up"])
    counts = hit.groupby(level=0).sum().reindex(range(len(low)), fill_value=0).astype(int)

    if return_binary:
        return (counts > 0).astype(np.int8).to_numpy()
    else:
        return counts.to_numpy()
    

def process_chunk(
    df: Union[pd.DataFrame], 
    fields: dict,
    fixed_features_columns: List[str], 
    timed_features_columns: List[str],
    mamm_field_suffix: Optional[str] = '_MAMOGRAFIA',
    anamnesis_field_suffix: Optional[str] = '_ANAMNESE'
) -> pd.DataFrame:
    col_cd_pessoa = fields["person_id"]
    col_dt_atend_mamografia = fields["mammogram_date"]+mamm_field_suffix
    col_dt_atend_anamnese = fields["anamnesis_date"]+anamnesis_field_suffix
    col_cd_atend = fields["mammogram_id_final"]
    col_dt_nasc = fields["person_birthdate"]
    col_birads = "birads_labels"
    
    
    out_rows = []
    # itertuples is much faster; set name=None for simple tuples
    for row in df.itertuples(index=False, name=None):
        # Map tuple positions to names once for readability
        r = df.columns
        row = dict(zip(r, row))
        #print(row)
        #print("PAUSE")
        
        person_id = row[col_cd_pessoa]
        mammo_ids = row[col_cd_atend]
        mammo_codes3 = row["mammogram_codes_upto3"]
        mammo_dates = row["mammogram_dates_upto3"]
        mammo_res = row["birads_upto3"]

        mammo_complete_dates = row[col_dt_atend_mamografia]
        mammo_complete_res = row[col_birads]
        anamnesis_dates = row[col_dt_atend_anamnese]

        fixed_features = {col: row.get(col) for col in fixed_features_columns}
        timed_features = {col: row.get(col) for col in timed_features_columns}

        biopsy_dates = row.get("biopsy_dates")
        biopsy_results = row.get("biopsy_results")

        # Skip rows without mammograms
        #if not isinstance(mammo_dates, (list, tuple)) or len(mammo_dates) == 0:
        #    continue

        for exam_date, exam_res, exam_id in zip(mammo_dates, mammo_res, mammo_ids):
            ## features available up to exam_date
            #temp_timed_features = {}
            #if pd.isna(anamnesis_dates) or not isinstance(anamnesis_dates, (list, tuple)):
            #    for nm in timed_features:
            #        temp_timed_features[nm] = np.nan
            #else:
            #    for nm, anam_info in timed_features.items():
            #        if isinstance(anam_info, (list, tuple)):
            #            temp_timed_features[nm] = [
            #                val for d, val in zip(anamnesis_dates, anam_info) if d <= exam_date
            #            ]
            #        else:
            #            temp_timed_features[nm] = np.nan
            
            temp_timed_features = {}            
            for nm, anamnesis_info in timed_features.items():
                    if (type(anamnesis_dates)==float or anamnesis_dates is None) and pd.isna(anamnesis_dates):
                        temp_timed_features.update({nm:np.nan})
                    else:
                        #print(anamnesis_dates, anamnesis_info)
                        temp_timed_features.update({
                            nm : [ val for date, val in zip(anamnesis_dates, anamnesis_info) if date <= exam_date ]
                        })

            birads_time = [d for d, _ in zip(mammo_dates, mammo_res) if d <= exam_date]
            birads_arr  = [x for d, x in zip(mammo_dates, mammo_res) if d <= exam_date]
            mammograms_previous = [x for d, x in zip(mammo_dates, mammo_codes3) if d <= exam_date]

            transformed_row = {
                'person_id': person_id,
                'mammogram_id': exam_id,
                'mammogram_current_date': exam_date,
                'mammogram_current_result': exam_res,
                'mammogram_prior_codes': mammograms_previous,
                'mammogram_prior_dates': birads_time,
                'mammogram_prior_birads': birads_arr,
                'mammogram_complete_dates': mammo_complete_dates,
                'mammogram_complete_birads': mammo_complete_res,
                'mammogram_complete_codes': mammo_ids,
                'biopsy_dates': biopsy_dates,
                'biopsy_results': biopsy_results,
                **fixed_features,
                **temp_timed_features,
            }
            out_rows.append(transformed_row)
                
    if len(out_rows)==0:
        print('ZERO')

    # return as Arrow table for zero-copy append
    return pd.DataFrame(out_rows)
    if out_rows:
        return pa.Table.from_pandas(pd.DataFrame(out_rows), preserve_index=False)
    return None