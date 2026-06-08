import pandas as pd
import numpy as np
from pathlib import Path

def load_and_prepare_data(config):
    """
    Load and prepare data for modeling.
    
    Parameters:
    -----------
    config : dict
        Configuration from model_config.yaml
    
    Returns:
    --------
    pd.DataFrame
        Prepared data for modeling
    """
    
    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    
    raw_path = Path(config['data']['raw_path'])
    
    # Load data
    print("\n Loading data...")
    clients_df = pd.read_csv(raw_path / 'clients.csv', dtype={'client_dwh_id': str})
    calls_df = pd.read_csv(raw_path / 'calls.csv', dtype={'client_dwh_id': str, 'call_id': str})
    campaign_uploads_df = pd.read_csv(raw_path / 'campaign_uploads.csv', dtype={'client_dwh_id': str})
    
    # Convert dates
    calls_df['start_time'] = pd.to_datetime(calls_df['start_time'])
    calls_df['dialogue_stage_time'] = pd.to_datetime(calls_df['dialogue_stage_time'])
    
    print(f"  ✓ Clients: {len(clients_df):,}")
    print(f"  ✓ Calls: {calls_df['call_id'].nunique():,} unique calls")
    print(f"  ✓ Campaign uploads: {len(campaign_uploads_df):,}")
    
    # Take only unique calls (one call = one row)
    print("\n Preparing call-level dataset...")
    
    # Take the last stage of each call
    last_stages = calls_df.sort_values('dialogue_stage_time').groupby('call_id').last().reset_index()
    
    # Take the first stage for start time
    first_stages = calls_df.sort_values('dialogue_stage_time').groupby('call_id').first().reset_index()
    
    # Merge
    calls_unique = last_stages[['call_id', 'client_dwh_id', 'sieb_id', 'is_result', 
                                 'client_call_number', 'client_prev_result_rate', 
                                 'client_prev_contact_rate', 'days_since_last_call']].merge(
        first_stages[['call_id', 'start_time']],
        on='call_id'
    )
    
    # Add time features
    calls_unique['hour'] = calls_unique['start_time'].dt.hour
    calls_unique['day_of_week'] = calls_unique['start_time'].dt.dayofweek
    calls_unique['is_weekend'] = (calls_unique['day_of_week'] >= 5).astype(int)
    
    print(f"  ✓ Unique calls: {len(calls_unique):,}")
    
    # Join with clients
    print("\n Joining with clients...")
    df = calls_unique.merge(clients_df, on='client_dwh_id', how='left')
    
    # Join with campaigns to get segment and treename
    campaign_info = campaign_uploads_df[['client_dwh_id', 'sieb_id', 'segment', 'treename']].drop_duplicates()
    df = df.merge(campaign_info, on=['client_dwh_id', 'sieb_id'], how='left')
    
    print(f"  ✓ Final dataset: {len(df):,} rows")
    
    # Check target variable distribution
    print(f"\n Target variable distribution:")
    print(f"  is_result=1 (positive): {df['is_result'].sum():,} ({df['is_result'].mean():.2%})")
    print(f"  is_result=0 (negative): {(df['is_result'] == 0).sum():,} ({(df['is_result'] == 0).mean():.2%})")
    
    # Save the prepared dataset
    processed_path = Path(config['data']['processed_path'])
    processed_path.mkdir(parents=True, exist_ok=True)
    
    output_file = processed_path / 'dataset.csv'
    df.to_csv(output_file, index=False)
    print(f"\n Saved to: {output_file}")
    
    return df