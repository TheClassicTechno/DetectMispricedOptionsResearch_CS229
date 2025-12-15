#!/usr/bin/env python3
"""
Advanced Feature Engineering for Options Mispricing
Implement additional features that could improve model accuracy
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def create_advanced_features(dataframe):
    """Create advanced features for options pricing"""
    
    # Greek and volatility interactions
    dataframe['moneyness_iv'] = dataframe['moneyness'] * dataframe['iv']
    dataframe['delta_gamma'] = dataframe['delta'] * dataframe['gamma']
    dataframe['vega_iv'] = dataframe['vega'] * dataframe['iv']
    dataframe['theta_tau'] = dataframe['theta'] * dataframe['tau_days']
    dataframe['vix_iv_spread'] = dataframe['vix'] - dataframe['iv']
    
    # Polynomial transformations
    dataframe['moneyness_squared'] = dataframe['moneyness'] ** 2
    dataframe['iv_squared'] = dataframe['iv'] ** 2
    dataframe['tau_sqrt'] = np.sqrt(dataframe['tau_days'])
    dataframe['tau_log'] = np.log(dataframe['tau_days'] + 1)
    
    # Option elasticity
    dataframe['elasticity'] = dataframe['delta'] * dataframe['S'] / (dataframe['mkt_price'] + 1e-6)
    
    # Moneyness bins - categorical approach
    dataframe['deep_otm'] = (dataframe['moneyness'] < 0.9).astype(int)
    dataframe['otm'] = ((dataframe['moneyness'] >= 0.9) & (dataframe['moneyness'] < 0.95)).astype(int)
    dataframe['near_atm'] = ((dataframe['moneyness'] >= 0.95) & (dataframe['moneyness'] <= 1.05)).astype(int)
    dataframe['itm'] = ((dataframe['moneyness'] > 1.05) & (dataframe['moneyness'] <= 1.1)).astype(int)
    dataframe['deep_itm'] = (dataframe['moneyness'] > 1.1).astype(int)
    
    # Expiration time buckets
    dataframe['very_short'] = (dataframe['tau_days'] <= 7).astype(int)
    dataframe['short'] = ((dataframe['tau_days'] > 7) & (dataframe['tau_days'] <= 30)).astype(int)
    dataframe['medium'] = ((dataframe['tau_days'] > 30) & (dataframe['tau_days'] <= 90)).astype(int)
    dataframe['long'] = (dataframe['tau_days'] > 90).astype(int)
    
    # Volatility regime indicators
    vix_lower_quartile = dataframe['vix'].quantile(0.25)
    vix_upper_quartile = dataframe['vix'].quantile(0.75)
    dataframe['low_vol_regime'] = (dataframe['vix'] <= vix_lower_quartile).astype(int)
    dataframe['high_vol_regime'] = (dataframe['vix'] >= vix_upper_quartile).astype(int)
    
    # Greek ratios
    dataframe['gamma_vega_ratio'] = dataframe['gamma'] / (dataframe['vega'] + 1e-6)
    dataframe['theta_delta_ratio'] = dataframe['theta'] / (abs(dataframe['delta']) + 1e-6)
    
    # Pricing deviation metrics
    dataframe['price_deviation_pct'] = (dataframe['mkt_price'] - dataframe['bs_price']) / (dataframe['bs_price'] + 1e-6)
    dataframe['abs_price_deviation'] = abs(dataframe['price_deviation_pct'])
    
    return dataframe

def select_best_features(features, labels, num_features=20):
    """Select top features using mutual information"""
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    
    selector = SelectKBest(score_func=mutual_info_classif, k=num_features)
    selected_features = selector.fit_transform(features, labels)
    
    feature_names = features.columns[selector.get_support()]
    print(f"Selected {num_features} best features:")
    for index, name in enumerate(feature_names):
        print(f"  {index+1}. {name}")
    
    return selected_features, feature_names, selector

def improve_feature_engineering():
    """Main function to improve feature engineering"""
    
    # Load data
    data_path = Path("svm/data/clean_options_synth.csv")
    df = pd.read_csv(data_path)
    
    print(f"Original features: {len(df.columns)}")
    
    # Create advanced features
    df_enhanced = create_advanced_features(df.copy())
    
    # Remove non-feature columns
    feature_cols = [col for col in df_enhanced.columns 
                   if col not in ['date', 'ticker', 'option_type', 'S', 'K', 
                                'bs_price', 'mkt_price', 'residual', 'label_uf_over', 
                                'fwd_option_return']]
    
    X = df_enhanced[feature_cols]
    y = df_enhanced['label_uf_over']
    
    print(f"Enhanced features: {len(X.columns)}")
    print(f"New features added: {len(X.columns) - 8}")
    
    # Feature selection
    X_selected, selected_features, selector = select_best_features(X, y, k=15)
    
    # Save enhanced dataset
    enhanced_path = data_path.parent / "enhanced_options_data.csv"
    df_enhanced.to_csv(enhanced_path, index=False)
    
    print(f"Enhanced dataset saved to: {enhanced_path}")
    
    return df_enhanced, selected_features

if __name__ == "__main__":
    df_enhanced, selected_features = improve_feature_engineering()