# Crypto Trading Sentiment Analysis with Machine Learning

## Overview
Comprehensive analysis of 211,224 trades from Hyperliquid platform (May 2023 - May 2025) combining statistical analysis with machine learning to predict profitable trades based on market sentiment and trading patterns.

## Project Structure
```
.

├── complete_ml_analysis.ipynb           # ML models + clustering + advanced stats
├── ml_performance_complete.png          # 6-panel ML dashboard
└── README.md                            # This file
```

## Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

### Run Analysis

**Option 1: Statistical Analysis Only**
```bash
jupyter notebook complete_analysis.ipynb
```
Output: 9-panel dashboard + timing heatmaps + 6 insights

**Option 2: Streamlined Pipeline**
```bash
jupyter notebook master_execution_script.ipynb
```
Output: 4-panel dashboard + text report

**Option 3: Complete ML Analysis (Recommended)**
```bash
jupyter notebook complete_ml_analysis.ipynb
```
Output: All visualizations + ML models + clustering + statistical tests

## Key Findings

### 1. Market Sentiment Impact
| Sentiment | Avg PnL | Win Rate | Trades | Statistical Significance |
|-----------|---------|----------|--------|-------------------------|
| **Extreme Greed** | **$67.89** | **46.0%** | 39,992 | p < 0.0001 ✅ |
| Fear | $54.29 | 42.0% | 61,837 | - |
| Greed | $42.74 | 38.0% | 50,303 | - |
| Extreme Fear | $34.54 | 37.0% | 21,400 | - |
| Neutral | $34.31 | 40.0% | 37,686 | - |

**T-test**: Extreme Fear vs Extreme Greed: t = -4.32, p = 0.000016 (highly significant)

### 2. Machine Learning Results

**Best Model: Random Forest**
| Metric | Score |
|--------|-------|
| Accuracy | 88.72% |
| Precision | 81.03% |
| Recall | 94.74% |
| F1-Score | 87.35% |
| AUC-ROC | 97.70% |
| MCC | 0.7805 |

**Cross-Validation**: 97.83% ± 0.09% (5-fold)

**Model Comparison**:
| Model | Accuracy | AUC-ROC | F1-Score |
|-------|----------|---------|----------|
| Random Forest | 88.72% | 97.70% | 87.35% |
| Gradient Boosting | 89.88% | 97.53% | 87.40% |
| Logistic Regression | 88.59% | 96.99% | 86.28% |

### 3. Feature Importance (Top 10)
| Feature | Importance | Contribution |
|---------|------------|--------------|
| trader_recent_winrate | 88.11% | Recent performance |
| trader_cumulative_pnl | 2.79% | Historical profits |
| log_price | 2.05% | Asset price |
| trader_trade_count | 1.62% | Experience |
| fear_greed_score | 1.20% | Market sentiment |
| coin_encoded | 0.85% | Cryptocurrency |
| hour | 0.76% | Time of day |
| log_trade_size | 0.70% | Position size |
| month | 0.55% | Seasonality |
| day_of_week_num | 0.50% | Day pattern |

**Key Insight**: Top 3 features account for 92.9% of predictive power

### 4. Optimal Trading Windows
- **Best Hour**: 12:00 ($131.17 avg PnL)
- **Worst Hour**: 23:00 ($18.75 avg PnL)
- **Best Day**: Saturday ($65.36 avg PnL)
- **Peak Performance**: Saturday at 12:00 during Extreme Greed

### 5. Winner vs Loser Characteristics
| Metric | Winners (41.1%) | Losers (58.9%) | Difference |
|--------|-----------------|----------------|------------|
| Avg Sentiment | 52.4 | 51.2 | +1.2 points |
| Avg Trade Size | $5,779.71 | $5,541.47 | +$238.24 |
| Peak Hour | 19:00 | 20:00 | -1 hour |
| Total PnL | $13.2M | -$2.9M | $16.1M |

### 6. Trader Clustering (K-Means, k=4)

**Cluster 0: Conservative Traders** (5 traders, 15.6%)
- Avg PnL: $44.08
- Win Rate: 36.8%
- Trade Size: $2,197
- Sentiment: 51.6 (neutral)

**Cluster 1: Greed-Seeking Winners** (12 traders, 37.5%)
- Avg PnL: $54.47
- Win Rate: 46.5%
- Trade Size: $2,099
- Sentiment: 59.9 (greedy)

**Cluster 2: High-Risk Players** (5 traders, 15.6%)
- Avg PnL: $365.73
- Win Rate: 34.6%
- Trade Size: $6,619
- Sentiment: 51.4 (neutral)

**Cluster 3: Institutional Scale** (10 traders, 31.2%)
- Avg PnL: $42.33
- Win Rate: 37.5%
- Trade Size: $12,298
- Sentiment: 40.7 (fearful)

### 7. Advanced Statistical Tests

**Pearson Correlation (Sentiment vs PnL)**
- Correlation: 0.0081
- P-value: 0.000190
- Result: ✅ Statistically significant (weak positive correlation)

**One-Way ANOVA (PnL across sentiments)**
- F-statistic: 9.06
- P-value: < 0.000001
- Result: ✅ Significant difference between sentiment groups

### 8. Top Cryptocurrencies
| Coin | Avg PnL | Total PnL | Win Rate | Trades |
|------|---------|-----------|----------|--------|
| @107 | $92.82 | $2.78M | 47% | 29,992 |
| HYPE | $28.65 | $1.95M | 41% | 68,005 |
| SOL | $153.36 | $1.64M | 39% | 10,691 |
| ETH | $118.30 | $1.32M | 36% | 11,158 |
| BTC | $33.30 | $868K | 35% | 26,064 |

## Analysis Notebooks Comparison

### complete_analysis.ipynb
**Purpose**: Deep statistical insights  
**Output**: 
- 9-panel comprehensive dashboard
- Timing heatmaps
- 6 detailed insights with statistical tests

**Key Features**:
- Sentiment performance analysis
- Winner vs loser patterns
- Optimal trading times
- Trader profiles
- Extreme conditions analysis
- Coin-specific performance

### master_execution_script.ipynb
**Purpose**: Quick automated analysis  
**Output**:
- 4-panel simplified dashboard
- Text-based executive report

**Key Features**:
- Streamlined 5-step pipeline
- Essential insights only
- Automated report generation
- Fast execution (~1 minute)

### complete_ml_analysis.ipynb (⭐ Most Comprehensive)
**Purpose**: Full ML + statistical analysis  
**Output**:
- 6-panel ML dashboard
- All statistical visualizations
- Model comparison
- Feature importance
- Clustering analysis

**Key Features**:
- 3 ML models (Logistic, RF, GBM)
- 16 engineered features
- 5-fold cross-validation
- Advanced statistical tests (Pearson, ANOVA)
- K-Means clustering (4 trader types)
- ROC curves & confusion matrices

## Machine Learning Methodology

### Feature Engineering (16 Features)
```python
# Sentiment features
- fear_greed_score          # Raw sentiment (0-100)
- sentiment_extreme         # Binary: extreme condition
- high_greed               # Binary: score > 70
- high_fear                # Binary: score < 30

# Trade features
- log_trade_size           # Log-transformed position size
- log_price                # Log-transformed price
- is_buy                   # Direction (Buy=1, Sell=0)
- is_crossed               # Order type
- coin_encoded             # Cryptocurrency ID

# Temporal features
- hour                     # Hour of day (0-23)
- day_of_week_num          # Day (0=Monday, 6=Sunday)
- is_weekend               # Binary weekend flag
- month                    # Month (1-12)

# Trader history features
- trader_trade_count       # Cumulative trades
- trader_cumulative_pnl    # Running total PnL
- trader_recent_winrate    # Rolling 10-trade win rate
```

### Model Training Pipeline
1. **Data Split**: 80% train, 20% test (stratified)
2. **Feature Scaling**: StandardScaler for numerical features
3. **Class Balancing**: Balanced class weights
4. **Cross-Validation**: 5-fold stratified CV
5. **Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC, MCC

### Model Configurations
```python
# Logistic Regression
- max_iter: 1000
- class_weight: balanced
- solver: lbfgs

# Random Forest
- n_estimators: 100
- max_depth: 10
- class_weight: balanced
- n_jobs: -1

# Gradient Boosting
- n_estimators: 100
- max_depth: 5
- learning_rate: 0.1
```

## Trading Strategies

### Strategy 1: ML-Powered Trading
```python
# Use Random Forest predictions
if model.predict_proba(features)[1] > 0.7:
    trade_size = standard_size * 1.5  # Increase position
elif model.predict_proba(features)[1] < 0.3:
    skip_trade()  # Avoid low-confidence trades
```
**Expected Performance**: 88.7% accuracy, 97.7% AUC

### Strategy 2: Sentiment-Based Trading
```python
if sentiment == "Extreme Greed":
    position_multiplier = 1.5  # +50% size
    expected_pnl = $67.89
elif sentiment == "Neutral":
    position_multiplier = 0.7  # -30% size
    expected_pnl = $34.31
```
**Expected Improvement**: +97.9% over neutral

### Strategy 3: Time-Optimized Trading
```python
OPTIMAL_HOURS = [7, 8, 10, 11, 12]
OPTIMAL_DAYS = ["Saturday", "Friday"]

if hour in OPTIMAL_HOURS and day in OPTIMAL_DAYS:
    execute_trade()
    expected_pnl = $131.17 (hour 12) or $65.36 (Saturday avg)
```

### Strategy 4: Combined Approach (Best Performance)
```python
if (sentiment == "Extreme Greed" and 
    hour == 12 and 
    day == "Saturday" and
    model_confidence > 0.8 and
    trader_recent_winrate > 0.45):
    
    position_size = max_position
    expected_win_rate = 94.74%  # Model recall
    expected_profit = $131+ per trade
```

### Strategy 5: Cluster-Based Trading
```python
# Match your profile to cluster
if cluster == 1:  # Greed-Seeking Winners
    trade_during_sentiment = 59.9
    expected_winrate = 46.5%
elif cluster == 2:  # High-Risk Players
    use_larger_positions = True
    target_avg_pnl = $365.73
```

## Generated Visualizations

### complete_analysis_dashboard.png (9 panels)
1. Performance by Sentiment (bar)
2. Win Rate by Sentiment (bar)
3. Hourly Performance (line)
4. Performance by Day (bar)
5. Sentiment vs PnL (scatter)
6. Trader Profitability (histogram)
7. Winners vs Losers Sentiment (boxplot)
8. Trade Size Distribution (histogram)
9. Top 5 Coins by Profit (horizontal bar)

### comprehensive_insights_dashboard.png (4 panels)
1. Performance by Market Sentiment
2. Win Rate by Sentiment
3. Hourly Performance Pattern
4. Trader Profitability Distribution

### timing_heatmaps.png (2 heatmaps)
1. Win Rate Heatmap (Day × Hour) - Identify best times
2. Average PnL Heatmap (Day × Hour) - Profit optimization

### ml_performance_complete.png (6 panels)
1. Model Accuracy Comparison (bar)
2. Model AUC-ROC Comparison (bar)
3. ROC Curves (all models, line)
4. Top 10 Feature Importance (horizontal bar)
5. Confusion Matrix - Random Forest (heatmap)
6. All Metrics Comparison (grouped bar)

## Code Usage Examples

### Load and Predict with ML Model
```python
# Train model
from sklearn.ensemble import RandomForestClassifier

X_train, y_train = prepare_features(historical_data)
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Predict new trade
new_trade_features = extract_features(current_trade)
probability = model.predict_proba(new_trade_features)[0][1]

if probability > 0.7:
    print(f"High confidence: {probability:.1%} - EXECUTE TRADE")
else:
    print(f"Low confidence: {probability:.1%} - SKIP")
```

### Filter Optimal Trading Conditions
```python
optimal_trades = merged_df[
    (merged_df['market_sentiment'] == 'Extreme Greed') &
    (merged_df['hour'].isin([7, 8, 10, 11, 12])) &
    (merged_df['day_of_week'] == 'Saturday')
]

print(f"Optimal trades: {len(optimal_trades)}")
print(f"Avg PnL: ${optimal_trades['Closed PnL'].mean():.2f}")
print(f"Win rate: {optimal_trades['profitable'].mean()*100:.1f}%")
```

### Identify Your Trader Cluster
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Your trading stats
your_stats = {
    'Avg_PnL': 50.0,
    'Win_Rate': 0.45,
    'Avg_Trade_Size': 3000,
    'Avg_Sentiment': 55
}

# Find cluster
cluster = predict_cluster(your_stats)
print(f"You belong to Cluster {cluster}")
print(cluster_characteristics[cluster])
```

## Dataset Details

### Historical Trading Data
- **Records**: 211,224 trades
- **Period**: May 1, 2023 - May 1, 2025
- **Total Volume**: $1.19 billion
- **Traders**: 32 active accounts
- **Cryptocurrencies**: 280+ symbols

**Columns**: Account, Coin, Side, Execution Price, Size USD, Closed PnL, Leverage, Timestamp IST, Event, Crossed, Start Position

### Fear & Greed Index
- **Records**: 2,644 daily readings
- **Scale**: 0-100
  - 0-24: Extreme Fear
  - 25-49: Fear
  - 50-74: Greed
  - 75-100: Extreme Greed
- **Source**: Bitcoin Fear & Greed Index

### Engineered Features (36 total)
- **Temporal**: hour, day, month, weekend flag
- **Sentiment**: raw score, extreme flags, high/low thresholds
- **Trade**: log size, log price, direction, order type
- **Trader History**: cumulative PnL, trade count, rolling win rate
- **Encoded**: coin ID, account ID

## Performance Metrics

### Overall Statistics
- **Total PnL**: $10,296,958.94
- **Average PnL**: $48.75 per trade
- **Overall Win Rate**: 41.13%
- **Profitable Trades**: 86,869 / 211,224
- **Top Trader**: $2,143,382.60

### ML Model Performance
- **Best Accuracy**: 89.88% (Gradient Boosting)
- **Best AUC**: 97.70% (Random Forest)
- **Best Recall**: 94.74% (Random Forest) - Catches 95% of winners
- **Best Precision**: 89.50% (Gradient Boosting)
- **Most Consistent**: Random Forest (lowest variance)

### Statistical Significance
- **Sentiment Impact**: p < 0.0001 (ANOVA)
- **Winner vs Loser**: p = 0.00019 (Pearson correlation)
- **Extreme Conditions**: p = 0.000016 (T-test)

## Actionable Insights

### For Traders
1. **Trade during Extreme Greed** - 97.9% better than neutral
2. **Use ML predictions** - 88.7% accuracy, 97.7% AUC
3. **Focus on Saturday 12:00** - Peak performance window
4. **Monitor recent win rate** - 88% of prediction power
5. **Target $5,780 position size** - Matches winner average
6. **Identify your cluster** - Optimize strategy for your profile

### For Algorithms
1. **Prioritize trader history** - Recent win rate is #1 feature
2. **Use ensemble models** - Random Forest best overall
3. **Implement confidence thresholds** - Only trade >70% probability
4. **Combine sentiment + time** - Multiplicative effects
5. **Track cumulative PnL** - 2nd most important feature

### For Risk Management
1. **Avoid 23:00** - Worst hour ($18.75 avg)
2. **Reduce size in Neutral** - 30% lower performance
3. **Watch Extreme Fear** - Highest volatility (1136 std)
4. **Use stop-losses** - 58.9% of trades are unprofitable
5. **Diversify across clusters** - Different strategies work

## Limitations & Disclaimers

### Model Limitations
- **Data Period**: 2023-2025 crypto market specific
- **Sample Size**: 32 traders (may not generalize)
- **Feature Dominance**: 88% from recent win rate alone
- **Overfitting Risk**: High tree-based model performance
- **Temporal Bias**: Recent data may not predict future

### Statistical Caveats
- **Weak Correlation**: r = 0.008 (sentiment vs PnL)
- **Multiple Testing**: No Bonferroni correction applied
- **Survivorship Bias**: Only includes active traders
- **Market Regime**: Bull market 2023-2025

### Risk Warnings
- **Past ≠ Future**: Historical performance doesn't guarantee results
- **High Leverage**: Data includes leveraged positions
- **Volatility**: Crypto markets are highly volatile
- **Model Decay**: Retraining required as markets evolve
- **Not Financial Advice**: Educational purposes only

## Requirements

### Python Packages
```bash
pip install pandas==1.5.3
pip install numpy==1.24.3
pip install matplotlib==3.7.1
pip install seaborn==0.12.2
pip install scipy==1.10.1
pip install scikit-learn==1.2.2
```

### System Requirements
- **RAM**: 8GB minimum (16GB recommended)
- **CPU**: Multi-core for parallel processing
- **Storage**: 500MB for data + outputs
- **Python**: 3.8 or higher

## Future Enhancements

### Planned Features
- [ ] Real-time prediction API
- [ ] Deep learning models (LSTM, Transformers)
- [ ] Sentiment from social media (Twitter, Reddit)
- [ ] Multi-exchange data integration
- [ ] Backtesting framework with slippage
- [ ] Risk-adjusted metrics (Sharpe, Sortino)
- [ ] Portfolio optimization
- [ ] Automated trading bot integration

### Research Ideas
- [ ] Non-linear sentiment effects
- [ ] Interaction terms between features
- [ ] Time-series forecasting (ARIMA, Prophet)
- [ ] Reinforcement learning agents
- [ ] Volatility prediction models
- [ ] Market regime detection

## Contributing
For improvements or bug reports, please follow student-appropriate practices:
1. Document your methodology
2. Include statistical tests
3. Validate on test data
4. Compare against baselines

## Author
Analysis completed for Hyperliquid hiring assignment  
Focus: Data science + ML + statistical analysis

## License

Educational use only. Data sourced from Hyperliquid and Fear & Greed Index.
