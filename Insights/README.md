# Crypto Trading Sentiment Analysis

## Overview
Analysis of 211,224 trades from Hyperliquid platform (May 2023 - May 2025) correlating trader performance with Bitcoin Fear & Greed Index to identify profitable trading patterns.

## Project Structure
```
.

├── master_execution_script.ipynb        # Streamlined analysis pipeline
├── complete_analysis_dashboard.png      # 9-panel visualization
├── comprehensive_insights_dashboard.png # 4-panel simplified dashboard
├── timing_heatmaps.png                  # Day×Hour heatmaps
├── analysis_report.txt                  # Executive summary
└── README.md                            # This file
```

## Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scipy
```

### Run Analysis
**Option 1: Complete Analysis (Recommended)**
```bash
jupyter notebook complete_analysis.ipynb
```
Generates: 9-panel dashboard + timing heatmaps + console insights

**Option 2: Streamlined Pipeline**
```bash
jupyter notebook master_execution_script.ipynb
```
Generates: 4-panel dashboard + analysis report

## Key Findings

### 1. Market Sentiment Impact
| Sentiment | Avg PnL | Win Rate | Trades |
|-----------|---------|----------|--------|
| **Extreme Greed** | **$67.89** | **46.0%** | 39,992 |
| Fear | $54.29 | 42.0% | 61,837 |
| Greed | $42.74 | 38.0% | 50,303 |
| Extreme Fear | $34.54 | 37.0% | 21,400 |
| Neutral | $34.31 | 40.0% | 37,686 |

**Key Insight**: Extreme Greed outperforms Neutral by 97.9% (p < 0.0001)

### 2. Optimal Trading Windows
- **Best Hour**: 12:00 ($131.17 avg PnL)
- **Worst Hour**: 23:00 ($18.75 avg PnL)
- **Best Day**: Saturday ($65.36 avg PnL)
- **Strategy**: Trade Saturdays at 12:00 for maximum profitability

### 3. Winner Characteristics
| Metric | Winners (41.1%) | Losers (58.9%) |
|--------|-----------------|----------------|
| Avg Sentiment | 52.4 | 51.2 |
| Avg Trade Size | $5,779.71 | $5,541.47 |
| Peak Trading Hour | 19:00 | 20:00 |

**Insight**: Winners trade during slightly greedier markets (+1.2 sentiment points)

### 4. Trader Performance
- **Active Traders**: 32 (≥10 trades)
- **Profitable**: 29 traders (90.6%)
- **Top Performer**: $2,143,382.60 total PnL
- **Top 20% Profile**: 41.3% win rate, $11,422 avg trade size

### 5. Extreme Conditions
| Condition | Avg PnL | Win Rate | Volatility |
|-----------|---------|----------|------------|
| Extreme Greed (>75) | $74.74 | 44.37% | 805.91 |
| Neutral (40-60) | $49.97 | 39.70% | 922.49 |
| Extreme Fear (<25) | $34.54 | 37.06% | 1136.06 |

### 6. Top Cryptocurrencies
| Coin | Avg PnL | Total PnL | Win Rate |
|------|---------|-----------|----------|
| @107 | $92.82 | $2.78M | 47% |
| HYPE | $28.65 | $1.95M | 41% |
| SOL | $153.36 | $1.64M | 39% |
| ETH | $118.30 | $1.32M | 36% |
| BTC | $33.30 | $868K | 35% |

## Generated Visualizations

### complete_analysis_dashboard.png (9 panels)
1. Performance by Sentiment (bar chart)
2. Win Rate by Sentiment (bar chart)
3. Hourly Performance (line chart)
4. Performance by Day (bar chart)
5. Sentiment vs PnL (scatter)
6. Trader Profitability Distribution (histogram)
7. Winners vs Losers Sentiment (box plot)
8. Trade Size Distribution (histogram)
9. Top 5 Coins by Profit (horizontal bar)

### comprehensive_insights_dashboard.png (4 panels)
1. Performance by Market Sentiment
2. Win Rate by Sentiment
3. Hourly Performance Pattern
4. Trader Profitability Distribution

### timing_heatmaps.png (2 heatmaps)
1. Win Rate Heatmap (Day × Hour)
2. Average PnL Heatmap (Day × Hour)

## Analysis Methodology

### Data Processing
```python
# 1. Load datasets
fear_greed = pd.read_csv('fear_greed_index.csv')
trades = pd.read_csv('historical_data.csv')

# 2. Clean & merge
merged_df = trades.merge(fear_greed, on='date', how='left')

# 3. Feature engineering
merged_df['profitable'] = merged_df['Closed PnL'] > 0
merged_df['hour'] = merged_df['Timestamp IST'].dt.hour
merged_df['day_of_week'] = merged_df['Timestamp IST'].dt.day_name()
```

### Statistical Testing
- T-test between Extreme Fear vs Extreme Greed: **t = -4.32, p = 0.000016**
- Result: Statistically significant difference (not random)

## Trading Strategies

### Strategy 1: Sentiment-Based
```
IF sentiment == "Extreme Greed":
    Increase position size by 50%
ELIF sentiment == "Neutral":
    Reduce position size by 30%
```
**Expected Improvement**: +97.9% over neutral trading

### Strategy 2: Time-Based
```
OPTIMAL_HOURS = [7, 8, 10, 11, 12]  # Best performing
AVOID_HOURS = [23, 14, 21, 6, 2]    # Worst performing
OPTIMAL_DAY = "Saturday"
```

### Strategy 3: Combined Approach
```
IF day == "Saturday" AND hour == 12 AND sentiment == "Extreme Greed":
    Max position size (e.g., $11,422)
    Expected avg PnL: $131+ per trade
```

### Strategy 4: Coin Selection
- **High Volume**: @107, HYPE (consistent profits)
- **High Value**: SOL, ETH (larger avg PnL)
- **Stable**: BTC (lower volatility)

## Dataset Summary

### Historical Trader Data
- **Trades**: 211,224
- **Period**: May 1, 2023 - May 1, 2025
- **Total Volume**: $1.19 billion
- **Traders**: 32 active accounts
- **Cryptocurrencies**: 280+ symbols

**Columns**: Account, Coin, Side, Execution Price, Size USD, Closed PnL, Leverage, Timestamp IST, Event

### Fear & Greed Index
- **Records**: 2,644 daily readings
- **Scale**: 0-100 (0=Extreme Fear, 100=Extreme Greed)
- **Classifications**: Extreme Fear, Fear, Neutral, Greed, Extreme Greed

## Performance Metrics

### Overall Statistics
- **Total PnL**: $10,296,958.94
- **Average PnL per Trade**: $48.75
- **Overall Win Rate**: 41.13%
- **Profitable Trades**: 86,869 / 211,224

### Risk-Adjusted Insights
- **Best Risk/Reward**: Extreme Greed (high avg PnL, low volatility)
- **Highest Volatility**: Extreme Fear (1136.06 std dev)
- **Safest Window**: Hour 12 on Saturdays

## Code Features

### complete_analysis.ipynb
- Full 6-insight analysis
- Statistical significance testing
- 9-panel comprehensive dashboard
- Timing heatmaps
- Coin-specific analysis
- Extreme conditions comparison

### master_execution_script.ipynb
- Streamlined 5-step pipeline
- Automated report generation
- 4-panel essential dashboard
- Console-based insights
- Text report output

## Usage Examples

### Filter Trades by Sentiment
```python
extreme_greed_trades = merged_df[merged_df['market_sentiment'] == 'Extreme Greed']
print(f"Avg PnL: ${extreme_greed_trades['Closed PnL'].mean():.2f}")
```

### Find Best Trading Hour
```python
hourly_perf = merged_df.groupby('hour')['Closed PnL'].mean()
best_hour = hourly_perf.idxmax()
print(f"Best hour: {best_hour}:00")
```

### Analyze Top Trader
```python
top_trader = merged_df.groupby('Account')['Closed PnL'].sum().idxmax()
trader_stats = merged_df[merged_df['Account'] == top_trader]
print(f"Total PnL: ${trader_stats['Closed PnL'].sum():,.2f}")
print(f"Win Rate: {(trader_stats['profitable'].mean()*100):.1f}%")
```

## Actionable Recommendations

1. **Trade during Extreme Greed** - 97.9% better than neutral conditions
2. **Focus on Saturday mornings** - Especially 12:00 (peak hour)
3. **Target position size: $5,780** - Matches winner average
4. **Avoid late night trading** - 23:00 shows worst performance
5. **Monitor sentiment in real-time** - Adjust positions as conditions shift
6. **Diversify across top coins** - @107, HYPE, SOL, ETH for balance

## Limitations & Disclaimers

- **Historical Data**: Past performance ≠ future results
- **Sample Size**: 32 traders may not represent all patterns
- **Market Conditions**: 2023-2025 crypto market specific
- **Risk Warning**: Always use stop-losses and proper risk management
- **Educational Purpose**: Not financial advice

## Future Enhancements

- [ ] Real-time sentiment API integration
- [ ] Machine learning prediction models
- [ ] Multi-exchange comparison
- [ ] Leverage optimization analysis
- [ ] Risk-adjusted return metrics (Sharpe ratio)
- [ ] Drawdown analysis
- [ ] Backtesting framework

## Author
Analysis completed for Hyperliquid hiring assignment

## License

Data sourced from Hyperliquid and Fear & Greed Index. For educational purposes only.
