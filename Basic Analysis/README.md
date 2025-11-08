# Crypto Trading Sentiment Analysis

## Overview
Analysis of trader performance on Hyperliquid platform correlating with Bitcoin market sentiment (Fear & Greed Index). This project identifies patterns between market psychology and trading outcomes to derive actionable trading strategies.

## Dataset
- **Historical Trader Data**: 211,224 trades from Hyperliquid
  - Time period: May 2023 - May 2025
  - Features: Account, Symbol, Execution Price, Size, Side, Closed PnL, Leverage
- **Fear & Greed Index**: 2,644 daily sentiment records
  - Classifications: Extreme Fear, Fear, Neutral, Greed, Extreme Greed

## Key Findings

### 1. Sentiment Performance
- **Best Performance**: Extreme Greed ($67.89 avg PnL)
- **Worst Performance**: Neutral ($34.31 avg PnL)
- **Performance Spread**: 97.9% difference
- **Statistical Significance**: p-value = 0.000016 (highly significant)

### 2. Optimal Trading Times
- **Best Hour**: 12:00 ($131.17 avg PnL)
- **Best Day**: Saturday ($65.36 avg PnL)
- **Recommendation**: Trade on Saturdays at 12:00

### 3. Winner vs Loser Characteristics
| Metric | Winners (41.1%) | Losers (58.9%) |
|--------|-----------------|----------------|
| Avg Sentiment Score | 52.4 | 51.2 |
| Avg Trade Size | $5,779.71 | $5,541.47 |
| Total PnL | $13.2M | -$2.9M |

**Insight**: Winners trade during slightly more greedy markets (+1.2 sentiment points)

### 4. Trader Success
- **Profitable Traders**: 90.6% of active traders
- **Top Trader PnL**: $2,143,382.60
- **Top Performers Average**:
  - Win Rate: 41.3%
  - Trade Size: $11,422.24
  - Preferred Sentiment: 46.1

### 5. Top Performing Coins
| Coin | Avg PnL | Total PnL | Win Rate |
|------|---------|-----------|----------|
| @107 | $92.82 | $2.78M | 47% |
| HYPE | $28.65 | $1.95M | 41% |
| SOL | $153.36 | $1.64M | 39% |
| ETH | $118.30 | $1.32M | 36% |
| BTC | $33.30 | $868K | 35% |

## Analysis Methodology

### Data Processing
1. Load and merge datasets by date
2. Clean missing values and outliers
3. Create temporal features (hour, day of week)
4. Calculate profitability metrics

### Statistical Analysis
- T-tests for sentiment comparison
- Win rate calculations by multiple dimensions
- Volatility analysis across conditions
- Correlation studies between variables

### Visualizations Generated
1. **complete_analysis_dashboard.png**: 9-panel comprehensive dashboard
   - Performance by sentiment
   - Win rates
   - Hourly/daily patterns
   - Trader profitability
   - Coin performance

2. **timing_heatmaps.png**: Time-based performance matrices
   - Win rate heatmap (Day × Hour)
   - Avg PnL heatmap (Day × Hour)

## How to Run

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scipy
```

### Execution
```bash
python complete_analysis.py
```

Or run the Jupyter notebook:
```bash
jupyter notebook complete_analysis.ipynb
```

### Expected Output
- Console report with all insights
- `complete_analysis_dashboard.png`
- `timing_heatmaps.png`
- `analysis_report.txt`

## Trading Strategy Recommendations

### 1. Sentiment-Based Strategy
- **Primary Focus**: Trade during Extreme Greed conditions
- **Avoid**: Neutral market conditions
- **Expected Improvement**: ~98% better performance

### 2. Timing Strategy
- **Optimal Window**: Saturdays at 12:00
- **Secondary Windows**: Weekday mornings (7:00-12:00)
- **Avoid**: Late night hours (23:00, worst hour at $18.75 avg)

### 3. Position Sizing
- Increase positions during Extreme Greed (>75 sentiment score)
- Reduce exposure during Extreme Fear (<25 sentiment score)
- Top performers use avg trade size of $11,422

### 4. Coin Selection
- Focus on @107, HYPE, SOL for volume
- Consider ETH, BTC for stability
- Monitor high win-rate coins: OGN (100%), ZETA (93%)

## Project Structure
```
.
├── complete_analysis.ipynb          # Main analysis notebook
├── fear_greed_index.csv             # Sentiment data
├── historical_data.csv              # Trading data
├── complete_analysis_dashboard.png  # Main visualizations
├── timing_heatmaps.png             # Time-based heatmaps
└── README.md                        # This file
```

## Key Insights Summary

**Overall Performance**:
- Total PnL: $10,296,958.94
- Overall Win Rate: 41.1%
- Active Traders: 32 (90.6% profitable)

**Critical Finding**: The difference between trading in optimal vs suboptimal conditions can result in nearly 2x better performance. Market sentiment is a statistically significant predictor of trade outcomes.

## Future Work
- Real-time sentiment integration
- Machine learning prediction models
- Risk management optimization
- Multi-exchange comparison
- Leverage strategy analysis

## Author
Analysis completed for Hyperliquid hiring assignment

## License
Data sourced from Hyperliquid and Fear & Greed Index. Analysis for educational purposes.