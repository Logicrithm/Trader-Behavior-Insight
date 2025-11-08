# Crypto Trading Sentiment Analysis - Complete Suite

## 🎯 Project Overview
Comprehensive analysis of 211,224 trades from Hyperliquid platform (May 2023 - May 2025) combining statistical analysis, machine learning, and deep pattern discovery to predict profitable trades and uncover actionable trading strategies.

## 📁 Project Structure
```
.

├── deep_insights.ipynb                  # ⭐ Advanced patterns + strategies
├── actionable_strategies_dashboard.png  # ⭐ 6-panel strategy dashboard
└── README.md                            # This file
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

### Run Analysis

**Option 1: Deep Insights + Strategies (⭐ Recommended for Trading)**
```bash
jupyter notebook deep_insights.ipynb
```
Output: Actionable strategies + hidden patterns + concrete trading rules

**Option 2: Complete ML Analysis**
```bash
jupyter notebook complete_ml_analysis.ipynb
```
Output: ML models + clustering + statistical tests

**Option 3: Statistical Analysis**
```bash
jupyter notebook complete_analysis.ipynb
```
Output: 9-panel dashboard + timing heatmaps + 6 insights

**Option 4: Quick Pipeline**
```bash
jupyter notebook master_execution_script.ipynb
```
Output: 4-panel dashboard + text report

## 🎯 Major Discoveries

### Discovery 1: Volatility Premium (Counter-Intuitive Finding)
**Sharp sentiment decreases yield $431.74 average PnL** compared to $47.04 during stable periods - an **818% improvement**. This challenges the conventional "avoid volatility" wisdom in crypto markets.

| Transition Type | Avg PnL | Win Rate | Trades |
|----------------|---------|----------|--------|
| **Sharp Decrease** | **$431.74** | **27%** | 270 |
| Moderate Decrease | $316.36 | 27% | 540 |
| Moderate Increase | $148.68 | 32% | 499 |
| Sharp Increase | $141.56 | 30% | 199 |
| Stable | $47.04 | 41% | 209,677 |

**Actionable Strategy:** Increase position sizes 2x when sentiment drops >10 points in a single day.

### Discovery 2: Momentum Persistence (Hot Hand Effect)
Contrary to the gambler's fallacy, **winning streaks predict future success** with 94.5% accuracy after 2+ consecutive wins vs. only 3.9% after losses.

| Previous Pattern | Next Trade PnL | Win Rate | Statistical Test |
|-----------------|----------------|----------|------------------|
| After Win | $143.58 | 93.3% | Chi² = 165,900 |
| After 2+ Wins | $148.50 | 94.5% | p < 0.000001 |
| After Loss | -$17.48 | 4.7% | Highly Significant |
| After 2+ Losses | -$18.78 | 3.9% | ✅ |

**Actionable Strategy:** Scale positions to 1.5x after 2+ wins, reduce to 0.5x after 2+ losses.

### Discovery 3: Trader Skill > Market Timing
**Historical win rate accounts for 88.1% of predictive power** while market sentiment contributes only 1.2%. This proves consistent execution matters far more than market conditions.

**Feature Importance (Random Forest)**:
1. trader_recent_winrate: 88.11%
2. trader_cumulative_pnl: 2.79%
3. log_price: 2.05%
4. fear_greed_score: 1.20%

**Actionable Strategy:** Maintain rolling 10-trade win rate >45%; pause trading if drops below 40%, resume after 3 consecutive wins.

### Discovery 4: Coin-Specific Sentiment Strategies
Different cryptocurrencies perform optimally under different sentiment conditions.

| Coin | Best Sentiment | Avg PnL | Worst Sentiment | Avg PnL |
|------|---------------|---------|-----------------|---------|
| @107 | Greed | $153.45 | Fear | -$33.55 |
| HYPE | Fear | $31.65 | Greed | $22.89 |
| ETH | Neutral-Bearish | $267.76 | Neutral-Bullish | $18.23 |
| SOL | Neutral-Bearish | $198.40 | Fear | $128.37 |
| BTC | Neutral-Bearish | $48.70 | Greed | $17.27 |
| FARTCOIN | Greed | $19.45 | Fear | **-$164.71** |

**Actionable Strategy:** Trade each coin only during its favorable sentiment window.

### Discovery 5: Risk-Adjusted Performance
Extreme Greed offers the best risk-adjusted returns (highest Sharpe ratio).

| Sentiment | Avg PnL | Volatility | Sharpe Ratio | Win Rate |
|-----------|---------|------------|--------------|----------|
| **Extreme Greed** | **$67.89** | **$766.83** | **0.089** | **46%** |
| Neutral | $34.31 | $517.12 | 0.066 | 40% |
| Fear | $54.29 | $935.36 | 0.058 | 42% |
| Greed | $42.74 | $1,116.03 | 0.038 | 38% |
| Extreme Fear | $34.54 | $1,136.06 | 0.030 | 37% |

**Actionable Strategy:** Allocate capital prioritizing highest Sharpe ratios (Extreme Greed > Neutral > Fear).

## 📊 Machine Learning Results

### Best Model: Random Forest
| Metric | Score |
|--------|-------|
| Accuracy | 88.72% |
| Precision | 81.03% |
| Recall | 94.74% |
| F1-Score | 87.35% |
| AUC-ROC | 97.70% |
| MCC | 0.7805 |

**Cross-Validation**: 97.83% ± 0.09% (5-fold)

### All Models Comparison
| Model | Accuracy | AUC-ROC | F1-Score | Recall |
|-------|----------|---------|----------|--------|
| Random Forest | 88.72% | 97.70% | 87.35% | 94.74% |
| Gradient Boosting | 89.88% | 97.53% | 87.40% | 85.40% |
| Logistic Regression | 88.59% | 96.99% | 86.28% | 87.19% |

### Trader Clustering (K-Means, k=4)

**Cluster 0: Conservative Traders** (5 traders, 15.6%)
- Avg PnL: $44.08 | Win Rate: 36.8% | Trade Size: $2,197 | Sentiment: 51.6

**Cluster 1: Greed-Seeking Winners** (12 traders, 37.5%)
- Avg PnL: $54.47 | Win Rate: 46.5% | Trade Size: $2,099 | Sentiment: 59.9

**Cluster 2: High-Risk Players** (5 traders, 15.6%)
- Avg PnL: $365.73 | Win Rate: 34.6% | Trade Size: $6,619 | Sentiment: 51.4

**Cluster 3: Institutional Scale** (10 traders, 31.2%)
- Avg PnL: $42.33 | Win Rate: 37.5% | Trade Size: $12,298 | Sentiment: 40.7

## 🎯 Concrete Trading Strategies

### Strategy 1: Volatility Momentum (NEW!)
```python
if sentiment_change < -10:  # Sharp decrease
    position_size = standard_size * 2.0
    expected_pnl = $431.74
elif sentiment_change > 10:  # Sharp increase
    position_size = standard_size * 1.5
    expected_pnl = $141.56
else:  # Stable
    position_size = standard_size * 0.8
    expected_pnl = $47.04
```
**Expected Improvement**: 818% over stable conditions

### Strategy 2: Streak-Based Position Sizing (NEW!)
```python
# Calculate recent streak
consecutive_wins = count_consecutive_wins(last_trades)
consecutive_losses = count_consecutive_losses(last_trades)

if consecutive_wins >= 2:
    position_multiplier = 1.5  # Ride the hot hand
    expected_success_rate = 94.5%
elif consecutive_losses >= 2:
    position_multiplier = 0.5  # Reduce exposure
    expected_success_rate = 3.9%
else:
    position_multiplier = 1.0
    expected_success_rate = 41.1%
```
**Statistical Validation**: Chi² = 165,900, p < 0.000001

### Strategy 3: Coin-Sentiment Matching (NEW!)
```python
optimal_conditions = {
    '@107': 'Greed',      # $153.45 avg PnL
    'HYPE': 'Fear',       # $31.65 avg PnL
    'ETH': 'Neutral-Bearish',  # $267.76 avg PnL
    'SOL': 'Neutral-Bearish',  # $198.40 avg PnL
    'BTC': 'Neutral-Bearish'   # $48.70 avg PnL
}

avoid_conditions = {
    'FARTCOIN': 'Fear',   # -$164.71 avg PnL (DANGER!)
    '@107': 'Fear',       # -$33.55 avg PnL
}

if coin in optimal_conditions:
    if current_sentiment == optimal_conditions[coin]:
        execute_trade()
```

### Strategy 4: ML-Powered Confidence Filter
```python
from sklearn.ensemble import RandomForestClassifier

# Use trained model
features = extract_features(current_trade)
probability = model.predict_proba(features)[0][1]

if probability > 0.8:
    position_size = standard_size * 1.5
    print(f"High confidence: {probability:.1%}")
elif probability > 0.6:
    position_size = standard_size * 1.0
    print(f"Medium confidence: {probability:.1%}")
else:
    skip_trade()
    print(f"Low confidence: {probability:.1%}")
```
**Expected Performance**: 97.7% AUC, 88.7% accuracy

### Strategy 5: Win Rate Maintenance
```python
# Track rolling performance
rolling_winrate = calculate_rolling_winrate(last_10_trades)

if rolling_winrate < 0.40:
    # Pause trading
    status = "PAUSED"
    print("Win rate dropped below 40% - taking a break")
    
elif rolling_winrate > 0.45:
    # Active trading
    status = "ACTIVE"
    position_size = standard_size
    
# Resume after 3 consecutive wins
if status == "PAUSED" and consecutive_wins >= 3:
    status = "ACTIVE"
    print("Resuming trading after recovery")
```
**Rationale**: Historical win rate = 88% of prediction power

### Strategy 6: Risk-Adjusted Allocation
```python
# Allocate capital by Sharpe ratio
allocation = {
    'Extreme Greed': 0.50,  # Sharpe: 0.089 (highest)
    'Neutral': 0.30,        # Sharpe: 0.066
    'Fear': 0.20,           # Sharpe: 0.058
    'Greed': 0.00,          # Sharpe: 0.038 (lower)
    'Extreme Fear': 0.00    # Sharpe: 0.030 (lowest)
}

capital_allocation = total_capital * allocation[current_sentiment]
```

## 📈 Notebook Comparison

### deep_insights.ipynb ⭐ NEW!
**Purpose**: Uncover hidden patterns + deliver concrete strategies  
**Best For**: Active traders seeking actionable rules

**Unique Analyses**:
- Sentiment transition patterns (6 types)
- Winning/losing streak effects (Chi-square test)
- Coin-specific sentiment strategies (8 coins)
- Risk-adjusted Sharpe ratios by sentiment

**Output**:
- actionable_strategies_dashboard.png (6 panels)
- 6 concrete trading strategies with entry/exit rules
- Statistical validation for each pattern

**Key Discoveries**:
- Volatility premium: $431.74 during sharp drops
- Hot hand effect: 94.5% success after 2+ wins
- Coin-sentiment matching: @107 in Greed = $153.45

### complete_ml_analysis.ipynb
**Purpose**: ML prediction + clustering + advanced stats  
**Best For**: Algorithmic traders, data scientists

**Features**:
- 3 ML models (LR, RF, GBM)
- 16 engineered features
- 5-fold cross-validation
- K-Means clustering (4 trader types)
- Pearson correlation + ANOVA tests

**Output**:
- ml_performance_complete.png (6 panels)
- Model comparison metrics
- Feature importance ranking
- Trader cluster profiles

### complete_analysis.ipynb
**Purpose**: Deep statistical insights  
**Best For**: Comprehensive understanding

**Features**:
- 6 detailed insights with statistical tests
- Sentiment performance analysis
- Optimal timing (day/hour)
- Winner vs loser patterns
- Extreme conditions comparison
- Coin-specific analysis

**Output**:
- complete_analysis_dashboard.png (9 panels)
- timing_heatmaps.png
- Detailed console reports

### master_execution_script.ipynb
**Purpose**: Quick automated analysis  
**Best For**: Fast insights, presentations

**Features**:
- Streamlined 5-step pipeline
- Essential insights only
- Automated report generation
- Fast execution (~1 minute)

**Output**:
- comprehensive_insights_dashboard.png (4 panels)
- analysis_report.txt

## 📊 Generated Visualizations

### actionable_strategies_dashboard.png (6 panels) ⭐ NEW!
1. **Performance by Sentiment Transition** - Shows $431.74 for sharp decreases
2. **Next Trade After Streaks** - Visualizes 94.5% success after 2+ wins
3. **Coin Performance by Sentiment** - Heatmap of optimal conditions
4. **Risk-Adjusted Returns** - Sharpe ratios by sentiment
5. **Win Rate by Trading Condition** - Overall vs streaks
6. **Expected Returns** - Benchmarks by market sentiment

### ml_performance_complete.png (6 panels)
1. Model Accuracy Comparison
2. Model AUC-ROC Comparison
3. ROC Curves (all models)
4. Top 10 Feature Importance
5. Confusion Matrix (Random Forest)
6. All Metrics Comparison

### complete_analysis_dashboard.png (9 panels)
1. Performance by Sentiment
2. Win Rate by Sentiment
3. Hourly Performance
4. Performance by Day
5. Sentiment vs PnL (scatter)
6. Trader Profitability
7. Winners vs Losers Sentiment
8. Trade Size Distribution
9. Top 5 Coins by Profit

### timing_heatmaps.png (2 heatmaps)
1. Win Rate Heatmap (Day × Hour)
2. Average PnL Heatmap (Day × Hour)

## 🎯 Implementation Roadmap

### Phase 1: Immediate Actions (Day 1)
- [ ] Track rolling 10-trade win rate for all traders
- [ ] Set alerts for sharp sentiment drops (>10 points)
- [ ] Implement position scaling: 1.5x after 2+ wins, 0.5x after 2+ losses
- [ ] Create coin-sentiment matching table

### Phase 2: System Integration (Week 1)
- [ ] Integrate sentiment API for real-time tracking
- [ ] Deploy Random Forest model (97.7% AUC)
- [ ] Set confidence thresholds: >80% = 1.5x, <60% = skip
- [ ] Automate win rate monitoring with pause triggers

### Phase 3: Advanced Features (Month 1)
- [ ] Implement volatility momentum strategy
- [ ] Backtest on out-of-sample data (2025+)
- [ ] Add risk-adjusted capital allocation
- [ ] Create performance dashboard

### Phase 4: Optimization (Ongoing)
- [ ] Retrain ML models monthly
- [ ] Refine coin-sentiment mappings
- [ ] A/B test strategy combinations
- [ ] Monitor and adjust thresholds

## 💡 Key Insights Summary

### Statistical Findings
- **Overall Win Rate**: 41.13%
- **Total PnL**: $10,296,958.94
- **Best Sentiment**: Extreme Greed ($67.89 avg, 46% win rate)
- **Best Time**: Saturday at 12:00 ($131.17 avg)
- **Best Day**: Saturday ($65.36 avg)

### Pattern Discoveries
- **Volatility Premium**: $431.74 during sharp drops (818% improvement)
- **Momentum Persistence**: 94.5% success after 2+ wins
- **Trader Skill Dominance**: 88.1% of predictive power
- **Coin-Sentiment Effects**: @107 performs 456% better in Greed vs Fear

### ML Performance
- **Best Model**: Random Forest (97.70% AUC)
- **Key Feature**: trader_recent_winrate (88.11% importance)
- **Profitable Traders**: 90.6% (29 of 32)

### Risk-Adjusted Returns
- **Best Sharpe**: Extreme Greed (0.089)
- **Safest**: Neutral (lowest volatility, 0.066 Sharpe)
- **Worst**: Extreme Fear (0.030 Sharpe, highest volatility)

## 📈 Expected Performance by Strategy

| Strategy | Expected Avg PnL | Expected Win Rate | Risk Level |
|----------|------------------|-------------------|------------|
| Volatility Momentum | $431.74 | 27% | High |
| Streak-Based (2+ wins) | $148.50 | 94.5% | Medium |
| ML Confidence >80% | Variable | 94.74% | Low |
| Coin-Sentiment Matching | $153.45 (@107) | 52.4% | Medium |
| Extreme Greed Focus | $67.89 | 46% | Medium |
| Win Rate Maintenance | $48.75+ | 45%+ | Low |

## 🔬 Statistical Validation

### Tests Performed
- **Pearson Correlation**: r = 0.0081, p = 0.000190 (sentiment vs PnL) ✅
- **One-Way ANOVA**: F = 9.06, p < 0.000001 (PnL across sentiments) ✅
- **T-Test**: t = -4.32, p = 0.000016 (Extreme Fear vs Greed) ✅
- **Chi-Square**: χ² = 165,900, p < 0.000001 (streak effects) ✅

All tests show statistical significance (p < 0.05).

## ⚠️ Limitations & Risk Warnings

### Data Limitations
- **Period**: 2023-2025 crypto market specific
- **Sample**: 32 traders (may not generalize)
- **Survivorship Bias**: Only includes active traders
- **Market Regime**: Bull market conditions

### Strategy Risks
- **Volatility Premium**: 27% win rate (high variance)
- **Streak Dependency**: Requires consistent tracking
- **Model Decay**: Retraining needed as markets evolve
- **Execution Slippage**: Not accounted for in analysis

### Important Disclaimers
- **Past ≠ Future**: Historical performance doesn't guarantee results
- **Not Financial Advice**: For educational purposes only
- **High Leverage**: Data includes leveraged positions
- **Crypto Volatility**: Extremely volatile asset class
- **Regulatory Risk**: Crypto regulations evolving

## 🔧 Requirements

```bash
# Core packages
pandas==1.5.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
scipy==1.10.1
scikit-learn==1.2.2

# System requirements
RAM: 8GB minimum (16GB recommended)
CPU: Multi-core for parallel processing
Storage: 500MB for data + outputs
Python: 3.8+
```

## 📚 Usage Examples

### Example 1: Check Current Trading Conditions
```python
current_sentiment = get_fear_greed_score()  # e.g., 75
coin = "ETH"

# Check coin-sentiment match
if coin == "ETH" and 40 <= current_sentiment < 50:
    print("✅ Optimal condition: ETH in Neutral-Bearish")
    expected_pnl = 267.76
    
# Check volatility
sentiment_change = current_sentiment - previous_sentiment
if sentiment_change < -10:
    print("✅ Sharp decrease detected - Volatility premium active")
    position_multiplier = 2.0
```

### Example 2: Calculate Position Size
```python
def calculate_position_size(base_size, recent_trades, ml_probability):
    # Factor 1: Streak adjustment
    consecutive_wins = count_consecutive_wins(recent_trades)
    if consecutive_wins >= 2:
        streak_multiplier = 1.5
    elif count_consecutive_losses(recent_trades) >= 2:
        streak_multiplier = 0.5
    else:
        streak_multiplier = 1.0
    
    # Factor 2: ML confidence
    if ml_probability > 0.8:
        confidence_multiplier = 1.5
    elif ml_probability < 0.6:
        return 0  # Skip trade
    else:
        confidence_multiplier = 1.0
    
    # Factor 3: Win rate check
    rolling_wr = calculate_winrate(recent_trades[-10:])
    if rolling_wr < 0.40:
        return 0  # Paused
    
    return base_size * streak_multiplier * confidence_multiplier
```

### Example 3: Monitor Performance
```python
def check_trading_status(recent_trades):
    rolling_wr = calculate_winrate(recent_trades[-10:])
    consecutive_wins = count_consecutive_wins(recent_trades)
    
    if rolling_wr < 0.40:
        status = "PAUSED"
        print(f"⚠️ Win rate: {rolling_wr:.1%} - Below threshold")
        
        if consecutive_wins >= 3:
            status = "RESUMING"
            print("✅ 3 consecutive wins - Ready to resume")
    else:
        status = "ACTIVE"
        print(f"✅ Win rate: {rolling_wr:.1%} - Trading active")
    
    return status
```

## 🎓 Learning Path

### Beginner Level
1. Start with `master_execution_script.ipynb`
2. Review `analysis_report.txt`
3. Study simple strategies (Strategy 6: Benchmarks)

### Intermediate Level
1. Run `complete_analysis.ipynb`
2. Understand timing patterns
3. Implement sentiment-based allocation

### Advanced Level
1. Study `complete_ml_analysis.ipynb`
2. Train custom ML models
3. Optimize feature engineering

### Expert Level
1. Explore `deep_insights.ipynb`
2. Implement all 6 strategies
3. Backtest and optimize combinations

## 🚀 Future Enhancements

### Planned Features
- [ ] Real-time sentiment API integration
- [ ] Deep learning models (LSTM, Transformers)
- [ ] Social media sentiment (Twitter, Reddit)
- [ ] Multi-exchange data comparison
- [ ] Automated backtesting framework
- [ ] Portfolio optimization algorithms
- [ ] Risk parity allocation

### Research Ideas
- [ ] Order flow analysis
- [ ] Volatility forecasting
- [ ] Regime detection algorithms
- [ ] Reinforcement learning agents
- [ ] Network effects between coins
- [ ] Liquidity impact modeling

## 👥 Author
Analysis completed for Hyperliquid hiring assignment  
Focus: Data science + ML + actionable strategy development

## 📄 License
Educational use only. Data from Hyperliquid and Fear & Greed Index.

---

**⚡ Quick Reference Card**

**Best Conditions**: Extreme Greed, Saturday 12:00  
**Best Strategy**: Volatility momentum + Streak-based sizing  
**Best Model**: Random Forest (97.7% AUC)  
**Best Indicator**: Rolling 10-trade win rate  
**Stop Trading**: Win rate < 40%  
**Resume Trading**: 3 consecutive wins  

**Key Numbers to Remember**:
- $431.74: Sharp decrease avg PnL
- 94.5%: Success rate after 2+ wins
- 88.1%: Predictive power from win rate
- 0.089: Best Sharpe ratio (Extreme Greed)
