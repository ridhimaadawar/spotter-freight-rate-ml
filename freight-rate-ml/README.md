# Freight Rate Prediction (Spotter ML Engineer assessment)

Predicts `posted_rate` for 12,000 future loads (Nov-Dec 2025) from 48,000 labeled loads (Jan-Oct 2025).

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py          # cleaning, backtest, final fit -> validation_predictions.csv
python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs_filled.csv
```
Outputs: `validation_predictions.csv`, `data/december_chart_inputs_filled.csv`, `scorer_results/candidate_december.png`,
`outputs/backtest_results.csv`, `outputs/data_quality.json`. Runtime is about 3 minutes on one CPU core.

## Approach (short)
- **Validation:** the task is a forecast (every validation load is dated after the last training load), so I use
  rolling-origin time splits (train on months <= k, test on the next two months), never a random split.
- **Data quality:** sign-flipped weights (`abs`), missing weight/market_index (median imputation), ~1.4% corrupted labels
  (rates multiplied/divided by roughly 2-6x; removed from training only), and 8 cities in validation never seen in training
  (handled with coordinate features instead of city IDs).
- **Model:** LightGBM (L1 loss) on `log($/mile)` using distance, haversine distance, equipment, weight, coordinates, weekday.
  `market_index` and `quote_signal` are intentionally NOT used: they lowered out-of-time accuracy.
- **Backtest MAE (clean loads):** about 47 USD vs 94 USD for a lane+equipment per-mile baseline.
Layout: `src/common.py` (cleaning + features), `src/train.py` (full pipeline), `src/experiments.py`, `src/exp2.py`, `src/exp3.py` (exploration).

## Data
Raw assessment CSVs are not committed. Place `train_test.csv`, `validation.csv`, `validation_predictions_template.csv`
and `december_chart_inputs.csv` in `data/` (underscored names) before running.
