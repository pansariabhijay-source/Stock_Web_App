# AlphaStock Debugging and Refactoring Plan

This plan addresses the runtime errors in the backend and refactors the frontend to eliminate hardcoded "mock" data (e.g. random numbers) to ensure it uses real prediction and historical data from the API.

## User Review Required

> [!IMPORTANT]
> This plan will introduce two new API endpoints (`/api/prices` and `/api/history/{ticker}`) to supply the frontend with live data and historical charts. It will also fix the `ValueError` by filling missing dataframe columns with `0.0` ensuring the shape aligns with the trained models. Please confirm this alignment approach is correct for your system.

## Proposed Changes

### Backend Machine Learning & Environment

#### [MODIFY] [requirements analysis / environment]
- **Environment**: The `hmmlearn` module is listed in `requirements.txt` but appears missing from the current Python environment (causing `ModuleNotFoundError: No module named 'hmmlearn'`). I will actively run `pip install hmmlearn` to ensure it is available.

#### [MODIFY] [api/model_registry.py](file:///c:/Users/ACER/Desktop/alpha_stock/api/model_registry.py)
- **Feature Alignment Fix (`predict` and `_get_regression_prediction`)**: 
  Currently, the code slices `feature_df` using `[f for f in feature_names if f in feature_df.columns]`. If some features are missing in the runtime inference DataFrame, the output array will have fewer columns than `feature_names`, resulting in a `ValueError` during the `predict()` call of models like LightGBM.
  We will reconstruct an exact DataFrame matching the expected `feature_names` array size, placing `0.0` (or `np.nan` if appropriate) for any missing features before mapping to the `X_latest` float32 numpy array.
- **New Methods for Frontend**:
  - `get_all_prices()`: return `{ticker: {"price": ..., "pct_change": ...}}`.
  - `get_history(ticker, days=30)`: return the last `N` close prices & dates for the stock.

---

### Backend API Endpoints

#### [MODIFY] [api/routes.py](file:///c:/Users/ACER/Desktop/alpha_stock/api/routes.py)
- Add a new endpoint `GET /api/prices` that calls `registry.get_all_prices()`.
- Add a new endpoint `GET /api/history/{ticker}` that calls `registry.get_history(ticker)`.

#### [MODIFY] [api/schemas.py](file:///c:/Users/ACER/Desktop/alpha_stock/api/schemas.py)
- Define new Pydantic Models for the response schemas of `PricesResponse` and `HistoryResponse` to maintain the robust FastAPI schema validation.

---

### Frontend API Layer & Components

#### [MODIFY] [frontend/src/api.js](file:///c:/Users/ACER/Desktop/alpha_stock/frontend/src/api.js)
- Wire up generic handlers `api.getPrices()` and `api.getHistory(ticker)`.

#### [MODIFY] [frontend/src/components/LiveTickerTape.jsx](file:///c:/Users/ACER/Desktop/alpha_stock/frontend/src/components/LiveTickerTape.jsx)
- **Remove Hardcoding**: Eliminate the `Math.random()` price logic.
- Switch the `fetchTicker` internal implementation to await `api.getPrices()`.

#### [MODIFY] [frontend/src/pages/Dashboard.jsx](file:///c:/Users/ACER/Desktop/alpha_stock/frontend/src/pages/Dashboard.jsx)
- **Remove Hardcoding**: Update `generateTrendChartPath` function which currently generates random walk data using `Math.random()`.
- Re-implement `generateTrendChartPath` (or replace it) by fetching the actual 30-day historical chart data via `api.getHistory(niftyPred.ticker)`. 
- Construct the combined chart line array linking the real historical prices matching to the new prediction target. 

## Open Questions

- Is there a preferred default value for missing features during prediction? I will use `0.0`, but another common fallback is `np.nan` or forward-filling from previous days if applicable.
- For the `LiveTickerTape` scrolling, is hitting the backend every 5 seconds for a refreshed price array acceptable, or should we just fetch it once at startup? 

## Verification Plan

### Automated Tests
- N/A - but we can verify the backend `/api/prices` using cURL or a browser query to ensure it outputs JSON successfully without errors.

### Manual Verification
- View `predict_error.log` while querying `/api/predict` via `api.js` wrapper to verify `hmmlearn` parses and `predict` handles exact feature dimensionality.
- Run `npm run dev` and open the app locally to confirm that elements like the `LiveTickerTape` and the `TrendChart` on `Dashboard.jsx` populate smoothly without math random glitches.
