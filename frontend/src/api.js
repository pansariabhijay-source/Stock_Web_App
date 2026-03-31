import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

export const api = {
  // Get health status
  checkHealth: async () => {
    const res = await axios.get(`${API_BASE}/health`);
    return res.data;
  },

  // Get current market regime
  getRegime: async () => {
    const res = await axios.get(`${API_BASE}/regime`);
    return res.data;
  },

  // Get available models and stocks
  getModels: async () => {
    const res = await axios.get(`${API_BASE}/models`);
    return res.data; // { total_stocks, stocks: [...] }
  },

  // Get prediction for a single stock and horizon
  getPrediction: async (ticker, horizon = '1d', model = 'lightgbm_clf') => {
    const res = await axios.post(`${API_BASE}/predict`, {
      ticker,
      horizon,
      model
    });
    return res.data;
  },

  // Get SHAP explanation
  explainPrediction: async (ticker, horizon = '1d') => {
    const res = await axios.post(`${API_BASE}/explain`, {
      ticker,
      horizon,
      top_n: 10
    });
    return res.data;
  }
};
