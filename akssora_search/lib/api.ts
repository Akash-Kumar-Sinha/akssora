import axios from "axios";
import { BACKEND_URL } from "./constant";

const api = axios.create({
  baseURL: `${BACKEND_URL}/auth`,
  withCredentials: true,
});

let isRefreshing = false;
let refreshFailCount = 0;
const MAX_REFRESH_ATTEMPTS = 3;

api.interceptors.response.use(
  (response) => {
    refreshFailCount = 0;
    return response;
  },
  async (error) => {
    const original = error.config;

    console.log(
      `[Interceptor] Status: ${error.response?.status} | URL: ${original.url} | RefreshFailCount: ${refreshFailCount}`,
    );

    if (error.response?.status === 401 && !original._retry) {
      if (refreshFailCount >= MAX_REFRESH_ATTEMPTS) {
        refreshFailCount = 0;
        isRefreshing = false;
        window.location.href = "/";
        return Promise.reject(error);
      }

      if (isRefreshing) return Promise.reject(error);

      original._retry = true;
      isRefreshing = true;

      try {
        await axios.post(
          `${BACKEND_URL}/auth/oauth/refresh`,
          {},
          { withCredentials: true },
        );
        isRefreshing = false;
        refreshFailCount = 0;
        return api(original);
      } catch (refreshError) {
        refreshFailCount++;
        isRefreshing = false;
        console.error(
          `[Interceptor] Refresh failed. Count: ${refreshFailCount}/${MAX_REFRESH_ATTEMPTS}`,
        );

        if (refreshFailCount >= MAX_REFRESH_ATTEMPTS) {
          console.warn("[Interceptor] Giving up. Redirecting to home.");
          refreshFailCount = 0;
          window.location.href = "/";
        }

        return Promise.reject(refreshError);
      }
    }

    return Promise.reject(error);
  },
);

export default api;
