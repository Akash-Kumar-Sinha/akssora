import axios from 'axios'
import { AUTH_URL } from './constant'

const api = axios.create({
  baseURL: AUTH_URL,
  withCredentials: true,
})

let isRefreshing = false

api.interceptors.response.use(
  (response) => response, 

  async (error) => {
    const original = error.config

    if (error.response?.status === 401 && !original._retry) {
      
      if (isRefreshing) return Promise.reject(error) 
      
      original._retry = true
      isRefreshing = true

      try {
        await api.post('/oauth/refresh')
        console.log("Token refreshed successfully")
        isRefreshing = false

        return api(original)

      } catch (refreshError) {
        isRefreshing = false
        window.location.href = '/'
        return Promise.reject(refreshError)
      }
    }

    return Promise.reject(error)
  }
)

export default api
