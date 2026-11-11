import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/villages': 'http://127.0.0.1:8000',
      '/risk-grid': 'http://127.0.0.1:8000',
      '/village-risk': 'http://127.0.0.1:8000',
      '/model-info': 'http://127.0.0.1:8000',
      '/tracks-sample': 'http://127.0.0.1:8000',
      '/feature-importance-chart': 'http://127.0.0.1:8000'
    }
  }
});

