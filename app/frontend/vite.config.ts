import react from '@vitejs/plugin-react'
import path from 'path'
import { defineConfig } from 'vite'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      '^/(auth|hedge-fund|language-models|flows|flow-runs|api-keys|storage|backtest|ollama|health)': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
