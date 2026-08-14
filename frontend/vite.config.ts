import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": {
          target: env.VITE_DEV_PROXY_TARGET ?? "http://127.0.0.1:8000",
          changeOrigin: true,
          // Allow long-running streaming responses (10 min)
          timeout: 600000,
          proxyTimeout: 600000,
        },
      },
    },
  };
});
