import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: /api/* is proxied to FastAPI, so no CORS setup is needed locally.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
