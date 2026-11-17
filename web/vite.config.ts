import { defineConfig } from "vite";

// MapLibre alone is ~800 kB minified; that is expected for a map app. Its worker is an ES module.
export default defineConfig({
  build: { chunkSizeWarningLimit: 1200 },
  worker: { format: "es" },
});
