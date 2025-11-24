import type { NextConfig } from "next";

const isProd = process.env.NODE_ENV === "production";
const basePath = isProd ? "/textbook" : "";

const nextConfig: NextConfig = {
  output: "export",

  basePath,

  assetPrefix: isProd ? "/textbook/" : undefined,

  images: {
    unoptimized: true,
  },

  env: {
    NEXT_PUBLIC_BASE_PATH: basePath,
  },
};

export default nextConfig;