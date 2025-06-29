/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}',  // Ensures Tailwind scans all JSX/TSX files
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};


