# Surveillance Dashboard Frontend

This Next.js project mirrors the Flask dashboard summary so you can deploy a lightweight operator UI on Vercel while the heavy OpenCV/YOLO workloads stay on Render.

## Commands

```bash
npm install       # once
npm run dev       # local preview at http://localhost:3000
npm run build     # production bundle
npm run start     # serve the production build
```

Set `NEXT_PUBLIC_API_BASE` to the Render backend URL (defaults to `http://localhost:5000` for local testing). The page polls `/status` every few seconds and renders MJPEG feeds directly via `<img>` tags, so no extra proxying is required.
