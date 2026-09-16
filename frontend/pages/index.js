import { useEffect, useMemo, useState } from 'react';
import StatusCard from '../components/StatusCard';
import VideoFeed from '../components/VideoFeed';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:5000';

const defaultFeeds = [
  { label: 'Camera 0 · All Detectors', path: '/video_feed/0/all' },
  { label: 'Camera 0 · Crowd Only', path: '/video_feed/0/crowd' },
  { label: 'Camera 1 · All Detectors', path: '/video_feed/1/all' },
];

export default function Home() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [pollMs, setPollMs] = useState(4000);

  useEffect(() => {
    let active = true;

    const fetchStatus = async () => {
      try {
        const response = await fetch(`${API_BASE}/status`);
        if (!response.ok) {
          throw new Error(`Backend responded with ${response.status}`);
        }
        const payload = await response.json();
        if (active) {
          setStatus(payload);
          setError(null);
        }
      } catch (err) {
        if (active) {
          setError(err.message);
        }
      }
    };

    fetchStatus();
    const id = setInterval(fetchStatus, pollMs);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [pollMs]);

  const crowdCards = useMemo(() => {
    if (!status?.crowd) return [];
    return Object.entries(status.crowd).map(([camId, snapshot]) => ({
      title: `Camera ${camId} Crowd`,
      value: snapshot.count ?? '—',
      helper: snapshot.high_density ? '⚠ High density' : 'Normal',
    }));
  }, [status]);

  const cameraCards = useMemo(() => {
    if (!status?.cameras) return [];
    return Object.entries(status.cameras).map(([camId, cam]) => ({
      title: `Camera ${camId}`,
      value: cam.running ? 'Online' : 'Offline',
      helper: cam.name,
    }));
  }, [status]);

  const faceCard = status?.face
    ? [{
        title: 'Reference Face',
        value: status.face.has_reference ? status.face.reference_name : 'Not loaded',
        helper: status.face.has_reference ? 'Matching active' : 'Upload via Flask UI',
      }]
    : [];

  const weaponCard = status?.weapon
    ? [{
        title: 'Weapon Model',
        value: status.weapon.model_loaded ? 'Ready' : 'Loading…',
        helper: 'YOLOv8n inference',
      }]
    : [];

  const feeds = status?.cameras
    ? Object.keys(status.cameras).flatMap((camId) => [
        { label: `Camera ${camId} · All Detectors`, path: `/video_feed/${camId}/all` },
        { label: `Camera ${camId} · Crowd`, path: `/video_feed/${camId}/crowd` },
      ])
    : defaultFeeds;

  return (
    <main className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">Integrated Smart Surveillance</p>
          <h1>Cloud-hosted Operator Console</h1>
          <p className="lede">
            This dashboard fetches `/status` telemetry and MJPEG feeds from the Flask backend running on Render.
            Deploy it on Vercel and point <code>NEXT_PUBLIC_API_BASE</code> to your backend URL.
          </p>
        </div>
        <div className="hero__actions">
          <label>
            Poll Interval (ms)
            <input
              type="number"
              min="1000"
              step="500"
              value={pollMs}
              onChange={(event) => setPollMs(Number(event.target.value) || 4000)}
            />
          </label>
          <p className="hero__api">API Base: {API_BASE}</p>
        </div>
      </header>

      {error ? (
        <div className="alert alert--error">Failed to reach backend: {error}</div>
      ) : null}

      <section className="status-grid">
        {[...cameraCards, ...crowdCards, ...faceCard, ...weaponCard].map((card) => (
          <StatusCard key={`${card.title}-${card.helper}`} {...card} />
        ))}
      </section>

      <section className="video-grid">
        {feeds.map((feed) => (
          <VideoFeed key={`${feed.label}-${feed.path}`} {...feed} />
        ))}
      </section>

      <section className="notes">
        <h2>Deployment Checklist</h2>
        <ol>
          <li>Deploy Flask backend to Render using `render.yaml` and capture the public URL.</li>
          <li>
            Set <code>NEXT_PUBLIC_API_BASE</code> in Vercel to that URL (e.g., https://smart-surveillance.onrender.com).
          </li>
          <li>
            Ensure <code>CORS_ORIGINS</code> on the backend allows the Vercel domain; `*` works for private demos.
          </li>
          <li>Use the original Flask dashboard for uploading reference faces and managing credentials.</li>
        </ol>
      </section>
    </main>
  );
}
