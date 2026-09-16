export default function VideoFeed({ label, path }) {
  const apiBase = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:5000';
  const url = `${apiBase}${path}`;
  return (
    <div className="video-feed">
      <div className="video-feed__header">
        <h3>{label}</h3>
        <span className="video-feed__url">{url}</span>
      </div>
      <div className="video-feed__body">
        <img src={url} alt={label} />
      </div>
    </div>
  );
}
