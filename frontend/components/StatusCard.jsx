export default function StatusCard({ title, value, helper }) {
  return (
    <div className="status-card">
      <p className="status-card__title">{title}</p>
      <p className="status-card__value">{value}</p>
      {helper ? <p className="status-card__helper">{helper}</p> : null}
    </div>
  );
}
