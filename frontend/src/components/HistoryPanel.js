import React from 'react';
import { QUERY_TYPE_LABELS, QUERY_TYPE_COLORS } from '../constants/examples';

export default function HistoryPanel({ history, onHistoryClick }) {
  if (!history || history.length === 0) return null;

  return (
    <div className="history-panel">
      <h4 className="history-title">Recent queries</h4>
      <div className="history-list">
        {history.slice(0, 5).map((item, i) => {
          const color = QUERY_TYPE_COLORS[item.type] || '#2563eb';
          const label = QUERY_TYPE_LABELS[item.type] || item.type;
          const time = new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
          return (
            <button key={i} className="history-item" onClick={() => onHistoryClick(item)}>
              <span className="history-badge" style={{ background: color + '18', color, border: `1px solid ${color}44` }}>
                {label}
              </span>
              <span className="history-q">{item.question.slice(0, 60)}{item.question.length > 60 ? '…' : ''}</span>
              <span className="history-time">{time}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
