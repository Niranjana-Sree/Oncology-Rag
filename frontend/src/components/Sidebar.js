import React from 'react';
import { QUERY_TYPE_COLORS, QUERY_TYPE_LABELS } from '../constants/examples';

function timeAgo(ts) {
  const diff = Math.floor((Date.now() - ts) / 1000);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

export default function Sidebar({ history, activeId, onNewChat, onHistoryClick, backendStatus }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-logo-row">
          <div className="sidebar-logo-circle">O</div>
          <span className="sidebar-logo-text">OncologyRAG</span>
        </div>
        <div className="sidebar-status">
          <span className={`status-pulse${backendStatus === 'offline' ? ' status-pulse--offline' : backendStatus === 'checking' ? ' status-pulse--checking' : ''}`} />
          <span>{backendStatus === 'online' ? 'Online' : backendStatus === 'offline' ? 'Offline' : 'Checking...'}</span>
        </div>
        <button className="new-chat-btn" onClick={onNewChat}>+ New Chat</button>
      </div>

      <div className="sidebar-divider" />

      <div className="sidebar-history">
        {history.length === 0
          ? <p className="sidebar-history-empty">No conversations yet</p>
          : history.map(item => {
              const color = QUERY_TYPE_COLORS[item.queryType] || '#2563eb';
              const label = QUERY_TYPE_LABELS[item.queryType] || item.queryType;
              return (
                <button
                  key={item.id}
                  className={`history-item${item.id === activeId ? ' history-item--active' : ''}`}
                  onClick={() => onHistoryClick(item)}
                >
                  <div className="history-item-top">
                    <span className="history-type-badge" style={{ background: color + '22', color }}>
                      {label}
                    </span>
                  </div>
                  <span className="history-item-q">{item.question.slice(0, 50)}{item.question.length > 50 ? '…' : ''}</span>
                  <span className="history-item-time">{timeAgo(item.timestamp)}</span>
                </button>
              );
            })
        }
      </div>

      <div className="sidebar-footer">OncologyRAG v1.0</div>
    </aside>
  );
}
