import React from 'react';

export default function TopBar({ title, onToggleSidebar }) {
  return (
    <div className="topbar">
      <button className="topbar-hamburger" onClick={onToggleSidebar} aria-label="Toggle sidebar">
        ☰
      </button>
      <span className="topbar-title">{title}</span>
    </div>
  );
}
