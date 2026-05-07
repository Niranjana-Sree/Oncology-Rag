import React, { useState, useEffect } from 'react';
import { checkHealth } from '../services/api';

export default function Header({ backendStatus, setBackendStatus }) {
  useEffect(() => {
    const check = () => {
      checkHealth().then(res => setBackendStatus(res.ok ? 'online' : 'offline'));
    };
    check();
    const interval = setInterval(check, 30000);
    return () => clearInterval(interval);
  }, [setBackendStatus]);

  return (
    <div className="header">
      <div className="header-left">
        <h1 className="header-title">Med Reasoning</h1>
        <p className="header-subtitle">AI-powered Medical Question Answering</p>
      </div>
      <div className="header-status">
        <span className={`status-dot ${backendStatus === 'online' ? 'status-dot--online' : backendStatus === 'offline' ? 'status-dot--offline' : 'status-dot--checking'}`} />
        <span className="status-text">
          {backendStatus === 'online' ? 'Backend online' : backendStatus === 'offline' ? 'Backend offline' : 'Checking...'}
        </span>
      </div>
    </div>
  );
}
