import React from 'react';
import { QUERY_TYPE_COLORS, QUERY_TYPE_LABELS } from '../constants/examples';

export default function MessageBubble({ question, queryType, language }) {
  const color = QUERY_TYPE_COLORS[queryType] || '#2563eb';
  const label = QUERY_TYPE_LABELS[queryType] || queryType;

  return (
    <div className="user-message-wrap">
      <div className="user-message-badges">
        <span className="msg-badge" style={{ background: color + '22', color }}>
          {label}
        </span>
        {language && language !== 'en' && (
          <span className="msg-badge" style={{ background: '#f3f4f6', color: '#6b7280' }}>
            {language.toUpperCase()}
          </span>
        )}
      </div>
      <div className="user-bubble">{question}</div>
    </div>
  );
}
