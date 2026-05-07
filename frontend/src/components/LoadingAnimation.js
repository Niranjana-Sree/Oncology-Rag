import React, { useState, useEffect } from 'react';

const MESSAGES = [
  'Analysing your query with LAQA...',
  'Searching oncology knowledge base...',
  'Cross-encoder reranking results...',
  'Generating medical answer...',
];

export default function LoadingAnimation({ isLoading }) {
  const [msgIndex, setMsgIndex] = useState(0);

  useEffect(() => {
    if (!isLoading) { setMsgIndex(0); return; }
    const interval = setInterval(() => {
      setMsgIndex(i => (i + 1) % MESSAGES.length);
    }, 1800);
    return () => clearInterval(interval);
  }, [isLoading]);

  if (!isLoading) return null;

  return (
    <div className="loading-box">
      <div className="loading-spinner" />
      <p className="loading-msg">{MESSAGES[msgIndex]}</p>
    </div>
  );
}
