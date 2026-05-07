import React from 'react';

const PILLS = [
  { label: 'Side effects of cisplatin',    qt: 'qa',         question: 'What are the main side effects of cisplatin and how are they managed?' },
  { label: 'How does pembrolizumab work',  qt: 'lfqa',       question: 'Explain in detail how pembrolizumab works as an immune checkpoint inhibitor.' },
  { label: 'HNSCC first-line treatment',   qt: 'mcqa',       question: 'What is the standard first-line treatment for recurrent or metastatic HNSCC?\nA. Carboplatin + paclitaxel\nB. Pembrolizumab monotherapy or with chemotherapy\nC. Cetuximab monotherapy\nD. Docetaxel + cisplatin' },
  { label: 'Verify: Cisplatin for HNSCC',  qt: 'fact_check', question: 'Cisplatin-based chemotherapy combined with pembrolizumab is approved as first-line treatment for recurrent or metastatic HNSCC. True or false?' },
  { label: 'Standard drug is ___',         qt: 'fill_blank', question: 'The anti-PD-1 monoclonal antibody ________ is approved for first-line treatment of recurrent or metastatic head and neck squamous cell carcinoma.' },
];

const PILL_COLORS = {
  qa: '#2563eb', lfqa: '#0891b2', mcqa: '#7c3aed',
  fact_check: '#d97706', fill_blank: '#16a34a', jeopardy: '#dc2626',
};

export default function EmptyState({ onPillClick }) {
  return (
    <div className="empty-state">
      <div className="empty-logo">O</div>
      <h2 className="empty-title">OncologyRAG</h2>
      <p className="empty-subtitle">Your AI-powered oncology assistant</p>
      <p className="empty-desc">
        Ask me anything about cancer treatment, medications, or oncology guidelines.
      </p>
      <div className="example-pills-row">
        {PILLS.map((pill, i) => (
          <button
            key={i}
            className="example-pill"
            onClick={() => onPillClick({ type: pill.qt, question: pill.question })}
          >
            <span className="example-pill-dot" style={{ background: PILL_COLORS[pill.qt] || '#2563eb' }} />
            {pill.label}
          </button>
        ))}
      </div>
    </div>
  );
}
