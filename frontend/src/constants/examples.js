export const EXAMPLES = [
  {
    type: 'qa',
    label: 'Cisplatin side effects',
    question: 'What are the main side effects of cisplatin and how are they managed?',
  },
  {
    type: 'qa',
    label: 'TNM staging',
    question: 'Explain the TNM staging system for cancer and what each component means.',
  },
  {
    type: 'mcqa',
    label: 'HNSCC first-line treatment',
    question: 'What is the standard first-line treatment for recurrent or metastatic head and neck squamous cell carcinoma (HNSCC)?\nA. Carboplatin + paclitaxel\nB. Pembrolizumab monotherapy or with chemotherapy\nC. Cetuximab monotherapy\nD. Docetaxel + cisplatin',
  },
  {
    type: 'lfqa',
    label: 'Pembrolizumab mechanism',
    question: 'Explain in detail how pembrolizumab works as an immune checkpoint inhibitor and what the PD-1/PD-L1 pathway does.',
  },
  {
    type: 'lfqa',
    label: 'Radiation therapy types',
    question: 'What are the different types of radiation therapy used in oncology and what are the advantages and disadvantages of each?',
  },
  {
    type: 'fact_verification',
    label: 'Pembrolizumab FDA approval',
    question: 'Pembrolizumab was approved by the FDA as a first-line treatment for metastatic non-small cell lung cancer with high PD-L1 expression (TPS ≥50%). True or false, and explain.',
  },
  {
    type: 'fill_blank',
    label: 'HNSCC drug fill-blank',
    question: 'The anti-PD-1 monoclonal antibody ________ is approved for first-line treatment of recurrent or metastatic head and neck squamous cell carcinoma based on the KEYNOTE-048 trial.',
  },
  {
    type: 'jeopardy',
    label: 'Checkpoint inhibitor jeopardy',
    question: 'This class of immunotherapy drugs works by blocking inhibitory receptors such as PD-1, PD-L1, or CTLA-4 on T cells, thereby restoring anti-tumour immune responses.',
  },
];

export const QUERY_TYPE_LABELS = {
  qa: 'Q&A',
  mcqa: 'MCQA',
  lfqa: 'Long Answer',
  fact_check: 'Fact Check',
  fill_blank: 'Fill Blank',
  jeopardy: 'Jeopardy',
};

export const QUERY_TYPE_COLORS = {
  qa: '#2563eb',
  mcqa: '#7c3aed',
  lfqa: '#0891b2',
  fact_check: '#d97706',
  fill_blank: '#16a34a',
  jeopardy: '#dc2626',
};

export const PLACEHOLDERS = {
  qa: 'Ask a medical question about oncology, e.g. "What is the standard treatment for stage 3 breast cancer?"',
  mcqa: 'Paste a multiple choice question with options A, B, C, D...',
  lfqa: 'Ask for a detailed explanation, e.g. "Explain how CAR-T cell therapy works..."',
  fact_check: 'State a medical claim to verify, e.g. "Pembrolizumab is approved for first-line NSCLC..."',
  fill_blank: 'Enter a sentence with a blank, e.g. "The drug ________ targets EGFR in lung cancer."',
  jeopardy: 'Provide a description and ask what the term/drug/procedure is...',
};
