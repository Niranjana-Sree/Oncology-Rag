import React, { useEffect, useRef } from 'react';
import MessageBubble from './MessageBubble';
import AnswerCard from './AnswerCard';
import EmptyState from './EmptyState';

const TYPING_MSGS = [
  'Analysing your query...',
  'Searching oncology knowledge...',
  'Ranking results...',
  'Generating answer...',
];

function TypingIndicator() {
  const [idx, setIdx] = React.useState(0);
  React.useEffect(() => {
    const t = setInterval(() => setIdx(i => (i + 1) % TYPING_MSGS.length), 1500);
    return () => clearInterval(t);
  }, []);
  return (
    <div className="typing-wrap">
      <div className="typing-card">
        <div className="typing-dots"><span /><span /><span /></div>
        <span className="typing-msg">{TYPING_MSGS[idx]}</span>
      </div>
    </div>
  );
}

export default function ConversationArea({ conversations, isLoading, onPillClick }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversations, isLoading]);

  if (conversations.length === 0 && !isLoading) {
    return (
      <div className="conversation-area">
        <EmptyState onPillClick={onPillClick} />
      </div>
    );
  }

  return (
    <div className="conversation-area">
      <div className="message-thread">
        {conversations.map(conv => (
          <React.Fragment key={conv.id}>
            <MessageBubble
              question={conv.question}
              queryType={conv.queryType}
              language={conv.language}
            />
            <AnswerCard
              patientAnswer={conv.patientAnswer}
              doctorAnswer={conv.doctorAnswer}
              sources={conv.sources}
              latency_ms={conv.latency_ms}
              iterations={conv.iterations}
            />
          </React.Fragment>
        ))}
        {isLoading && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
