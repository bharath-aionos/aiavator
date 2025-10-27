import React, { useState } from 'react';
import VoiceAgentConverted from './VoiceAgentConverted';

function DocumentQAStep({ onDone, ragQuery, ragAnswer, ragPagesUsed, ragCandidates, ragLoading, handleAsk, updateRagFromServer }) {
  const [pdfFile, setPdfFile] = useState(null);
  const [pdfInfo, setPdfInfo] = useState(null); // { pdf_path, filename }
  const [processInfo, setProcessInfo] = useState(null); // { pages, chunks }
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);

  const uploadPdf = async () => {
    if (!pdfFile) return;
    const formData = new FormData();
    formData.append('file', pdfFile);
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/pdf/upload', { method: 'POST', body: formData });
      if (!res.ok) throw new Error('Upload failed');
      const data = await res.json();
      setPdfInfo(data);
      setProcessInfo(null);
    } catch (e) {
      console.error(e);
      alert('Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const processPdf = async () => {
    if (!pdfInfo?.pdf_path) {
      alert('Please upload a PDF first');
      return;
    }
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/pdf/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pdf_path: pdfInfo.pdf_path }),
      });
      const data = await res.json();
      if (res.ok && data) {
        setProcessInfo(data);
      } else {
        throw new Error('Process failed');
      }
      // Shared RAG state is managed in App.js; no local setters to call here.
    } catch (e) {
      console.error(e);
      alert('Process failed');
    } finally {
      setLoading(false);
    }
  };

  // Use unified RAG handler from props
  const askRag = () => {
    if (!query.trim()) return;
    handleAsk(query);
  };

  return (
    <div style={{ minHeight: '100vh', width: '100%', background: 'linear-gradient(135deg, #0b1220 0%, #0a0a16 100%)', color: 'white' }}>
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <div style={{ fontWeight: 800, fontSize: 20 }}>ITQ TravelPort Smartpoint Tutor</div>
                </div>
              <div style={{ opacity: 0.8, fontSize: 12 }}>Powered by FAISS • LangChain • Groq</div>
            </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 0.9fr', gap: 20 }}>
          {/* Left column: Upload/Process/Query */}
          <div style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 14, padding: 18, boxShadow: '0 12px 40px rgba(0,0,0,0.35)' }}>
            <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 12 }}>Document Setup</div>
            <div style={{ marginBottom: 12 }}>
              <input type="file" accept="application/pdf" onChange={(e) => setPdfFile(e.target.files?.[0] || null)} />
              <button onClick={uploadPdf} disabled={!pdfFile || loading} style={{ marginLeft: 8 }}>Upload</button>
              {pdfInfo?.filename && <span style={{ color: '#9ae6b4', marginLeft: 8 }}>Uploaded: {pdfInfo.filename}</span>}
            </div>
            <div style={{ marginBottom: 12 }}>
              <button onClick={processPdf} disabled={!pdfInfo?.pdf_path || loading}>Process Document</button>
              {processInfo && (
                <span style={{ color: '#a0c4ff', marginLeft: 8 }}>Pages: {processInfo.pages} · Chunks: {processInfo.chunks}</span>
              )}
            </div>
            <div style={{ marginBottom: 12 }}>
              <input
                type="text"
                placeholder="Ask a question about the document"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                style={{ width: '70%' }}
              />
              <button onClick={askRag} disabled={!processInfo || ragLoading} style={{ marginLeft: 8 }}>Ask</button>
            </div>
            {ragAnswer && (
              <div style={{ marginTop: 8, padding: 12, background: 'rgba(255,255,255,0.06)', borderRadius: 8 }}>
                <div style={{ fontWeight: 600, marginBottom: 6 }}>Answer</div>
                <div>{ragAnswer}</div>
              </div>
            )}
            {ragPagesUsed && ragPagesUsed.length > 0 && (
              <div style={{ marginTop: 8 }}>
                <div style={{ fontWeight: 600, marginBottom: 6 }}>Relevant Pages</div>
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                  {ragPagesUsed.map((p) => (
                    <div key={p} style={{ width: 160 }}>
                      <img
                        src={`http://localhost:8000/pdf/page/${p}`}
                        alt={`Page ${p}`}
                        style={{ width: '100%', borderRadius: 6, border: '1px solid rgba(255,255,255,0.1)' }}
                      />
                      <div style={{ color: '#cbd5e1', fontSize: 12, marginTop: 4, textAlign: 'center' }}>Page {p}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <div style={{ marginTop: 16 }}>
              <button
                onClick={onDone}
                disabled={!processInfo}
                style={{ padding: '10px 16px', borderRadius: 8 }}
              >
                Continue to Assistant
              </button>
            </div>
          </div>

          {/* Right column: Voice Assistant (enabled after processing) */}
          <div style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 14, padding: 18, boxShadow: '0 12px 40px rgba(0,0,0,0.35)' }}>
            <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 12 }}>Voice Assistant</div>
            {!processInfo ? (
              <div style={{ opacity: 0.8 }}>Process a PDF to enable the assistant.</div>
            ) : (
              <VoiceAgentConverted
                ragQuery={ragQuery}
                ragAnswer={ragAnswer}
                ragPagesUsed={ragPagesUsed}
                ragCandidates={ragCandidates}
                ragLoading={ragLoading}
                handleAsk={handleAsk}
                updateRagFromServer={updateRagFromServer}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default DocumentQAStep;
