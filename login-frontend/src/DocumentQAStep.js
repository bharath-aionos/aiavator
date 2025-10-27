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
    <div style={{ minHeight: '100vh', width: '100%', backgroundColor: '#f5f5f5', padding: '20px' }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ 
                  fontSize: '24px', 
                  fontWeight: '800', 
                  color: '#3b82f6',
                  fontFamily: 'Arial, sans-serif',
                  letterSpacing: '-0.5px'
                }}>itq</span>
                <div style={{ fontSize: '24px', fontWeight: '700', color: '#111' }}>RAG Console</div>
              </div>
              <div style={{ color: '#666', fontSize: '14px' }}>FAISS • LangChain • Groq</div>
            </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 0.9fr', gap: 20 }}>
          {/* Left column: Upload/Process/Query */}
          <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', borderRadius: 12, padding: 24, boxShadow: '0 2px 8px rgba(0,0,0,0.08)' }}>
            <div style={{ fontSize: 18, fontWeight: 600, marginBottom: 20, color: '#111' }}>Document Setup</div>
            <div style={{ marginBottom: 20, display: 'flex', gap: 10 }}>
              <label htmlFor="file-upload" style={{
                padding: '10px 14px',
                border: '1px solid #e5e7eb',
                borderRadius: 8,
                cursor: 'pointer',
                backgroundColor: '#fff',
                flex: 1,
                display: 'flex',
                alignItems: 'center'
              }}>
                <input type="file" id="file-upload" accept="application/pdf" onChange={(e) => setPdfFile(e.target.files?.[0] || null)} style={{ display: 'none' }} />
                <span style={{ color: '#666', fontSize: 14 }}>{pdfInfo?.filename || 'Choose file'}</span>
              </label>
              <button onClick={uploadPdf} disabled={!pdfFile || loading} style={{ 
                background: 'linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%)',
                color: 'white',
                padding: '8px 20px',
                borderRadius: 8,
                border: 'none',
                cursor: 'pointer',
                fontWeight: 500,
                fontSize: 14
              }}>Upload</button>
            </div>
            <div style={{ marginBottom: 20 }}>
              <button onClick={processPdf} disabled={!pdfInfo?.pdf_path || loading} style={{
                width: '100%',
                background: 'linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%)',
                color: 'white',
                padding: '10px',
                borderRadius: 8,
                border: 'none',
                cursor: pdfInfo?.pdf_path && !loading ? 'pointer' : 'not-allowed',
                fontWeight: 500,
                fontSize: 14,
                opacity: (!pdfInfo?.pdf_path || loading) ? 0.5 : 1
              }}>Process Document</button>
              {processInfo && (
                <span style={{ color: '#666', marginLeft: 8, fontSize: 14 }}>Pages: {processInfo.pages} · Chunks: {processInfo.chunks}</span>
              )}
            </div>
            <div style={{ marginBottom: 12, display: 'flex', gap: 10 }}>
              <input
                type="text"
                placeholder="Ask a question about the document"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                style={{ 
                  flex: 1,
                  padding: '10px 14px',
                  borderRadius: 8,
                  border: '1px solid #e5e7eb',
                  fontSize: 14,
                  color: '#111',
                  outline: 'none'
                }}
              />
              <button onClick={askRag} disabled={!processInfo || ragLoading} style={{ 
                background: 'linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%)',
                color: 'white',
                padding: '10px 20px',
                borderRadius: 8,
                border: 'none',
                cursor: (!processInfo || ragLoading) ? 'not-allowed' : 'pointer',
                fontWeight: 500,
                fontSize: 14,
                opacity: (!processInfo || ragLoading) ? 0.5 : 1
              }}>Ask</button>
            </div>
            {ragAnswer && (
              <div style={{ marginTop: 16, padding: 16, background: '#f9fafb', borderRadius: 8, border: '1px solid #e5e7eb' }}>
                <div style={{ fontWeight: 600, marginBottom: 8, color: '#111' }}>Answer</div>
                <div style={{ color: '#333', fontSize: 14 }}>{ragAnswer}</div>
              </div>
            )}
            {ragPagesUsed && ragPagesUsed.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <div style={{ fontWeight: 600, marginBottom: 8, color: '#111' }}>Relevant Pages</div>
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                  {ragPagesUsed.map((p) => (
                    <div key={p} style={{ width: 160 }}>
                      <img
                        src={`http://localhost:8000/pdf/page/${p}`}
                        alt={`Page ${p}`}
                        style={{ width: '100%', borderRadius: 8, border: '1px solid #e5e7eb' }}
                      />
                      <div style={{ color: '#666', fontSize: 12, marginTop: 4, textAlign: 'center' }}>Page {p}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <div style={{ marginTop: 16 }}>
              <button
                onClick={onDone}
                disabled={!processInfo}
                style={{ 
                  width: '100%',
                  padding: '10px 16px', 
                  borderRadius: 8,
                  background: 'linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%)',
                  color: 'white',
                  border: 'none',
                  cursor: processInfo ? 'pointer' : 'not-allowed',
                  fontWeight: 500,
                  fontSize: 14,
                  opacity: processInfo ? 1 : 0.5
                }}
              >
                Continue to Assistant
              </button>
            </div>
          </div>

          {/* Right column: Voice Assistant (enabled after processing) */}
          <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', borderRadius: 12, padding: 24, boxShadow: '0 2px 8px rgba(0,0,0,0.08)', display: 'flex', flexDirection: 'column' }}>
            <div style={{ fontSize: 18, fontWeight: 600, marginBottom: 16, color: '#111' }}>Voice Assistant</div>
            {!processInfo ? (
              <div style={{ color: '#666', textAlign: 'center', padding: '40px 0' }}>Process a PDF to enable the assistant.</div>
            ) : (
              <div style={{ flex: 1 }}>
                <VoiceAgentConverted
                  ragQuery={ragQuery}
                  ragAnswer={ragAnswer}
                  ragPagesUsed={ragPagesUsed}
                  ragCandidates={ragCandidates}
                  ragLoading={ragLoading}
                  handleAsk={handleAsk}
                  updateRagFromServer={updateRagFromServer}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default DocumentQAStep;
