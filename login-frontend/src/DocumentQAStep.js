import React, { useState } from 'react';
import VoiceAgentConverted from './VoiceAgentConverted';

function DocumentQAStep({ onDone }) {
  const [pdfFile, setPdfFile] = useState(null);
  const [pdfInfo, setPdfInfo] = useState(null); // { pdf_path, filename }
  const [processInfo, setProcessInfo] = useState(null); // { pages, chunks }
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [pagesUsed, setPagesUsed] = useState([]);
  const [candidates, setCandidates] = useState([]);
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
      setAnswer('');
      setPagesUsed([]);
      setCandidates([]);
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
      if (!res.ok) throw new Error('Process failed');
      const data = await res.json();
      setProcessInfo(data);
      setAnswer('');
      setPagesUsed([]);
      setCandidates([]);
    } catch (e) {
      console.error(e);
      alert('Process failed');
    } finally {
      setLoading(false);
    }
  };

  const askRag = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/rag/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      if (!res.ok) throw new Error('Query failed');
      const data = await res.json();
      setAnswer(data.answer || '');
      setPagesUsed(Array.isArray(data.pages) ? data.pages : []);
      setCandidates(Array.isArray(data.candidates) ? data.candidates : []);
    } catch (e) {
      console.error(e);
      alert('Query failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', width: '100%', background: 'linear-gradient(135deg, #0b1220 0%, #0a0a16 100%)', color: 'white' }}>
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          <h2 style={{ margin: 0, fontSize: 24, letterSpacing: 0.5 }}>NYC Docs • RAG Console</h2>
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
                placeholder="Optional: Ask a test question about the document"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                style={{ width: '70%' }}
              />
              <button onClick={askRag} disabled={!processInfo || loading} style={{ marginLeft: 8 }}>Ask</button>
            </div>
            {answer && (
              <div style={{ marginTop: 8, padding: 12, background: 'rgba(255,255,255,0.06)', borderRadius: 8 }}>
                <div style={{ fontWeight: 600, marginBottom: 6 }}>Answer</div>
                <div>{answer}</div>
              </div>
            )}
            {pagesUsed && pagesUsed.length > 0 && (
              <div style={{ marginTop: 8 }}>
                <div style={{ fontWeight: 600, marginBottom: 6 }}>Relevant Pages</div>
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                  {pagesUsed.map((p) => (
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
              <VoiceAgentConverted />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default DocumentQAStep;
