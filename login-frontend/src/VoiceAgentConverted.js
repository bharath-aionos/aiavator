import React, { useState, useEffect } from 'react';

function VoiceAgentConverted() {
  const [status, setStatus] = useState('disconnected');
  const [statusText, setStatusText] = useState('Ready to Connect');
  const [connected, setConnected] = useState(false);
  const [peerConnection, setPeerConnection] = useState(null);
  const [lastPages, setLastPages] = useState([]);
  const [lastAnswer, setLastAnswer] = useState('');

  const waitForIceGatheringComplete = async (pc, timeoutMs = 2000) => {
    if (pc.iceGatheringState === 'complete') return;
    console.log("Waiting for ICE gathering to complete. Current state:", pc.iceGatheringState);
    return new Promise((resolve) => {
      let timeoutId;
      const checkState = () => {
        console.log("icegatheringstatechange:", pc.iceGatheringState);
        if (pc.iceGatheringState === 'complete') {
          cleanup();
          resolve();
        }
      };

  
      const onTimeout = () => {
        console.warn(`ICE gathering timed out after ${timeoutMs} ms.`);
        cleanup();
        resolve();
      };
      const cleanup = () => {
        pc.removeEventListener('icegatheringstatechange', checkState);
        clearTimeout(timeoutId);
      };
      pc.addEventListener('icegatheringstatechange', checkState);
      timeoutId = setTimeout(onTimeout, timeoutMs);
      checkState();
    });
  };

  const createSmallWebRTCConnection = async (audioTrack) => {
    const iceServers = [
      { urls: "stun:localhost:3479" },
      { urls: "turn:localhost:3479?transport=udp", username: "test", credential: "test123" },
      { urls: "turn:localhost:3479?transport=tcp", username: "test", credential: "test123" }
    ];
    const pc = new RTCPeerConnection({ iceServers });
    addPeerConnectionEventListeners(pc);
    pc.ontrack = e => {
      const audioEl = document.getElementById("audio-el");
      if (audioEl) audioEl.srcObject = e.streams[0];
    };
    pc.addTransceiver(audioTrack, { direction: 'sendrecv' });
    pc.addTransceiver('video', { direction: 'sendrecv' });
    await pc.setLocalDescription(await pc.createOffer());
    await waitForIceGatheringComplete(pc);
    const offer = pc.localDescription;
    const response = await fetch('http://localhost:8000/api/offer', {
      body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
      headers: { 'Content-Type': 'application/json' },
      method: 'POST',
    });
    const answer = await response.json();
    await pc.setRemoteDescription(answer);
    return pc;
  };

  const connect = async () => {
    setStatus('connecting');
    setStatusText('Connecting...');
    setConnected(true);
    try {
      const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const pc = await createSmallWebRTCConnection(audioStream.getAudioTracks()[0]);
      setPeerConnection(pc);
    } catch (error) {
      console.error("Connection failed:", error);
      setStatus('disconnected');
      setStatusText('Ready to Connect');
      setConnected(false);
    }
  };

  const addPeerConnectionEventListeners = (pc) => {
    pc.oniceconnectionstatechange = () => {
      console.log("oniceconnectionstatechange", pc?.iceConnectionState);
    };
    pc.onconnectionstatechange = () => {
      console.log("onconnectionstatechange", pc?.connectionState);
      let connectionState = pc?.connectionState;
      if (connectionState === 'connected') {
        setStatus('connected');
        setStatusText('Connected & Listening');
        setConnected(true);
      } else if (connectionState === 'disconnected') {
        setStatus('disconnected');
        setStatusText('Ready to Connect');
        setConnected(false);
      }
    };
    pc.onicecandidate = (event) => {
      if (event.candidate) {
        console.log("New ICE candidate:", event.candidate);
      } else {
        console.log("All ICE candidates have been sent.");
      }
    };
  };

  const disconnect = () => {
    if (!peerConnection) {
      return;
    }
    peerConnection.close();
    setPeerConnection(null);
    setStatus('disconnected');
    setStatusText('Ready to Connect');
    setConnected(false);
  };

  const handleConnect = async () => {
    if (!connected) {
      await connect();
    } else {
      disconnect();
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', alignItems: 'center', justifyContent: 'center', fontFamily: "'Inter', sans-serif", background: 'linear-gradient(135deg, #0f1419 0%, #1a1a2e 100%)', margin: 0, padding: 0, position: 'relative' }}>
      <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', background: 'rgba(255, 255, 255, 0.1)', backdropFilter: 'blur(20px)', zIndex: -1 }}></div>

      <div style={{ position: 'relative', width: '100%', maxWidth: '500px', padding: '40px', textAlign: 'center' }}>
        <div style={{ background: 'rgba(255, 255, 255, 0.15)', backdropFilter: 'blur(20px)', borderRadius: '30px', padding: '50px 40px', boxShadow: '0 25px 50px rgba(0, 0, 0, 0.25)', border: '1px solid rgba(255, 255, 255, 0.2)', transition: 'all 0.3s ease', position: 'relative', overflow: 'hidden' }}>
          <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '1px', background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent)' }}></div>

          <div style={{ position: 'absolute', top: '20px', right: '20px', width: '40px', height: '40px', background: 'rgba(255, 255, 255, 0.1)', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'rgba(255, 255, 255, 0.7)', fontSize: '16px', transition: 'all 0.3s ease', opacity: status === 'connected' ? 1 : 0.5 }}>
            🎤
          </div>

          <div style={{ width: '80px', height: '80px', margin: '0 auto 30px', background: 'linear-gradient(135deg, #ff6b6b, #4ecdc4)', borderRadius: '20px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '32px', color: 'white', fontWeight: 'bold', boxShadow: '0 10px 25px rgba(0, 0, 0, 0.2)', animation: 'float 3s ease-in-out infinite' }}>
            AI
          </div>

          <h1 style={{ color: 'white', fontSize: '32px', fontWeight: '700', marginBottom: '10px', textShadow: '0 2px 4px rgba(0, 0, 0, 0.3)' }}>Voice Agent</h1>
          <p style={{ color: 'rgba(255, 255, 255, 0.8)', fontSize: '16px', marginBottom: '40px', fontWeight: '400' }}>Intelligent voice assistant powered by AI</p>

          <div style={{ marginBottom: '40px', position: 'relative' }}>
            <div style={{ fontSize: '18px', fontWeight: '600', color: 'white', marginBottom: '20px', textShadow: '0 2px 4px rgba(0, 0, 0, 0.3)', transition: 'all 0.3s ease' }}>
              <span style={{ width: '12px', height: '12px', borderRadius: '50%', display: 'inline-block', marginRight: '10px', transition: 'all 0.3s ease', background: status === 'disconnected' ? '#ff6b6b' : status === 'connecting' ? '#feca57' : '#48cae4', boxShadow: status === 'connecting' ? '0 0 0 0 rgba(254, 202, 87, 0.7)' : status === 'connected' ? '0 0 0 0 rgba(72, 202, 228, 0.7)' : 'none', animation: status === 'connecting' ? 'pulse 1.5s infinite' : status === 'connected' ? 'pulse-success 2s infinite' : 'none' }}></span>
              <span>{statusText}</span>
            </div>
          </div>

          <button type="button" onClick={handleConnect} style={{ position: 'relative', zIndex: 2, width: '120px', height: '120px', border: 'none', borderRadius: '50%', background: 'linear-gradient(135deg, #ff6b6b, #ee5a52)', color: 'white', fontSize: '16px', fontWeight: '600', cursor: 'pointer', boxShadow: '0 15px 35px rgba(255, 107, 107, 0.4)', transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)', margin: '0 auto', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
            <span style={{ fontSize: '24px', transition: 'all 0.3s ease' }}>{connected ? '⏹' : '▶'}</span>
          </button>

          <div style={{ position: 'absolute', bottom: '-50px', left: '50%', transform: 'translateX(-50%)', width: '200px', height: '50px', opacity: status === 'connected' ? 1 : 0, transition: 'all 0.3s ease', pointerEvents: 'none' }}>
            <div style={{ width: '4px', height: '20px', background: 'rgba(255, 255, 255, 0.6)', margin: '0 2px', borderRadius: '2px', display: 'inline-block', animation: 'wave 1.2s ease-in-out infinite' }}></div>
            <div style={{ width: '4px', height: '20px', background: 'rgba(255, 255, 255, 0.6)', margin: '0 2px', borderRadius: '2px', display: 'inline-block', animation: 'wave 1.2s ease-in-out infinite', animationDelay: '0.1s' }}></div>
            <div style={{ width: '4px', height: '20px', background: 'rgba(255, 255, 255, 0.6)', margin: '0 2px', borderRadius: '2px', display: 'inline-block', animation: 'wave 1.2s ease-in-out infinite', animationDelay: '0.2s' }}></div>
            <div style={{ width: '4px', height: '20px', background: 'rgba(255, 255, 255, 0.6)', margin: '0 2px', borderRadius: '2px', display: 'inline-block', animation: 'wave 1.2s ease-in-out infinite', animationDelay: '0.3s' }}></div>
            <div style={{ width: '4px', height: '20px', background: 'rgba(255, 255, 255, 0.6)', margin: '0 2px', borderRadius: '2px', display: 'inline-block', animation: 'wave 1.2s ease-in-out infinite', animationDelay: '0.4s' }}></div>
          </div>
          {/* Live RAG result preview while assistant is speaking */}
          {connected && (lastAnswer || (lastPages && lastPages.length > 0)) && (
            <div style={{ marginTop: 24, textAlign: 'left' }}>
              {lastAnswer && (
                <div style={{ marginBottom: 10, padding: 12, background: 'rgba(255,255,255,0.06)', borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, marginBottom: 6, color: '#e2e8f0' }}>Answer</div>
                  <div style={{ color: 'white' }}>{lastAnswer}</div>
                </div>
              )}
              {lastPages && lastPages.length > 0 && (
                <div>
                  <div style={{ fontWeight: 600, marginBottom: 6, color: '#e2e8f0' }}>Relevant Pages</div>
                  <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                    {lastPages.map((p) => (
                      <div key={p} style={{ width: 120 }}>
                        <img
                          src={`http://localhost:8000/pdf/page/${p}`}
                          alt={`Page ${p}`}
                          style={{ width: '100%', borderRadius: 6, border: '1px solid rgba(255,255,255,0.12)' }}
                        />
                        <div style={{ color: '#cbd5e1', fontSize: 12, marginTop: 4, textAlign: 'center' }}>Page {p}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
       
      </div>

      <div style={{ marginTop: 'auto', marginBottom: '20px', color: 'rgba(255, 255, 255, 0.6)', fontSize: '14px', textAlign: 'center' }}>
        Secure • Private • Real-time
      </div>
     

      <audio id="audio-el" autoPlay></audio>

      <style>
        {`
          @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
          }
          @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(254, 202, 87, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(254, 202, 87, 0); }
            100% { box-shadow: 0 0 0 0 rgba(254, 202, 87, 0); }
          }
          @keyframes pulse-success {
            0% { box-shadow: 0 0 0 0 rgba(72, 202, 228, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(72, 202, 228, 0); }
            100% { box-shadow: 0 0 0 0 rgba(72, 202, 228, 0); }
          }
          @keyframes wave {
            0%, 100% { transform: scaleY(1); }
            50% { transform: scaleY(2); }
          }
        `}
      </style>
    </div>
  );
}

export default VoiceAgentConverted;
