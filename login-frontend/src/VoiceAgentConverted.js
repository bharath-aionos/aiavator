import { useState, useEffect } from 'react';

function VoiceAgentConverted({ ragQuery, ragAnswer, ragPagesUsed, ragCandidates, ragLoading, handleAsk, updateRagFromServer }) {
  const [status, setStatus] = useState('disconnected');
  const [statusText, setStatusText] = useState('Ready to Connect');
  const [connected, setConnected] = useState(false);
  const [peerConnection, setPeerConnection] = useState(null);
  const [lastAnswer, setLastAnswer] = useState('');
  const [lastPages, setLastPages] = useState([]);

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

  useEffect(() => {
    let interval = null;
    let lastTs = null;
    let isPolling = false;
    let consecutiveEmptyResponses = 0;
    const MAX_EMPTY_RESPONSES = 3;

    const poll = async () => {
      if (!connected || isPolling) return;
      
      try {
        isPolling = true;
        const res = await fetch('http://localhost:8000/rag/last');
        if (!res.ok) return;
        
        const data = await res.json();
        const ts = data.timestamp;
        const hasAnswer = data.answer && data.answer.trim().length > 0;
        
        if (ts && ts !== lastTs && hasAnswer) {
          lastTs = ts;
          consecutiveEmptyResponses = 0;
          if (updateRagFromServer) {
            updateRagFromServer(data);
          }
        } else {
          consecutiveEmptyResponses++;
          if (consecutiveEmptyResponses >= MAX_EMPTY_RESPONSES) {
            // Stop polling after several empty responses
            if (interval) {
              clearInterval(interval);
              interval = null;
            }
          }
        }
      } catch (e) {
        console.error('Polling error:', e);
      } finally {
        isPolling = false;
      }
    };

    if (connected && !interval) {
      // Initial poll with delay to avoid immediate hammering
      setTimeout(poll, 1000);
      // Poll less frequently and only when connected
      interval = setInterval(poll, 5000);
    }

    return () => {
      if (interval) {
        clearInterval(interval);
        interval = null;
      }
      isPolling = false;
    };
  }, [connected, updateRagFromServer]);

  return (
    <div style={{ 
      width: '100%',
      height: '100%',
      display: 'flex',
      flexDirection: 'column'
    }}>
      <div style={{ 
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        flex: 1,
        width: '100%',
        padding: '20px'
      }}>
        <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '16px', color: '#111' }}>Voice Assistant</h2>
        <p style={{ color: '#666', marginBottom: '24px', textAlign: 'center' }}>
          Process a PDF to enable the assistant.
        </p>

        {/* Status and Connect Button */}
        <div style={{ marginBottom: '16px', textAlign: 'center' }}>
          <div style={{ 
            width: '8px', 
            height: '8px', 
            borderRadius: '50%', 
            display: 'inline-block', 
            marginRight: '8px',
            background: status === 'disconnected' ? '#dc2626' : status === 'connecting' ? '#3b82f6' : '#10b981'
          }}></div>
          <span style={{ color: '#666', fontSize: '14px' }}>{statusText}</span>
        </div>

        {connected && (
          <div style={{ marginTop: '16px', color: '#666', fontSize: '14px', textAlign: 'center' }}>
            {ragLoading ? 'Listening and processing...' : 'Ready for voice input'}
          </div>
        )}

        <button
          onClick={handleConnect}
          style={{
            padding: '10px 32px',
            borderRadius: '8px',
            background: connected ? 'linear-gradient(135deg, #dc2626 0%, #b91c1c 100%)' : 'linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%)',
            color: 'white',
            border: 'none',
            cursor: 'pointer',
            marginTop: '16px',
            fontWeight: '500',
            fontSize: '14px',
            transition: 'opacity 0.2s'
          }}
        >
          {connected ? 'Stop' : 'Continue to Assistant'}
        </button>
      </div>

      <audio id="audio-el" autoPlay></audio>
    </div>
  );
}

export default VoiceAgentConverted;