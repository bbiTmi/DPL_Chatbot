import React, { useRef, useEffect, useState } from 'react';
import Webcam from './Webcam'; // Adjust path if needed

const FaceAuth = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [webcam, setWebcam] = useState(null);
  const [mode, setMode] = useState('login'); // or 'register'
  const [response, setResponse] = useState(null);

  useEffect(() => {
    if (videoRef.current && canvasRef.current) {
      const webcamInstance = new Webcam(videoRef.current, 'user', canvasRef.current);
      setWebcam(webcamInstance);

      webcamInstance.start()
        .then(() => console.log('Webcam started'))
        .catch(err => console.error('Error starting webcam:', err));
    }

    return () => {
      webcam?.stop();
    };
  }, []);

  const captureAndSend = async () => {
    try {
      const imageBase64 = webcam.snap();

      const res = await fetch(`http://localhost:5000/face/${mode}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageBase64 }),
      });

      const data = await res.json();
      setResponse(data);
    } catch (err) {
      console.error('Error during capture/send:', err);
    }
  };

  return (
    <div style={{ textAlign: 'center' }}>
      <h2>Face {mode === 'login' ? 'Login' : 'Register'}</h2>

      <video ref={videoRef} autoPlay playsInline style={{ width: 400, height: 300, border: '1px solid #ccc' }} />
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      <div style={{ marginTop: 20 }}>
        <button onClick={captureAndSend}>Capture & {mode}</button>
        <button onClick={() => setMode(mode === 'login' ? 'register' : 'login')} style={{ marginLeft: 10 }}>
          Switch to {mode === 'login' ? 'Register' : 'Login'}
        </button>
      </div>

      {response && (
        <div style={{ marginTop: 20 }}>
          <p><strong>Status:</strong> {response.status}</p>
          {response.user && <p><strong>User:</strong> {response.user}</p>}
        </div>
      )}
    </div>
  );
};

export default FaceAuth;