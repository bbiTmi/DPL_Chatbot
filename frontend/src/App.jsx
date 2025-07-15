import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainPage from './components/MainPage';
import FaceAuth from './components/FaceAuth';
import ManagerLogin from './components/ManagerLogin';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/customer" element={<FaceAuth />} />
        <Route path="/manager" element={<ManagerLogin />} />
      </Routes>
    </Router>
  );
}

export default App;