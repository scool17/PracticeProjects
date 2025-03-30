import './App.css';
import React, { useState, useEffect } from "react";
import TensorFlowDashboard from './TensorFlowComponent/TensorFlowDashboard.js';
import NeuralNetworkDashboard from './NeuralNetworkComponent/NeuralNetworkDashboard.js';

function App() {


  return (
    <>
      <NeuralNetworkDashboard />
        <div>
          <TensorFlowDashboard />
        </div>
    </>
  );
}

export default App;
