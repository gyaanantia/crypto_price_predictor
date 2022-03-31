import React, {Component} from "react"
import graph from './example-graph.png';
import Posts from "./components/Posts";
import './App.css';


function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Cryptocurrency Price Predictor</h1>
      </header>

      <div class="left-side">
        <img src={graph} className="App-logo" alt="logo" />
        <h6>Bitcoin is a decentralized digital currency, without a central bank or single administrator, that can be sent from user to user on the peer-to-peer bitcoin network without the need for intermediaries.</h6>
      </div>

      <div class="right-side">
        <h4>Bitcoin</h4>
        <aside class="right-side-buttons">
          <div class="time-button">
            <button>
              <h5>1D</h5>
            </button>
          </div>
          <div class="time-button">
            <button>
              <h5>1W</h5>
            </button>
          </div>
          <div class="time-button">
            <button>
              <h5>1M</h5>
            </button>
          </div>
          <div class="time-button">
            <button>
              <h5>3M</h5>
            </button>
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;
