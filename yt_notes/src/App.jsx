import "./App.css";
import GoogleMapComponent from "./components/googlemap/GoogleMapComponent";
import SearchBar from "./components/searchbar/SearchBar";

function App() {
  return (
    <div className="App">
      {/* <div className="logo-container">
        <img src="/logo.png" className="logo" />
        <h1 className="logo-text">가볼지도</h1>
      </div> */}
      <SearchBar />
      <GoogleMapComponent />
    </div>
  );
}

export default App;
