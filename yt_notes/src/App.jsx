import './App.css';
import GoogleMapComponent from "./components/googlemap/GoogleMapComponent";
import SearchBar from "./components/searchbar/SearchBar";

function App() {
  return (
    <div className="App">
      <SearchBar />
      <GoogleMapComponent />
    </div>
  );
}

export default App;
