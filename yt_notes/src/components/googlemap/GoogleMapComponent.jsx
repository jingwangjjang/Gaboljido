import { Map, GoogleApiWrapper } from 'google-maps-react';
import './GoogleMapComponent.css';

const GoogleMapComponent = (props) => {
return (
  <div className="map-wrapper">
    <div className="map-container">
      <Map
        google={props.google}
        zoom={14}
        initialCenter={{
          lat: 37.5665,
          lng: 126.9780,
        }}
      />
    </div>
    <div className="list-container">
      <ul>
        <li>Location 1</li>
        <li>Location 2</li>
        <li>Location 3</li>
      </ul>
    </div>
  </div>
);
};

export default GoogleApiWrapper({
  apiKey: 'AIzaSyBchX_q-BfWjKnLM0oj6Nt1QR3gafQESrg',
  loadingElement: <div>Loading map...</div>
})(GoogleMapComponent);