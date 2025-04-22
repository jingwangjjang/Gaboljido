import { Map, GoogleApiWrapper, Marker, InfoWindow } from "google-maps-react";
import "./GoogleMapComponent.css";
import ListContainer from "./ListContainer";
import { useState } from "react";

const GoogleMapComponent = (props) => {
  const [selectedPlace, setSelectedPlace] = useState(null);
  const [activeMarker, setActiveMarker] = useState(null);
  const [showInfoWindow, setShowInfoWindow] = useState(false);

  const handleMapClick = (mapProps, map, clickEvent) => {
    setShowInfoWindow(false); // Close the info window on map click
  };

  const handleMarkerClick = (props, marker) => {
    setSelectedPlace(props);
    setActiveMarker(marker);
    setShowInfoWindow(true);
  };

  return (
    <>
      <div className="map-wrapper">
        <div className="map-container">
          <Map
            google={props.google}
            zoom={14}
            // styles={mapStyles}
            initialCenter={{
              lat: 37.5665,
              lng: 126.978,
            }}
            onClick={handleMapClick}
          >
            <Marker
              position={{ lat: 37.5665, lng: 126.978 }}
              name="Seoul City Hall"
              onClick={handleMarkerClick}
            />
            <InfoWindow marker={activeMarker} visible={showInfoWindow}>
              <div>
                <h4>{selectedPlace?.name || "Unknown Place"}</h4>
                <p>Latitude: {selectedPlace?.position.lat}</p>
                <p>Longitude: {selectedPlace?.position.lng}</p>
              </div>
            </InfoWindow>
          </Map>
        </div>
        <ListContainer />
      </div>
    </>
  );
};

export default GoogleApiWrapper({
  apiKey: "AIzaSyBchX_q-BfWjKnLM0oj6Nt1QR3gafQESrg",
  loadingElement: <div>Loading map...</div>,
})(GoogleMapComponent);
