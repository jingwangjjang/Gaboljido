// import { Map, GoogleApiWrapper, Marker, InfoWindow } from "google-maps-react";
import "./GoogleMapComponent.css";
import ListContainer from "./ListContainer";
import { useState, useEffect } from "react";
import {
  GoogleMap,
  LoadScript,
  Marker,
  InfoWindow,
} from "@react-google-maps/api";

const GoogleMapComponent = ({ mapData }) => {
  const [selectedPlace, setSelectedPlace] = useState(null);
  const [activeMarker, setActiveMarker] = useState(null);
  const [locations, setLocations] = useState([]); // Store Place ID and Lat/Lng
  const [isLoaded, setIsLoaded] = useState(false); // Track if Google Maps API is loaded

  useEffect(() => {
    if (isLoaded) {
      fetchAllLocations(); // Fetch locations only after API is loaded
    }
  }, [mapData, isLoaded]);

  const fetchLocation = async (address) => {
    // if (!window.google || !window.google.maps) {
    //   console.error("Google Maps API is not loaded.");
    //   return null;
    // }
    const geocoder = new window.google.maps.Geocoder();
    address = `${address} 성수`; // temporary fix for address
    return new Promise((resolve, reject) => {
      geocoder.geocode({ address }, (results, status) => {
        if (status === "OK" && results[0]) {
          const placeId = results[0].place_id;
          const location = results[0].geometry.location;
          resolve({
            placeId,
            lat: location.lat(),
            lng: location.lng(),
            name: address,
          });
        } else {
          console.error(`Geocoding failed for address "${address}":`, status);
          resolve(null); // Return null on failure
        }
      });
    });
  };

  const fetchAllLocations = async () => {
    if (!mapData?.data) return;
    const storeNames = mapData.data.map((store) => store.store_name);
    const locationPromises = storeNames.map((storeName) =>
      fetchLocation(storeName)
    );
    const fetchedLocations = await Promise.all(locationPromises);
    setLocations(fetchedLocations.filter((location) => location !== null));
    console.log("Fetched Locations:", fetchedLocations);
  };

  const handleMapClick = () => {
    setSelectedPlace(null); // Close the info window on map click
    setActiveMarker(null);
  };

  const handleMarkerClick = (location, marker) => {
    setSelectedPlace(location);
    setActiveMarker(marker);
  };

  return (
    <LoadScript
      googleMapsApiKey={process.env.REACT_APP_GOOGLE_MAP_API_KEY}
      onLoad={() => setIsLoaded(true)} // Set isLoaded to true when API is loaded
    >
      <div className="map-wrapper">
        <div className="map-container">
          <GoogleMap
            mapContainerStyle={{ width: "100%", height: "100%" }}
            zoom={14}
            center={{
              lat: 37.544206,
              lng: 127.057101,
            }}
            onClick={handleMapClick}
          >
            {locations.map((location, index) => (
              <Marker
                key={index}
                position={{ lat: location.lat, lng: location.lng }}
                onClick={(e) => handleMarkerClick(location, e)}
              />
            ))}
            {selectedPlace && (
              <InfoWindow
                position={{ lat: selectedPlace.lat, lng: selectedPlace.lng }}
                onCloseClick={() => setSelectedPlace(null)}
              >
                <div>
                  <h4 className="info-window-title">
                    {/* temp fix for address */}
                    {selectedPlace.name.replace(/ 성수$/, "") ||
                      "Unknown Place"}
                  </h4>
                  <a
                    href={`https://www.google.com/maps/place/?q=place_id:${selectedPlace.placeId}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Google Map
                  </a>
                </div>
              </InfoWindow>
            )}
          </GoogleMap>
        </div>
        <ListContainer mapData={mapData} />
      </div>
    </LoadScript>
  );
};

export default GoogleMapComponent;
