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
  const [locations, setLocations] = useState([]);
  // Sample data for locations
  //   [
  //     {
  //         "placeId": "ChIJM2dMgpakfDURtL7E74IE1QU",
  //         "lat": 37.544577,
  //         "lng": 127.055991,
  //         "name": "클로버 성수 성수"
  //     },
  // ]
  const [isLoaded, setIsLoaded] = useState(false); // Google Maps API load status

  useEffect(() => {
    if (isLoaded) {
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
      fetchAllLocations();
    }
  }, [mapData, isLoaded]);

  const fetchLocation = async (address) => {
    const geocoder = new window.google.maps.Geocoder();
    address = `${address}`;
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
          resolve(null);
        }
      });
    });
  };

  const handleMarkerClick = (location, marker) => {
    setSelectedPlace(location);
  };

  const handleMapClick = () => {
    setSelectedPlace(null); // Close InfoWindow
  };

  // Handling click events in ListContainer
  const handleSectionClick = (storeId) => {
    const location = locations.find(
      (loc) =>
        mapData.data.find((store) => store.store_id === storeId)?.store_name ===
        loc.name
    );
    if (location) {
      setSelectedPlace(location);
    }
  };

  return (
    <LoadScript
      googleMapsApiKey={process.env.REACT_APP_GOOGLE_MAP_API_KEY}
      onLoad={() => setIsLoaded(true)}
    >
      <div className="map-wrapper">
        <div className="map-container">
          <GoogleMap
            mapContainerStyle={{ width: "100%", height: "100%" }}
            zoom={14}
            // Default map center = 성수
            center={{
              lat: selectedPlace?.lat || 37.544206,
              lng: selectedPlace?.lng || 127.057101,
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
                    {selectedPlace.name || "Unknown Place"}
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
        <ListContainer
          mapData={mapData}
          onSectionClick={handleSectionClick} // Handle click event in ListContainer
        />
      </div>
    </LoadScript>
  );
};

export default GoogleMapComponent;
