import { useState } from "react";
import "./SearchBar.css";
import VideosContainer from "./VideosContainer";
import LinksContainer from "./LinksContainer";
import GoogleMapComponent from "../googlemap/GoogleMapComponent";

const SearchBar = () => {
  const [query, setQuery] = useState("");
  const [videos, setVideos] = useState([]);
  const [links, setLinks] = useState([]);
  const [idTitles, setidTitles] = useState([]);
  const [mapData, setMapData] = useState(null); // Datas for GoogleMapComponent
  const [isLoading, setIsLoading] = useState(false); // Loading state for handleSearch

  const extractVideoId = (url) => {
    try {
      const trimmedUrl = url.trim(); // Remove the space
      if (!trimmedUrl) {
        return null; // Handle empty URL case
      }
      const urlObj = new URL(trimmedUrl);
      if (urlObj.hostname === "www.youtube.com") {
        if (urlObj.pathname.startsWith("/shorts/")) {
          return urlObj.pathname.split("/shorts/")[1];
        } else if (urlObj.searchParams.get("v")) {
          return urlObj.searchParams.get("v");
        }
      } else if (urlObj.hostname === "youtu.be") {
        return urlObj.pathname.slice(1);
      }
      return null; // Unvalid YouTube URL
    } catch (error) {
      console.error("Error parsing URL:", error);
      return null;
    }
  };

  const fetchVideoTitle = async (videoId) => {
    try {
      const response = await fetch(
        `https://www.googleapis.com/youtube/v3/videos?part=snippet&id=${videoId}&key=${process.env.REACT_APP_GOOGLE_YOUTUBE_API_KEY}`
      );
      const data = await response.json();
      if (data.items && data.items.length > 0) {
        return data.items[0].snippet.title; // Get the video title
      }
      return "Unknown Title";
    } catch (error) {
      console.error("Error fetching video title:", error);
      return "Error fetching title";
    }
  };

  const handleInputChange = (e) => {
    setQuery(e.target.value);
  };

  const handleAddLink = async () => {
    if (idTitles.length > 4) {
      alert("You can only add up to 5 links.");
      setQuery("");
      return;
    }
    const videoId = extractVideoId(query);
    if (idTitles.some((idTitle) => idTitle.id === videoId)) {
      alert("This link has already been added.");
      return;
    }
    if (videoId) {
      const title = await fetchVideoTitle(videoId);
      setidTitles((prevIdTitles) => [...prevIdTitles, { id: videoId, title }]);
      setLinks(query);
      setQuery("");
    } else {
      alert("Please enter a valid YouTube link.");
      setQuery("");
    }
  };

  const handleSearch = async () => {
    if (idTitles.length > 0) {
      const selectedOption = document.querySelector(".dropdown").value; // Selected option value
      if (selectedOption === "default") {
        alert("Please select a region.");
        return;
      }
      const payload = {
        url: links,
        region_code: selectedOption,
      };
      try {
        setIsLoading(true);
        const response = await fetch("http://34.22.100.60:8000/analyze-url/", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        });
        if (response.ok) {
          const data = await response.json();
          console.log("Response from API:", data);
          setMapData(data); // Save response data to state
        } else {
          console.error("API request failed:", response.statusText);
          alert("Failed to send data to the API.");
        }
      } catch (error) {
        console.error("Error during API request:", error);
        alert("An error occurred while sending the request.");
      } finally {
        setIsLoading(false);
      }
      setVideos(idTitles);
      setidTitles([]);
      setLinks([]);
      setQuery("");
    } else {
      alert("Please enter a valid YouTube link.");
    }
  };

  return (
    <>
      <div
        className="search-bar-container"
        style={{ height: videos.length > 0 ? "150vh" : "100vh" }}
      >
        {!videos.length && (
          <h1 className="search-bar-title">
            Unearth Hidden <p>Gems</p>, Map Your Adventures
          </h1>
        )}
        <form
          className="search-bar"
          onSubmit={(e) => {
            e.preventDefault();
            handleSearch();
          }}
        >
          <div className="messageBox">
            <div className="inputs">
              <textarea
                required=""
                placeholder="Enter YouTube link here..."
                type="text"
                value={query}
                onChange={handleInputChange}
                className="messageInput"
              />
              <select className="dropdown">
                <option value="default" selected disabled>
                  구 선택
                </option>
                <option value="11">강남구</option>
                <option value="12">강동구</option>
                <option value="13">강북구</option>
                <option value="14">강서구</option>
                <option value="15">관악구</option>
                <option value="16">광진구</option>
                <option value="17">구로구</option>
                <option value="18">금천구</option>
                <option value="19">노원구</option>
                <option value="20">도봉구</option>
                <option value="21">동대문구</option>
                <option value="22">동작구</option>
                <option value="23">마포구</option>
                <option value="24">서대문구</option>
                <option value="25">서초구</option>
                <option value="26">성동구</option>
                <option value="27">성북구</option>
                <option value="28">송파구</option>
                <option value="29">양천구</option>
                <option value="30">영등포구</option>
                <option value="31">용산구</option>
                <option value="32">은평구</option>
                <option value="33">종로구</option>
                <option value="34">중구</option>
                <option value="35">중랑구</option>
              </select>
            </div>
            <div className="buttons">
              <div className="addButton" onClick={handleAddLink}>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 337 337"
                >
                  <circle
                    stroke-width="20"
                    stroke="#6c6c6c"
                    fill="none"
                    r="158.5"
                    cy="168.5"
                    cx="168.5"
                  ></circle>
                  <path
                    stroke-linecap="round"
                    stroke-width="25"
                    stroke="#6c6c6c"
                    d="M167.759 79V259"
                  ></path>
                  <path
                    stroke-linecap="round"
                    stroke-width="25"
                    stroke="#6c6c6c"
                    d="M79 167.138H259"
                  ></path>
                </svg>
              </div>
              <button className="sendButton" type="submit">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 664 663"
                >
                  <path
                    fill="none"
                    d="M646.293 331.888L17.7538 17.6187L155.245 331.888M646.293 331.888L17.753 646.157L155.245 331.888M646.293 331.888L318.735 330.228L155.245 331.888"
                  ></path>
                  <path
                    stroke-linejoin="round"
                    stroke-linecap="round"
                    stroke-width="33.67"
                    stroke="#6c6c6c"
                    d="M646.293 331.888L17.7538 17.6187L155.245 331.888M646.293 331.888L17.753 646.157L155.245 331.888M646.293 331.888L318.735 330.228L155.245 331.888"
                  ></path>
                </svg>
              </button>
            </div>
          </div>
        </form>
        <LinksContainer idTitles={idTitles} />
        {isLoading ? ( // While loading, show the loading screen
          <section class="loader">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
          </section>
        ) : null}
        <VideosContainer videos={videos} />
        {videos.length > 0 && <GoogleMapComponent mapData={mapData} />}
      </div>
    </>
  );
};

export default SearchBar;
