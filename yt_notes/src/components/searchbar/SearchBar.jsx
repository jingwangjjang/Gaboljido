import { useState } from "react";
import "./SearchBar.css";
import YouTube from "react-youtube";

const SearchBar = () => {
  const [query, setQuery] = useState("");
  const [videos, setVideos] = useState([]);
  const [links, setLinks] = useState([]);

  const extractVideoId = (url) => {
    try {
      const trimmedUrl = url.trim(); // Remove the space
      if (!trimmedUrl) {
        return null; // Hnadle empty URL case
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

  const handleInputChange = (e) => {
    setQuery(e.target.value);
  };

  const handleAddLink = () => {
    if (links.length >= 4) {
      alert("You can only add up to 5 links.");
      return;
    }
    const videoId = extractVideoId(query);
    if (links.includes(videoId)) {
      alert("This link has already been added.");
      return;
    }
    if (videoId) {
      setLinks((prevLinks) => [...prevLinks, videoId]);
      setQuery("");
    } else {
      alert("Please enter a valid YouTube link.");
    }
  };

  const handleSearch = () => {
    if (links.length > 0) {
      setVideos(links);
      setLinks([]);
      setQuery("");
    } else {
      alert("Please enter a valid YouTube link.");
    }
  };

  return (
    <>
      <div className="links-list">
        {links.length > 0 ? (
          links.map((link, index) => (
            <div key={index} className="link-item">
              {link}
            </div>
          ))
        ) : (
          <p>No links added yet.</p>
        )}
      </div>
      <h1 className="search-bar-title">
        Unearth Hidden <p>Gems</p>, Map Your Adventures
      </h1>
      <form
        className="search-bar"
        onSubmit={(e) => {
          e.preventDefault();
          handleSearch();
        }}
      >
        <div className="messageBox">
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
          <textarea
            required=""
            placeholder="Add a YouTube link..."
            type="text"
            value={query}
            onChange={handleInputChange}
            className="messageInput"
          />
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
      </form>
      <div
        className="video-list"
        style={{ height: videos.length > 0 ? "500px" : "0" }}
      >
        {videos.length > 0
          ? videos.map((videoId) => (
              <div key={videoId} className="video-wrapper">
                <YouTube videoId={videoId} />
              </div>
            ))
          : null}
      </div>
    </>
  );
};

export default SearchBar;
