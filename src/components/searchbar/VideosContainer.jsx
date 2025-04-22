import YouTube from "react-youtube";
import "./VideosContainer.css";

const VideosContainer = ({ videos }) => {
  return (
    <div
      className="video-list"
      style={{ height: videos.length > 0 ? "25rem" : "0" }}
    >
      {videos.length > 0
        ? videos.map((video) => (
            <div key={video.id} className="video-wrapper">
              <YouTube videoId={video.id} />
            </div>
          ))
        : null}
    </div>
  );
};

export default VideosContainer;
