import "./LinksContainer.css";

const LinksContainer = ({ idTitles = [] }) => {
  const formatTitle = (title) => {
    // Remove hashtags and trim whitespace
    title = title.replace(/#[^\s#]+/g, "").trim();
    // if title is longer than 30 characters, truncate it
    return title.length > 30 ? title.slice(0, 30) + "..." : title;
  };

  return (
    <div className="links-list">
      {idTitles.length > 0
        ? idTitles.map((idTitle, index) => (
            <div key={index} className="link-item">
              <img
                src="https://upload.wikimedia.org/wikipedia/commons/4/42/YouTube_icon_%282013-2017%29.png"
                alt="YouTube Logo"
                className="youtube-logo"
              />
              <p>{formatTitle(idTitle.title)}</p>
            </div>
          ))
        : null}
    </div>
  );
};

export default LinksContainer;
