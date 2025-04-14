import { useState } from 'react';
import './SearchBar.css';
import YouTube from 'react-youtube';

const SearchBar = () => {
    const [query, setQuery] = useState('');
    const [videos, setVideos] = useState([]);
    const [links, setLinks] = useState([]);

    const extractVideoId = (url) => {
        try {
            const trimmedUrl = url.trim(); // Remove the space
            if (!trimmedUrl) {
                return null; // Hnadle empty URL case
            }
            const urlObj = new URL(trimmedUrl);
            if (urlObj.hostname === 'www.youtube.com') {
                if (urlObj.pathname.startsWith('/shorts/')) {
                    return urlObj.pathname.split('/shorts/')[1];
                } else if (urlObj.searchParams.get('v')) {
                    return urlObj.searchParams.get('v');
                }
            } else if (urlObj.hostname === 'youtu.be') {
                return urlObj.pathname.slice(1);
            }
            return null; // Unvalid YouTube URL
        } catch (error) {
            console.error('Error parsing URL:', error);
            return null; 
        }
    };

    const handleInputChange = (e) => {
        setQuery(e.target.value);
    };
    
    const handleAddLink = () => {
        if (links.length >= 4) {
            alert('You can only add up to 5 links.');
            return;
        }
        const videoId = extractVideoId(query);
        if (links.includes(videoId)) {
            alert('This link has already been added.');
            return;
        }
        if (videoId) {
            setLinks((prevLinks) => [...prevLinks, videoId]);
            setQuery('');
        } else {
            alert('Please enter a valid YouTube link.');
        }
    };

    const handleSearch = () => {
        if (links.length > 0) {
            setVideos(links);
            setLinks([]);
            setQuery('');
        } else {
            alert('Please enter a valid YouTube link.');
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
            <form
                className="search-bar"
                onSubmit={(e) => {
                    e.preventDefault();
                    handleSearch();
                }}
            >
                <input
                    type="text"
                    className="search-input"
                    placeholder="Enter YouTube link..."
                    value={query}
                    onChange={handleInputChange}
                />
                <button
                    type="button"
                    className="add-button"
                    onClick={handleAddLink}
                >
                    +
                </button>
                <button type="submit" className="search-button">
                    🔍
                </button>
            </form>
            <div className="video-list" style={{ height: videos.length > 0 ? '500px' : '0' }}>
                {videos.length > 0 ? (
                    videos.map((videoId) => (
                        <div key={videoId} className="video-wrapper">
                            <YouTube videoId={videoId} />
                        </div>
                    ))
                ) : null}
            </div>
        </>
    );
}

export default SearchBar;