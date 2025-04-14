import { useState } from 'react';
import './SearchBar.css';

const SearchBar = ({ onSearch }) => {
    const [query, setQuery] = useState('');

    const handleInputChange = (e) => {
        setQuery(e.target.value);
    };

    const handleSearch = () => {
        if (onSearch) {
            onSearch(query);
        }
    };

    return (
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
                placeholder="Search..."
                value={query}
                onChange={handleInputChange}
            />
            <button type="submit" className="search-button">
                🔍
            </button>
        </form>
    );
};

export default SearchBar;