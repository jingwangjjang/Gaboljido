import "./ListContainer.css";

const ListContainer = ({ mapData, onSectionClick }) => {
  const stores = mapData?.data || [];
  const personEmojis = [
    "(ˈⰙˈ)",
    "(ˈ᎑ˈ)",
    "(°͈̅​ ᢐ °͈̅)",
    "( ° ͜ʖ °)",
    "(˙ꈊ˙)",
    "(╹3╹)",
    "(╹ᗜ╹)",
    "(´•᎑•`)",
    "( ‾ʖ̫‾)",
  ];

  const getRandomEmoji = () => {
    const randomIndex = Math.floor(Math.random() * personEmojis.length);
    return personEmojis[randomIndex];
  };

  return (
    <div className="list-container">
      <h1>Places in videos</h1>
      {stores.length > 0 ? (
        stores.map((store, index) => (
          <section
            className="accordion"
            id={`store-${store.store_id}`}
            key={index}
            onClick={() => onSectionClick(store.store_id)} // Passing data to the parent container
          >
            <h1 className="title">
              <a href={`#store-${store.store_id}`}>{store.store_name}</a>
            </h1>
            <div className="content">
              <div className="wrapper">
                <p>
                  {getRandomEmoji()}: {store.review_1}
                </p>
                <p>
                  {getRandomEmoji()}: {store.review_2}
                </p>
                <p>
                  {getRandomEmoji()}: {store.review_3}
                </p>
              </div>
            </div>
          </section>
        ))
      ) : (
        <p>데이터가 없습니다.</p>
      )}
    </div>
  );
};

export default ListContainer;
