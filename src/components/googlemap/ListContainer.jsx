import "./ListContainer.css";

const ListContainer = ({ mapData }) => {
  // Check if mapData and its nested structure exist
  const stores = mapData?.data?.data || []; // Safely access the data array

  return (
    <div className="list-container">
      <h1>Places in videos</h1>
      {stores.length > 0 ? (
        stores.map((store, index) => (
          <section className="accordion" key={index}>
            <h1 className="title">
              <a href={`#store-${store.store_id}`}>{store.store_name}</a>
            </h1>
            <div className="content">
              <div className="wrapper">
                <p>{store.review_1}</p>
                <p>{store.review_2}</p>
                <p>{store.review_3}</p>
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
