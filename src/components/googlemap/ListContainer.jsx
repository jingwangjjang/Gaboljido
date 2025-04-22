import "./ListContainer.css";

const ListContainer = () => {
  return (
    <div className="list-container">
      <h1>장소들</h1>
      <section class="accordion" id="overview">
        <h1 class="title">
          <a href="#overview">1번 장소</a>
        </h1>
        <div class="content">
          <div class="wrapper">
            <p>
              약하자면, 깃허브 풀 리퀘스트는 단순한 코드 병합 요청을 넘어, 코드
              품질 관리, 협업 증진, 변경 사항 추적을 위한 강력한 도구입니다.
              체계적인 풀 리퀘스트 워크플로우를 통해 프로젝트의 안정성과 개발
              효율성을 크게 향상시킬 수 있습니다.
            </p>
          </div>
        </div>
      </section>

      <section class="accordion" id="how-does-it-work">
        <h1 class="title">
          <a href="#how-does-it-work">2번 장소</a>
        </h1>
        <div class="content">
          <div class="wrapper">
            <p>
              약하자면, 깃허브 풀 리퀘스트는 단순한 코드 병합 요청을 넘어, 코드
              품질 관리, 협업 증진, 변경 사항 추적을 위한 강력한 도구입니다.
              체계적인 풀 리퀘스트 워크플로우를 통해 프로젝트의 안정성과 개발
              효율성을 크게 향상시킬 수 있습니다.
            </p>
          </div>
        </div>
      </section>

      <section class="accordion" id="inspiration">
        <h1 class="title">
          <a href="#inspiration">3번 장소</a>
        </h1>
        <div class="content">
          <div class="wrapper">
            <p>
              약하자면, 깃허브 풀 리퀘스트는 단순한 코드 병합 요청을 넘어, 코드
              품질 관리, 협업 증진, 변경 사항 추적을 위한 강력한 도구입니다.
              체계적인 풀 리퀘스트 워크플로우를 통해 프로젝트의 안정성과 개발
              효율성을 크게 향상시킬 수 있습니다.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default ListContainer;
