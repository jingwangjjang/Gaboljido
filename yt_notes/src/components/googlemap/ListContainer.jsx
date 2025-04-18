import "./ListContainer.css";

const ListContainer = () => {
  return (
    <div className="list-container">
      <h1>장소들들</h1>
      <section class="accordion" id="overview">
        <h1 class="title">
          <a href="#overview">1번 장소소</a>
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
          <a href="#how-does-it-work">How does it work?</a>
        </h1>
        <div class="content">
          <div class="wrapper">
            <p>
              We cannot transition height or any CSS property from zero to auto.
              If we want to smoothly expand a collapsed accordion, we could set
              a<code>max-height</code> and transition the <code>height</code>{" "}
              property to a large value when we expand the section. However,
              <a
                href="https://css-tricks.com/using-css-transitions-auto-dimensions/#aa-there-are-two-crucial-downsides-to-this"
                target="_blank"
              >
                it may not properly work
              </a>
              if the content is larger than the <code>max-height</code>.
            </p>
            <p>
              So, here we create a CSS grid with a grid item. Then we transition
              the
              <code>grid-template-rows</code> property from <code>0fr</code> to
              <code>1fr</code> and the grid item transitions to its content
              height.
            </p>
            <p>
              I used CSS <code>clip-path</code> and <code>mix-blend-mode</code>{" "}
              to animate the background and text color of the accordion. Each
              accordion has two pseudo-elements. We know that{" "}
              <code>opacity</code>,<code>clip-path</code> and,{" "}
              <code>visibility</code> are animatable CSS properties. So when the
              accordion is selected using the
              <code>:target</code> pseudo-class, we transition the clip-path
              circle radius to 200% of the
              <code>::before</code>
              pseudo-element. When the accordion is not selected, we do the same
              with the
              <code>::after</code> but this time with a delay. This creates an
              illusion as it appears to clip from inside.
            </p>
            <p>
              Since the accordion's title is an anchor element, users can also
              navigate through the items with their keyboard.
            </p>
          </div>
        </div>
      </section>

      <section class="accordion" id="inspiration">
        <h1 class="title">
          <a href="#inspiration">Inspiration</a>
        </h1>
        <div class="content">
          <div class="wrapper">
            <p>
              In a recent video, Kevin showed us a way to animate height from
              zero to auto value. Seeing that, I couldn't resist experimenting
              with the idea. Then I came up with this CSS-only accordion
              component and added the reveal animation too!
            </p>
            <p>
              I would highly recommend watching
              <a href="https://youtu.be/B_n4YONte5A" target="_blank">
                his video
              </a>
              , where he beautifully explains the technique and points out how
              he discovered it. Also, read
              <a
                href="https://keithjgrant.com/posts/2023/04/transitioning-to-height-auto/"
                target="_blank"
              >
                Keith J. Grant's post
              </a>
              and
              <a
                href="https://nemzes.net/posts/animating-height-auto/"
                target="_blank"
              >
                Nelson Menezes's post
              </a>
              on the same topic.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default ListContainer;
