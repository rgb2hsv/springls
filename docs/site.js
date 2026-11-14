// Sidebar scrollspy, mobile nav toggle, copy buttons, and one-video-at-a-time playback.
(function () {
  const links = Array.from(document.querySelectorAll('.toc a[href^="#"]'));
  const targets = links
    .map((a) => document.getElementById(a.getAttribute('href').slice(1)))
    .filter(Boolean);

  function setActive(id) {
    links.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === '#' + id));
    const active = links.find((a) => a.classList.contains('active'));
    if (active) {
      const toc = document.querySelector('.toc');
      const r = active.getBoundingClientRect();
      const tr = toc.getBoundingClientRect();
      if (r.top < tr.top || r.bottom > tr.bottom) active.scrollIntoView({ block: 'nearest' });
    }
  }

  function onScroll() {
    const y = window.scrollY + 120;
    let current = targets[0];
    for (const t of targets) {
      if (t.getBoundingClientRect().top + window.scrollY <= y) current = t;
      else break;
    }
    if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 4) current = targets[targets.length - 1];
    if (current) setActive(current.id);
  }
  let ticking = false;
  window.addEventListener('scroll', () => {
    if (!ticking) { requestAnimationFrame(() => { onScroll(); ticking = false; }); ticking = true; }
  }, { passive: true });
  onScroll();

  // Mobile nav
  const btn = document.querySelector('.menu-toggle');
  const close = () => { document.body.classList.remove('nav-open'); btn.setAttribute('aria-expanded', 'false'); };
  btn.addEventListener('click', () => {
    const open = document.body.classList.toggle('nav-open');
    btn.setAttribute('aria-expanded', String(open));
  });
  document.querySelector('.scrim').addEventListener('click', close);
  links.forEach((a) => a.addEventListener('click', close));

  // Copy buttons on code blocks
  document.querySelectorAll('pre:not(.pseudo)').forEach((pre) => {
    const b = document.createElement('button');
    b.className = 'copy-btn';
    b.type = 'button';
    b.textContent = 'Copy';
    b.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(pre.querySelector('code').innerText);
        b.textContent = 'Copied';
      } catch (e) {
        b.textContent = 'Failed';
      }
      setTimeout(() => (b.textContent = 'Copy'), 1200);
    });
    pre.appendChild(b);
  });

  // Only one video plays at a time
  const videos = Array.from(document.querySelectorAll('video'));
  videos.forEach((v) => v.addEventListener('play', () => {
    videos.forEach((o) => { if (o !== v) o.pause(); });
  }));
})();
