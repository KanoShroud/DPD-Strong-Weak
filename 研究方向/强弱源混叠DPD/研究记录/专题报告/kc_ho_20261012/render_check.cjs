// Render the delivered Markdown's Pandoc HTML in Chromium and validate all math.
const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');
const {chromium} = require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async () => {
  const out = path.join(__dirname, process.argv[2] || 'latex_markdown_render_qa');
  if (fs.existsSync(out)) throw new Error('Use a new render output directory.');
  fs.mkdirSync(out, {recursive: true});
  const browser = await chromium.launch({headless: true, channel: 'msedge'});
  try {
    const page = await browser.newPage({viewport: {width: 1400, height: 1050}, deviceScaleFactor: 1});
    await page.goto(pathToFileURL(path.join(__dirname, '报告渲染核验.html')).href);
    await page.waitForFunction(() => typeof katex !== 'undefined' && document.querySelectorAll('.katex').length > 0, {timeout: 30000});
    await page.evaluate(() => document.fonts.ready);
    const result = await page.evaluate(() => {
      const maths = [...document.querySelectorAll('.math')];
      const failures = [];
      for (const [index, element] of maths.entries()) {
        const annotation = element.querySelector('annotation');
        const source = annotation ? annotation.textContent : element.textContent;
        try { katex.renderToString(source, {displayMode: element.classList.contains('display'), throwOnError: true, strict: 'error'}); }
        catch(error) { failures.push({index, source, error: String(error)}); }
      }
      const overflow = [...document.querySelectorAll('.katex-display')].filter(el => el.scrollWidth > el.clientWidth + 2).map(el => el.innerText);
      const captions = [...document.querySelectorAll('.report-caption')];
      return {total: maths.length, display: document.querySelectorAll('.math.display').length,
        chapter_headings: [...document.querySelectorAll('h2')].map(el => el.textContent),
        inline: document.querySelectorAll('.math.inline').length, failures, overflow,
        captions: captions.length,
        centered_captions: captions.filter(el => getComputedStyle(el.querySelector('p') || el).textAlign === 'center').length,
        error_nodes: document.querySelectorAll('.katex-error').length,
        broken_images: [...document.images].filter(i => !i.complete || !i.naturalWidth).map(i => i.src)};
    });
    for (const [id, heading] of [['intro', '1.2 两步定位'], ['symbols', null], ['objective', '2.3 未知波形'], ['discussion', '4.1 窄峰']]) {
      const loc = id === 'symbols' ? page.locator('table').first() : page.locator('h3').filter({hasText: heading});
      await loc.evaluate(el => window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - 100));
      await page.screenshot({path: path.join(out, id+'.png')});
    }
    await page.locator('.math.display').first().screenshot({path:path.join(out,'equation_2_1.png')});
    const source = fs.readFileSync(path.join(__dirname, '学术交流报告_中文.md'), 'utf8');
    const inlineExpected = [...source.replace(/\$\$[\s\S]*?\$\$/g, '').matchAll(/\$[^$\n]+\$/g)].length;
    result.expected_inline = inlineExpected;
    result.no_boldsymbol = !source.includes('\\boldsymbol');
    result.status = result.chapter_headings.length === 5 && result.no_boldsymbol && result.centered_captions === 17 && result.captions === 17 && result.inline === inlineExpected && result.display === 22 && !result.failures.length && !result.error_nodes && !result.overflow.length && !result.broken_images.length ? 'PASS' : 'FAIL';
    fs.writeFileSync(path.join(out, 'render_validation.json'), JSON.stringify(result,null,2), 'utf8');
    console.log(JSON.stringify(result,null,2));
    if(result.status !== 'PASS') process.exitCode=1;
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode=1;});
