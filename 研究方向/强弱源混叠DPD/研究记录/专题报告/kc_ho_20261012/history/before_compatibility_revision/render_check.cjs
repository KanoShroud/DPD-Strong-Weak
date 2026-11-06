// Render the delivered Markdown's Pandoc HTML in Chromium and validate all math.
const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');
const {chromium} = require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async () => {
  const out = path.join(__dirname, 'analysis_v3_external', 'render_qa_v2');
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
      return {total: maths.length, display: document.querySelectorAll('.math.display').length,
        inline: document.querySelectorAll('.math.inline').length, failures, overflow,
        error_nodes: document.querySelectorAll('.katex-error').length,
        broken_images: [...document.images].filter(i => !i.complete || !i.naturalWidth).map(i => i.src)};
    });
    for (const [id, selector] of [['symbols', 'table'], ['likelihood', 'h3'], ['gradient', 'h3']]) {
      const loc = id === 'symbols' ? page.locator(selector).nth(1) : page.locator(selector).filter({hasText: id === 'likelihood' ? '3.1 从复高斯' : '3.4 连续位置'});
      await loc.evaluate(el => window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - 30));
      await page.screenshot({path: path.join(out, id+'.png')});
    }
    await page.locator('.math.display').first().screenshot({path:path.join(out,'equation_2_1.png')});
    const source = fs.readFileSync(path.join(__dirname, '学术交流报告_中文.md'), 'utf8');
    const inlineExpected = [...source.replace(/\$\$[\s\S]*?\$\$/g, '').matchAll(/\$[^$\n]+\$/g)].length;
    result.expected_inline = inlineExpected;
    result.status = result.inline === inlineExpected && result.display === 48 && !result.failures.length && !result.error_nodes && !result.overflow.length && !result.broken_images.length ? 'PASS' : 'FAIL';
    fs.writeFileSync(path.join(out, 'render_validation.json'), JSON.stringify(result,null,2), 'utf8');
    console.log(JSON.stringify(result,null,2));
    if(result.status !== 'PASS') process.exitCode=1;
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode=1;});
