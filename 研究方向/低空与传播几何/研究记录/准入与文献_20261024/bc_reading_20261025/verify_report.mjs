import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
const require = createRequire(import.meta.url);
const { chromium } = require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root = path.dirname(fileURLToPath(import.meta.url));
const html = path.join(path.dirname(root), '准入报告.html');
const browser = await chromium.launch({channel:'msedge', headless:true});
const page = await browser.newPage({viewport:{width:1360,height:1000},deviceScaleFactor:1});
const errors = [];
page.on('pageerror', e => errors.push(String(e)));
await page.goto(pathToFileURL(html).href);
const linkAudit = await page.locator('a').evaluateAll(elements => elements.map(e=>({href:e.href,text:e.textContent})));
const broken = [];
for(const link of linkAudit){
  const u = new URL(link.href);
  if(u.protocol === 'file:'){
    const local = fileURLToPath(u);
    if(!fs.existsSync(local)) broken.push(link);
    if(path.resolve(local) === path.resolve(html) && u.hash){
      if(!await page.locator(`[id="${decodeURIComponent(u.hash.slice(1))}"]`).count()) broken.push(link);
    }
  }
}
await page.screenshot({path:path.join(root,'html_desktop.png')});
await page.locator('#papers').scrollIntoViewIfNeeded();
await page.locator('details.paper').first().locator('summary').click();
const detailsOpen = await page.locator('details.paper').first().getAttribute('open') !== null;
await page.screenshot({path:path.join(root,'html_papers.png')});
await page.locator('#next').scrollIntoViewIfNeeded();
const hiddenMainHeadings = await page.locator('main h2').evaluateAll(es=>es.filter(e=>e.closest('details')).map(e=>e.id));
await page.screenshot({path:path.join(root,'html_plan.png')});
await page.setViewportSize({width:390,height:844});
await page.evaluate(()=>scrollTo(0,0));
await page.screenshot({path:path.join(root,'html_mobile.png')});
const dimensions = await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));
const audit = {browser:'Microsoft Edge / Playwright headless', paper_cards:await page.locator('details.paper').count(),
  local_links_checked:linkAudit.filter(x=>x.href.startsWith('file:')).length,broken_links:broken,
  console_errors:errors,details_interaction_pass:detailsOpen,mobile_dimensions:dimensions,
  main_headings_inside_details:hiddenMainHeadings};
audit.pass = !broken.length && !errors.length && detailsOpen && dimensions.scroll === dimensions.width && audit.paper_cards === 10 && !hiddenMainHeadings.length;
fs.writeFileSync(path.join(root,'HTML核验.json'),JSON.stringify(audit,null,2),'utf8');
console.log(JSON.stringify(audit,null,2));
await browser.close();
if(!audit.pass) process.exitCode=1;
