import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath,pathToFileURL} from 'node:url';
const require=createRequire(import.meta.url);
const {chromium}=require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=path.dirname(fileURLToPath(import.meta.url));
const browser=await chromium.launch({channel:'msedge',headless:true});
const page=await browser.newPage({viewport:{width:1360,height:1000}});
const errors=[];page.on('pageerror',e=>errors.push(String(e)));
await page.goto(pathToFileURL(path.join(root,'第一步核验报告.html')).href);
const broken=[];
for(const href of await page.locator('a').evaluateAll(es=>es.map(e=>e.href))){
 const u=new URL(href);if(u.protocol==='file:'&&!fs.existsSync(fileURLToPath(u)))broken.push(href);
}
await page.screenshot({path:path.join(root,'report_desktop.png')});
await page.locator('img').scrollIntoViewIfNeeded();
await page.screenshot({path:path.join(root,'report_figure.png')});
await page.setViewportSize({width:390,height:844});
await page.evaluate(()=>scrollTo(0,0));
await page.screenshot({path:path.join(root,'report_mobile.png')});
const dims=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));
await page.goto(pathToFileURL(path.join(path.dirname(root),'准入报告.html')).href);
const banner=await page.locator('#step1-latest').count();
const audit={broken_links:broken,page_errors:errors,mobile:dims,parent_banner:banner,pass:!broken.length&&!errors.length&&dims.width===dims.scroll&&banner===1};
fs.writeFileSync(path.join(root,'HTML核验.json'),JSON.stringify(audit,null,2),'utf8');
console.log(audit);await browser.close();if(!audit.pass)process.exitCode=1;
