import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {createRequire} from 'node:module';
import {fileURLToPath,pathToFileURL} from 'node:url';
const require=createRequire(import.meta.url);
const {chromium}=require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=path.dirname(fileURLToPath(import.meta.url));
const file=path.join(root,'五分钟看懂.html');
const html=fs.readFileSync(file,'utf8');
for(const script of html.matchAll(/<script>([\s\S]*?)<\/script>/g))new vm.Script(script[1]);
const browser=await chromium.launch({channel:'msedge',headless:true});
const page=await browser.newPage({viewport:{width:1360,height:1050}});
const errors=[],broken=[],checks={};page.on('pageerror',e=>errors.push(String(e)));
await page.goto(pathToFileURL(file).href);
await page.locator('details').evaluateAll(es=>es.forEach(e=>e.open=true));
for(const href of await page.locator('a').evaluateAll(es=>es.map(e=>e.href))){const u=new URL(href);if(u.protocol==='file:'&&!fs.existsSync(fileURLToPath(u)))broken.push(href);}
for(let i=0;i<5;i++){await page.locator(`[data-stage="${i}"]`).click();checks[`stage_${i}`]=await page.locator(`[data-stage="${i}"]`).getAttribute('aria-pressed')==='true'&&(await page.locator('#history-svg title').textContent())===(await page.locator('#stage-title').textContent());}
for(let i=0;i<3;i++){await page.locator(`[data-wall="${i}"]`).click();checks[`wall_${i}`]=await page.locator(`[data-wall="${i}"]`).getAttribute('aria-pressed')==='true';}
checks.finite_wall=await page.locator('#wall-line').getAttribute('d')==='M350 190V310';
await page.locator('[data-wall="1"]').click();checks.blocked_cross_visible=await page.locator('#collision').getAttribute('visibility')==='visible';
await page.locator('#mirror-toggle').click();checks.mirror_visible=await page.locator('#mirror-group').isVisible();
await page.locator('#play').click();await page.waitForFunction(()=>document.getElementById('pulse-status').textContent.startsWith('到达接收站'),null,{timeout:5000});checks.animation_finished=true;
await page.locator('#mirror').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(root,'镜像动画.png')});
await page.locator('#wall').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(root,'墙面检查.png')});
await page.locator('[data-stage="0"]').click();await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:path.join(root,'desktop.png')});
const widths=[];
for(const width of [390,320]){await page.setViewportSize({width,height:900});await page.evaluate(()=>scrollTo(0,0));widths.push(await page.evaluate(()=>({viewport:innerWidth,width:document.documentElement.scrollWidth})));if(width===390)await page.screenshot({path:path.join(root,'mobile.png')});}
const audit={checks,errors,broken_links:broken,widths,pass:Object.values(checks).every(Boolean)&&!errors.length&&!broken.length&&widths.every(w=>w.viewport===w.width)};
fs.writeFileSync(path.join(root,'交付核验.json'),JSON.stringify(audit,null,2),'utf8');console.log(audit);await browser.close();if(!audit.pass)process.exitCode=1;
