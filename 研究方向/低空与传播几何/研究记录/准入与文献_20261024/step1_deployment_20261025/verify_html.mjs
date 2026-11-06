import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath,pathToFileURL} from 'node:url';
const require=createRequire(import.meta.url);
const {chromium}=require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=path.dirname(fileURLToPath(import.meta.url));
const browser=await chromium.launch({channel:'msedge',headless:true});
const page=await browser.newPage({viewport:{width:1360,height:1000}});
const errors=[], broken=[], widths=[];
page.on('pageerror',e=>errors.push(String(e)));
for(const name of ['部署与多径歧义报告','推导与文献对应']){
  await page.goto(pathToFileURL(path.join(root,name+'.html')).href);
  await page.locator('details').evaluateAll(es=>es.forEach(e=>e.open=true));
  for(const href of await page.locator('a').evaluateAll(es=>es.map(e=>e.href))){
    const u=new URL(href);if(u.protocol==='file:'&&!fs.existsSync(fileURLToPath(u)))broken.push(href);
  }
  await page.setViewportSize({width:1360,height:1000});
  await page.screenshot({path:path.join(root,name+'_desktop.png')});
  if(name==='部署与多径歧义报告'){
    await page.getByRole('heading',{name:'3 看一个具体假位置',exact:true}).scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(root,'反例条件_desktop.png')});
    await page.evaluate(()=>scrollTo(0,0));
  }
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:path.join(root,name+'_mobile.png')});
  widths.push(await page.evaluate(()=>({client:innerWidth,scroll:document.documentElement.scrollWidth})));
}
await page.goto(pathToFileURL(path.join(path.dirname(root),'准入报告.html')).href);
const latest=await page.locator('#step1-latest').innerText();
const audit={broken_links:broken,errors,mobile:widths,latest_banner:latest,
  pass:!broken.length&&!errors.length&&widths.every(w=>w.client===w.scroll)&&latest.includes('部署与多径')};
fs.writeFileSync(path.join(root,'HTML核验.json'),JSON.stringify(audit,null,2),'utf8');
console.log(audit);await browser.close();if(!audit.pass)process.exitCode=1;
