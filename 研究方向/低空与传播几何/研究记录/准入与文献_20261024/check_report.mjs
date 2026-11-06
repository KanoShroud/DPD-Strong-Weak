import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
const require=createRequire('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/package.json');
const {chromium}=require('playwright');
const root=path.dirname(fileURLToPath(import.meta.url));
const folder=path.join(root,process.argv.includes('--followup')?'followup_20261025/html_qa':'html_qa');fs.mkdirSync(folder,{recursive:true});
const browser=await chromium.launch({channel:'msedge',headless:true});
const errors=[];
try{
 const page=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1});
 page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(pathToFileURL(path.join(root,'准入报告.html')).href,{waitUntil:'load'});
 const links=await page.locator('a').evaluateAll(nodes=>nodes.map(a=>({href:a.href,label:a.textContent.trim()})));
 const broken=[];
 for(const a of links){
  const url=new URL(a.href);
  if(url.protocol!=='file:')continue;
  const filename=fileURLToPath(url);
  if(!fs.existsSync(filename)){broken.push(a);continue;}
  if(filename.endsWith('准入报告.html')&&url.hash){
   const exists=await page.evaluate(id=>Boolean(document.getElementById(id)),decodeURIComponent(url.hash.slice(1)));
   if(!exists)broken.push(a);
  }
 }
 const desktopOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
 await page.screenshot({path:path.join(folder,'desktop_top.png')});
 await page.locator('#decision').scrollIntoViewIfNeeded();
 await page.screenshot({path:path.join(folder,'desktop_decision.png')});
 await page.locator('[data-filter="motion"]').click();
 const motionRows=await page.locator('#papers-table tbody tr:visible').count();
 await page.locator('[data-filter="all"]').click();
 await page.locator('#search').fill('Chai');
 const searchRows=await page.locator('#papers-table tbody tr:visible').count();
 await page.locator('#search').fill('');
 await page.locator('#ma-record summary').click();
 const maOpen=await page.locator('#ma-record').evaluate(e=>e.open);
 await page.locator('#ma-record').scrollIntoViewIfNeeded();
 await page.screenshot({path:path.join(folder,'desktop_reading.png')});
 await page.locator('#ma-record summary').click();
 await page.locator('#structure-record summary').click();
 const structureOpen=await page.locator('#structure-record').evaluate(e=>e.open);
 await page.locator('#structure-record').scrollIntoViewIfNeeded();
 await page.screenshot({path:path.join(folder,'desktop_followup.png')});
 await page.locator('#structure-record summary').click();
 await page.setViewportSize({width:390,height:844});
 await page.evaluate(()=>scrollTo(0,0));
 await page.screenshot({path:path.join(folder,'mobile_top.png')});
 const mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
 await page.locator('#papers').scrollIntoViewIfNeeded();
 await page.screenshot({path:path.join(folder,'mobile_papers.png')});
 const result={local_link_count:links.filter(a=>a.href.startsWith('file:')).length,broken,errors,
  desktopOverflow,mobileOverflow,motionRows,searchRows,maOpen,structureOpen,rendered_in:'Microsoft Edge / Playwright',
  scope:'Local navigation, filters, expandable records and responsive layout; external links not availability-certified.'};
 result.pass=!broken.length&&!errors.length&&!desktopOverflow&&!mobileOverflow&&motionRows===2&&searchRows===1&&maOpen&&structureOpen;
 fs.writeFileSync(path.join(folder,'checks.json'),JSON.stringify(result,null,2),'utf8');
 console.log(JSON.stringify(result));
 if(!result.pass)process.exitCode=1;
}finally{await browser.close();}
