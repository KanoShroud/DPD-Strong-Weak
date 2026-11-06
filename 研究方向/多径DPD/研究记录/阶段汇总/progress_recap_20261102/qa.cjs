const {chromium}=require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs');const path=require('path');const {pathToFileURL,fileURLToPath}=require('url');
(async()=>{
 const dest=__dirname;const target=path.join(dest,'最近几轮研究进展.html');
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(pathToFileURL(target).href);await page.waitForLoadState('load');
 const checks={};
 for(const width of [1280,390,320]){
  await page.setViewportSize({width,height:950});
  checks['no_horizontal_overflow_'+width]=await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1);
  await page.screenshot({path:path.join(dest,'preview-'+width+'.png'),fullPage:false});
 }
 await page.setViewportSize({width:1280,height:950});
 await page.locator('#results').scrollIntoViewIfNeeded();
 await page.screenshot({path:path.join(dest,'preview-results.png')});
 const detail=page.locator('#results details');await detail.locator('summary').click();
 checks.details_open=await detail.getAttribute('open')!==null;
 checks.table_rows=await page.locator('#results tbody tr').count()===7;
 await page.evaluate(()=>window.print=()=>{window.__printed=true});await page.locator('#print').click();
 checks.print_action=await page.evaluate(()=>window.__printed===true);
 const links=await page.locator('a').evaluateAll(a=>a.map(x=>x.href));
 checks.local_links=links.filter(x=>x.startsWith('file:')).every(x=>fs.existsSync(fileURLToPath(new URL(x))));
 checks.anchors=await page.evaluate(()=>[...document.querySelectorAll('a[href^="#"]')].every(a=>document.getElementById(a.getAttribute('href').slice(1))));
 checks.no_script_errors=errors.length===0;
 fs.writeFileSync(path.join(dest,'qa.json'),JSON.stringify({checks,errors,passed:Object.values(checks).every(Boolean)},null,2),'utf8');
 console.log(checks);await browser.close();
 if(!Object.values(checks).every(Boolean))process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1});
