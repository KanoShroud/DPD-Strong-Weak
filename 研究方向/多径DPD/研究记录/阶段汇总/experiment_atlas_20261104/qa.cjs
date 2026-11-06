const {chromium} = require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('fs');
const path = require('path');
const {pathToFileURL,fileURLToPath} = require('url');
(async()=>{
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 const page=await browser.newPage();
 const errors=[]; page.on('pageerror',e=>errors.push(String(e)));
 const target=path.join(__dirname,'实验场景方法与结果.html');
 await page.goto(pathToFileURL(target).href);
 const checks={};
 for(const width of [1440,390]){
  await page.setViewportSize({width,height:1000});
  checks['no_page_overflow_'+width]=await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1);
  await page.screenshot({path:path.join(__dirname,'preview-'+width+'.png')});
 }
 await page.setViewportSize({width:1440,height:1000});
 await page.locator('#results').scrollIntoViewIfNeeded();
 await page.screenshot({path:path.join(__dirname,'preview-results.png')});
 await page.locator('button').filter({hasText:'展开所有细节'}).click();
 checks.details_open=await page.locator('details').evaluateAll(xs=>xs.every(x=>x.open));
 const urls=await page.locator('a').evaluateAll(xs=>xs.map(x=>x.href));
 checks.files=urls.every(u=>fs.existsSync(fileURLToPath(u)));
 checks.anchors=await page.locator('a[href^="#"]').evaluateAll(xs=>xs.every(x=>document.getElementById(x.getAttribute('href').slice(1))));
 checks.table_count=(await page.locator('table').count())===16;
 checks.errors=errors.length===0;
 fs.writeFileSync(path.join(__dirname,'page_qa.json'),JSON.stringify({checks,errors},null,2),'utf8');
 console.log(checks);
 await browser.close();
 if(!Object.values(checks).every(Boolean))process.exitCode=1;
})();
