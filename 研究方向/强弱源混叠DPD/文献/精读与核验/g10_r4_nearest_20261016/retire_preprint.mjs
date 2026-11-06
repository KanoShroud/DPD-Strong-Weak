import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {Workbook} from '@oai/artifact-tool';

const out=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(out,'../../..');
const auditDir=path.resolve(out,'../qnomp_catalog_cleanup_20261016');
const quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const replacement='NOMP联合精修的重要近邻，统一引用2025年期刊版。';
const audit=[];
for(const file of ['文献精读结果.csv','参考文献编号对照.csv']) {
  const target=path.join(root,file), before=await fs.readFile(target);
  const wb=await Workbook.fromCSV(before.toString('utf8').replace(/^\uFEFF/,''),{sheetName:'文献目录'});
  const sheet=wb.worksheets.getItem('文献目录');
  const values=sheet.getUsedRange().values.filter(row=>row[0]!==null && row[0]!=='');
  assert.equal(values.length,120);
  const h=values[0], id=h.indexOf('编号');
  assert.equal(values[119][id],'R119');
  assert.equal(values[118][id],'R118');
  // CSV has no presentation formatting. These widths affect verification previews only.
  for(const col of ['A','B']) sheet.getRange(col+'1:'+col+'120').format.columnWidth=12;
  for(const col of ['C','D']) sheet.getRange(col+'1:'+col+'120').format.columnWidth=29;
  const imageBefore=await wb.render({sheetName:sheet.name,range:'A117:D120',scale:1,format:'png'});
  await fs.writeFile(path.join(auditDir,file+'.before.png'),new Uint8Array(await imageBefore.arrayBuffer()));
  sheet.getRangeByIndexes(119,0,1,h.length).clear({applyTo:'contents'});
  const field=h.indexOf('总体评价');
  if(field>=0) sheet.getRangeByIndexes(118,field,1,1).values=[[replacement]];
  wb.recalculate();
  const finalRows=sheet.getRangeByIndexes(0,0,119,h.length).values;
  const expected=values.slice(0,119).map(row=>[...row]);
  if(field>=0)expected[118][field]=replacement;
  assert.deepEqual(finalRows,expected);
  const imageAfter=await wb.render({sheetName:sheet.name,range:'A117:D119',scale:1,format:'png'});
  await fs.writeFile(path.join(auditDir,file+'.after.png'),new Uint8Array(await imageAfter.arrayBuffer()));
  const lines=before.toString('utf8').split('\r\n');
  assert.equal(lines.length,121); // Header + 119 single-line records + trailing newline.
  assert.equal(lines[120],'');
  const unchangedPrefix=lines.slice(0,118).join('\r\n')+'\r\n';
  const after=Buffer.from(unchangedPrefix+finalRows[118].map(quote).join(',')+'\r\n','utf8');
  assert.equal(after.subarray(0,3).toString('hex'),'efbbbf');
  const reopened=await Workbook.fromCSV(after.toString('utf8').replace(/^\uFEFF/,''),{sheetName:'回读'});
  assert.deepEqual(reopened.worksheets.getItem('回读').getUsedRange().values.filter(row=>row[0]!==null && row[0]!==''),expected);
  assert((await fs.readFile(target)).equals(before),'Unexpected concurrent edit');
  await fs.writeFile(target,after);
  audit.push({file,before_sha256:sha(before),after_sha256:sha(after),records:118,
    unaffected_records_byte_preserved:true,removed_item:'R119',updated_reference:'R118',
    artifact_readback:true,encoding:'UTF-8 BOM',line_endings:'CRLF'});
}
const catalogPath=path.join(out,'catalog_additions.json');
const additions=JSON.parse(await fs.readFile(catalogPath,'utf8')).filter(row=>row['编号']!=='R119');
assert.equal(additions.length,2);
additions.find(row=>row['编号']==='R118')['总体评价']=replacement;
await fs.writeFile(catalogPath,JSON.stringify(additions,null,2)+'\n','utf8');
const manifestPath=path.join(out,'source_manifest.json');
const manifest=JSON.parse(await fs.readFile(manifestPath,'utf8'));
delete manifest.R119;
assert.deepEqual(Object.keys(manifest),['R117','R118']);
await fs.writeFile(manifestPath,JSON.stringify(manifest,null,2)+'\n','utf8');
await fs.writeFile(path.join(auditDir,'catalog_cleanup_audit.json'),JSON.stringify(audit,null,2)+'\n','utf8');
console.log(JSON.stringify(audit));
