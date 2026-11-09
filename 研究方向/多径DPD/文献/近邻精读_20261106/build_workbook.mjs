import fs from 'node:fs/promises';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const root = 'F:/DPD_Strong_Weak/研究方向/多径DPD/文献';
const out = `${root}/近邻精读_20261106`;
const previewDir = 'C:/Users/Administrator/AppData/Local/Temp/dpd_reading_20261106';
const all = JSON.parse(await fs.readFile(`${root}/近邻重审_20261105/文献矩阵.json`, 'utf8'));
const rows = all.filter(r => r.reading_date === '2026-11-06');
if (rows.length !== 69) throw new Error(`Expected 69 readings, found ${rows.length}`);
const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(`${out}/近邻文献精读矩阵.xlsx`));
const levels = [
  ['A','一级_核心定位','#176B69'], ['B','二级_关联与盲分离','#235B88'],
  ['C','三级_不同信息定位','#705787'], ['D','四级_选模与理论','#7D642D'],
];
const widths = [64,350,285,320,330,320,290,340,350,300,270,285,360];
for (const [grade,name,color] of levels) {
  const sh = wb.worksheets.getItem(name);
  const group = rows.filter(r=>r.grade===grade);
  const before = sh.getUsedRange().values.slice(5).filter(r=>r[0]);
  const oldIds = new Set(before.map(r=>r[0]));
  const pending = group.filter(r=>!oldIds.has(r.id));
  const makeRow = e=>{
    const r=e.close_reading;
    return [e.id,`${e.title}\n${e.authors||''}\n${e.year} · ${e.venue||''}`,r.scene,r.observations_priors,r.method,r.theory,r.overlap,r.insight,`${r.baseline_role}\n限制：${r.limits}`,r.experiments,r.grade_reason,r.read_scope,`${r.evidence}\n${e.doi ? 'https://doi.org/'+e.doi : e.url}\nZotero：${e.zotero_keys.join(', ')}`];
  };
  // Preserve existing readings, including any user edits: stop rather than overwrite.
  for (const row of before) {
    const entry=group.find(e=>e.id===row[0]);
    if(!entry || JSON.stringify(row)!==JSON.stringify(makeRow(entry))) throw new Error(`Existing row differs: ${row[0]}`);
  }
  sh.getRange('A1').values = [[`${name.replace('_',' · ')}｜${group.length}篇`]];
  sh.getRange('A2').values = [['2026-11-06 · 69篇正文精读已齐；本次续读第二批12篇、第三批9篇。']];
  const data=pending.map(makeRow);
  if (data.length) sh.tables.items[0].rows.add(null,data);
  data.forEach((row,i)=>{
    const rowNo=before.length+i+6;
    const target=sh.getRange(`A${rowNo}:M${rowNo}`);
    target.copyFrom(sh.getRange(`A${before.length+5}:M${before.length+5}`),'all');
    target.values=[row];
    const lines = Math.max(...row.map((v,j)=>String(v).split('\n').reduce((sum,line)=>sum+Math.max(1,Math.ceil([...line].reduce((n,c)=>n+(c.charCodeAt(0)>255?1:0.53),0)/((widths[j]-16)/14))),0)));
    const height=Math.max(150,lines*21+22);
    target.format.rowHeightPx=height;
    sh.getRange(`A${rowNo}:L${rowNo}`).format.fill=(before.length+i)%2?'#F0F5F8':'#FFFFFF';
    sh.getRange(`M${rowNo}`).format.fill='#F5F6F7';
    sh.getRange(`A${rowNo}`).format.font={name:'Microsoft YaHei',size:11,bold:true,color};
  });
  console.log(`${grade}: preserved ${before.length}, appended ${data.length}`);
}
wb.recalculate();
console.log((await wb.inspect({kind:'sheet',include:'id,name',maxChars:2500})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:20},maxChars:2000})).ndjson);
for(const [g,name] of levels){
  const previews=[['left','A1:D8'],['analysis','G5:J8']];
  if(g==='B') previews.push(['new','C14:F16']);
  if(g==='C') previews.push(['new','G31:J33']);
  if(g==='D') previews.push(['new','J18:M18']);
  for (const [tag,range] of previews) {
    const image=await wb.render({sheetName:name,range,scale:1,format:'png'});
    await fs.writeFile(`${previewDir}/workbook_${g}_${tag}.png`,new Uint8Array(await image.arrayBuffer()));
  }
}
const tail=await wb.render({sheetName:'三级_不同信息定位',range:'I40:M42',scale:1,format:'png'});
await fs.writeFile(`${previewDir}/workbook_C_tail.png`,new Uint8Array(await tail.arrayBuffer()));
const result=await SpreadsheetFile.exportXlsx(wb);
await result.save(`${out}/近邻文献精读矩阵.xlsx`);
console.log(`Exported ${rows.length} papers / ${levels.length} sheets`);
