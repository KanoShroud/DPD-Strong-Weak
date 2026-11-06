import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {Workbook} from '@oai/artifact-tool';

const out = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(out, '../../..');
const additions = JSON.parse(await fs.readFile(path.join(out, 'catalog_additions.json'), 'utf8'));
const files = ['文献精读结果.csv', '参考文献编号对照.csv'];
const quote = x => '"'+String(x ?? '').replaceAll('"', '""')+'"';
const digest = x => crypto.createHash('sha256').update(x).digest('hex');
const audit = [];
await fs.mkdir(path.join(out, 'catalog_before'), {recursive: true});
for (const name of files) {
  const target = path.join(root, name);
  const before = await fs.readFile(target);
  assert.equal(before.subarray(0, 3).toString('hex'), 'efbbbf');
  const text = before.toString('utf8').replace(/^\uFEFF/, '');
  assert(text.endsWith('\r\n'));
  const wb = await Workbook.fromCSV(text, {sheetName: '文献目录'});
  const sheet = wb.worksheets.getItem('文献目录');
  const previous = sheet.getUsedRange().values.filter(row => row[0] !== null && row[0] !== '');
  assert.equal(previous.length, 117);
  const headers = previous[0];
  assert(!previous.some(row => row.includes('R117')));
  const beforePreview = await wb.render({sheetName: sheet.name, range: 'A1:D3', scale: 1, format: 'png'});
  await fs.writeFile(path.join(out, name+'.before.png'), new Uint8Array(await beforePreview.arrayBuffer()));
  const rows = additions.map(item => headers.map(key => item[key] ?? ''));
  sheet.getRangeByIndexes(previous.length, 0, 3, headers.length).values = rows;
  wb.recalculate();
  const authored = sheet.getRangeByIndexes(previous.length, 0, 3, headers.length).values;
  assert.deepEqual(authored, rows);
  const preview = await wb.render({sheetName: sheet.name, range: 'A118:D120', scale: 1, format: 'png'});
  await fs.writeFile(path.join(out, name+'.after.png'), new Uint8Array(await preview.arrayBuffer()));
  // Serialize only the new authored rows. The 116 existing records remain byte-for-byte intact.
  const after = Buffer.concat([before, Buffer.from(authored.map(row=>row.map(quote).join(',')).join('\r\n')+'\r\n', 'utf8')]);
  const reopened = await Workbook.fromCSV(after.toString('utf8').replace(/^\uFEFF/, ''), {sheetName:'回读'});
  const actual = reopened.worksheets.getItem('回读').getUsedRange().values.filter(row=>row[0]!==null && row[0]!=='');
  assert.deepEqual(actual, [...previous, ...rows]);
  const backup = path.join(out, 'catalog_before', name);
  await fs.writeFile(backup, before, {flag:'wx'});
  assert((await fs.readFile(target)).equals(before), 'File changed during authoring');
  await fs.writeFile(target, after);
  assert((await fs.readFile(target)).subarray(0, before.length).equals(before));
  audit.push({file:name, before_sha256:digest(before), after_sha256:digest(after), old_records:116,
    new_records:119, columns:headers.length, original_bytes_unchanged:true, artifact_value_readback:true,
    utf8_bom:true, line_endings:'CRLF', added:['R117','R118','R119']});
}
await fs.writeFile(path.join(out, 'catalog_validation.json'), JSON.stringify(audit, null, 2), 'utf8');
console.log(JSON.stringify(audit));
