import {Workbook} from '@oai/artifact-tool';
const w=Workbook.create();
console.log(w.help('CSV export',{search:'csv',include:'index,examples,notes',maxChars:2500}).ndjson);
