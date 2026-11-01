/* Apply the declared OOXML colour palette to a generated presentation.
 * Uses only the public JSZip dependency; no workstation-specific helpers.
 */
const fs=require('fs');
const JSZip=require('jszip');

async function applyTheme(file,theme){
  const zip=await JSZip.loadAsync(fs.readFileSync(file));
  const scheme='(?:bg[12]|tx[12]|dk[12]|lt[12]|accent[1-6]|hlink|folHlink)';
  const pattern=new RegExp('<a:srgbClr\\b([^>]*?)val="('+scheme+')"([^>]*?)(?:\\/>|>([\\s\\S]*?)<\\/a:srgbClr>)','g');
  for(const [name,entry] of Object.entries(zip.files)){
    if(entry.dir||!name.endsWith('.xml'))continue;
    let xml=await entry.async('string');
    xml=xml.replace(pattern,(_,before,value,after,children)=>children===undefined
      ?`<a:schemeClr${before}val="${value}"${after}/>`
      :`<a:schemeClr${before}val="${value}"${after}>${children}</a:schemeClr>`);
    if(name==='ppt/theme/theme1.xml'){
      const colors=['dk1','lt1','dk2','lt2','accent1','accent2','accent3','accent4','accent5','accent6','hlink','folHlink'];
      const palette='<a:clrScheme name="BPS Geometry">'+colors.map(k=>`<a:${k}><a:srgbClr val="${theme.colors[k]}"/></a:${k}>`).join('')+'</a:clrScheme>';
      if(!/<a:clrScheme\b/.test(xml))throw new Error('Theme colour scheme is missing');
      xml=xml.replace(/<a:clrScheme\b[^>]*>[\s\S]*?<\/a:clrScheme>/,palette);
    }
    zip.file(name,xml);
  }
  fs.writeFileSync(file,await zip.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
}
module.exports={applyTheme};
