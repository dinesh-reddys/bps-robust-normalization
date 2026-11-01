/* Editable PowerPoint generated from completed BTP experiment outputs. */
const fs=require('fs');
const path=require('path');
const pptxgen=require('pptxgenjs');
const {applyTheme}=require('./pptx_theme.js');
const root=path.resolve(__dirname,'..');
const summary=JSON.parse(fs.readFileSync(path.join(root,'results/summary.json')));
const project=JSON.parse(fs.readFileSync(path.join(root,'project.json')));
const pres=new pptxgen(); pres.layout='LAYOUT_WIDE';
pres.author=project.author;pres.subject='BTP 1 study of normalization for BPS classification';
pres.title='Effect of robust normalization on BPS based 3D object classification';
pres.company='Indian Institute of Technology Kharagpur';pres.lang='en-IN';
const THEME={name:'BPS Geometry',headFontFace:'Cambria',bodyFontFace:'Arial',colors:{dk1:'182A24',lt1:'FFFFFF',dk2:'FFFFFF',lt2:'182A24',accent1:'14816A',accent2:'BD6B2B',accent3:'4D6475',accent4:'DDECE6',accent5:'A7B9B0',accent6:'B33030',hlink:'14816A',folHlink:'4D6475'}};
pres.theme={headFontFace:THEME.headFontFace,bodyFontFace:THEME.bodyFontFace};
const C=pres.SchemeColor;
const P={A:'4D6475',B:'BD6B2B',C:'14816A'};
for(const [name,dark,titleSize,titleY,titleH] of [['CONTENT',false,36,.38,1.05],['TITLE',true,40,.8,2.0],['DARK',true,36,.38,1.05]]){
 pres.defineSlideMaster({title:name,background:{color:dark?C.background2:C.background1},
  objects:[{placeholder:{options:{name:'title',type:'title',x:.62,y:titleY,w:12.05,h:titleH,fontSize:titleSize,bold:true,color:dark?C.text2:C.text1,margin:0,breakLine:false}}},
   {text:{text:'Somavarapu Dinesh Reddy  |  23EC39025',options:{x:.62,y:7.10,w:8,h:.16,fontSize:9,color:dark?C.accent5:C.accent3,margin:0,isTextBox:true}}}],
  slideNumber:{x:12.1,y:7.06,w:.5,h:.25,fontSize:9,color:dark?C.text2:C.text1,margin:0}});
}
const timing=[20,50,60,55,55,60,75,65,60,40];
const notes=[];
function text(slide,value,x,y,w,h,size=20,extra={}){slide.addText(value,{x,y,w,h,fontSize:size,color:C.text1,margin:0,breakLine:false,isTextBox:true,...extra});}
function box(slide,label,x,y,w,h,extra={}){slide.addShape(pres.ShapeType.rect,{x,y,w,h,fill:{color:C.accent4},line:{color:C.accent4},...extra});text(slide,label,x+.18,y+.16,w-.36,h-.32,22,{bold:true,align:'center',valign:'mid'});}
function line(slide,x1,y1,x2,y2){slide.addShape(pres.ShapeType.line,{x:x1,y:y1,w:x2-x1,h:y2-y1,line:{color:C.accent1,width:2,beginArrowType:'none',endArrowType:'triangle'}});}
function source(slide,value){text(slide,value,.64,6.72,11.9,.22,10,{color:C.accent3});}
function image(slide,name,x,y,w,h){const file=path.join(root,'results/figures',name+'.png');const bytes=fs.readFileSync(file);const iw=bytes.readUInt32BE(16),ih=bytes.readUInt32BE(20);const scale=Math.min(w/iw,h/ih),dw=iw*scale,dh=ih*scale;slide.addImage({path:file,x:x+(w-dw)/2,y:y+(h-dh)/2,w:dw,h:dh,altText:name.replaceAll('_',' '),objectName:name});}
let section='Problem and method';pres.addSection({title:section});
function slide(title,layout='CONTENT'){let s=pres.addSlide({masterName:layout,sectionTitle:section});s.addText(title,{placeholder:'title'});return s;}
function speaker(s,index,script){const item={slide:index+1,seconds:timing[index],script};notes.push(item);s.addNotes(`Target ${timing[index]} seconds.\n\n${script}`);}
const lookup=(p,f,r)=>summary.overall_accuracy.find(v=>v.pipeline===p&&v.fraction===f&&v.radius===r);
const percent=x=>(100*x).toFixed(2);

let s=slide('Normalization for BPS based 3D object classification','TITLE');
text(s,'A controlled study of distant outliers',.65,3.05,11.8,.6,25,{color:C.text2});
text(s,'SOMAVARAPU DINESH REDDY  |  23EC39025',.65,4.0,11.8,.35,19,{color:C.text2});
text(s,'Supervisor  Prof. Prabir Kumar Biswas\nE&ECE  ·  IIT Kharagpur',.65,4.55,11.6,.75,19,{color:C.accent5});
image(s,'cover_object',9.25,3.05,3.55,3.55);
speaker(s,0,'My BTP study examines how normalization affects 3D object classification when point clouds contain distant outliers. I use the Basis Point Sets representation and a small multilayer perceptron, then compare three preprocessing rules.');

s=slide('A distant point can change the object scale');
box(s,'Clean object\nRadius about 1',.8,1.85,3.25,1.35);
box(s,'One far point\nMaximum radius grows',5.2,1.85,3.25,1.35);
line(s,4.15,2.52,5.0,2.52);
text(s,'Maximum based scaling shrinks the genuine object relative to the fixed BPS basis.',.85,3.8,10.8,1.15,27);
text(s,'Question: can a robust scale estimate preserve classification accuracy?',.85,5.38,11.7,.7,23,{color:C.accent1,bold:true});
source(s,'Motivation: Prokudin et al., ICCV 2019, supplementary discussion of normalization sensitivity');
speaker(s,1,'The original preprocessing subtracts the mean of the cloud and divides coordinates by the largest distance from that centre. If an outlier lies five times farther away, the genuine object can become approximately five times smaller after normalization. BPS measures distances from fixed reference locations, so this changes many features together. The authors already identify this normalization sensitivity in their supplementary discussion. My question is whether a specific robust normalization rule reduces the classification damage, and what clean performance or failure limits accompany it.');

s=slide('BPS turns points into a fixed feature vector');
image(s,'bps_explanation',.65,1.62,6.35,4.5);
text(s,'512 fixed reference points',7.35,1.95,5.2,.6,25,{bold:true});
text(s,'For each reference point, record the nearest distance to the object.',7.35,2.85,5.1,1.25,23);
text(s,'Point order does not change the minimum distance.',7.35,4.55,5.1,1.0,23,{color:C.accent1});
source(s,'Prokudin, Lassner and Romero, ICCV 2019  |  Diagram is a 2D illustration; the implementation uses 3D coordinates');
speaker(s,2,'A point cloud is an unordered collection of xyz coordinates. BPS samples one fixed set of reference points and uses exactly that set for every object. For each reference point, I find the nearest cloud point and store the Euclidean distance. With 512 reference points, every cloud becomes a vector of 512 numbers. The diagram is a two-dimensional illustration, but the implementation is three-dimensional. Reordering the object points does not change the minimum over the set, so the representation is permutation invariant. It is not generally rotation invariant because the reference locations remain fixed.');

s=slide('A small MLP predicts the object category');
const layers=[['512 distances',.7],['512 hidden units',3.85],['512 hidden units',7.0],['40 class scores',10.15]];
for(let i=0;i<layers.length;i++){box(s,layers[i][0],layers[i][1],2.05,2.5,1.35);if(i<3)line(s,layers[i][1]+2.57,2.72,layers[i+1][1]-.12,2.72);}
text(s,'Weighted sums → ReLU → batch normalization → dropout',.75,4.0,11.8,.7,24);
text(s,'Train with cross entropy and Adam\nPredict the class with the largest output score',.75,5.05,11.8,1.0,22);
source(s,'Architecture adapted from the public BPS MLP example; identical dimensions and training settings across A, B and C');
speaker(s,3,'The classifier extends the perceptron idea using two hidden layers. Each linear layer computes weighted sums, and ReLU introduces a nonlinearity. Batch normalization and dropout follow the author example. The output has 40 scores, one for each category. Cross entropy measures classification error during training, backpropagation computes gradients, and Adam updates the weights. At inference, dropout is disabled and the largest output score determines the category. The BPS reference points are fixed rather than learned. All three pipelines have the same classifier architecture; each is trained using its own preprocessing.');

s=slide('Three pipelines isolate centring and scaling');
s.addTable([['Pipeline','Centre','Scale'].map(value=>({text:value,options:{fill:C.accent1,color:C.text2,bold:true}})),['A  Original','Mean','Maximum radius'],['B  Centring control','Coordinate wise median','Maximum radius'],['C  Robust variant','Coordinate wise median','95th percentile radius']],{x:.75,y:1.8,w:11.8,h:2.55,colW:[3.4,4.4,4.0],fontSize:21,color:C.text1,border:{type:'solid',color:'D9D9D9',pt:1},fill:C.background1,margin:.16,bold:false,autoPage:false,rowH:.63,
  ...{},});
text(s,'A → B tests centring\nB → C tests scaling',.85,4.85,6.1,1.0,25,{bold:true,color:C.accent1});
text(s,'All points are retained.\nThe percentile rule does not remove outliers.',7.1,4.85,5.1,1.0,21);
source(s,'All models use the same saved BPS basis; the 95th percentile is fixed before final test evaluation');
speaker(s,4,'Pipeline A uses the original mean centre and maximum radius. B changes only the centre to the coordinate-wise median. C also replaces maximum radius by the 95th percentile of radii. A versus B tells us whether centring alone helps. B versus C isolates the scale change. The percentile can ignore a small upper tail of extreme radii when estimating the object scale. However, all points remain in the cloud. This is a preprocessing investigation motivated by the source paper, not a claim that robust normalization is a new general algorithm.');

s=slide('Models are selected before test evaluation');
for(const [x,n,label] of [[.8,'7,872','Fitting objects'],[4.95,'1,968','Validation objects'],[9.1,'2,468','Test objects']]){box(s,n,x,1.85,3.35,1.1);text(s,label,x,3.15,3.35,.5,21,{align:'center'});}
text(s,'ModelNet40  ·  2048 points per object  ·  40 categories',.8,4.12,11.8,.5,23,{bold:true});
text(s,'3 training seeds × 3 corruption draws\nReplacement: 1%, 2%, 5%, 10%  |  Distances: 3×, 5×, 10×',.8,5.05,11.8,1.0,22);
source(s,'PointNet HDF5 conversion: 9840 training clouds; validation is drawn only from that split. Matched corruptions across pipelines.');
speaker(s,5,'The processed ModelNet40 archive contains 9840 training clouds and 2468 test clouds. I reserve 1968 training objects for validation and fit on 7872. Validation chooses checkpoints; the test set is used only after training. Each pipeline is trained with three seeds. Evaluation replaces selected points with points on a distant spherical shell, keeping the total point count fixed. I test four replacement fractions and three distances, with three corruption draws. The same corrupted objects are supplied to all pipelines. This is a controlled diagnostic protocol, not the official PointCloud-C benchmark or realistic sensor noise.');

section='Measured results';pres.addSection({title:section});
s=slide('What happens to classification accuracy');
text(s,'Accuracy (%)',.72,1.47,3,.22,12,{color:C.accent3});
const labels=['Clean','2% at radius 5','10% at radius 5'];
s.addChart(pres.ChartType.bar,['A','B','C'].map(p=>({name:p,labels,values:[lookup(p,0,0).mean*100,lookup(p,.02,5).mean*100,lookup(p,.1,5).mean*100]})),{x:.65,y:1.7,w:8.6,h:4.6,barDir:'col',grouping:'clustered',showLegend:true,legendPos:'b',legendFontSize:12,showTitle:false,showValue:true,dataLabelPosition:'outEnd',dataLabelFormatCode:'0.0',dataLabelColor:'182A24',dataLabelBkgrdColor:'FFFFFF',dataLabelFormatCodeSourceLinked:false,chartColors:[P.A,P.B,P.C],showCatName:false,catAxisLabelFontSize:12,catAxisLabelColor:'182A24',valAxisLabelFontSize:11,valAxisLabelColor:'182A24',valAxisMinVal:0,valAxisMaxVal:100,valAxisTitle:'Accuracy (%)',showValue:true,showBorder:false,showCatName:false,valGridLine:{color:'D9D9D9',width:1},catGridLine:{style:'none'},showCatName:false,showSerName:false,showPercent:false,showLeaderLines:false,showValue:true,legendFontFace:'+mn-lt',catAxisLabelFontFace:'+mn-lt',valAxisLabelFontFace:'+mn-lt',dataLabelFontFace:'+mn-lt',dataLabelFontSize:10});
const gain=100*(lookup('C',.02,5).mean-lookup('A',.02,5).mean);
text(s,(gain>=0?'+':'')+gain.toFixed(2)+' pp',9.65,2.15,3.0,.85,38,{bold:true,color:C.accent1});
text(s,'C minus A\n2% replacement\nRadius 5',9.65,3.35,3.0,1.65,23);
source(s,'Measured test means; corruption draws averaged within training seeds. Report contains SD across seeds and the full distance sweep.');
speaker(s,6,`The clean accuracies are ${percent(lookup('A',0,0).mean)} percent for A, ${percent(lookup('B',0,0).mean)} for B and ${percent(lookup('C',0,0).mean)} for C. At two-percent replacement and radius five, the corresponding accuracies are ${percent(lookup('A',.02,5).mean)}, ${percent(lookup('B',.02,5).mean)} and ${percent(lookup('C',.02,5).mean)} percent. C changes accuracy relative to A by ${gain.toFixed(2)} percentage points. The right group shows the ten-percent condition, which is important for interpreting the method's limit. These bars show means; the report also gives training-seed standard deviations and the complete radius sweep. The comparison does not establish performance against other architectures or real sensor corruptions.`);

s=slide('Geometry explains the prediction changes');
const geometry=JSON.parse(fs.readFileSync(path.join(root,'results/geometry_summary.json')));
for(const [x,key,title,maxValue] of [[.65,'inlier_radius_ratio','Main object size ratio',1.1],[6.95,'feature_drift','Relative BPS feature drift',.55]]){
 text(s,title,x,1.65,5.7,.4,21,{bold:true});
 s.addChart(pres.ChartType.bar,['A','B','C'].map(p=>({name:p,labels:['0%','1%','2%','5%','10%'],values:[0,.01,.02,.05,.1].map(f=>geometry.find(v=>v.pipeline===p&&v.fraction===f)[key])})),{x,y:2.18,w:5.7,h:3.95,barDir:'col',grouping:'clustered',showLegend:true,legendPos:'b',legendFontSize:11,showValue:false,chartColors:[P.A,P.B,P.C],catAxisLabelFontSize:12,valAxisLabelFontSize:11,catAxisLabelColor:'182A24',valAxisLabelColor:'182A24',valAxisMinVal:0,valAxisMaxVal:maxValue,valGridLine:{color:'D9D9D9',width:1},catGridLine:{style:'none'},showBorder:false,legendFontFace:'+mn-lt',catAxisLabelFontFace:'+mn-lt',valAxisLabelFontFace:'+mn-lt'});
}
source(s,'Radius-five diagnostic. Main-object size is measured on retained inliers; BPS feature drift is measured before classification.');
speaker(s,7,'The geometric diagnostics help explain the accuracy results. On the left, I measure the size of the genuine retained object after normalization relative to its clean size. A value close to one means that its scale is preserved. With a far outlier, maximum-based scaling can produce a value near one fifth at radius five. The right plot measures how much the 512-dimensional distance vector changes. Together these measurements connect preprocessing to the classifier input. They do not prove that every changed feature causes an error, but they support the proposed scale mechanism independently of the predicted labels.');

s=slide('Percentile scaling has an operating limit');
text(s,'Heavy contamination can enter the percentile itself',.8,1.9,11.7,.75,27,{bold:true});
box(s,'10% extreme points',.8,3.05,3.3,1.15);line(s,4.25,3.61,4.85,3.61);box(s,'95th percentile\ncan be an outlier radius',5.05,3.05,4.0,1.15);line(s,9.2,3.61,9.8,3.61);box(s,'Object shrinks',10.0,3.05,2.55,1.15);
text(s,'Nearby clutter can also distort nearest distances.\nNormalization does not remove those points.',.85,5.1,11.7,1.0,24,{color:C.accent1});
source(s,'Observed high-contamination results are reported alongside successful conditions; no universal robustness threshold is claimed.');
speaker(s,8,'The percentile rule is not immune to contamination. If enough points have extreme radii, the 95th percentile can itself be determined by outliers, and the genuine object can shrink again. The exact transition depends on finite sampling and the radial distribution. There is also a second mechanism: an outlier near a reference point can directly become its nearest neighbour and change that feature. Normalization does not remove it. My distant-shell experiment mainly investigates scale sensitivity. Nearby clutter, Gaussian jitter, missing regions and real scans remain separate tests.');

section='Conclusions';pres.addSection({title:section});
s=slide('Findings and next experiments','DARK');
text(s,`At radius 5, C retains ${percent(lookup('C',.02,5).mean)}% with 2% outliers\nbut falls to ${percent(lookup('C',.1,5).mean)}% with 10% outliers`,.7,1.95,11.8,1.1,29,{color:C.text2,bold:true});
text(s,'Completed  BPS classifier · A/B/C ablation · repeated-seed evaluation',.7,3.65,11.8,.75,22,{color:C.text2});
text(s,'Next  Outlier filtering · noise augmentation · broader corruptions',.7,4.7,11.8,.75,22,{color:C.accent5});
text(s,'References: Prokudin et al., ICCV 2019; Qi et al., CVPR 2017;\nRen et al., ICML 2022; Wu et al., CVPR 2015.',.7,5.95,11.8,.6,12,{color:C.accent5});
speaker(s,9,`The main result is conditional. At radius five, C retains ${percent(lookup('C',.02,5).mean)} percent accuracy with two-percent replacement, but falls to ${percent(lookup('C',.1,5).mean)} percent with ten-percent replacement. Its clean accuracy is about 0.8 percentage points below A. The contribution is the implemented comparison and explanation of an acknowledged failure mechanism. The next stage is outlier filtering, followed by augmentation and broader corruptions. Thank you.`);

(async()=>{
 const dest=path.join(root,'deliverables');fs.mkdirSync(dest,{recursive:true});
 const file=path.join(dest,'BTP1_Presentation.pptx');
 await pres.writeFile({fileName:file});await applyTheme(file,THEME);
 if(notes.reduce((a,b)=>a+b.seconds,0)!==540)throw new Error('Presentation timing must be 540 seconds');
 fs.writeFileSync(path.join(root,'results/presentation_notes.json'),JSON.stringify(notes,null,2));
 fs.writeFileSync(path.join(dest,'Presentation_Timing.md'),'# Presentation timing\n\nTarget: 9 minutes. Strict limit: 10 minutes. Q&A: 2 minutes, separate.\n\n'+notes.map(n=>`## Slide ${n.slide} — ${n.seconds} seconds\n\n${n.script}\n`).join('\n'));
 console.log('Created 10-slide editable deck; speaker target 540 seconds.');
})().catch(e=>{console.error(e);process.exit(1)});
