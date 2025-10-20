/* Inventory Demand Forecasting — supports `const dataset = [...]` with columns like:
   Date, Store ID, Product ID, Region, Category, Units Sold, Units Ordered, Demand Forecast, etc.

   Features:
   - Column mapping UI (Date, Target) with auto-detect (tolerant to spaces/case/punctuation).
   - Filters for Store/Product/Region; aggregate by date (sum/mean/last).
   - EDA: time series + rolling mean, histogram, day-of-week averages, ACF(1–30).
   - Model: GRU/LSTM via TensorFlow.js, z-score scaling, 80/20 chronological split.
   - Prediction: rolling one-step horizon; Evaluation: MAE, RMSE, MAPE on test set.
*/

// ---------- Globals ----------
let RAW_ROWS = [];          // as loaded from file or data.js (not filtered)
let FILTERED_ROWS = [];     // after applying filters
let SERIES = [];            // aggregated numeric series for target
let DATES = [];             // matching Date objects for SERIES
let MEAN = 0, STD = 1;
let MODEL = null;
let TRAIN_SPLIT_INDEX = null;
let CHARTS = {};
let HISTORY_LOSS = [];

// UI els
const E = id => document.getElementById(id);
const els = {
  dateCol:    () => E('dateCol'),
  targetCol:  () => E('targetCol'),
  aggFunc:    () => E('aggFunc'),
  store:      () => E('storeFilter'),
  product:    () => E('productFilter'),
  region:     () => E('regionFilter'),
};

function log(msg){
  const time = new Date().toLocaleTimeString();
  const logEl = E('log');
  logEl.innerHTML += `[${time}] ${msg}<br/>`;
  logEl.scrollTop = logEl.scrollHeight;
}

// ---------- Utils ----------
const norm = k => (k||'').toString().toLowerCase().replace(/[^a-z0-9]/g,'');
const tryParseNum = v => {
  if (typeof v === 'number') return v;
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
};
const asDate = v => {
  const d = new Date(v);
  return isNaN(d.getTime()) ? null : d;
};
function basicStats(arr){
  const n = arr.length || 1;
  const mean = arr.reduce((a,b)=>a+b,0)/n;
  const variance = arr.reduce((s,x)=>s+(x-mean)**2,0)/n;
  const std = Math.sqrt(variance);
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  return {mean,std,min,max};
}
function zfit(arr){ const s=basicStats(arr); MEAN=s.mean; STD=s.std||1; }
const z = arr => arr.map(x => (x-MEAN)/STD);
const iz = arr => arr.map(x => x*STD + MEAN);

function histogram(arr, bins=24){
  const min = Math.min(...arr), max = Math.max(...arr);
  const step = (max - min) / (bins||1) || 1;
  const edges = Array.from({length:bins+1},(_,i)=>min + i*step);
  const counts = new Array(bins).fill(0);
  for (const v of arr){
    let idx = Math.min(Math.floor((v - min)/step), bins-1);
    if (idx < 0) idx = 0;
    counts[idx]++;
  }
  const labels = counts.map((_,i)=> (edges[i]).toFixed(1) + "–" + (edges[i+1]).toFixed(1));
  return {labels, counts};
}
function rollingMean(arr, w=7){
  const out = [];
  for (let i=0;i<arr.length;i++){
    const s = Math.max(0, i-w+1);
    const slice = arr.slice(s, i+1);
    out.push(slice.reduce((a,b)=>a+b,0)/slice.length);
  }
  return out;
}
function dayOfWeekAverages(dates, series){
  const sums = [0,0,0,0,0,0,0], counts = [0,0,0,0,0,0,0];
  for (let i=0;i<series.length;i++){
    const d = dates[i];
    if (!(d instanceof Date)) continue;
    const dow = d.getDay(); // 0=Sun..6=Sat
    sums[dow] += series[i];
    counts[dow]++;
  }
  const avgs = sums.map((s,i)=> counts[i] ? s/counts[i] : 0);
  const labels = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  return {labels, avgs};
}
function acf(arr, maxLag=30){
  const n = arr.length;
  const mean = arr.reduce((a,b)=>a+b,0)/n;
  const denom = arr.reduce((s,x)=> s + (x-mean)*(x-mean), 0) || 1;
  const out = [];
  for (let lag=1;lag<=maxLag;lag++){
    let num = 0;
    for (let t=lag;t<n;t++){
      num += (arr[t]-mean)*(arr[t-lag]-mean);
    }
    out.push(num/denom);
  }
  return out;
}

// ---------- Column detection & mapping ----------
function detectColumns(rows){
  if (!rows.length) return {dateKey:null, targetKey:null};
  const names = Object.keys(rows[0]);
  const normalized = names.map(norm);

  // candidates (normalized)
  const dateCands   = ['date','ds','day','timestamp','time'];
  const targetCands = [
    'demand','qty','quantity','sales','units','unitssold','sold',
    'demandforecast','orders','unitsordered','consumption'
  ];

  const pick = (cands) => {
    for (const c of cands){
      const idx = normalized.indexOf(c);
      if (idx !== -1) return names[idx];
    }
    // fallback fuzzy: contains substring
    for (const [i, n] of normalized.entries()){
      if (cands.some(c => n.includes(c))) return names[i];
    }
    return null;
  };

  const dateKey = pick(dateCands);
  let targetKey = pick(targetCands);

  // Heuristic: if both "unitssold" and "demandforecast" exist, prefer Units Sold as target
  const hasUnitsSold = normalized.includes('unitssold');
  const hasForecast  = normalized.includes('demandforecast');
  if (hasUnitsSold) targetKey = names[normalized.indexOf('unitssold')];
  else if (!targetKey && hasForecast) targetKey = names[normalized.indexOf('demandforecast')];

  return {dateKey, targetKey};
}

// ---------- UI population ----------
function populateSelect(select, options, selected=null){
  select.innerHTML = '';
  for (const opt of options){
    const o = document.createElement('option');
    o.value = opt.value;
    o.textContent = opt.label;
    if (selected !== null && selected === opt.value) o.selected = true;
    select.appendChild(o);
  }
}

function uniqueValues(rows, key){
  const set = new Set();
  for (const r of rows){
    if (r[key] !== undefined && r[key] !== null && r[key] !== '') set.add(String(r[key]));
  }
  return Array.from(set).sort();
}

// ---------- Data loading ----------
async function loadFromDataJS(){
  // Supports window.INVENTORY_DATA or top-level `dataset` (let/const) from data.js
  if (typeof window !== 'undefined' && typeof window.INVENTORY_DATA !== 'undefined'){
    return window.INVENTORY_DATA;
  }
  // Try top-level binding `dataset`
  try {
    // eslint-disable-next-line no-undef
    if (typeof dataset !== 'undefined') return dataset;
  } catch(e){/* ignore */}
  log("No global data detected; using synthetic demo data.");
  return syntheticData();
}

function syntheticData(n=365){
  const start = new Date("2024-01-01");
  const rows = [];
  for (let i=0;i<n;i++){
    const d = new Date(start.getTime() + i*24*3600*1000);
    const dow = d.getDay();
    const seasonal = 80 + 10*Math.sin(2*Math.PI*i/7) + (dow===6?12:0);
    const trend = 0.05*i;
    const y = Math.max(0, seasonal + trend + (Math.random()-0.5)*10);
    rows.push({
      "Date": d.toISOString().slice(0,10),
      "Store ID": "S001",
      "Product ID": "P0001",
      "Region": "North",
      "Units Sold": Number(y.toFixed(2)),
      "Price": 30 + Math.sin(i/20)*2,
    });
  }
  return rows;
}

function parseCSV(text){
  const lines = text.trim().split(/\r?\n/);
  const header = lines.shift().split(',').map(s=>s.trim());
  const out = [];
  for (const line of lines){
    if (!line.trim()) continue;
    const cols = line.split(',').map(s=>s.trim());
    const obj = {};
    header.forEach((h,i)=> obj[h] = cols[i]);
    out.push(obj);
  }
  return out;
}

function toTable(rows, mountId="tableMount", limit=200){
  const mount = E(mountId);
  if (!rows.length){ mount.innerHTML = ""; return; }
  const keys = Object.keys(rows[0]);
  let html = `<table><thead><tr>${keys.map(k=>`<th>${k}</th>`).join("")}</tr></thead><tbody>`;
  for (let i=0;i<Math.min(limit, rows.length);i++){
    const r = rows[i];
    html += `<tr>${keys.map(k=>`<td>${(r[k]??'')}</td>`).join("")}</tr>`;
  }
  html += "</tbody></table>";
  mount.innerHTML = html;
}

function summarize(rows, dateKey, targetKey){
  const n = rows.length;
  const first = rows[0]?.[dateKey];
  const last  = rows[n-1]?.[dateKey];
  const stats = basicStats(rows.map(r => tryParseNum(r[targetKey])).filter(Number.isFinite));
  E('dataSummary').innerHTML = `
    <div><b>Rows:</b> ${n}</div>
    <div><b>Date range:</b> ${first} → ${last}</div>
    <div><b>Target:</b> ${targetKey} | <b>mean:</b> ${stats.mean.toFixed(2)} <b>std:</b> ${stats.std.toFixed(2)} <b>min:</b> ${stats.min.toFixed(2)} <b>max:</b> ${stats.max.toFixed(2)}</div>
  `;
}

// ---------- Filtering & Aggregation ----------
function applyFilters(){
  const store = els.store().value;
  const product = els.product().value;
  const region = els.region().value;
  FILTERED_ROWS = RAW_ROWS.filter(r =>
    (store   === '__ALL__' || String(r.__store)   === store) &&
    (product === '__ALL__' || String(r.__product) === product) &&
    (region  === '__ALL__' || String(r.__region)  === region)
  );
}

function aggregateByDate(dateKey, targetKey){
  const agg = els.aggFunc().value; // sum | mean | last
  const map = new Map(); // dateStr -> [sum, count, last]
  for (const r of FILTERED_ROWS){
    const d = r[dateKey];
    const v = tryParseNum(r[targetKey]);
    if (!Number.isFinite(v)) continue;
    if (!map.has(d)) map.set(d, [0,0,null]);
    const a = map.get(d);
    a[0] += v; a[1] += 1; a[2] = v;
  }
  const dates = Array.from(map.keys()).sort();
  const series = dates.map(ds => {
    const [sum,cnt,last] = map.get(ds);
    if (agg==='sum') return sum;
    if (agg==='mean') return sum / (cnt||1);
    return last;
  });
  // convert to Date objs
  const dateObjs = dates.map(ds => asDate(ds));
  return {dates: dateObjs, series};
}

function rebuildSeries(){
  const dateKey = els.dateCol().value;
  const targetKey = els.targetCol().value;
  applyFilters();
  const {dates, series} = aggregateByDate(dateKey, targetKey);
  DATES = dates; SERIES = series;
}

// ---------- Charts ----------
function lineChart(id, labels, data, label, extraDatasets=[]){
  if (CHARTS[id]) CHARTS[id].destroy();
  const ctx = E(id).getContext('2d');
  CHARTS[id] = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: [{ label, data, fill:false, tension:0.2 }, ...extraDatasets ]},
    options: {
      responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{labels:{color:'#e2e8f0'}}, tooltip:{mode:'index', intersect:false}},
      scales:{ x:{ticks:{color:'#cbd5e1'}}, y:{ticks:{color:'#cbd5e1'}} }
    }
  });
}
function barChart(id, labels, data, label){
  if (CHARTS[id]) CHARTS[id].destroy();
  const ctx = E(id).getContext('2d');
  CHARTS[id] = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label, data }]},
    options: {
      responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{labels:{color:'#e2e8f0'}}},
      scales:{ x:{ticks:{color:'#cbd5e1'}}, y:{ticks:{color:'#cbd5e1'}} }
    }
  });
}

// ---------- EDA ----------
function renderEDA(){
  if (!SERIES.length){ log("Load data and check filters/columns first."); return; }
  const rw = Math.max(3, Number(E('rollingWindow').value) || 7);
  const labels = DATES.map(d => d ? d.toISOString().slice(0,10) : "");
  const roll = rollingMean(SERIES, rw);
  lineChart('tsChart', labels, SERIES, 'Target', [
    { label:`Rolling mean (${rw})`, data: roll, fill:false, borderDash:[6,6], tension:0.2 }
  ]);
  const {labels: bins, counts} = histogram(SERIES, 24);
  barChart('histChart', bins, counts, "Histogram");
  const {labels: dLabels, avgs} = dayOfWeekAverages(DATES, SERIES);
  barChart('dowChart', dLabels, avgs, "Mean by Day of Week");
  const ac = acf(SERIES, 30);
  barChart('acfChart', Array.from({length:30},(_,i)=>`lag ${i+1}`), ac, "Autocorrelation (1–30)");
  log("EDA rendered.");
}

// ---------- Windowing ----------
function makeWindows(series, seqLen){
  const X = [], y = [];
  for (let i=0; i+seqLen < series.length; i++){
    X.push(series.slice(i, i+seqLen));
    y.push(series[i+seqLen]);
  }
  return {X, y};
}

function trainTestSplit(X, y, testRatio=0.2){
  const n = X.length;
  const nTest = Math.max(1, Math.floor(n * testRatio));
  const nTrain = n - nTest;
  TRAIN_SPLIT_INDEX = nTrain;
  return {
    Xtrain: X.slice(0, nTrain),
    ytrain: y.slice(0, nTrain),
    Xtest:  X.slice(nTrain),
    ytest:  y.slice(nTrain)
  };
}

// ---------- Model ----------
function buildModel(kind, seqLen, units, dropout){
  const model = tf.sequential();
  const layer = (kind === 'LSTM')
    ? tf.layers.lstm({ units, inputShape:[seqLen,1], returnSequences:false, dropout })
    : tf.layers.gru ({ units, inputShape:[seqLen,1], returnSequences:false, dropout });
  model.add(layer);
  model.add(tf.layers.dense({units:32, activation:'relu'}));
  model.add(tf.layers.dense({units:1}));
  model.compile({optimizer: tf.train.adam(), loss: 'meanSquaredError'});
  return model;
}

async function trainModel(){
  if (!SERIES.length){ log("Load data first."); return; }
  const seqLen = Number(E('seqLen').value);
  const units  = Number(E('units').value);
  const dropout= Number(E('dropout').value);
  const epochs = Number(E('epochs').value);
  const batch  = Number(E('batch').value);
  const kind   = E('modelType').value;

  // Scale + windows
  zfit(SERIES);
  const zs = z(SERIES);
  const {X, y} = makeWindows(zs, seqLen);
  if (X.length < 10){ log("Not enough data for chosen sequence length."); return; }
  const {Xtrain, ytrain, Xtest, ytest} = trainTestSplit(X, y, 0.2);

  const tXtr = tf.tensor3d(Xtrain.map(x=>x.map(v=>[v])));
  const tytr = tf.tensor2d(ytrain.map(v=>[v]));
  const tXte = tf.tensor3d(Xtest.map(x=>x.map(v=>[v])));
  const tyte = tf.tensor2d(ytest.map(v=>[v]));

  if (MODEL) { MODEL.dispose(); MODEL = null; }
  MODEL = buildModel(kind, seqLen, units, dropout);

  // Loss chart
  if (CHARTS.lossChart) CHARTS.lossChart.destroy();
  CHARTS.lossChart = new Chart(E('lossChart').getContext('2d'), {
    type:'line',
    data:{ labels: [], datasets:[
      {label:'Train Loss', data:[], fill:false, tension:0.1},
      {label:'Val Loss', data:[], fill:false, borderDash:[6,6], tension:0.1}
    ]},
    options:{
      responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{labels:{color:'#e2e8f0'}} },
      scales:{ x:{ticks:{color:'#cbd5e1'}}, y:{ticks:{color:'#cbd5e1'}} }
    }
  });

  log(`Training ${kind}(${units}) for ${epochs} epochs, batch ${batch}...`);
  await MODEL.fit(tXtr, tytr, {
    epochs, batchSize: batch, validationSplit: 0.1, shuffle:false,
    callbacks:{
      onEpochEnd: async (epoch, logs) => {
        CHARTS.lossChart.data.labels.push(epoch+1);
        CHARTS.lossChart.data.datasets[0].data.push(Number(logs.loss.toFixed(6)));
        CHARTS.lossChart.data.datasets[1].data.push(Number((logs.val_loss??NaN).toFixed(6)));
        CHARTS.lossChart.update();
        log(`Epoch ${epoch+1}/${epochs} — loss=${logs.loss.toFixed(6)} val=${(logs.val_loss??0).toFixed(6)}`);
        await tf.nextFrame();
      }
    }
  });

  // keep test tensors for evaluation
  tXtr.dispose(); tytr.dispose();
  window.__tXtest = tXte;
  window.__tytest = tyte;
  E('btnPredict').disabled = false;
  E('btnEvaluate').disabled = false;
  log("✅ Training complete.");
}

// ---------- Prediction ----------
function oneStepPredictSequence(model, lastWindow, horizon){
  const out = [];
  let win = lastWindow.slice();
  for (let i=0;i<horizon;i++){
    const x = tf.tensor3d([win.map(v=>[v])]);
    const yhat = model.predict(x);
    const val = yhat.dataSync()[0];
    x.dispose(); yhat.dispose();
    out.push(val);
    win = win.slice(1).concat(val);
  }
  return out;
}

function predictFuture(){
  if (!MODEL){ log("Train model first."); return; }
  const seqLen = Number(E('seqLen').value);
  const horizon = Number(E('horizon').value);
  const zs = z(SERIES);
  const lastWindow = zs.slice(zs.length - seqLen);
  if (lastWindow.length < seqLen){ log("Not enough data for this sequence length."); return; }
  const zForecast = oneStepPredictSequence(MODEL, lastWindow, horizon);
  const forecast = iz(zForecast);

  const lastDate = DATES[DATES.length-1] || new Date();
  const labels = [];
  for (let i=1;i<=horizon;i++){
    const d = new Date(lastDate.getTime() + i*24*3600*1000);
    labels.push(d.toISOString().slice(0,10));
  }
  const combinedLabels = DATES.map(d=> d? d.toISOString().slice(0,10) : "").concat(labels);
  const combinedSeries = SERIES.concat(Array(horizon).fill(null));
  lineChart('predChart', combinedLabels, combinedSeries, 'Actual', [
    { label:'Forecast', data: Array(SERIES.length).fill(null).concat(forecast), fill:false, borderDash:[4,4], tension:0.2 }
  ]);
  log("✅ Forecast plotted.");
}

// ---------- Evaluation ----------
function evaluateTest(){
  if (!MODEL || !window.__tXtest || !window.__tytest){
    log("Train model first."); return;
  }
  const tXte = window.__tXtest;
  const tyte = window.__tytest;
  const predsZ = MODEL.predict(tXte);
  const preds = predsZ.arraySync().map(r => r[0]);
  const truthZ = tyte.arraySync().map(r => r[0]);
  const pred = iz(preds);
  const truth = iz(truthZ);

  const mae = meanAbsError(truth, pred);
  const rmse = rootMeanSquaredError(truth, pred);
  const mape = meanAbsPercError(truth, pred);
  E('metrics').innerHTML = `
    <div class="kpi"><span>MAE</span><span>${mae.toFixed(3)}</span></div>
    <div class="kpi"><span>RMSE</span><span>${rmse.toFixed(3)}</span></div>
    <div class="kpi"><span>MAPE</span><span>${(100*mape).toFixed(2)}%</span></div>
  `;

  // Align test labels to the original timeline
  const seqLen = Number(E('seqLen').value);
  const {X, y} = makeWindows(z(SERIES), seqLen);
  const testStartIdx = TRAIN_SPLIT_INDEX + seqLen;
  const testDates = DATES.slice(testStartIdx, testStartIdx + truth.length).map(d => d? d.toISOString().slice(0,10):"");
  lineChart('predChart', testDates, truth, "Actual (Test)", [
    {label:"Predicted (Test)", data: pred, fill:false, borderDash:[6,2], tension:0.2}
  ]);
  predsZ.dispose();
  log("✅ Evaluation complete.");
}

function meanAbsError(yTrue, yPred){
  const n = yTrue.length || 1;
  let s=0; for (let i=0;i<n;i++) s+= Math.abs(yTrue[i]-yPred[i]);
  return s/n;
}
function rootMeanSquaredError(yTrue, yPred){
  const n = yTrue.length || 1;
  let s=0; for (let i=0;i<n;i++){ const e=yTrue[i]-yPred[i]; s+= e*e; }
  return Math.sqrt(s/n);
}
function meanAbsPercError(yTrue, yPred){
  const n = yTrue.length || 1;
  let s=0; let cnt=0;
  for (let i=0;i<n;i++){
    const yt = yTrue[i];
    if (Math.abs(yt) < 1e-8) continue;
    s += Math.abs((yPred[i]-yt)/yt);
    cnt++;
  }
  return cnt? s/cnt : 0;
}

// ---------- Wiring ----------
async function handleLoad(){
  try{
    let rows = [];
    const file = E('fileInput').files[0];
    if (file){
      const txt = await file.text();
      rows = file.name.endsWith(".json") ? JSON.parse(txt) : parseCSV(txt);
      log(`Loaded ${rows.length} rows from ${file.name}`);
    } else {
      rows = await loadFromDataJS();
      log(`Loaded ${rows.length} rows from data.js (or synthetic).`);
    }
    if (!Array.isArray(rows) || !rows.length) throw new Error("No rows found.");

    // Normalize convenient meta-keys
    const names = Object.keys(rows[0]);
    const {dateKey, targetKey} = detectColumns(rows);

    // Prepare RAW_ROWS with helper copies
    RAW_ROWS = rows.map(r => ({
      ...r,
      __date: r[dateKey],
      __store: r["Store ID"] ?? r["store"] ?? r["store_id"],
      __product: r["Product ID"] ?? r["product"] ?? r["product_id"],
      __region: r["Region"] ?? r["region"]
    }));

    // Populate column selects (list numeric columns for target)
    const dateOptions = names.map(n => ({value:n, label:n}));
    const numericCandidates = names.filter(n => rows.some(rr => Number.isFinite(tryParseNum(rr[n]))));
    const targetOptions = numericCandidates.map(n => ({value:n, label:n}));
    populateSelect(E('dateCol'), dateOptions, dateKey || names[0]);
    populateSelect(E('targetCol'), targetOptions, targetKey || (numericCandidates[0] || names[0]));

    // Populate filters
    const stores = ["__ALL__"].concat(uniqueValues(RAW_ROWS, "__store"));
    const products = ["__ALL__"].concat(uniqueValues(RAW_ROWS, "__product"));
    const regions = ["__ALL__"].concat(uniqueValues(RAW_ROWS, "__region"));
    populateSelect(E('storeFilter'), stores.map(v=>({value:v,label:v==="__ALL__"?"All":v})), "__ALL__");
    populateSelect(E('productFilter'), products.map(v=>({value:v,label:v==="__ALL__"?"All":v})), "__ALL__");
    populateSelect(E('regionFilter'), regions.map(v=>({value:v,label:v==="__ALL__"?"All":v})), "__ALL__");

    // Initial filter + aggregate
    applyFilters();
    const dk = E('dateCol').value;
    const tk = E('targetCol').value;
    ({dates: DATES, series: SERIES} = aggregateByDate(dk, tk));

    // Sort RAW_ROWS chronologically for table, and show a preview
    RAW_ROWS.sort((a,b)=> (asDate(a.__date) || 0) - (asDate(b.__date) || 0));
    toTable(RAW_ROWS);

    summarize(RAW_ROWS, E('dateCol').value, E('targetCol').value);
    E('btnTrain').disabled = false;
    E('btnEDA').disabled = false;
    E('btnPredict').disabled = true;
    E('btnEvaluate').disabled = true;
  }catch(e){
    console.error(e); log(`❌ ${e.message}`);
  }
}

function reprepare(){
  if (!RAW_ROWS.length) return;
  rebuildSeries();
  summarize(RAW_ROWS, E('dateCol').value, E('targetCol').value);
  E('btnPredict').disabled = true;
  E('btnEvaluate').disabled = true;
  log("Filters/columns changed. Re-run EDA and (optionally) retrain model.");
}

// Event listeners
E('btnLoad').addEventListener('click', handleLoad);
E('btnEDA').addEventListener('click', renderEDA);
E('btnTrain').addEventListener('click', ()=>{
  trainModel().catch(e=>{console.error(e); log(`❌ ${e.message}`);});
});
E('btnPredict').addEventListener('click', ()=>{
  try{ predictFuture(); }catch(e){ console.error(e); log(`❌ ${e.message}`); }
});
E('btnEvaluate').addEventListener('click', ()=>{
  try{ evaluateTest(); }catch(e){ console.error(e); log(`❌ ${e.message}`); }
});
['dateCol','targetCol','aggFunc','storeFilter','productFilter','regionFilter'].forEach(id => {
  document.getElementById(id).addEventListener('change', reprepare);
});

log("App loaded. Click 'Load Dataset' to start. If no data.js is present, a synthetic dataset will be used.");
