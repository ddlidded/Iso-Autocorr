// Iso-Autocorr: browser-only port of Autocorr.m
// Produces Excel with sheets: original, cor_pct, cor_abs, total

const TR_SYMBOLS = ["C", "N", "D", "O"]; // for isotopeLabel parsing
const ABUNDANCE = [0.0107, 0.00364, 0.00001, 0.00187]; // natural isotope abundances (MATLAB Autocorr.m)

function setStatus(msg, isError = false) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.classList.toggle("error", isError);
}

function comb(n, k) {
  if (k < 0 || k > n) return 0;
  k = Math.min(k, n - k);
  let res = 1;
  for (let i = 1; i <= k; i++) {
    res = (res * (n - k + i)) / i;
  }
  return res;
}

function dbinom(a, b, r) {
  return comb(a, b) * Math.pow(1 - r, a - b) * Math.pow(r, b);
}

function labelAutodetect(labels) {
  const blob = labels.join("");
  const out = [];
  if (blob.includes("C13")) out.push(1);
  if (blob.includes("N15")) out.push(2);
  if (blob.includes("D2")) out.push(3);
  if (blob.includes("O18")) out.push(4);
  return out;
}

function str2ab(label, A, B) {
  const parts = String(label).split("-");
  if (parts.length === 1) return [0, 0];
  if (parts.length === 3) {
    const first = String(label)[0];
    if (first === A) return [parseInt(parts[2], 10), 0];
    if (first === B) return [0, parseInt(parts[2], 10)];
    throw new Error(`Unexpected tracer in isotopeLabel '${label}' (expected ${A}/${B})`);
  }
  if (parts.length === 4) {
    return [parseInt(parts[2], 10), parseInt(parts[3], 10)];
  }
  throw new Error(`Cannot parse isotopeLabel '${label}'`);
}

function abGetCounts(labels, A, B) {
  const counts = new Array(labels.length);
  for (let i = 0; i < labels.length; i++) {
    const [a, b] = str2ab(labels[i], A, B);
    counts[i] = [a, b];
  }
  return counts;
}

function formulaCounts(formula) {
  // Returns [C,N,H,O,S,P] counts.
  let f = String(formula ?? "").trim();
  f = f.replace(/^"|"$/g, "");
  f = f.replace(/[+-]/g, "");

  function parseIntAt(i) {
    if (i >= f.length || f[i] < "0" || f[i] > "9") return [1, i];
    let j = i;
    while (j < f.length && f[j] >= "0" && f[j] <= "9") j++;
    return [Number(f.slice(i, j)), j];
  }

  function parseGroup(i) {
    const counts = Object.create(null);
    while (i < f.length) {
      const ch = f[i];
      if (ch === "(") {
        const inner = parseGroup(i + 1);
        i = inner.i;
        if (i >= f.length || f[i] !== ")") throw new Error(`Unmatched '(' in formula '${formula}'`);
        const [mult, ni] = parseIntAt(i + 1);
        i = ni;
        for (const k of Object.keys(inner.counts)) {
          counts[k] = (counts[k] ?? 0) + inner.counts[k] * mult;
        }
        continue;
      }
      if (ch === ")") return { counts, i };
      if (ch >= "A" && ch <= "Z") {
        let sym = ch;
        i++;
        if (i < f.length && f[i] >= "a" && f[i] <= "z") {
          sym += f[i];
          i++;
        }
        const [mult, ni] = parseIntAt(i);
        i = ni;
        counts[sym] = (counts[sym] ?? 0) + mult;
        continue;
      }
      if (" []{},;".includes(ch)) {
        i++;
        continue;
      }
      throw new Error(`Unexpected character '${ch}' in formula '${formula}'`);
    }
    return { counts, i };
  }

  const parsed = parseGroup(0);
  if (parsed.i !== f.length) throw new Error(`Unexpected trailing input in formula '${formula}'`);

  const symbols = ["C", "N", "H", "O", "S", "P"];
  return symbols.map((s) => Number(parsed.counts[s] ?? 0));
}

function abGetFullDt(dtRows, counts, A_num, B_num, sampleCount) {
  // Mirrors AB_getfulldt.m and respects MATLAB column-major reshape behavior.
  // cn list ordering: A repeats blocks, B cycles within.
  const v1 = [];
  for (let a = 0; a < A_num + 1; a++) {
    for (let b = 0; b < B_num + 1; b++) v1.push(b);
  }
  const v2 = [];
  for (let a = 0; a < A_num + 1; a++) {
    for (let b = 0; b < B_num + 1; b++) v2.push(a);
  }
  // cn = [v1;v2]' then swap columns => [A,B]
  const cn = v1.map((b, idx) => [v2[idx], b]);

  const fulldt = new Array(cn.length);
  const idx_r = [];
  const idx_j = [];

  for (let j = 0; j < cn.length; j++) {
    const [A, B] = cn[j];
    let tp = -1;
    for (let r = 0; r < counts.length; r++) {
      if (counts[r][0] === A && counts[r][1] === B) {
        tp = r;
        break;
      }
    }
    if (tp === -1) {
      fulldt[j] = new Float64Array(sampleCount); // zeros
    } else {
      idx_r.push(tp);
      idx_j.push(j);
      fulldt[j] = dtRows[tp];
    }
  }

  // reduced_idx: idx_j ordered by idx_r ascending
  const order = idx_r.map((val, i) => [val, i]).sort((a, b) => a[0] - b[0]);
  const reduced_idx = order.map(([, i]) => idx_j[i]);

  return { fulldt, reduced_idx };
}

function matMul(A, B, n) {
  // A: Float64Array (n*n) row-major, B: Float64Array (n*n)
  const C = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    const ioff = i * n;
    for (let k = 0; k < n; k++) {
      const a = A[ioff + k];
      if (a === 0) continue;
      const koff = k * n;
      for (let j = 0; j < n; j++) {
        C[ioff + j] += a * B[koff + j];
      }
    }
  }
  return C;
}

function matVec(A, x, n) {
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const off = i * n;
    for (let j = 0; j < n; j++) s += A[off + j] * x[j];
    y[i] = s;
  }
  return y;
}

function aTResidual(A, r, n) {
  const w = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += A[i * n + j] * r[i];
    w[j] = s;
  }
  return w;
}

function solveLinearSystem(G, c) {
  // Gaussian elimination with partial pivoting.
  const n = c.length;
  const A = new Float64Array(G); // copy
  const b = new Float64Array(c);

  for (let k = 0; k < n; k++) {
    // pivot
    let piv = k;
    let max = Math.abs(A[k * n + k]);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(A[i * n + k]);
      if (v > max) {
        max = v;
        piv = i;
      }
    }
    if (max === 0) return null;
    if (piv !== k) {
      for (let j = k; j < n; j++) {
        const tmp = A[k * n + j];
        A[k * n + j] = A[piv * n + j];
        A[piv * n + j] = tmp;
      }
      const tb = b[k];
      b[k] = b[piv];
      b[piv] = tb;
    }

    const akk = A[k * n + k];
    for (let i = k + 1; i < n; i++) {
      const factor = A[i * n + k] / akk;
      if (factor === 0) continue;
      for (let j = k; j < n; j++) A[i * n + j] -= factor * A[k * n + j];
      b[i] -= factor * b[k];
    }
  }

  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let j = i + 1; j < n; j++) s -= A[i * n + j] * x[j];
    x[i] = s / A[i * n + i];
  }
  return x;
}

function leastSquaresNormalEq(A, P, b, n) {
  // Solve min ||A_P z - b|| via normal equations.
  const k = P.length;
  const G = new Float64Array(k * k);
  const c = new Float64Array(k);

  for (let ii = 0; ii < k; ii++) {
    const colI = P[ii];
    let ci = 0;
    for (let r = 0; r < n; r++) ci += A[r * n + colI] * b[r];
    c[ii] = ci;

    for (let jj = 0; jj <= ii; jj++) {
      const colJ = P[jj];
      let s = 0;
      for (let r = 0; r < n; r++) s += A[r * n + colI] * A[r * n + colJ];
      G[ii * k + jj] = s;
      G[jj * k + ii] = s;
    }
  }

  const z = solveLinearSystem(G, c);
  if (!z) return null;
  return z;
}

function nnls(A, b, n) {
  // Lawson–Hanson NNLS.
  const tol = 1e-12;
  const x = new Float64Array(n);
  const P = [];
  const Z = new Set([...Array(n).keys()]);

  function residual() {
    const Ax = matVec(A, x, n);
    const r = new Float64Array(n);
    for (let i = 0; i < n; i++) r[i] = b[i] - Ax[i];
    return r;
  }

  let r = residual();
  let w = aTResidual(A, r, n);

  while (true) {
    // pick t in Z with max w
    let t = -1;
    let wmax = tol;
    for (const j of Z) {
      if (w[j] > wmax) {
        wmax = w[j];
        t = j;
      }
    }
    if (t === -1) break;

    Z.delete(t);
    P.push(t);

    let xOld = new Float64Array(x);
    while (true) {
      const z = leastSquaresNormalEq(A, P, b, n);
      if (!z) break;

      // xNew is zero except on P
      const xNew = new Float64Array(n);
      for (let i = 0; i < P.length; i++) xNew[P[i]] = z[i];

      // check positivity
      let anyNeg = false;
      for (const idx of P) {
        if (xNew[idx] <= tol) {
          anyNeg = true;
          break;
        }
      }
      if (!anyNeg) {
        x.set(xNew);
        break;
      }

      // find alpha
      let alpha = Infinity;
      for (const idx of P) {
        if (xNew[idx] <= tol) {
          const denom = xOld[idx] - xNew[idx];
          if (denom > 0) alpha = Math.min(alpha, xOld[idx] / denom);
        }
      }
      if (!Number.isFinite(alpha)) alpha = 0;

      // step and move zeros back to Z
      for (let i = 0; i < n; i++) x[i] = xOld[i] + alpha * (xNew[i] - xOld[i]);

      // remove near-zeros from P
      for (let i = P.length - 1; i >= 0; i--) {
        const idx = P[i];
        if (x[idx] <= tol) {
          x[idx] = 0;
          P.splice(i, 1);
          Z.add(idx);
        }
      }

      xOld = new Float64Array(x);
      if (P.length === 0) break;
    }

    r = residual();
    w = aTResidual(A, r, n);
  }

  return x;
}

function isocorrA(distInRows, n, abA, imA, sampleCount) {
  const dim = n + 1;
  const M = new Float64Array(dim * dim);
  for (let i = 1; i <= dim; i++) {
    for (let j = 1; j <= i; j++) {
      const a = dim - j;
      const b = i - j;
      M[(i - 1) * dim + (j - 1)] = comb(a, b) * Math.pow(1 - abA, a - b) * Math.pow(abA, b);
    }
  }
  const Mp = new Float64Array(dim * dim);
  for (let i = 1; i <= dim; i++) {
    for (let j = i; j <= dim; j++) {
      const a = j - 1;
      const b = j - i;
      Mp[(i - 1) * dim + (j - 1)] = comb(a, b) * Math.pow(1 - imA, a - b) * Math.pow(imA, b);
    }
  }
  const N = matMul(M, Mp, dim);

  const distOutAbs = new Array(dim);
  const distOutPct = new Array(dim);
  for (let r = 0; r < dim; r++) {
    distOutAbs[r] = new Float64Array(sampleCount);
    distOutPct[r] = new Float64Array(sampleCount);
  }

  for (let c = 0; c < sampleCount; c++) {
    const b = new Float64Array(dim);
    for (let r = 0; r < dim; r++) b[r] = distInRows[r][c];
    const x = nnls(N, b, dim);
    let s = 0;
    for (let r = 0; r < dim; r++) {
      distOutAbs[r][c] = x[r];
      s += x[r];
    }
    s += 1e-10;
    for (let r = 0; r < dim; r++) distOutPct[r][c] = distOutAbs[r][c] / s;
  }

  return { abs: distOutAbs, pct: distOutPct };
}

function isocorrAB(distInRows, n, m, abA, abB, imA, imB, sampleCount) {
  const dim = (n + 1) * (m + 1);
  const M = new Float64Array(dim * dim);

  for (let i = 1; i <= dim; i++) {
    const Arow = Math.floor((i - 1) / (m + 1));
    const Brow = (i - 1) % (m + 1);
    for (let j = 1; j <= i; j++) {
      const Acol = Math.floor((j - 1) / (m + 1));
      const Bcol = (j - 1) % (m + 1);

      const a1 = n - Acol;
      const b1 = Arow - Acol;
      const a2 = m - Bcol;
      const b2 = Brow - Bcol;
      if (b1 >= 0 && b2 >= 0) {
        M[(i - 1) * dim + (j - 1)] = dbinom(a1, b1, abA) * dbinom(a2, b2, abB);
      }
    }
  }

  const Mp = new Float64Array(dim * dim);
  for (let i = 1; i <= dim; i++) {
    const Arow = Math.floor((i - 1) / (m + 1));
    const Brow = (i - 1) % (m + 1);
    for (let j = i; j <= dim; j++) {
      const Acol = Math.floor((j - 1) / (m + 1));
      const Bcol = (j - 1) % (m + 1);

      const a1 = Acol;
      const b1 = Acol - Arow;
      const a2 = Bcol;
      const b2 = Bcol - Brow;
      if (b1 >= 0 && b2 >= 0) {
        Mp[(i - 1) * dim + (j - 1)] = dbinom(a1, b1, imA) * dbinom(a2, b2, imB);
      }
    }
  }

  const N = matMul(M, Mp, dim);

  const distOutAbs = new Array(dim);
  const distOutPct = new Array(dim);
  for (let r = 0; r < dim; r++) {
    distOutAbs[r] = new Float64Array(sampleCount);
    distOutPct[r] = new Float64Array(sampleCount);
  }

  for (let c = 0; c < sampleCount; c++) {
    const b = new Float64Array(dim);
    for (let r = 0; r < dim; r++) b[r] = distInRows[r][c];
    const x = nnls(N, b, dim);
    let s = 0;
    for (let r = 0; r < dim; r++) {
      distOutAbs[r][c] = x[r];
      s += x[r];
    }
    s += 1e-10;
    for (let r = 0; r < dim; r++) distOutPct[r][c] = distOutAbs[r][c] / s;
  }

  return { abs: distOutAbs, pct: distOutPct };
}

function toNumber(x) {
  const v = Number(x);
  return Number.isFinite(v) ? v : 0;
}

function buildSheetAoA(fields, rows) {
  const aoa = [fields.slice()];
  for (const r of rows) {
    aoa.push(fields.map((f) => (r[f] === undefined ? "" : r[f])));
  }
  return aoa;
}

async function run() {
  const file = document.getElementById("file").files[0];
  if (!file) throw new Error("Please select a CSV file.");

  const impurity = [
    Number(document.getElementById("impC").value || 0.01),
    Number(document.getElementById("impN").value || 0.01),
    Number(document.getElementById("impD").value || 0.01),
    Number(document.getElementById("impO").value || 0.01),
  ];

  setStatus("Parsing CSV...");

  const parsed = await new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: resolve,
      error: reject,
    });
  });

  if (parsed.errors && parsed.errors.length) {
    throw new Error(parsed.errors[0].message || "CSV parse failed");
  }

  const fields = parsed.meta.fields;
  let rows = parsed.data;
  if (!fields || !fields.length) throw new Error("No header row detected.");

  // Trim whitespace from all string-like fields (matches MATLAB readtable/writetable behavior seen in examples).
  rows = rows.map((row) => {
    const out = { ...row };
    for (const k of Object.keys(out)) {
      if (typeof out[k] === "string") out[k] = out[k].trim();
    }
    return out;
  });

  // Cut empty trailing rows based on medMz > 0
  let lastPos = -1;
  for (let i = 0; i < rows.length; i++) {
    if (toNumber(rows[i].medMz) > 0) lastPos = i;
  }
  if (lastPos >= 0) rows = rows.slice(0, lastPos + 1);

  const parentIdx = fields.indexOf("parent");
  if (parentIdx === -1) throw new Error("Missing required column 'parent'.");
  const sampleFields = fields.slice(parentIdx + 1);
  const sampleCount = sampleFields.length;

  // Group heads: isotopeLabel == 'C12 PARENT'
  const grpHeads = [];
  for (let i = 0; i < rows.length; i++) {
    if (String(rows[i].isotopeLabel) === "C12 PARENT") grpHeads.push(i);
  }
  if (!grpHeads.length) throw new Error("Could not find any 'C12 PARENT' rows in isotopeLabel.");
  grpHeads.push(rows.length);

  // Precompute dt rows as Float64Array per row
  const dtAll = rows.map((row) => {
    const v = new Float64Array(sampleCount);
    for (let j = 0; j < sampleCount; j++) v[j] = toNumber(row[sampleFields[j]]);
    return v;
  });

  setStatus(`Running corrections... (groups: ${grpHeads.length - 1})`);

  const catAbs = [];
  const catPct = [];
  const totalRows = [];

  for (let g = 0; g < grpHeads.length - 1; g++) {
    const start = grpHeads[g];
    const end = grpHeads[g + 1];

    for (let i = start; i < end; i++) rows[i].metaGroupId = String(g + 1);

    const labels = [];
    const groupDt = [];
    for (let i = start; i < end; i++) {
      labels.push(String(rows[i].isotopeLabel));
      groupDt.push(dtAll[i]);
    }

    const out = labelAutodetect(labels);
    let corrAbsRows;
    let corrPctRows;

    if (out.length === 0) {
      // MATLAB sets corr_pct to ones(1, samples); practical case is a single-row group.
      corrAbsRows = groupDt;
      if (groupDt.length === 1) {
        corrPctRows = [new Float64Array(sampleCount).fill(1)];
      } else {
        corrPctRows = groupDt.map((row) => {
          const pct = new Float64Array(sampleCount);
          for (let c = 0; c < sampleCount; c++) {
            let s = 0;
            for (let r = 0; r < groupDt.length; r++) s += groupDt[r][c];
            pct[c] = s > 0 ? row[c] / s : 0;
          }
          return pct;
        });
      }
    } else if (out.length === 1) {
      const trA = out[0];
      const tp = formulaCounts(rows[start].formula);
      const A_num = tp[trA - 1];
      const abA = ABUNDANCE[trA - 1];
      const imA = impurity[trA - 1];
      const counts = abGetCounts(labels, TR_SYMBOLS[trA - 1], TR_SYMBOLS[trA - 1]);
      const { fulldt, reduced_idx } = abGetFullDt(groupDt, counts, A_num, 0, sampleCount);
      const corrFull = isocorrA(fulldt, A_num, abA, imA, sampleCount);
      corrAbsRows = reduced_idx.map((idx) => corrFull.abs[idx]);
      corrPctRows = reduced_idx.map((idx) => corrFull.pct[idx]);
    } else if (out.length === 2) {
      const trA = out[0];
      const trB = out[1];
      const tp = formulaCounts(rows[start].formula);
      const A_num = tp[trA - 1];
      const B_num = tp[trB - 1];
      const abA = ABUNDANCE[trA - 1];
      const abB = ABUNDANCE[trB - 1];
      const imA = impurity[trA - 1];
      const imB = impurity[trB - 1];
      const counts = abGetCounts(labels, TR_SYMBOLS[trA - 1], TR_SYMBOLS[trB - 1]);
      const { fulldt, reduced_idx } = abGetFullDt(groupDt, counts, A_num, B_num, sampleCount);
      const corrFull = isocorrAB(fulldt, A_num, B_num, abA, abB, imA, imB, sampleCount);
      corrAbsRows = reduced_idx.map((idx) => corrFull.abs[idx]);
      corrPctRows = reduced_idx.map((idx) => corrFull.pct[idx]);
    } else {
      throw new Error("More than 2 labeled terms detected in a group.");
    }

    // Concatenate into full-table order
    for (let i = 0; i < corrAbsRows.length; i++) {
      catAbs.push(corrAbsRows[i]);
      catPct.push(corrPctRows[i]);
    }

    // Total row (corr_tic)
    const tic = new Float64Array(sampleCount);
    for (let c = 0; c < sampleCount; c++) {
      let s = 0;
      for (let r = 0; r < corrAbsRows.length; r++) s += corrAbsRows[r][c];
      tic[c] = s;
    }
    const tr = { ID: g + 1, Name: String(rows[start].compound ?? ""), formula: String(rows[start].formula ?? "") };
    for (let c = 0; c < sampleCount; c++) tr[sampleFields[c]] = tic[c];
    totalRows.push(tr);
  }

  // Build corrected row objects by copying original and swapping sample columns.
  const corAbsRows = rows.map((row, i) => {
    const out = { ...row };
    for (let c = 0; c < sampleCount; c++) out[sampleFields[c]] = catAbs[i][c];
    return out;
  });
  const corPctRows = rows.map((row, i) => {
    const out = { ...row };
    for (let c = 0; c < sampleCount; c++) out[sampleFields[c]] = catPct[i][c];
    return out;
  });

  setStatus("Generating Excel...");

  const wb = XLSX.utils.book_new();

  const wsOriginal = XLSX.utils.aoa_to_sheet(buildSheetAoA(fields, rows));
  const wsCorPct = XLSX.utils.aoa_to_sheet(buildSheetAoA(fields, corPctRows));
  const wsCorAbs = XLSX.utils.aoa_to_sheet(buildSheetAoA(fields, corAbsRows));

  const totalFields = ["ID", "Name", "formula", ...sampleFields];
  const wsTotal = XLSX.utils.aoa_to_sheet(buildSheetAoA(totalFields, totalRows));

  XLSX.utils.book_append_sheet(wb, wsOriginal, "original");
  XLSX.utils.book_append_sheet(wb, wsCorPct, "cor_pct");
  XLSX.utils.book_append_sheet(wb, wsCorAbs, "cor_abs");
  XLSX.utils.book_append_sheet(wb, wsTotal, "total");

  const array = XLSX.write(wb, { bookType: "xlsx", type: "array" });
  const blob = new Blob([array], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });

  const base = file.name.replace(/\.[^.]+$/, "");
  const outName = `${base}_cor.xlsx`;

  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = outName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);

  setStatus(`Done. Downloaded ${outName}`);
}

document.getElementById("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = document.getElementById("run");
  btn.disabled = true;
  try {
    await run();
  } catch (err) {
    setStatus(String(err?.message || err), true);
  } finally {
    btn.disabled = false;
  }
});
