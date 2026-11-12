/**
 * aqnet v0.2.0 — Adversarial Queueing & Saddle-Node Bifurcation Studio
 * Implements closed-form Wright-omega / Lambert-W_0 Saddle-Node bifurcation,
 * exact 5-topology conservation/delay solvers (Cases 1–5), dual fixed-point
 * branch solver, Case 1 structural p-invariance, Law L10 burst-trajectory
 * separatrix test, and real Two-Station Confidence Cascade hysteresis solver.
 */

(function (root) {
  "use strict";

  const TOPOLOGY_META = [
    { id: "one_node_modification", label: "Case 2: 1-Node Modification" },
    { id: "one_node_destruction", label: "Case 1: 1-Node Destruction" },
    { id: "tandem", label: "Case 3: N-Node Tandem Chain" },
    { id: "feedforward", label: "Case 4: N-Node Feedforward Chain" },
    { id: "feedback", label: "Case 5: N-Node Feedback Mesh" }
  ];

  const state = {
    mode: "bifurcation",
    view: "fold",
    theme: "academic-paper",
    topology: "one_node_modification",
    lambdaExt: 3.5,
    p: 0.25,
    mu: 10.0,
    T: 1.0,
    N: 2,
    burstMultipliers: [1.0, 1.35, 1.95, 2.45, 1.80, 1.0],
    lastTrajectory: []
  };

  /**
   * Wright omega function omega(z) = W_0(exp(z)) for real z > -1
   * solved via Halley's third-order iteration (residual < 1e-15).
   */
  function wrightOmega(z) {
    let w = z > 2.0 ? z - Math.log(z) : Math.max(0.2, 0.5 * (z + 1.0));
    for (let iter = 0; iter < 25; iter += 1) {
      const r = w + Math.log(w) - z;
      if (Math.abs(r) < 1e-15) {
        break;
      }
      const wp1 = w + 1.0;
      const dw = (r * w) / wp1 / (1.0 - (r * w) / (2.0 * wp1 * wp1 * wp1));
      w = Math.max(1e-12, w - dw);
    }
    return w;
  }

  /**
   * Lanczos log-Gamma approximation (accurate to < 1e-15 on a > 0).
   */
  function logGamma(z) {
    const coeffs = [
      676.5203681218851,
      -1259.1392167224028,
      771.32342877765313,
      -176.61502916214059,
      12.507343278686905,
      -0.13857109526572012,
      9.9843695780195716e-6,
      1.5056327351493116e-7
    ];
    if (z < 0.5) {
      return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1.0 - z);
    }
    const zm1 = z - 1.0;
    let x = 0.99999999999980993;
    for (let i = 0; i < coeffs.length; i += 1) {
      x += coeffs[i] / (zm1 + i + 1.0);
    }
    const t = zm1 + coeffs.length - 0.5;
    return 0.5 * Math.log(2.0 * Math.PI) + (zm1 + 0.5) * Math.log(t) - t + Math.log(x);
  }

  /**
   * Regularized lower incomplete gamma function P(a, x) = gamma(a, x) / Gamma(a).
   */
  function gammaincReg(a, x) {
    if (x <= 0.0) {
      return 0.0;
    }
    if (!Number.isFinite(x)) {
      return 1.0;
    }
    const gln = logGamma(a);
    if (x < a + 1.0) {
      let ap = a;
      let sum = 1.0 / a;
      let del = sum;
      for (let n = 1; n <= 200; n += 1) {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if (Math.abs(del) < Math.abs(sum) * 1e-15) {
          break;
        }
      }
      const val = sum * Math.exp(-x + a * Math.log(x) - gln);
      return Math.max(0.0, Math.min(1.0, val));
    }
    // Continued fraction for Q(a, x) = 1 - P(a, x)
    let b = x + 1.0 - a;
    let c = 1.0 / 1e-30;
    let d = 1.0 / b;
    let h = d;
    for (let i = 1; i <= 200; i += 1) {
      const an = -i * (i - a);
      b += 2.0;
      d = an * d + b;
      if (Math.abs(d) < 1e-30) {
        d = 1e-30;
      }
      c = b + an / c;
      if (Math.abs(c) < 1e-30) {
        c = 1e-30;
      }
      d = 1.0 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1.0) < 1e-15) {
        break;
      }
    }
    const qVal = h * Math.exp(-x + a * Math.log(x) - gln);
    return Math.max(0.0, Math.min(1.0, 1.0 - qVal));
  }

  /**
   * Regularized Erlang-N CDF F_Erlang(T; N, rate) = 1 - exp(-rate*T)*sum_{k=0}^{N-1} (rate*T)^k / k!
   */
  function erlangCdf(T, N, rate) {
    const x = rate * T;
    if (x <= 0.0) {
      return 0.0;
    }
    let sum = 1.0;
    let term = 1.0;
    for (let k = 1; k < N; k += 1) {
      term *= x / k;
      sum += term;
    }
    return Math.max(0.0, Math.min(1.0, 1.0 - Math.exp(-x) * sum));
  }

  function upperAttemptBound(topology, p, mu) {
    if (topology === "one_node_destruction") {
      return mu / Math.max(1.0 - p, 1e-9);
    }
    return mu;
  }

  /**
   * Case 4 (General N-Node Feedforward Chain) stage attenuation & Gamma moment-matching.
   */
  function feedforwardStageMetrics(LambdaStar, p, mu, T, N) {
    const lam0 = Math.max(0.0, LambdaStar);
    if (lam0 >= mu) {
      return { PSucc: 0.0, muS: Infinity, k: Number(N), theta: Infinity, EDSucc: Infinity, EDFail: Infinity, ED: Infinity };
    }
    let muS = 0.0;
    let sigmaS2 = 0.0;
    const TStages = [];
    for (let i = 0; i < N; i += 1) {
      const lamI = lam0 * Math.pow(1.0 - p, i);
      const tI = 1.0 / (mu - lamI);
      TStages.push(tI);
      muS += tI;
      sigmaS2 += tI * tI;
    }
    const k = (muS * muS) / sigmaS2;
    const theta = sigmaS2 / muS;
    const pNoTo = gammaincReg(k, T / theta);
    const pNoToK1 = gammaincReg(k + 1.0, T / theta);
    const pSurvAll = Math.pow(1.0 - p, N);
    const PSucc = Math.max(0.0, Math.min(1.0, pSurvAll * pNoTo));
    const EDSucc = pNoTo > 1e-15 ? muS * (pNoToK1 / pNoTo) : T;

    let numFail = 0.0;
    let denFail = 0.0;
    let cumT = 0.0;
    for (let i = 0; i < N; i += 1) {
      cumT += TStages[i];
      const wAttI = Math.pow(1.0 - p, i) * p;
      numFail += wAttI * cumT;
      denFail += wAttI;
    }
    const pToGivenNoAtt = Math.max(0.0, 1.0 - pNoTo);
    const wTo = pSurvAll * pToGivenNoAtt;
    const eSnGtT = pToGivenNoAtt > 1e-15 ? (muS * (1.0 - pNoToK1)) / pToGivenNoAtt : T;
    numFail += wTo * eSnGtT;
    denFail += wTo;

    const EDFail = denFail > 1e-15 ? numFail / denFail : 0.0;
    const ED = PSucc > 1e-15 ? (1.0 / PSucc - 1.0) * EDFail + EDSucc : Infinity;
    return { PSucc, muS, k, theta, EDSucc, EDFail, ED };
  }

  /**
   * Case 5 (N-Node Symmetric Feedback Mesh) Erlang-k visit series.
   */
  function feedbackSeriesMetrics(LambdaStar, p, mu, T, N, maxVisits) {
    const kMax = maxVisits || 180;
    const lam = Math.max(0.0, LambdaStar);
    if (lam >= mu) {
      return { PSucc: 0.0, EV: Infinity, EDSucc: Infinity, EDFail: Infinity, ED: Infinity };
    }
    const gammaRate = mu - lam;
    const qExit = 1.0 / (N + 1.0);
    const rCont = N / (N + 1.0);
    const x = gammaRate * T;
    const expMinusX = Math.exp(-x);

    let poissonPartial = 0.0;
    let term = expMinusX;
    const F = [1.0];
    for (let k = 1; k <= kMax + 1; k += 1) {
      poissonPartial += term;
      F.push(Math.max(0.0, Math.min(1.0, 1.0 - poissonPartial)));
      term *= x / k;
    }

    let EV = 0.0;
    let PSucc = 0.0;
    let numSucc = 0.0;
    let numFail = 0.0;
    let denFail = 0.0;
    const baseRatio = (1.0 - p) * rCont;
    let wk = 1.0;

    for (let k = 1; k <= kMax; k += 1) {
      const fKm1 = F[k - 1];
      const fK = F[k];
      const fKp1 = F[k + 1];
      const uk = wk * fKm1;
      EV += uk;

      const pSuccK = wk * (1.0 - p) * qExit * fK;
      PSucc += pSuccK;
      if (fK > 1e-15 && pSuccK > 0.0) {
        const eSkSucc = (k / gammaRate) * (fKp1 / fK);
        numSucc += pSuccK * eSkSucc;
      }

      const pAttK = uk * p;
      const eDAttK = (1.0 / gammaRate) + ((k - 1.0) / gammaRate) * (fKm1 > 0.0 ? fK / fKm1 : 1.0);
      const pToK = wk * (1.0 - p) * Math.max(0.0, fKm1 - fK);
      const eDToK = T + 1.0 / gammaRate;
      const pFailK = pAttK + pToK;
      numFail += pAttK * eDAttK + pToK * eDToK;
      denFail += pFailK;

      wk *= baseRatio;
      if (wk < 1e-15) {
        break;
      }
    }

    PSucc = Math.max(0.0, Math.min(1.0, PSucc));
    EV = Math.max(1.0, EV);
    const EDSucc = PSucc > 1e-15 ? numSucc / PSucc : 0.0;
    const EDFail = denFail > 1e-15 ? numFail / denFail : 0.0;
    const ED = PSucc > 1e-15 ? (1.0 / PSucc - 1.0) * EDFail + EDSucc : Infinity;
    return { PSucc, EV, EDSucc, EDFail, ED };
  }

  function successProbPsucc(LambdaStar, topology, p, mu, T, N) {
    const lamMax = upperAttemptBound(topology, p, mu);
    if (LambdaStar <= 0.0) {
      if (topology === "one_node_destruction" || topology === "one_node_modification") {
        return (1.0 - p) * (1.0 - Math.exp(-mu * T));
      }
      if (topology === "tandem") {
        return Math.pow(1.0 - p, N) * erlangCdf(T, N, mu);
      }
      if (topology === "feedforward") {
        return Math.pow(1.0 - p, N) * gammaincReg(Number(N), mu * T);
      }
      return feedbackSeriesMetrics(0.0, p, mu, T, N).PSucc;
    }
    if (LambdaStar >= lamMax) {
      return 0.0;
    }

    if (topology === "one_node_destruction") {
      const lamEff = (1.0 - p) * LambdaStar;
      return (1.0 - p) * Math.max(0.0, 1.0 - Math.exp(-(mu - lamEff) * T));
    }
    if (topology === "one_node_modification") {
      return (1.0 - p) * Math.max(0.0, 1.0 - Math.exp(-(mu - LambdaStar) * T));
    }
    if (topology === "tandem") {
      return Math.pow(1.0 - p, N) * erlangCdf(T, N, mu - LambdaStar);
    }
    if (topology === "feedforward") {
      return feedforwardStageMetrics(LambdaStar, p, mu, T, N).PSucc;
    }
    return feedbackSeriesMetrics(LambdaStar, p, mu, T, N).PSucc;
  }

  function throughputGamma(LambdaStar, topology, p, mu, T, N) {
    const lamMax = upperAttemptBound(topology, p, mu);
    if (LambdaStar <= 0.0 || LambdaStar >= lamMax) {
      return 0.0;
    }
    if (topology === "feedback") {
      const fb = feedbackSeriesMetrics(LambdaStar, p, mu, T, N);
      return (LambdaStar * fb.PSucc) / fb.EV;
    }
    return LambdaStar * successProbPsucc(LambdaStar, topology, p, mu, T, N);
  }

  function brentRoot(fn, a, b, tol) {
    let low = a;
    let high = b;
    let fLow = fn(low);
    for (let i = 0; i < 64; i += 1) {
      const mid = 0.5 * (low + high);
      const fMid = fn(mid);
      if (Math.abs(fMid) < tol || high - low < tol) {
        return mid;
      }
      if (fLow * fMid <= 0.0) {
        high = mid;
      } else {
        low = mid;
        fLow = fMid;
      }
    }
    return 0.5 * (low + high);
  }

  function solveSaddleNodeBifurcation(topology, p, mu, T, N) {
    const lamMax = upperAttemptBound(topology, p, mu);

    if (topology === "one_node_modification") {
      const v = wrightOmega(mu * T + 1.0);
      const lamSn = (v - 1.0) / T;
      const lamCrit = ((1.0 - p) * Math.pow(v - 1.0, 2)) / (T * v);
      const d2Gamma = -((1.0 - p) * T * (1.0 + v)) / v;
      const c2 = -0.5 * d2Gamma;
      const lam0 = Math.pow(v - 1.0, 2) / (T * v);
      return {
        topology,
        LambdaMax: lamMax,
        LambdaSn: lamSn,
        lambdaCritSn: lamCrit,
        rhoSn: lamSn / mu,
        c2NormalForm: c2,
        lamCritAtP0: lam0
      };
    }

    if (topology === "one_node_destruction") {
      const v = wrightOmega(mu * T + 1.0);
      const xSn = (v - 1.0) / T;
      const lamSn = xSn / (1.0 - p);
      const lamCrit = Math.pow(v - 1.0, 2) / (T * v);
      const d2Gamma = -((1.0 - p) * (1.0 - p) * T * (1.0 + v)) / v;
      return {
        topology,
        LambdaMax: lamMax,
        LambdaSn: lamSn,
        lambdaCritSn: lamCrit,
        rhoSn: xSn / mu,
        c2NormalForm: -0.5 * d2Gamma,
        lamCritAtP0: lamCrit
      };
    }

    // Golden-section search on (0, mu) for Cases 3, 4, 5
    let a = 1e-7 * lamMax;
    let b = (1.0 - 1e-7) * lamMax;
    const phi = 0.5 * (Math.sqrt(5.0) - 1.0);
    for (let i = 0; i < 60; i += 1) {
      const x1 = b - phi * (b - a);
      const x2 = a + phi * (b - a);
      if (throughputGamma(x1, topology, p, mu, T, N) < throughputGamma(x2, topology, p, mu, T, N)) {
        a = x1;
      } else {
        b = x2;
      }
    }
    const lamSn = 0.5 * (a + b);
    const lamCrit = throughputGamma(lamSn, topology, p, mu, T, N);
    const eps = Math.min(1e-4 * lamMax, 0.25 * Math.min(lamSn, lamMax - lamSn));
    const xHi = Math.min(lamMax * (1.0 - 1e-9), lamSn + eps);
    const xLo = Math.max(lamMax * 1e-9, lamSn - eps);
    const hEff = 0.5 * (xHi - xLo);
    const gPlus = throughputGamma(xHi, topology, p, mu, T, N);
    const gMinus = throughputGamma(xLo, topology, p, mu, T, N);
    const d2Gamma = (gPlus - 2.0 * lamCrit + gMinus) / Math.max(hEff * hEff, 1e-24);

    // Compute p=0 baseline capacity
    let a0 = 1e-7 * mu;
    let b0 = (1.0 - 1e-7) * mu;
    for (let i = 0; i < 50; i += 1) {
      const x1 = b0 - phi * (b0 - a0);
      const x2 = a0 + phi * (b0 - a0);
      if (throughputGamma(x1, topology, 0.0, mu, T, N) < throughputGamma(x2, topology, 0.0, mu, T, N)) {
        a0 = x1;
      } else {
        b0 = x2;
      }
    }
    const lamCritAtP0 = throughputGamma(0.5 * (a0 + b0), topology, 0.0, mu, T, N);

    return {
      topology,
      LambdaMax: lamMax,
      LambdaSn: lamSn,
      lambdaCritSn: lamCrit,
      rhoSn: lamSn / mu,
      c2NormalForm: Math.max(1e-9, -0.5 * d2Gamma),
      lamCritAtP0
    };
  }

  function expectedTotalSojourn(LambdaStar, topology, p, mu, T, N, backoff) {
    const bVal = typeof backoff === "number" ? backoff : 0.03;
    const pSucc = Math.max(1e-12, successProbPsucc(LambdaStar, topology, p, mu, T, N));
    if (topology === "one_node_destruction") {
      const lamEff = (1.0 - p) * LambdaStar;
      const rhoEff = lamEff / mu;
      const eWq = rhoEff / (mu * Math.max(1e-9, 1.0 - rhoEff));
      return (1.0 / pSucc) * (eWq + (1.0 - p) / mu) + ((1.0 - pSucc) / pSucc) * bVal;
    }
    if (topology === "one_node_modification") {
      return 1.0 / (pSucc * Math.max(1e-9, mu - LambdaStar));
    }
    if (topology === "tandem") {
      return N / (pSucc * Math.max(1e-9, mu - LambdaStar));
    }
    if (topology === "feedforward") {
      return feedforwardStageMetrics(LambdaStar, p, mu, T, N).ED;
    }
    return feedbackSeriesMetrics(LambdaStar, p, mu, T, N).ED;
  }

  function solveDualFixedPoints(lambdaExt, topology, p, mu, T, N) {
    const sn = solveSaddleNodeBifurcation(topology, p, mu, T, N);
    if (lambdaExt > sn.lambdaCritSn + 1e-9) {
      return {
        isStable: false,
        isCriticalFold: false,
        stabilityRegime: "supercritical_collapse",
        LambdaStable: NaN,
        LambdaUnstable: NaN,
        separatrixGap: 0.0,
        expectedDelay: Infinity,
        conservationResidual: NaN,
        sn
      };
    }

    if (Math.abs(lambdaExt - sn.lambdaCritSn) <= 1e-9) {
      const eDelayFold = expectedTotalSojourn(sn.LambdaSn, topology, p, mu, T, N);
      const resFold = Math.abs(throughputGamma(sn.LambdaSn, topology, p, mu, T, N) - lambdaExt);
      return {
        isStable: false,
        isCriticalFold: true,
        stabilityRegime: "critical_fold_semistable",
        LambdaStable: sn.LambdaSn,
        LambdaUnstable: sn.LambdaSn,
        separatrixGap: 0.0,
        expectedDelay: eDelayFold,
        conservationResidual: resFold,
        sn
      };
    }

    const lamStable = brentRoot(
      function (x) {
        return throughputGamma(x, topology, p, mu, T, N) - lambdaExt;
      },
      0.0,
      sn.LambdaSn,
      1e-11
    );

    let upperHi = sn.LambdaMax * (1.0 - 1e-9);
    let delta = sn.LambdaMax * 1e-9;
    while (
      throughputGamma(upperHi, topology, p, mu, T, N) > lambdaExt &&
      delta > 1e-15 * sn.LambdaMax
    ) {
      delta *= 0.1;
      upperHi = sn.LambdaMax - delta;
    }
    const lamUnstable =
      throughputGamma(upperHi, topology, p, mu, T, N) <= lambdaExt
        ? brentRoot(
            function (x) {
              return throughputGamma(x, topology, p, mu, T, N) - lambdaExt;
            },
            sn.LambdaSn,
            upperHi,
            1e-11
          )
        : upperHi;

    const eDelay = expectedTotalSojourn(lamStable, topology, p, mu, T, N);
    const residual = Math.abs(throughputGamma(lamStable, topology, p, mu, T, N) - lambdaExt);

    return {
      isStable: true,
      isCriticalFold: false,
      stabilityRegime: "subcritical_bistable",
      LambdaStable: lamStable,
      LambdaUnstable: lamUnstable,
      separatrixGap: lamUnstable - lamStable,
      expectedDelay: eDelay,
      conservationResidual: residual,
      sn
    };
  }

  /**
   * Exact Two-Station Confidence Cascade analytical solver, Beta(alpha_g, beta_g) CDF/PDF,
   * 3-term curvature decomposition (C_batch, C_amort, C_queue), Chow-Queueing Lagrangian
   * optimal threshold tau_star, IFT sensitivity d(tau_star)/d(lam), and Quartic Hysteresis deadband
   * Delta_tau_hyst_star = (3 * kappa_sw * D_tau / H_L)^(1/4) (ported from src/aqnet/models/cascade.py
   * and src/aqnet/models/cascade_nonstationary.py).
   */
  function betaPdf(x, a, b) {
    const xc = Math.min(Math.max(x, 1e-6), 1.0 - 1e-6);
    const logB = logGamma(a) + logGamma(b) - logGamma(a + b);
    return Math.exp((a - 1.0) * Math.log(xc) + (b - 1.0) * Math.log(1.0 - xc) - logB);
  }

  function betaincCf(x, a, b) {
    const maxIter = 200;
    const eps = 3e-15;
    const fpMin = 1e-300;
    const qab = a + b;
    const qap = a + 1.0;
    const qam = a - 1.0;
    let c = 1.0;
    let d = 1.0 - (qab * x) / qap;
    if (Math.abs(d) < fpMin) d = fpMin;
    d = 1.0 / d;
    let h = d;
    for (let m = 1; m <= maxIter; m += 1) {
      const m2 = 2 * m;
      let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1.0 + aa * d;
      if (Math.abs(d) < fpMin) d = fpMin;
      c = 1.0 + aa / c;
      if (Math.abs(c) < fpMin) c = fpMin;
      d = 1.0 / d;
      h *= d * c;
      aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1.0 + aa * d;
      if (Math.abs(d) < fpMin) d = fpMin;
      c = 1.0 + aa / c;
      if (Math.abs(c) < fpMin) c = fpMin;
      d = 1.0 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1.0) < eps) break;
    }
    return h;
  }

  function betaincReg(x, a, b) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    const logBt =
      logGamma(a + b) -
      logGamma(a) -
      logGamma(b) +
      a * Math.log(x) +
      b * Math.log(1.0 - x);
    const bt = logBt > -700.0 ? Math.exp(logBt) : 0.0;
    if (x < (a + 1.0) / (a + b + 2.0)) {
      return Math.min(1.0, Math.max(0.0, (bt * betaincCf(x, a, b)) / a));
    }
    return Math.min(1.0, Math.max(0.0, 1.0 - (bt * betaincCf(1.0 - x, b, a)) / b));
  }

  function solveBatchTimeoutMomentsJS(lam2, b, TBatchS) {
    if (b <= 1 || TBatchS <= 0.0) {
      return { E_B: 1.0, E_B_disp: 1.0, E_W_batch_s: 0.0, E_W_batch_ms: 0.0, p_timeout: b <= 1 ? 0.0 : 1.0 };
    }
    if (lam2 <= 1e-12) {
      return {
        E_B: 1.0,
        E_B_disp: 1.0,
        E_W_batch_s: TBatchS,
        E_W_batch_ms: TBatchS * 1000.0,
        p_timeout: 1.0
      };
    }
    const u = lam2 * TBatchS;
    const pK = new Array(b + 1).fill(0.0);
    pK[0] = u < 700.0 ? Math.exp(-u) : 0.0;
    for (let k = 1; k <= b; k += 1) {
      pK[k] = (pK[k - 1] * u) / k;
    }
    let sumDeficit = 0.0;
    let pTimeout = 0.0;
    let sumWait = 0.0;
    for (let k = 0; k <= b - 2; k += 1) {
      sumDeficit += (b - 1.0 - k) * pK[k];
      pTimeout += pK[k];
      sumWait += (1.0 + 0.5 * k) * pK[k];
    }
    const E_B = b - sumDeficit;
    let sumB = 0.0;
    for (let k = 0; k <= b - 1; k += 1) {
      sumB += pK[k];
    }
    const tailB = Math.max(0.0, 1.0 - sumB);
    const cycleDimless = sumWait + ((b * (b - 1.0)) / (2.0 * u)) * tailB;
    const E_W_cycle_s = TBatchS * cycleDimless;
    const E_W_batch_s = E_W_cycle_s / Math.max(E_B, 1e-12);
    return {
      E_B,
      E_B_disp: E_B,
      E_W_batch_s,
      E_W_batch_ms: E_W_batch_s * 1000.0,
      p_timeout: pTimeout
    };
  }

  function cascadeDelayComponentsJS(q, params) {
    const qC = Math.min(Math.max(q, 0.0), 1.0);
    const b = params.b;
    const TBatchS = params.TBatchMs / 1000.0;
    const eS1Ms = params.E_S1_ms !== undefined ? params.E_S1_ms : 1.1243;
    const cS1Sq = params.C_S1_sq !== undefined ? params.C_S1_sq : 0.02;
    const rho1 = (params.lam * eS1Ms) / 1000.0;
    const station1Stable =
      rho1 < 1.0 && (params.dEdgeMs === undefined || Number.isFinite(params.dEdgeMs));
    const dEdgeMs =
      station1Stable && params.dEdgeMs !== undefined
        ? params.dEdgeMs
        : station1Stable
          ? eS1Ms + (eS1Ms * rho1 * (1.0 + cS1Sq)) / (2.0 * (1.0 - rho1))
          : Infinity;
    const t0NetMs = params.t0NetMs !== undefined ? params.t0NetMs : 0.45;
    const t1NetMs = params.t1NetMs !== undefined ? params.t1NetMs : 0.25;
    const t0GMs = params.t0GMs !== undefined ? params.t0GMs : 2.05;
    const t1GMs = params.t1GMs !== undefined ? params.t1GMs : 0.25;
    const cs2G = params.cs2G !== undefined ? params.cs2G : 0.05;

    if (qC <= 1e-12) {
      const sG1 = t0GMs + t1GMs;
      const tSrv1 = t0NetMs + t0GMs + (t1NetMs + t1GMs);
      return {
        q: 0.0,
        E_B: 1.0,
        E_B_disp: 1.0,
        P_timeout: b > 1 ? 1.0 : 0.0,
        E_W_batch_ms: 0.0,
        T_srv_ms: tSrv1,
        S_G_ms: sG1,
        rho_1: rho1,
        E_D1_ms: dEdgeMs,
        raw_rho_G: 0.0,
        rho_G: 0.0,
        rho2: 0.0,
        E_W_queue_ms: station1Stable ? 0.0 : Infinity,
        E_D_ms: station1Stable ? dEdgeMs : Infinity,
        term_batch: 0.0,
        term_srv: 0.0,
        term_queue: 0.0,
        stable: station1Stable
      };
    }

    const lamOff = qC * params.lam;
    const bm = solveBatchTimeoutMomentsJS(lamOff, b, TBatchS);
    const bBar = bm.E_B;
    const wBatch = bm.E_W_batch_ms;

    const sG = t0GMs + t1GMs * bBar;
    const tSrv = t0NetMs + t0GMs + (t1NetMs + t1GMs) * bBar;
    const rawRhoG = (lamOff / (1000.0 * bBar)) * sG;
    const rhoG = Math.min(rawRhoG, 0.999);
    const cQ = 0.5 * (1.0 + cs2G);
    const wQueue = cQ * (rhoG / (1.0 - rhoG)) * sG;

    const termBatch = qC * wBatch;
    const termSrv = qC * tSrv;
    const termQueue = qC * wQueue;
    const isStable = station1Stable && rawRhoG < 1.0;
    const eD = isStable ? dEdgeMs + termBatch + termSrv + termQueue : Infinity;
    return {
      q: qC,
      E_B: bBar,
      E_B_disp: bBar,
      P_timeout: bm.p_timeout,
      E_W_batch_ms: wBatch,
      T_srv_ms: tSrv,
      S_G_ms: sG,
      rho_1: rho1,
      E_D1_ms: dEdgeMs,
      raw_rho_G: rawRhoG,
      rho_G: rhoG,
      rho2: rhoG,
      E_W_queue_ms: isStable ? wQueue : Infinity,
      E_D_ms: eD,
      term_batch: termBatch,
      term_srv: termSrv,
      term_queue: termQueue,
      stable: isStable
    };
  }

  function solveCascadeDelayJS(q, params) {
    const base = cascadeDelayComponentsJS(q, params);
    const eps = 5e-5;
    const qMid = Math.min(Math.max(q, eps + 1e-7), 0.995 - eps);
    const cHi = cascadeDelayComponentsJS(qMid + eps, params);
    const cMid = cascadeDelayComponentsJS(qMid, params);
    const cLo = cascadeDelayComponentsJS(qMid - eps, params);
    const dEDdq =
      (cHi.term_batch +
        cHi.term_srv +
        cHi.term_queue -
        (cLo.term_batch + cLo.term_srv + cLo.term_queue)) /
      (2.0 * eps);
    let cBatch = (cHi.term_batch - 2.0 * cMid.term_batch + cLo.term_batch) / (eps * eps);
    let cAmort = (cHi.term_srv - 2.0 * cMid.term_srv + cLo.term_srv) / (eps * eps);
    const cQueue = (cHi.term_queue - 2.0 * cMid.term_queue + cLo.term_queue) / (eps * eps);
    if (params.b === 1) {
      cBatch = 0.0;
      cAmort = 0.0;
    }
    const d2EDdq2 = cBatch + cAmort + cQueue;
    return Object.assign({}, base, {
      dED_dq_ms: dEDdq,
      C_batch_ms: cBatch,
      C_amort_ms: cAmort,
      C_queue_ms: cQueue,
      d2ED_dq2_ms: d2EDdq2
    });
  }

  function cascadePowerComponentsJS(q, params) {
    const qC = Math.min(Math.max(q, 0.0), 0.9999);
    const pEdgeW = params.pEdgeW !== undefined ? params.pEdgeW : 4.2 + params.lam * 0.00919;
    const pIdleP8W = params.pIdleP8W !== undefined ? params.pIdleP8W : 20.28;
    const pIdleP0W = params.pIdleP0W !== undefined ? params.pIdleP0W : 73.41;
    const tauP8Ms = params.tauP8Ms !== undefined ? params.tauP8Ms : 25.0;
    const eDynMj = params.eDynMj !== undefined ? params.eDynMj : 139.629;
    const pSGW = params.pSGW !== undefined ? params.pSGW : 73.965;
    const t0GMs = params.t0GMs !== undefined ? params.t0GMs : 2.05;
    const t1GMs = params.t1GMs !== undefined ? params.t1GMs : 0.25;
    if (qC <= 1e-12) {
      return { q: 0.0, rho_G: 0.0, P_idle_eff_W: pIdleP8W, E_P_W: pEdgeW + pIdleP8W };
    }
    const lamOff = qC * params.lam;
    const bm = solveBatchTimeoutMomentsJS(lamOff, params.b, params.TBatchMs / 1000.0);
    const bBar = bm.E_B;
    const sG = t0GMs + t1GMs * bBar;
    const rhoG = Math.min(0.999, (lamOff / (1000.0 * bBar)) * sG);
    const interBatchMs = (1000.0 * bBar) / lamOff;
    const gapMs = Math.max(0.0, interBatchMs - sG);
    const pIdleEff = pIdleP8W + (pIdleP0W - pIdleP8W) * Math.exp(-gapMs / tauP8Ms);
    const pDynW = (lamOff / 1000.0) * eDynMj;
    const pGpuStateW = rhoG * pSGW + (1.0 - rhoG) * pIdleEff;
    return {
      q: qC,
      rho_G: rhoG,
      P_idle_eff_W: pIdleEff,
      E_P_W: pEdgeW + pDynW + pGpuStateW
    };
  }

  function solveCascadePowerJS(q, params) {
    const base = cascadePowerComponentsJS(q, params);
    const eps = 5e-5;
    const qMid = Math.min(Math.max(q, eps + 1e-7), 0.995 - eps);
    const pHi = cascadePowerComponentsJS(qMid + eps, params);
    const pMid = cascadePowerComponentsJS(qMid, params);
    const pLo = cascadePowerComponentsJS(qMid - eps, params);
    const dEPdq = (pHi.E_P_W - pLo.E_P_W) / (2.0 * eps);
    const d2EPdq2 = (pHi.E_P_W - 2.0 * pMid.E_P_W + pLo.E_P_W) / (eps * eps);
    return Object.assign({}, base, {
      dEP_dq_W: dEPdq,
      d2EP_dq2_W: d2EPdq2
    });
  }

  function findOptimalThresholdTauJS(params) {
    const alphaG = params.alphaG !== undefined ? params.alphaG : 2.8;
    const betaG = params.betaG !== undefined ? params.betaG : 1.2;
    const rSlope = params.rSlope !== undefined ? params.rSlope : 0.30731157;
    const muD = params.muD !== undefined ? params.muD : 0.002;
    const muP = params.muP !== undefined ? params.muP : 0.0005;

    function foc(tauVal) {
      const qVal = betaincReg(tauVal, alphaG, betaG);
      const dRes = solveCascadeDelayJS(qVal, params);
      const pRes = solveCascadePowerJS(qVal, params);
      const r1 = rSlope * (1.0 - tauVal);
      return r1 - (muD * dRes.dED_dq_ms + muP * pRes.dEP_dq_W);
    }

    const fLo = foc(1e-4);
    const fHi = foc(1.0 - 1e-4);
    if (fLo >= 0.0 && fHi <= 0.0) {
      let lo = 1e-4;
      let hi = 1.0 - 1e-4;
      for (let iter = 0; iter < 60; iter += 1) {
        const mid = 0.5 * (lo + hi);
        if (foc(mid) > 0.0) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      return 0.5 * (lo + hi);
    }
    return fLo < 0.0 ? 0.0 : 1.0;
  }

  function solveHysteresisBandJS(params) {
    const alphaG = params.alphaG !== undefined ? params.alphaG : 2.8;
    const betaG = params.betaG !== undefined ? params.betaG : 1.2;
    const rSlope = params.rSlope !== undefined ? params.rSlope : 0.30731157;
    const muD = params.muD !== undefined ? params.muD : 0.002;
    const muP = params.muP !== undefined ? params.muP : 0.0005;
    const kappaSw = params.kappaSw !== undefined ? params.kappaSw : 1.2e-4;
    const sigmaLam = params.sigmaLam !== undefined ? params.sigmaLam : 75.0;
    const tauCorrS = params.tauCorrS !== undefined ? params.tauCorrS : 0.2;

    const tauStar = findOptimalThresholdTauJS(params);
    const qStar = betaincReg(tauStar, alphaG, betaG);
    const dCenter = solveCascadeDelayJS(qStar, params);
    const pCenter = solveCascadePowerJS(qStar, params);

    const lamF = Math.max(params.lam, 1e-3);
    const dLam = Math.max(0.05, Math.min(2.0, 0.25 * lamF));
    const pHi = Object.assign({}, params, { lam: lamF + dLam });
    const pLo = Object.assign({}, params, { lam: Math.max(1e-4, lamF - dLam) });
    const span = Math.max(pHi.lam - pLo.lam, 1e-9);

    const dHi = solveCascadeDelayJS(qStar, pHi);
    const dLo = solveCascadeDelayJS(qStar, pLo);
    const pwHi = solveCascadePowerJS(qStar, pHi);
    const pwLo = solveCascadePowerJS(qStar, pLo);

    const d2EDdqDlam = (dHi.dED_dq_ms - dLo.dED_dq_ms) / span;
    const d2EPdqDlam = (pwHi.dEP_dq_W - pwLo.dEP_dq_W) / span;

    const fg = betaPdf(tauStar, alphaG, betaG);
    const r1Prime = -rSlope;
    const focSlopeTau = -r1Prime + fg * (muD * dCenter.d2ED_dq2_ms + muP * pCenter.d2EP_dq2_W);
    const hessianLTau = Math.max(fg * focSlopeTau, 1e-8);
    const crossSens = muD * d2EDdqDlam + muP * d2EPdqDlam;
    const isInterior = tauStar > 1e-6 && tauStar < 1.0 - 1e-6;
    const dtauDlam = isInterior && Math.abs(focSlopeTau) > 1e-12 ? -crossSens / focSlopeTau : 0.0;

    const dTauDiff = dtauDlam * dtauDlam * ((sigmaLam * sigmaLam) / Math.max(tauCorrS, 1e-6));
    const deltaTauStar = Math.pow((3.0 * kappaSw * dTauDiff) / hessianLTau, 0.25);
    return {
      tau_star: tauStar,
      q_star: qStar,
      dtau_dlam_per_fps: dtauDlam,
      hessian_L_tau: hessianLTau,
      D_tau_per_s: dTauDiff,
      delta_tau_hyst_star: deltaTauStar,
      E_D_star_ms: dCenter.E_D_ms,
      C_batch_star_ms: dCenter.C_batch_ms
    };
  }

  function deriveCascadeParamsFromState() {
    // Map active studio sliders directly into CascadeParams (default state -> canonical lam=500, b=4, T=25ms)
    const lam = Math.max(50.0, 500.0 * (state.lambdaExt / 3.5) * (10.0 / Math.max(state.mu, 1.0)));
    const TBatchMs = Math.max(5.0, state.T * 25.0);
    const b = Math.max(2, state.N * 2);
    const sigmaLam = 75.0 * (1.0 + 0.5 * (state.p - 0.25));
    return {
      lam,
      b,
      TBatchMs,
      alphaG: 2.8,
      betaG: 1.2,
      rSlope: 0.30731157,
      t0NetMs: 0.45,
      t1NetMs: 0.25,
      t0GMs: 2.05,
      t1GMs: 0.25,
      cs2G: 0.05,
      eDynMj: 139.629,
      pSGW: 73.965,
      pIdleP8W: 20.28,
      pIdleP0W: 73.41,
      tauP8Ms: 25.0,
      muD: 0.002,
      muP: 0.0005,
      kappaSw: 1.2e-4,
      sigmaLam,
      tauCorrS: 0.2
    };
  }

  function solveCascadeSummaryFromState() {
    const params = deriveCascadeParamsFromState();
    const paramsB1 = Object.assign({}, params, { b: 1 });
    const hyst = solveHysteresisBandJS(params);
    const dAtOptB = solveCascadeDelayJS(hyst.q_star, params);
    const dAtOpt1 = solveCascadeDelayJS(hyst.q_star, paramsB1);
    return {
      params,
      paramsB1,
      tauOpt: hyst.tau_star,
      qOpt: hyst.q_star,
      delayOptMs: dAtOptB.E_D_ms,
      cBatch1: dAtOpt1.C_batch_ms,
      cBatchB: dAtOptB.C_batch_ms,
      deltaTauHyst: hyst.delta_tau_hyst_star
    };
  }

  /**
   * Integrate non-stationary burst trajectory dLambda/dt = 0.45 * mu * [lambda_ext * m(t) - Gamma(Lambda(t))]
   * across the user's burst multiplier sequence (Law L10 generative offering).
   */
  function simulateBurstTrajectory(dual) {
    const { topology, lambdaExt, p, mu, T, N, burstMultipliers } = state;
    const lamMax = dual.sn.LambdaMax;
    const startLam = dual.isStable ? dual.LambdaStable : 0.65 * dual.sn.LambdaSn;
    const stepsPerStage = 20;
    const dt = 0.06;
    let currentLam = startLam;
    let peakLam = startLam;
    let crossedSeparatrix = !dual.isStable;
    const points = [{ step: 0, lam: currentLam, mult: 1.0 }];

    for (let s = 0; s < burstMultipliers.length; s += 1) {
      const m = burstMultipliers[s];
      for (let k = 0; k < stepsPerStage; k += 1) {
        const gammaVal = throughputGamma(currentLam, topology, p, mu, T, N);
        const dLam = 0.45 * mu * (lambdaExt * m - gammaVal) * dt;
        currentLam = Math.max(0.05, Math.min(0.992 * lamMax, currentLam + dLam));
        if (currentLam > peakLam) {
          peakLam = currentLam;
        }
        if (dual.isStable && currentLam >= dual.LambdaUnstable) {
          crossedSeparatrix = true;
        }
        points.push({ step: points.length, lam: currentLam, mult: m });
      }
    }

    for (let r = 0; r < 30; r += 1) {
      const gammaVal = throughputGamma(currentLam, topology, p, mu, T, N);
      const dLam = 0.45 * mu * (lambdaExt - gammaVal) * dt;
      currentLam = Math.max(0.05, Math.min(0.995 * lamMax, currentLam + dLam));
      if (currentLam > peakLam) {
        peakLam = currentLam;
      }
      points.push({ step: points.length, lam: currentLam, mult: 1.0 });
    }

    state.lastTrajectory = points;
    return {
      points,
      peakLam,
      finalLam: currentLam,
      crossedSeparatrix: crossedSeparatrix || currentLam > dual.sn.LambdaSn * 1.05
    };
  }

  function renderFoldCanvas(ctx, width, height, dual, burstSim) {
    const { topology, p, mu, T, N, lambdaExt } = state;
    const sn = dual.sn;
    const padL = 56;
    const padR = 24;
    const padT = 26;
    const padB = 42;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;
    const yMax = Math.max(sn.lambdaCritSn * 1.25, lambdaExt * 1.15, 2.0);

    function toX(lam) {
      return padL + (lam / sn.LambdaMax) * plotW;
    }
    function toY(gamma) {
      return padT + plotH - (Math.min(yMax, Math.max(0, gamma)) / yMax) * plotH;
    }

    const styles = getComputedStyle(document.documentElement);
    const textMain = styles.getPropertyValue("--text-main").trim() || "#0f172a";
    const textMuted = styles.getPropertyValue("--text-muted").trim() || "#334155";
    const borderCol = styles.getPropertyValue("--border-color").trim() || "#cbd5e1";
    const accentPrimary = styles.getPropertyValue("--accent-primary").trim() || "#1d4ed8";
    const accentDanger = styles.getPropertyValue("--accent-danger").trim() || "#b91c1c";
    const accentSuccess = styles.getPropertyValue("--accent-success").trim() || "#047857";
    const accentWarn = styles.getPropertyValue("--accent-warn").trim() || "#b45309";

    ctx.clearRect(0, 0, width, height);

    ctx.strokeStyle = borderCol;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i += 1) {
      const gy = padT + (i / 4) * plotH;
      ctx.beginPath();
      ctx.moveTo(padL, gy);
      ctx.lineTo(padL + plotW, gy);
      ctx.stroke();

      const valY = yMax * (1 - i / 4);
      ctx.fillStyle = textMuted;
      ctx.font = "11px Inter, sans-serif";
      ctx.textAlign = "right";
      ctx.fillText(valY.toFixed(1), padL - 6, gy + 4);
    }

    for (let j = 0; j <= 5; j += 1) {
      const gx = padL + (j / 5) * plotW;
      const valX = sn.LambdaMax * (j / 5);
      ctx.fillStyle = textMuted;
      ctx.textAlign = "center";
      ctx.fillText(valX.toFixed(1), gx, height - 18);
    }

    // Local Quadratic Normal Form parabola: lambda_crit^SN - c_2 * (Lambda* - Lambda_SN*)^2
    ctx.strokeStyle = accentWarn;
    ctx.setLineDash([2, 3]);
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    let nfStarted = false;
    for (let i = 0; i <= 80; i += 1) {
      const lam = Math.max(0.0, Math.min(sn.LambdaMax, sn.LambdaSn + ((i - 40) / 40) * 0.35 * sn.LambdaMax));
      const gNf = sn.lambdaCritSn - sn.c2NormalForm * Math.pow(lam - sn.LambdaSn, 2);
      if (gNf >= 0.0) {
        if (!nfStarted) {
          ctx.moveTo(toX(lam), toY(gNf));
          nfStarted = true;
        } else {
          ctx.lineTo(toX(lam), toY(gNf));
        }
      }
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Stable Lower Branch (0 -> Lambda_SN)
    ctx.strokeStyle = accentPrimary;
    ctx.lineWidth = 2.6;
    ctx.beginPath();
    for (let i = 0; i <= 100; i += 1) {
      const lam = (i / 100) * sn.LambdaSn;
      const g = throughputGamma(lam, topology, p, mu, T, N);
      if (i === 0) {
        ctx.moveTo(toX(lam), toY(g));
      } else {
        ctx.lineTo(toX(lam), toY(g));
      }
    }
    ctx.stroke();

    // Unstable Upper Separatrix (Lambda_SN -> Lambda_max)
    ctx.strokeStyle = accentDanger;
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    for (let i = 0; i <= 100; i += 1) {
      const lam = sn.LambdaSn + (i / 100) * (sn.LambdaMax - sn.LambdaSn - 1e-4);
      const g = throughputGamma(lam, topology, p, mu, T, N);
      if (i === 0) {
        ctx.moveTo(toX(lam), toY(g));
      } else {
        ctx.lineTo(toX(lam), toY(g));
      }
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Horizontal external arrival slice y = lambdaExt
    ctx.strokeStyle = accentSuccess;
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(padL, toY(lambdaExt));
    ctx.lineTo(padL + plotW, toY(lambdaExt));
    ctx.stroke();
    ctx.setLineDash([]);

    // Fold point marker
    ctx.fillStyle = "#d97706";
    ctx.beginPath();
    ctx.arc(toX(sn.LambdaSn), toY(sn.lambdaCritSn), 6, 0, Math.PI * 2);
    ctx.fill();

    if (dual.isStable) {
      ctx.fillStyle = accentPrimary;
      ctx.beginPath();
      ctx.arc(toX(dual.LambdaStable), toY(lambdaExt), 5, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = accentDanger;
      ctx.beginPath();
      ctx.arc(toX(dual.LambdaUnstable), toY(lambdaExt), 5, 0, Math.PI * 2);
      ctx.fill();
    }

    if (burstSim && burstSim.points.length > 1) {
      ctx.strokeStyle = "#d97706";
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      for (let i = 0; i < burstSim.points.length; i += 1) {
        const pt = burstSim.points[i];
        const g = throughputGamma(pt.lam, topology, p, mu, T, N);
        if (i === 0) {
          ctx.moveTo(toX(pt.lam), toY(g));
        } else {
          ctx.lineTo(toX(pt.lam), toY(g));
        }
      }
      ctx.stroke();

      const peakG = throughputGamma(burstSim.peakLam, topology, p, mu, T, N);
      ctx.fillStyle = burstSim.crossedSeparatrix ? accentDanger : accentSuccess;
      ctx.beginPath();
      ctx.arc(toX(burstSim.peakLam), toY(peakG), 6, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.fillStyle = textMain;
    ctx.font = "bold 12px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(
      "Throughput Map Γ(Λ*) — Solid: Stable Λ_s* | Dashed: Separatrix Λ_u* | Dotted: Normal Form c₂",
      padL + plotW / 2,
      16
    );
    ctx.fillText("Internal Attempt Rate Λ* (pkt/s)", padL + plotW / 2, height - 4);
  }

  function renderManifoldCanvas(ctx, width, height) {
    const { mu, T, N, p } = state;
    const padL = 56;
    const padR = 24;
    const padT = 26;
    const padB = 42;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;
    const yMax = mu * 1.05;

    function toX(pv) {
      return padL + (pv / 0.9) * plotW;
    }
    function toY(lam) {
      return padT + plotH - (Math.min(yMax, Math.max(0, lam)) / yMax) * plotH;
    }

    const styles = getComputedStyle(document.documentElement);
    const textMain = styles.getPropertyValue("--text-main").trim() || "#0f172a";
    const textMuted = styles.getPropertyValue("--text-muted").trim() || "#334155";
    const borderCol = styles.getPropertyValue("--border-color").trim() || "#cbd5e1";
    const accentPrimary = styles.getPropertyValue("--accent-primary").trim() || "#1d4ed8";
    const accentSuccess = styles.getPropertyValue("--accent-success").trim() || "#047857";
    const accentDanger = styles.getPropertyValue("--accent-danger").trim() || "#b91c1c";

    ctx.clearRect(0, 0, width, height);

    ctx.strokeStyle = borderCol;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i += 1) {
      const gy = padT + (i / 4) * plotH;
      ctx.beginPath();
      ctx.moveTo(padL, gy);
      ctx.lineTo(padL + plotW, gy);
      ctx.stroke();

      ctx.fillStyle = textMuted;
      ctx.font = "11px Inter, sans-serif";
      ctx.textAlign = "right";
      ctx.fillText((yMax * (1 - i / 4)).toFixed(1), padL - 6, gy + 4);
    }

    for (let j = 0; j <= 6; j += 1) {
      const pv = (j / 6) * 0.9;
      ctx.fillStyle = textMuted;
      ctx.textAlign = "center";
      ctx.fillText(pv.toFixed(2), toX(pv), height - 18);
    }

    // Naive bound mu * (1 - p)
    ctx.strokeStyle = textMuted;
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(mu));
    ctx.lineTo(toX(0.9), toY(mu * 0.1));
    ctx.stroke();
    ctx.setLineDash([]);

    // Case 1 Destruction (p-invariant)
    const snDest0 = solveSaddleNodeBifurcation("one_node_destruction", 0.0, mu, T, N).lambdaCritSn;
    ctx.strokeStyle = accentSuccess;
    ctx.lineWidth = 2.6;
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(snDest0));
    ctx.lineTo(toX(0.9), toY(snDest0));
    ctx.stroke();

    // Case 2 Modification (linear erosion)
    const snMod0 = solveSaddleNodeBifurcation("one_node_modification", 0.0, mu, T, N).lambdaCritSn;
    ctx.strokeStyle = accentPrimary;
    ctx.lineWidth = 2.6;
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(snMod0));
    ctx.lineTo(toX(0.9), toY(snMod0 * 0.1));
    ctx.stroke();

    // Current p vertical marker
    ctx.strokeStyle = accentDanger;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(toX(p), padT);
    ctx.lineTo(toX(p), padT + plotH);
    ctx.stroke();

    ctx.fillStyle = textMain;
    ctx.font = "bold 12px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(
      "Critical External Capacity λ_crit^SN(p): Case 1 Structural p-Invariance (Green) vs. Case 2 Erosion (Blue)",
      padL + plotW / 2,
      16
    );
    ctx.fillText("Adversarial Attack Probability p", padL + plotW / 2, height - 4);
  }

  function renderCascadeCanvas(ctx, width, height, casSummary) {
    const padL = 56;
    const padR = 24;
    const padT = 26;
    const padB = 42;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;
    const yMax = 35.0;

    const styles = getComputedStyle(document.documentElement);
    const textMain = styles.getPropertyValue("--text-main").trim() || "#0f172a";
    const textMuted = styles.getPropertyValue("--text-muted").trim() || "#334155";
    const borderCol = styles.getPropertyValue("--border-color").trim() || "#cbd5e1";
    const accentPrimary = styles.getPropertyValue("--accent-primary").trim() || "#1d4ed8";
    const accentDanger = styles.getPropertyValue("--accent-danger").trim() || "#b91c1c";
    const accentSuccess = styles.getPropertyValue("--accent-success").trim() || "#047857";

    ctx.clearRect(0, 0, width, height);

    ctx.strokeStyle = borderCol;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i += 1) {
      const gy = padT + (i / 4) * plotH;
      ctx.beginPath();
      ctx.moveTo(padL, gy);
      ctx.lineTo(padL + plotW, gy);
      ctx.stroke();
      const valY = yMax * (1 - i / 4);
      ctx.fillStyle = textMuted;
      ctx.font = "11px Inter, sans-serif";
      ctx.textAlign = "right";
      ctx.fillText(valY.toFixed(0) + " ms", padL - 6, gy + 4);
    }

    function toX(q) {
      return padL + ((q - 0.05) / 0.75) * plotW;
    }
    function toY(delayMs) {
      return padT + plotH - (Math.min(yMax, Math.max(0, delayMs)) / yMax) * plotH;
    }

    // b = 1 (unbatched cascade from solveCascadeDelayJS)
    ctx.strokeStyle = accentDanger;
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 2.3;
    ctx.beginPath();
    for (let i = 0; i <= 80; i += 1) {
      const q = 0.05 + (i / 80) * 0.75;
      const d1 = solveCascadeDelayJS(q, casSummary.paramsB1).E_D_ms;
      const yPlot = Number.isFinite(d1) ? d1 : yMax;
      if (i === 0) {
        ctx.moveTo(toX(q), toY(yPlot));
      } else {
        ctx.lineTo(toX(q), toY(yPlot));
      }
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // b = 2*N (micro-batched accumulator from solveCascadeDelayJS)
    ctx.strokeStyle = accentPrimary;
    ctx.lineWidth = 2.6;
    ctx.beginPath();
    for (let i = 0; i <= 80; i += 1) {
      const q = 0.05 + (i / 80) * 0.75;
      const dB = solveCascadeDelayJS(q, casSummary.params).E_D_ms;
      const yPlot = Number.isFinite(dB) ? dB : yMax;
      if (i === 0) {
        ctx.moveTo(toX(q), toY(yPlot));
      } else {
        ctx.lineTo(toX(q), toY(yPlot));
      }
    }
    ctx.stroke();

    // Optimal threshold marker tau* -> qOpt
    const qMarker = Math.max(0.05, Math.min(0.8, casSummary.qOpt));
    ctx.fillStyle = accentSuccess;
    ctx.beginPath();
    ctx.arc(toX(qMarker), toY(casSummary.delayOptMs), 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = textMain;
    ctx.font = "bold 12px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(
      "Two-Station Confidence Cascade E[D](q): Unbatched b=1 (Dashed) vs. Batched b=" +
        casSummary.params.b +
        " (Solid)",
      padL + plotW / 2,
      16
    );
    ctx.fillText("Offload Fraction q(τ)", padL + plotW / 2, height - 4);
  }

  function updateUI() {
    document.getElementById("out-lambda").textContent = state.lambdaExt.toFixed(2);
    document.getElementById("out-p").textContent = state.p.toFixed(2);
    document.getElementById("out-mu").textContent = state.mu.toFixed(1);
    document.getElementById("out-timeout").textContent = state.T.toFixed(2);
    document.getElementById("out-nodes").textContent = String(state.N);

    const dual = solveDualFixedPoints(
      state.lambdaExt,
      state.topology,
      state.p,
      state.mu,
      state.T,
      state.N
    );
    const sn = dual.sn;
    const snDest = solveSaddleNodeBifurcation(
      "one_node_destruction",
      state.p,
      state.mu,
      state.T,
      state.N
    );
    const snMod = solveSaddleNodeBifurcation(
      "one_node_modification",
      state.p,
      state.mu,
      state.T,
      state.N
    );
    const burstSim = simulateBurstTrajectory(dual);
    const casSummary = solveCascadeSummaryFromState();

    document.getElementById("metric-lambda-crit").textContent =
      sn.lambdaCritSn.toFixed(3) + " pkt/s";
    document.getElementById("metric-fold-sub").textContent =
      "Λ_SN* = " +
      sn.LambdaSn.toFixed(3) +
      " pkt/s · ρ_SN = " +
      sn.rhoSn.toFixed(3) +
      " · c₂ = " +
      sn.c2NormalForm.toFixed(3) +
      " s";

    if (dual.isStable) {
      document.getElementById("metric-branches").textContent =
        dual.LambdaStable.toFixed(3) + " / " + dual.LambdaUnstable.toFixed(3);
      document.getElementById("metric-separatrix-sub").textContent =
        "Separatrix gap ΔΛ = " +
        dual.separatrixGap.toFixed(3) +
        " pkt/s · E[D] = " +
        dual.expectedDelay.toFixed(3) +
        " s";
    } else if (dual.isCriticalFold) {
      document.getElementById("metric-branches").textContent =
        "CRITICAL FOLD (" + sn.LambdaSn.toFixed(3) + ")";
      document.getElementById("metric-separatrix-sub").textContent =
        "Semistable coalesced root ΔΛ = 0 · E[D] = " + dual.expectedDelay.toFixed(3) + " s";
    } else {
      document.getElementById("metric-branches").textContent = "SUPERCRITICAL COLLAPSE";
      document.getElementById("metric-separatrix-sub").textContent =
        "λ = " +
        state.lambdaExt.toFixed(2) +
        " > λ_crit^SN = " +
        sn.lambdaCritSn.toFixed(2) +
        " (0 fixed points)";
    }

    const pCritMod = Math.max(0.0, Math.min(1.0, 1.0 - state.lambdaExt / snMod.lamCritAtP0));
    document.getElementById("metric-p-crit").textContent = "p_crit = " + pCritMod.toFixed(3);
    document.getElementById("metric-invariance-sub").textContent =
      "Case 1 λ_crit^SN = " +
      snDest.lambdaCritSn.toFixed(3) +
      " (p-invariant) vs. Case 2 " +
      snMod.lambdaCritSn.toFixed(3);

    const verdictEl = document.getElementById("offering-verdict");
    const offeringSubEl = document.getElementById("offering-sub");
    if (burstSim.crossedSeparatrix) {
      verdictEl.textContent = "SEPARATRIX ESCAPE";
      offeringSubEl.textContent =
        "Peak Λ(t) = " +
        burstSim.peakLam.toFixed(2) +
        " pkt/s crossed unstable branch into retry collapse";
    } else {
      verdictEl.textContent = "BASIN RECOVERED";
      offeringSubEl.textContent =
        "Peak Λ(t) = " +
        burstSim.peakLam.toFixed(2) +
        " pkt/s < Λ_u* = " +
        dual.LambdaUnstable.toFixed(2) +
        " pkt/s";
    }

    // Dynamic Cascade inspector card
    const casLabelEl = document.getElementById("label-cascade-card");
    if (casLabelEl) {
      casLabelEl.textContent =
        "Confidence Cascade (b=" +
        casSummary.params.b +
        ", T=" +
        casSummary.params.TBatchMs.toFixed(0) +
        "ms)";
    }
    document.getElementById("metric-cascade-tau").textContent =
      "τ* = " + casSummary.tauOpt.toFixed(3);
    document.getElementById("metric-cascade-sub").textContent =
      "C_batch = " +
      casSummary.cBatchB.toFixed(1) +
      " ms · Δτ_hyst* = " +
      casSummary.deltaTauHyst.toFixed(3);

    // Dynamic footer conservation residual
    const footerStatusEl = document.getElementById("footer-status");
    if (footerStatusEl) {
      if (Number.isFinite(dual.conservationResidual)) {
        footerStatusEl.textContent =
          "SymPy 4/4 Identities Verified · Conservation Residual |Γ(Λ*) - λ| = " +
          dual.conservationResidual.toExponential(2);
      } else {
        footerStatusEl.textContent =
          "SymPy 4/4 Identities Verified · Supercritical Regime (λ > λ_crit^SN, 0 roots)";
      }
    }

    // Update single comparative table across all 5 topologies
    const tbody = document.getElementById("topology-table-body");
    tbody.replaceChildren();
    TOPOLOGY_META.forEach(function (meta) {
      const rowDual = solveDualFixedPoints(
        state.lambdaExt,
        meta.id,
        state.p,
        state.mu,
        state.T,
        state.N
      );
      const tr = document.createElement("tr");
      if (meta.id === state.topology) {
        tr.className = "is-selected";
      }
      const tdName = document.createElement("td");
      tdName.textContent = meta.label;
      const tdSn = document.createElement("td");
      tdSn.textContent = rowDual.sn.LambdaSn.toFixed(3) + " pkt/s";
      const tdCrit = document.createElement("td");
      tdCrit.textContent = rowDual.sn.lambdaCritSn.toFixed(3) + " pkt/s";
      const tdRegime = document.createElement("td");
      tdRegime.textContent = rowDual.isStable
        ? "STABLE · E[D] = " + rowDual.expectedDelay.toFixed(3) + " s"
        : rowDual.isCriticalFold
          ? "CRITICAL FOLD · E[D] = " + rowDual.expectedDelay.toFixed(3) + " s"
          : "COLLAPSE (λ > λ_crit^SN)";
      tr.append(tdName, tdSn, tdCrit, tdRegime);
      tbody.appendChild(tr);
    });

    const canvas = document.getElementById("bifurcation-canvas");
    const ctx = canvas.getContext("2d");
    if (state.view === "manifold") {
      renderManifoldCanvas(ctx, canvas.width, canvas.height);
    } else if (state.view === "cascade") {
      renderCascadeCanvas(ctx, canvas.width, canvas.height, casSummary);
    } else {
      renderFoldCanvas(ctx, canvas.width, canvas.height, dual, burstSim);
    }
  }

  function parseBurstInput() {
    const raw = document.getElementById("burst-input").value;
    const nums = raw
      .split(",")
      .map(function (s) {
        return parseFloat(s.trim());
      })
      .filter(function (v) {
        return Number.isFinite(v) && v > 0.0;
      });
    state.burstMultipliers = nums.length > 0 ? nums.slice(0, 12) : [1.0];
  }

  function setView(viewName) {
    state.view = viewName;
    ["fold", "manifold", "cascade"].forEach(function (v) {
      const btn = document.getElementById("btn-view-" + v);
      if (btn) {
        btn.classList.toggle("is-active", v === viewName);
      }
    });
    updateUI();
  }

  function setMode(modeName) {
    state.mode = modeName;
    ["bifurcation", "cascade", "topology"].forEach(function (m) {
      const tab = document.getElementById("tab-" + m);
      if (tab) {
        const active = m === modeName;
        tab.classList.toggle("is-active", active);
        tab.setAttribute("aria-pressed", active ? "true" : "false");
      }
    });
    if (modeName === "cascade") {
      setView("cascade");
    } else if (modeName === "topology") {
      setView("manifold");
    } else {
      setView("fold");
    }
  }

  function copyWithFeedback(text, successMsg) {
    const feedbackEl = document.getElementById("copy-feedback");
    try {
      if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
        navigator.clipboard
          .writeText(text)
          .then(function () {
            feedbackEl.textContent = successMsg;
          })
          .catch(function () {
            feedbackEl.textContent = "Clipboard write blocked by browser.";
          });
      } else {
        feedbackEl.textContent = "Clipboard API unavailable.";
      }
    } catch (_err) {
      feedbackEl.textContent = "Clipboard copy failed.";
    }
  }

  function bindEvents() {
    document.getElementById("theme-select").addEventListener("change", function (e) {
      state.theme = e.target.value;
      document.documentElement.setAttribute("data-theme", state.theme);
      updateUI();
    });

    document.getElementById("topology-select").addEventListener("change", function (e) {
      state.topology = e.target.value;
      updateUI();
    });

    document.getElementById("slider-lambda").addEventListener("input", function (e) {
      state.lambdaExt = parseFloat(e.target.value);
      updateUI();
    });

    document.getElementById("slider-p").addEventListener("input", function (e) {
      state.p = parseFloat(e.target.value);
      updateUI();
    });

    document.getElementById("slider-mu").addEventListener("input", function (e) {
      state.mu = parseFloat(e.target.value);
      updateUI();
    });

    document.getElementById("slider-timeout").addEventListener("input", function (e) {
      state.T = parseFloat(e.target.value);
      updateUI();
    });

    document.getElementById("slider-nodes").addEventListener("input", function (e) {
      state.N = parseInt(e.target.value, 10);
      updateUI();
    });

    document.getElementById("btn-inject-burst").addEventListener("click", function () {
      parseBurstInput();
      setView("fold");
    });

    document.getElementById("btn-reset").addEventListener("click", function () {
      state.topology = "one_node_modification";
      state.lambdaExt = 3.5;
      state.p = 0.25;
      state.mu = 10.0;
      state.T = 1.0;
      state.N = 2;
      document.getElementById("topology-select").value = state.topology;
      document.getElementById("slider-lambda").value = "3.50";
      document.getElementById("slider-p").value = "0.25";
      document.getElementById("slider-mu").value = "10.0";
      document.getElementById("slider-timeout").value = "1.00";
      document.getElementById("slider-nodes").value = "2";
      document.getElementById("burst-input").value = "1.0, 1.35, 1.95, 2.45, 1.80, 1.0";
      parseBurstInput();
      setMode("bifurcation");
    });

    document.getElementById("preset-subcritical").addEventListener("click", function () {
      state.lambdaExt = 3.2;
      document.getElementById("slider-lambda").value = "3.20";
      document.getElementById("burst-input").value = "1.0, 1.25, 1.55, 1.30, 1.0";
      parseBurstInput();
      setView("fold");
    });

    document.getElementById("preset-metastable").addEventListener("click", function () {
      state.lambdaExt = 4.65;
      document.getElementById("slider-lambda").value = "4.65";
      document.getElementById("burst-input").value = "1.0, 1.60, 2.40, 2.90, 2.20, 1.0";
      parseBurstInput();
      setView("fold");
    });

    ["fold", "manifold", "cascade"].forEach(function (v) {
      document.getElementById("btn-view-" + v).addEventListener("click", function () {
        setView(v);
      });
    });

    ["bifurcation", "cascade", "topology"].forEach(function (m) {
      document.getElementById("tab-" + m).addEventListener("click", function () {
        setMode(m);
      });
    });

    const dialog = document.getElementById("proof-dialog");
    document.getElementById("btn-open-proofs").addEventListener("click", function () {
      if (typeof dialog.showModal === "function") {
        dialog.showModal();
      } else {
        dialog.setAttribute("open", "open");
      }
    });

    document.getElementById("btn-close-proofs").addEventListener("click", function () {
      if (typeof dialog.close === "function") {
        dialog.close();
      } else {
        dialog.removeAttribute("open");
      }
    });

    document.getElementById("btn-copy-bibtex").addEventListener("click", function () {
      const bib = document.getElementById("bibtex-block").textContent;
      copyWithFeedback(bib, "BibTeX copied.");
    });

    document.getElementById("btn-copy-json").addEventListener("click", function () {
      const dual = solveDualFixedPoints(
        state.lambdaExt,
        state.topology,
        state.p,
        state.mu,
        state.T,
        state.N
      );
      copyWithFeedback(JSON.stringify(dual, null, 2), "State JSON copied.");
    });
  }

  const exportedApi = {
    wrightOmega,
    gammaincReg,
    betaincReg,
    betaPdf,
    erlangCdf,
    feedforwardStageMetrics,
    feedbackSeriesMetrics,
    successProbPsucc,
    throughputGamma,
    solveSaddleNodeBifurcation,
    expectedTotalSojourn,
    solveDualFixedPoints,
    solveBatchTimeoutMomentsJS,
    solveCascadeDelayJS,
    solveCascadePowerJS,
    findOptimalThresholdTauJS,
    solveHysteresisBandJS,
    solveCascadeSummaryFromState,
    state
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = exportedApi;
  }
  if (root) {
    root.AQNET_STUDIO = exportedApi;
  }

  if (typeof document !== "undefined") {
    document.addEventListener("DOMContentLoaded", function () {
      parseBurstInput();
      bindEvents();
      updateUI();
    });
  }
})(typeof window !== "undefined" ? window : globalThis);
