 Review complete — the multi-agent pass plus my own numeric replication converge. The engine is mostly sound: the
  scalar LP path (partialOls/partialOlsMulti) reproduces full-model OLS coefficients and HC1 SEs to machine precision,
  the proxy-SVAR identification and wild-bootstrap sign/alignment are correct, and several of my own candidate concerns
  were refuted on inspection. Per your instruction, HC1-vs-HAC is excluded entirely.

  Here's what's actually wrong.

  Confirmed bugs

  1. The "whitening" estimator is biased — wrong point estimate. buildWhitenedShock/whitenedOls (2793–2878)
  The whitened regressor r is built by sequential partialling — residualize x on contemporaneous controls (full sample),
  then residualize on lagged-tilde — and the first p rows are dropped. After that truncation r is no longer orthogonal
  to the controls, yet whitenedOls forms beta = ⟨r, raw y2⟩/⟨r,r⟩, which is valid FWL only if r ⟂ all controls. I
  verified this independently: with the true partial effect = 1.0, the whitened estimate came out 0.85 / 0.73 in stress
  cases (correct = ~1.0); the agents reproduced ~8% error at n=100 with the default 8 lags, growing with persistence.
  With p=0 (no whitening) it matches partialOls exactly — so the bias is introduced specifically by whitening lags ≥ 1
  with any control selected.
  Fix: the function already builds the correct fullDesign = [x, wRows, lRows] for its residuals — just take beta as the
  x-coefficient of that full OLS (or residualize x and y jointly on [wRows, lRows], exactly as partialOls does).

  2. The whitening-path SE is also wrong. whitenedOls 2858–2867
  The HC1 sandwich uses the partial r in both the meat (Σ r²·resid²) and denom², while resid comes from the correct full
  design — so the regressor in the sandwich doesn't match the residual model. Both the center and the width of the band
  are off in the whitening path. (Fix falls out of #1: compute the sandwich with the jointly-residualized rx.)
  → Bottom line: "Use whitening" is unreliable for lags ≥ 1. It's off by default, so the default LP view is unaffected.

  3. Proxy-VARX uses positional lags over a calendar-discontiguous sample. buildVarData 2235–2248, fitVarx 2289,
  simulateVarx 2403, IRF recursion 2356
  The VAR stacks lags by array position (varsData[t-lag]) but the sample drops interior months (COVID/year/ZLB/NA
  filters). So "lag 1" is the previous retained row, not the previous calendar month, and the IRF/horizon axis (labeled
  monthly: "Horizon", h=) is measured in retained-rows. Verified against the shipped data: the VAR tab's default (COVID
  excluded) splices 2020-02 → 2021-04 as one step (~2.8% of design rows); unchecking "include ZLB" splices 2008-11 →
  2016-01 (86 months) as one step. No disclosure anywhere.
  Fix: estimate on the longest contiguous block, or drop observations whose lag window crosses a calendar gap, or at
  minimum warn that the VAR sample is non-contiguous.

  Notable (affects the default main LP)

  4. "Series lag" controls are event-spaced, not calendar-spaced. attachSeriesHistory 1852–1869, shockLagControls
  1871–1883
  With imputeZeros off (default) the analysis rows are event months only, so "shock lag k" partials out the k-th
  previous FOMC event's shock — which can be several calendar months back — while the dependent-variable lags and macro
  lags (dependentDelta, macroLagValue) use true calendar months (baseIdx − lag). So within one regression "lag 1" means
  different time spans for different controls, and the meaning silently flips when imputeZeros is toggled. Default
  shockLags = 8, so this is live in the default specification. Defensible as an event-study choice, but it's an
  undocumented inconsistency that changes the residualized shock and hence the displayed IRF.

  Minor

  - VAR diagnostics mis-scaled (covarianceMatrix denom omits kExog, 2322): the "Raw 2Y impact" / "Scale factor" readouts
  are off by a constant when controls are selected; the IRF chart and bands are fine.
  - Silent ridge in VARX only (solveLinearStable 2499–2511): on a near-singular VAR it adds ridge with no warning; LP
  paths instead drop columns / refuse. (Truly rank-deficient VARs are caught by the b11sq>EPS gate, so this is narrow.)
  - Dead config: meta.defaults.ci=0.9 is written by the builder but never read — the LP band is hardcoded 90% (Z90).
  Honest label, just a misleading config field.
  - Cross-tab band levels differ: LP 90% vs proxy-VARX 95% (both labeled); and market-vs-Eurodollar response tables use
  different residual-SD scaling in the no-controls (non-default) case.
