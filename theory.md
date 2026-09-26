# RADON mathematical scope

Let \(L(\theta)=\ell(f(\theta))\), with twice differentiable feature map \(f\) and outer loss \(\ell\). At a fixed parameter vector,

\[
\nabla^2_\theta L = J^\top \nabla^2_z\ell\,J + \sum_a (\partial_a\ell)\nabla^2_\theta f_a = S+R.
\]

For cross-entropy over logits, \(\nabla^2_z\ell\) is positive semidefinite, hence \(S\) is positive semidefinite. The residual is linear in the **logit-space** gradient \(\nabla_z\ell\). A zero parameter gradient \(J^\top\nabla_z\ell=0\) does not generally make \(R=0\). The Lean carrier formalization and fp64 numerical checks cover the algebraic split and probe identities; they do not prove convergence or optimizer superiority.

## Structural diagonal estimate

`fisher_diag_sample` samples labels from the model softmax and squares the resulting parameter gradient with the batch-size correction. For a cross-entropy model with independent sampled labels, its expectation equals \(\operatorname{diag}(S)\). A finite number of samples has nonzero variance. The implementation uses an exponential moving average of these samples, so its total curvature-estimation noise does not generally vanish when the residual vanishes.

## Coded residual probes

For a matrix \(M\) and sign vector \(v\), one diagonal estimate satisfies

\[
v_i(Mv)_i=M_{ii}+\sum_{j\ne i}M_{ij}v_iv_j.
\]

A cycle of \(m\) Hadamard probes uses a fixed random sign vector \(s\) and code matrix \(C\), with \(v_i^k=s_i C_{ki}\). Defining \(\bar C_{ij}=m^{-1}\sum_k C_{ki}C_{kj}\), the estimator is unbiased over \(s\), and

\[
\operatorname{Var}(\hat M_{ii})=\sum_{j\ne i}M_{ij}^2\bar C_{ij}^2.
\]

This identity assumes the same matrix \(M\) for every probe in the cycle. In training, probes occur on different steps while parameters change, so exact cancellation and this variance formula need not hold for the resulting moving-target average.

For a 2D weight tensor, `code(a,b)=(a+b) mod m` gives distinct orthogonal codes to immediate row and column neighbors. Coordinates with the same code still interfere. The code design therefore removes specific cross-talk terms; it does not guarantee a fixed improvement factor against \(m\) independent Hutchinson probes for every matrix. An antithetic pair \(v,-v\) yields the same product \(v\odot Mv\); pairing alone does not halve diagonal-estimator variance.

If the structural diagonal were exact, probing only \(R\) would leave variance governed by \(R\). In the implemented optimizer the structural diagonal is sampled, so both channels' errors must be measured. Scaling the outer loss gradient by \(c\) scales the ideal residual variance by \(c^2\), holding the model derivatives fixed.

## Adaptive residual projection: implemented finite-budget policy

The central question is to choose a probe count, directions, and refresh schedule from observations at each training state, subject to a stated error or compute target. The target must be specified: residual diagonal error, total Hessian-diagonal error, error in the preconditioned update, or held-out loss. These targets need not select the same policy. Count all pilot and diagnostic operator calls in the cost.

For a **fixed** residual matrix \(R\), a complete \(m\)-row Hadamard cycle, and code assignment \(c(i)\), the coherence is one when two coordinates share a code and zero otherwise. For nonnegative coordinate weights \(w_i\), the existing variance identity gives the weighted residual risk

\[
\mathcal E_m(c;R,w)=\sum_i w_i\sum_{j\ne i}R_{ij}^2\mathbf 1\{c(i)=c(j)\}.
\]

This makes alignment a weighted collision problem: directions should separate coordinates with large residual coupling and high update sensitivity. Latin coloring is one fixed candidate, not a solution for every \(R\). Probe count can be posed as the smallest measured or certified \(m\) whose risk reaches a prespecified threshold once all probe costs are included. Without assumptions on the Hessian family and accuracy target, there is no instance-independent cheap optimum.

The separate `radon/adaptive.py` path makes one pilot residual query \(y=R\omega\) at a **frozen** model and batch. If \(y\ne0\), it sets \(q=y/\|y\|_2\) and uses another query for \(Rq\). The pilot therefore changes the next probe directions according to the observed residual. Let \(P=qq^\top\) if projection is selected, or \(P=0\) if the policy declines it, and define \(B=R(I-P)\). Then

\[
\operatorname{diag}(R)=\operatorname{diag}(RP)+\operatorname{diag}(B),\qquad
\hat d_R=\operatorname{diag}(RP)+\frac1s\sum_{k=1}^{s}z_k\odot Bz_k.
\]

The \(z_k\) are **fresh independent Rademacher signs**, separate from pilot and calibration signs. Conditional on everything used to select \(P\) and \(s\), with \(R\) fixed, linearity and \(\mathbb E[z_i z_j]=\mathbf1\{i=j\}\) give

\[
\mathbb E[\hat d_{R,i}\mid P,s]=R_{ii},\qquad
\operatorname{Var}(\hat d_{R,i}\mid P,s)=\frac1s\sum_{j\ne i}B_{ij}^{2}.
\]

This applies to indefinite \(R\); projection is not guaranteed to reduce the rowwise variance. The Lean theorems `Radon.adaptive_diag_split`, `Radon.adaptive_probe_unbiased`, `Radon.adaptive_probe_variance`, and `Radon.adaptive_selected_variance` machine-check the diagonal identity and exact **single-final-probe** mean and variance for every pilot-selected direction. `Radon.adaptive_core_fusion_unbiased` and `Radon.adaptive_core_fusion_variance` additionally show that an exact structural-core diagonal adds no residual-probe variance. The \(1/s\) scaling follows from independence of the final probes and is not separately formalized in Lean.

Two independent calibration signs produce \(x_1=z_1\odot Bz_1\) and \(x_2=z_2\odot Bz_2\). For fixed \(P\) and nonnegative weights \(w_i\),

\[
\hat V_P=\tfrac12\sum_i w_i(x_{1i}-x_{2i})^2,
\qquad
\mathbb E[\hat V_P\mid P]=\sum_i w_i\sum_{j\ne i}B_{ij}^2.
\]

The same two HVPs also supply a full-probe risk estimate via \(Rz=Bz+(q^\top z)Rq\), so the implementation chooses projection or no projection without another operator call. It then chooses \(s\) between preset bounds by comparing the selected variance proxy with a relative-error target; training uses \(w_i=g_i^2\) from the current gradient. The actual cost is \(4+s\) residual HVPs when \(q\ne0\), or \(3+s\) when the pilot product is zero. Every pilot, calibration, and final query is counted. A fixed-count independent Rademacher estimator and the original coded cycle remain available as controls.

The two-sample proxy is noisy, and choosing the lower of two proxies introduces selection bias in the **risk estimate**. Consequently, `target_relative_rms` is a heuristic request, not a certified error threshold or an optimal stopping guarantee. Fresh final signs still preserve conditional unbiasedness. The formal result is for a frozen matrix; training-level EMA lag, sampled structural-core noise, hardware cost, and optimizer convergence need separate analysis. The adaptive path makes all of its HVPs before the parameter update, while the fixed coded cycle spans changing training states.

Prior art includes [stochastic diagonal estimation and Diag++](https://arxiv.org/abs/2201.10684), [hierarchical Hadamard probing](https://arxiv.org/abs/1302.4018), and [adaptive selection of projection dimension and query count](https://arxiv.org/abs/2410.11613). Projection plus adaptive query allocation is established. This implementation is a residual-aware comparison path, not a demonstrated novel or optimal method. A contribution would require stronger stopping guarantees or convincing cost-quality evidence on changing neural Hessians against these references.

## Optimizer step

`Radon.step()` combines momentum with the sampled core and coded residual exponential averages, clamps the curvature denominator below by `eps`, clips each update coordinate to `[-1,1]`, and applies decoupled weight decay. This is a bounded diagonal preconditioned update. The one-step exact Newton property in the formal file applies to an ideal diagonal quadratic with exact curvature, not to the clipped stochastic optimizer.

## Evidence boundary

The ten fp64 gates run on small examples. Lean proves mathematical statements under their stated hypotheses. CPU smoke tests check that small-model training paths execute. Validation perplexity, throughput, hardware scaling, and relative optimizer quality require the owner's planned full FineWeb-Edu experiments.
