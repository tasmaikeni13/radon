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

## Adaptive probe design: proposed research problem

The central question is to choose a probe count, directions, and refresh schedule from observations at each training state, subject to a stated error or compute target. The target must be specified: residual diagonal error, total Hessian-diagonal error, error in the preconditioned update, or held-out loss. These targets need not select the same policy. Count all pilot and diagnostic operator calls in the cost.

For a **fixed** residual matrix \(R\), a complete \(m\)-row Hadamard cycle, and code assignment \(c(i)\), the coherence is one when two coordinates share a code and zero otherwise. For nonnegative coordinate weights \(w_i\), the existing variance identity gives the weighted residual risk

\[
\mathcal E_m(c;R,w)=\sum_i w_i\sum_{j\ne i}R_{ij}^2\mathbf 1\{c(i)=c(j)\}.
\]

This makes alignment a weighted collision problem: directions should separate coordinates with large residual coupling and high update sensitivity. Latin coloring is one fixed candidate, not a solution for every \(R\). Probe count can be posed as the smallest measured or certified \(m\) whose risk reaches a prespecified threshold once all probe costs are included. Without assumptions on the Hessian family and accuracy target, there is no instance-independent cheap optimum.

An adaptive design may estimate useful coupling information from previous observations, then choose \(m_t\) and \(c_t\) for the current state. To reuse the fixed-matrix unbiasedness argument, the signs used to evaluate a chosen design must be fresh and independent of the observations used to choose it. If directions or stopping decisions depend on results from the same sign cycle, a new sequential estimator and guarantee are needed. A cycle spanning training steps additionally sees different \(R_t\), so its error includes curvature drift. Structural-core sampling error is a separate term. None of these adaptive guarantees is currently proved in Lean or implemented in the optimizer.

Prior art includes [stochastic diagonal estimation and Diag++](https://arxiv.org/abs/2201.10684), [hierarchical Hadamard probing](https://arxiv.org/abs/1302.4018), and [adaptive selection of projection dimension and query count](https://arxiv.org/abs/2410.11613). The general diagonal-from-matvec problem, structured probing, and some adaptive query allocation are established. A contribution here would need a distinct residual-aware policy for changing neural Hessians, an appropriate conditional guarantee, or convincing cost-quality evidence against those references.

## Optimizer step

`Radon.step()` combines momentum with the sampled core and coded residual exponential averages, clamps the curvature denominator below by `eps`, clips each update coordinate to `[-1,1]`, and applies decoupled weight decay. This is a bounded diagonal preconditioned update. The one-step exact Newton property in the formal file applies to an ideal diagonal quadratic with exact curvature, not to the clipped stochastic optimizer.

## Evidence boundary

The ten fp64 gates run on small examples. Lean proves mathematical statements under their stated hypotheses. CPU smoke tests check that small-model training paths execute. Validation perplexity, throughput, hardware scaling, and relative optimizer quality require the owner's planned full FineWeb-Edu experiments.
