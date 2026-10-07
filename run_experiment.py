"""CPU demonstration, not reproduction of trained DDH/DySurv clinical experiments.

Oracle in the extension experiment means the known sequential masking game,
NOT the full observational conditional E[f(X)|X_A=x_A].
"""
from pathlib import Path
import sys, json, math, time, platform, contextlib, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'upstream'))
from dynshap.dynshap_package.model_wrapper import DynamicSurvivalModel
from dynshap.dynshap_package.estimators.kernel import KernelSHAP
from dynshap.dynshap_package.estimators.sampling import SamplingSHAP
from dynshap.dynshap_package.estimators.temporal import TemporalSHAP

K, F, P = 3, 2, 6
H = np.array([1., 3., 5.])
MASKS = np.array([[(s >> p) & 1 for p in range(P)] for s in range(2**P)], bool)

def transition(x, nonlinear):
    if nonlinear:
        return .55*x + .55*(x*x-1) + .15*x[:, ::-1]
    return .75*x + .15*x[:, ::-1]

def generate(n, seed, nonlinear=False):
    rng = np.random.default_rng(seed)
    x = np.empty((n, K, F)); x[:, 0] = rng.normal(size=(n,F))
    for t in range(1,K):
        x[:,t] = transition(x[:,t-1], nonlinear) + rng.normal(0,.25,(n,F))
    return x

def predict(x):
    # Valid survival curves, a fixed known predictor (not trained).
    x = np.asarray(x, float)
    eta = .55*x[:,-1,0] - .35*x[:,-1,1]
    rate = .15*np.exp(np.clip(eta,-8,8))
    return np.exp(-rate[:,None]*H)

class SurvivalModel:
    def predict(self, observations, timestamps):
        x=np.asarray(observations,float)
        eta=.55*x[:,-1,0]-.35*x[:,-1,1]
        return np.exp(-.15*np.exp(np.clip(eta,-8,8))[:,None]*np.asarray(timestamps))

def exact_shap(values):
    phi=np.zeros((P,len(H)))
    for p in range(P):
        for s in range(2**P):
            if (s>>p)&1: continue
            a=s.bit_count()
            w=math.factorial(a)*math.factorial(P-a-1)/math.factorial(P)
            phi[p]+=w*(values[s|(1<<p)]-values[s])
    return phi

def marginal_values(bg, patient):
    values=[]
    for mask in MASKS:
        x=bg.copy(); m=mask.reshape(K,F);x[:,m]=patient[m]
        values.append(predict(x).mean(axis=0))
    return np.array(values)

def ordered_phi(result):
    df=result.shap_df
    cols=[c for c in df if c.startswith('t = ')]
    return df.groupby(['time_idx','feature'])[cols].mean().sort_index().to_numpy()

def official_experiment():
    bg=generate(128,2026); patient=generate(1,2027)[0]
    # Upstream bins are indexed by time-to-event: oldest row is largest bin.
    obs_times=np.array([2.5,1.5,.5]);ts=[obs_times.copy() for _ in bg]
    wrapper=DynamicSurvivalModel(SurvivalModel(),bg,ts,resolution=1.)
    gt=exact_shap(marginal_values(bg,patient)).reshape(K,F,len(H))[::-1].reshape(P,len(H))
    rows=[];philist={}
    for seed in range(5):
        for name, cls, kw in [('Kernel',KernelSHAP,dict(num_coalitions=512)),
                              ('Sampling',SamplingSHAP,dict(num_permutations=200)),
                              ('Temporal',TemporalSHAP,dict(num_coalitions=512,n_conditional_samples=128))]:
            start=time.perf_counter()
            est=cls(wrapper,prediction_horizons=H,random_state=seed,background_size=1.,**kw)
            with contextlib.redirect_stderr(io.StringIO()):
                result=est.explain((patient,obs_times))
            phi=ordered_phi(result);philist[name]=phi
            target=predict(patient[None])[0];base=predict(bg).mean(axis=0)
            rows.append(dict(seed=seed,method=name,
                rmse_to_marginal=float(np.sqrt(np.mean((phi-gt)**2))),
                residual_to_empirical_baseline=float(np.abs(target-base-phi.sum(axis=0)).mean()),
                historical_abs_share=float(np.abs(phi[2:]).sum()/max(np.abs(phi).sum(),1e-12)),
                seconds=time.perf_counter()-start))
    pd.DataFrame(rows).to_csv(OUT/'official_runs.csv',index=False)
    np.savez(OUT/'official_attributions.npz',patient=patient,background=bg,ground_truth_marginal=gt,**philist)
    return pd.DataFrame(rows)

def fit_sampler(bg, degree):
    poly=PolynomialFeatures(degree=degree,include_bias=False)
    models=[]
    for t in range(1,K):
        a=poly.fit_transform(bg[:,t-1])
        reg=Ridge(alpha=1.).fit(a,bg[:,t])
        residual=bg[:,t]-reg.predict(a)
        models.append((reg,np.maximum(residual.std(axis=0),1e-6)))
    return poly,models

def sequential_values(bg,patient,n,seed,nonlinear=False,fitted=None):
    # Common random numbers across coalitions and methods reduce MC noise.
    rng=np.random.default_rng(seed)
    initial=rng.normal(size=(n,F)) if fitted is None else bg[rng.integers(len(bg),size=n),0]
    noise=rng.normal(size=(K-1,n,F));values=[]
    for mask in MASKS:
        m=mask.reshape(K,F);x=np.empty((n,K,F));x[:,0]=initial
        x[:,0,m[0]]=patient[0,m[0]]
        for t in range(1,K):
            if fitted is None:
                x[:,t]=transition(x[:,t-1],nonlinear)+.25*noise[t-1]
            else:
                poly,models=fitted;reg,sigma=models[t-1]
                x[:,t]=reg.predict(poly.transform(x[:,t-1]))+noise[t-1]*sigma
            x[:,t,m[t]]=patient[t,m[t]]
        values.append(predict(x).mean(axis=0))
    return np.array(values)

def extension_experiment():
    rows=[];examples={}
    for nonlinear in [False,True]:
        scenario='nonlinear' if nonlinear else 'linear'
        bg=generate(2000,3100,nonlinear); val=generate(500,3101,nonlinear)
        patients=generate(10,3102,nonlinear)
        fits={degree:fit_sampler(bg,degree) for degree in [1,2]}
        # Held-out transition mean squared error.
        for degree,fit in fits.items():
            poly,models=fit
            mse=np.mean([(val[:,t]-models[t-1][0].predict(poly.transform(val[:,t-1])))**2 for t in range(1,K)])
            examples[f'{scenario}_transition_mse_degree{degree}']=float(mse)
        for idx,patient in enumerate(patients):
            reference=exact_shap(sequential_values(bg,patient,16384,10000+idx,nonlinear))
            # Estimate oracle Monte Carlo sensitivity independently.
            ref2=exact_shap(sequential_values(bg,patient,16384,20000+idx,nonlinear))
            examples[f'{scenario}_oracle_repeat_rmse_{idx}']=float(np.sqrt(np.mean((reference-ref2)**2)))
            for seed in range(5):
                for method,degree in [('Linear temporal',1),('Quadratic temporal',2)]:
                    values=sequential_values(bg,patient,512,seed,nonlinear,fits[degree])
                    phi=exact_shap(values)
                    rows.append(dict(scenario=scenario,patient=idx,seed=seed,method=method,
                        rmse_to_sequential_oracle=float(np.sqrt(np.mean((phi-reference)**2))),
                        local_accuracy_residual=float(np.max(np.abs(values[-1]-values[0]-phi.sum(axis=0))))))
                    if nonlinear and idx==0 and seed==0:
                        examples[method]=phi.tolist()
            if nonlinear and idx==0:
                examples['oracle_phi']=reference.tolist();examples['patient']=patient.tolist()
                examples['target']=predict(patient[None])[0].tolist()
    df=pd.DataFrame(rows);df.to_csv(OUT/'extension_runs.csv',index=False)
    (OUT/'extension_details.json').write_text(json.dumps(examples,indent=2))
    return df,examples

def plots(official,extended,details):
    fig,axes=plt.subplots(1,3,figsize=(11,3.4),constrained_layout=True)
    limit=max(np.abs(np.array(details[key])[:,1]).max() for key in ['oracle_phi','Linear temporal','Quadratic temporal'])
    for ax,key in zip(axes,['oracle_phi','Linear temporal','Quadratic temporal']):
        data=np.array(details[key])[:,1].reshape(K,F).T
        im=ax.imshow(data,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
        ax.set_title(key+' (h=3)');ax.set_xlabel('Chronological visit');ax.set_yticks([0,1],['Feature 0','Feature 1'])
        ax.set_xticks(range(3))
    fig.colorbar(im,ax=axes,label='Survival contribution');fig.savefig(OUT/'heatmap.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.6),constrained_layout=True)
    per_patient=extended.groupby(['scenario','method','patient']).rmse_to_sequential_oracle.mean()
    mean=per_patient.groupby(['scenario','method']).mean().unstack()
    std=per_patient.groupby(['scenario','method']).std().unstack()
    mean.plot.bar(ax=ax,yerr=std,capsize=3);ax.set_ylabel('RMSE to sequential oracle');ax.set_xlabel('Synthetic scenario')
    ax.tick_params(axis='x',rotation=0);fig.savefig(OUT/'extension_rmse.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    start=time.perf_counter();official=official_experiment();extended,details=extension_experiment()
    plots(official,extended,details)
    summary={'official_mean':official.groupby('method').mean(numeric_only=True).to_dict(),
             'extension_mean':extended.groupby(['scenario','method']).rmse_to_sequential_oracle.mean().to_string(),
             'total_seconds':time.perf_counter()-start,'python':platform.python_version()}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(official.groupby('method').mean(numeric_only=True).to_string())
    print(extended.groupby(['scenario','method']).rmse_to_sequential_oracle.agg(['mean','std']).to_string())
    print('runtime',summary['total_seconds'])
