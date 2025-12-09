from functools import partial
from math import log, ceil, exp
import math
import numpy as np
import torch
import cma
from utils import compute_fitness, is_less_eq, record, tic,toc
import scipy.stats as stats

# 1+1 ES  用内在维度eff_dim设置为500 否者51200
def oneplusoneES(env, ds, opts, x0):
    sigma0 = opts["sigma0"]
    mytimer = tic()
    x = x0.clone()
    fx=env.objective2(ds["train"], x)
    sigma = sigma0 * 1.0
    # effective dimension 
    tau = opts["eff_dim"] ** 0.5
    FEs = 1
    callback = partial(record, env, ds["val"], ds["test"])
    out = callback(0, 1, toc(mytimer), x, fx, {}, opts)
    if opts["verbose"]:
        print(f"opts: {opts}")
    while FEs < opts["maxFEs"]:
        u = torch.randn_like(x)
        y = x + u * sigma
        fy = env.objective2(ds["train"], y)

        succ, current_iter = is_less_eq(fy, fx, env.coef, FEs - 1, opts["maxFEs"])
        sigma *= exp((succ - 0.2) / tau)
        if succ:
            x, fx = y, fy

        FEs += 1
        elapsed = toc(mytimer)
        out["val_x"] = y
        if FEs % opts["verbose"] == 1:
            print(
                f"[(1+1)-ES] FEs = {FEs}: f = ({fx[0]:g}, {fx[1]:g}), metric = {fx[2]:g}, s = {sigma:.4f}, elapsed = {elapsed:.1f}s, coef = {current_iter:g}"
            )
        out = callback(FEs, FEs, elapsed, x, fx, out, opts)
    out = callback(FEs, FEs, toc(mytimer), x, fx, out, opts)
    out["best_x"] = x
    out["best_f"] = fx[0]
    return out

def SAES_id(env, ds, opts, x0):
    mytimer = tic()
    id = opts.get("eff_dim")
    d = x0.numel()
    opts = {
        "lambda": 4 + ceil(3 * log(id)),
        "mu": ceil((4 + ceil(3 * log(id))) / 2),
        "tau":1/(2*id)**0.5
    } | opts

    x = x0.clone()
    fx = env.objective2(ds["train"], x)
    callback = partial(record, env, ds["val"], ds["test"])
    out = callback(0, 1, toc(mytimer), x, fx, {}, opts)

    FEs=1
    t = 0
    sigma = opts["sigma0"]
    lambda_ = opts["lambda"]
    mu = opts["mu"]
    tau = opts["tau"]

    while FEs < opts["maxFEs"]:
        t += 1
        Y = []
        fY = []
        Sigma = []
        for _ in range(lambda_):
            u = torch.randn(1, device=x.device, dtype=x.dtype)
            sigma_tilde = sigma * (1 + tau * u)
            u_ = torch.randn_like(x)
            y = x + sigma_tilde * u_
            fy = env.objective2(ds["train"], y)
            FEs += 1

            Y.append(y)
            fY.append(fy)
            Sigma.append(sigma_tilde)
            succ, current_iter = is_less_eq(fy, fx, env.coef, t, opts["maxFEs"]//lambda_)
            if succ:
                x, fx = y, fy
        fY_=compute_fitness(fY, env.coef, t, opts["maxFEs"]//lambda_)
        # selection
        idxs = torch.tensor(fY_).argsort()[:mu]
        
        # recombination
        x=torch.zeros_like(y)
        sigma=0
        for idx in idxs:
            x+=1/mu*Y[idx.item()]
            sigma+=1/mu*Sigma[idx.item()]
        out["val_x"] = Y[idxs[0].item()]
        elapsed = toc(mytimer)
        if t % opts["verbose"] == 1:
            print(
                f"[SA-ES with id] FEs = {FEs}: f = ({fY[idxs[0].item()][0]:g}, {fY[idxs[0].item()][1]:g}), metric = {fY[idxs[0].item()][2]:g}, s = {sigma.item():.4f}, elapsed = {elapsed:.1f}s, coef = {current_iter:g}"
            )
        out = callback(t, FEs, elapsed, x, fx, out, opts)
    out = callback(t, FEs, toc(mytimer), x, fx, out, opts)
    out["best_x"] = x
    out["best_f"] = fx[0]
    return out

def SAES_id_e(env, ds, opts, x0):
    mytimer = tic()
    id = opts.get("eff_dim")
    d = x0.numel()
    opts = {
        "lambda": 4 + ceil(3 * log(id)),
        "mu": ceil((4 + ceil(3 * log(id))) / 2),
        "tau":1/(2*id)**0.5
    } | opts

    x = x0.clone()
    fx = env.objective2(ds["train"], x)
    callback = partial(record, env, ds["val"], ds["test"])
    out = callback(0, 1, toc(mytimer), x, fx, {}, opts)

    FEs=1
    t = 0
    sigma = opts["sigma0"]
    lambda_ = opts["lambda"]
    mu = opts["mu"]
    tau = opts["tau"]

    while FEs < opts["maxFEs"]:
        t += 1
        Y = []
        fY = []
        Sigma = []
        for _ in range(lambda_):
            u = torch.randn(1, device=x.device, dtype=x.dtype)
            sigma_tilde = sigma * torch.exp(tau*u)
            u_ = torch.randn_like(x)
            y = x + sigma_tilde*u_
            fy = env.objective2(ds["train"], y)
            FEs += 1

            Y.append(y)
            fY.append(fy)
            Sigma.append(sigma_tilde)
            succ, current_iter = is_less_eq(fy, fx, env.coef, t, opts["maxFEs"]//lambda_)
            if succ:
                x, fx = y, fy
        fY_=compute_fitness(fY, env.coef, t, opts["maxFEs"]//lambda_)
        # selection
        idxs = torch.tensor(fY_).argsort()[:mu]
        
        # recombination
        x=torch.zeros_like(y)
        sigma=0
        for idx in idxs:
            x+=1/mu*Y[idx.item()]
            sigma+=1/mu*Sigma[idx.item()]
        out["val_x"] = Y[idxs[0].item()]
        elapsed = toc(mytimer)
        if t % opts["verbose"] == 1:
            print(
                f"[SA-ES-e with id] FEs = {FEs}: f = ({fY[idxs[0].item()][0]:g}, {fY[idxs[0].item()][1]:g}), metric = {fY[idxs[0].item()][2]:g}, s = {sigma.item():.4f}, elapsed = {elapsed:.1f}s, coef = {current_iter:g}"
            )
        out = callback(t, FEs, elapsed, x, fx, out, opts)
    out = callback(t, FEs, toc(mytimer), x, fx, out, opts)
    out["best_x"] = x
    out["best_f"] = fx[0]
    return out

def SAES(env, ds, opts, x0):
    mytimer = tic()
    d = x0.numel()
    opts = {
        "lambda": 4 + ceil(3 * log(d)),
        "mu": ceil((4 + ceil(3 * log(d))) / 2),
        "tau":1/(2*d)**0.5
    } | opts

    x = x0.clone()
    fx = env.objective2(ds["train"], x)
    callback = partial(record, env, ds["val"], ds["test"])
    out = callback(0, 1, toc(mytimer), x, fx, {}, opts)

    FEs=1
    t = 0
    sigma = opts["sigma0"]
    lambda_ = opts["lambda"]
    mu = opts["mu"]
    tau = opts["tau"]

    while FEs < opts["maxFEs"]:
        t += 1
        Y = []
        fY = []
        Sigma = []
        for _ in range(lambda_):
            u = torch.randn(1)
            # sigma_tilde = sigma * math.exp(tau*u)
            sigma_tilde = sigma * (1+tau*u)
            u_ = torch.randn_like(x)
            y = x + sigma_tilde*u_
            fy = env.objective2(ds["train"], y)
            FEs += 1

            Y.append(y)
            fY.append(fy)
            Sigma.append(sigma_tilde)
            succ, current_iter = is_less_eq(fy, fx, env.coef, t, opts["maxFEs"]//lambda_)
            if succ:
                x, fx = y, fy
        fY_=compute_fitness(fY, env.coef, t, opts["maxFEs"]//lambda_)
        fY_ = torch.tensor(fY_)
        # selection
        idxs = fY_.argsort()[:mu]
        
        # recombination
        x=torch.zeros_like(y)
        sigma=0
        for idx in idxs:
            x+=1/mu*Y[idx.item()]
            sigma+=1/mu*Sigma[idx.item()]
        out["val_x"] = Y[idxs[0].item()]
        elapsed = toc(mytimer)
        if t % opts["verbose"] == 1:
            print(
                f"[SA-ES with id] FEs = {FEs}: f = ({fY[idxs[0].item()][0]:g}, {fY[idxs[0].item()][1]:g}), metric = {fY[idxs[0].item()][2]:g}, s = {sigma.item():.4f}, elapsed = {elapsed:.1f}s, coef = {current_iter:g}"
            )
        out = callback(t, FEs, elapsed, x, fx, out, opts)
    out = callback(t, FEs, toc(mytimer), x, fx, out, opts)
    out["best_x"] = x
    out["best_f"] = fx[0]
    return out

def bbt(env, ds, in_opts, x0):
    sigma0 = in_opts["sigma0"]
    eff_dim = in_opts["eff_dim"]
    mytimer=tic()
    linear_ = torch.nn.Linear(eff_dim, env.n*env.d, bias=False).to(x0.device).to(x0.dtype)
    linear=lambda x:linear_(x.unsqueeze(0)).cpu().detach().reshape(-1)+x0
    fobj=lambda x: env.objective(ds["train"],linear(x))[0]

    x = torch.zeros(eff_dim, dtype=x0.dtype, device=x0.device)
    # 配置CMAES的参数
    self_opts = {
        "popsize": in_opts["lambda"],
        "maxiter": in_opts["maxFEs"] // in_opts["lambda"],
        # 设置每一代选择的个体数量
        'CMA_mu': in_opts["mu"],
        "verbose": -1,
    }

    best_x = linear(x)#callback不受控制，使用env.objective
    best_f = fobj(x)
    t=0
    FEs = 1
    if in_opts["verbose"]:
        print(f"opts: {in_opts}")
    # 初始化CMAES
    CMAES = cma.CMAEvolutionStrategy(x.tolist(), sigma0, self_opts)
    callback = partial(record, env, ds["val"], ds["test"])
    out = callback(0, 1, toc(mytimer),x0,best_f,{}, in_opts)
    # 开始迭代
    for i in range(self_opts["maxiter"]):
        # 获取当前种群
        population = CMAES.ask()
        # 计算种群的适应度
        fitness = [fobj(torch.tensor(p).reshape(-1).to(x0.dtype)) for p in population]
        FEs += self_opts["popsize"]
        t+=1
        elapsed=toc(mytimer)
        # 更新CMAES
        CMAES.tell(population, fitness)
        x = linear(torch.tensor(CMAES.result.xbest).reshape(-1).to(dtype=x0.dtype))
        
        if CMAES.result.fbest < best_f:
            best_x = x
            best_f = CMAES.result.fbest
        out["val_x"] = x
        if t % in_opts["verbose"] == 1:
            print(f"[BBT] FEs = {FEs}, f = {best_f:g}, elapsed={elapsed:.1f}s ")
        
        out = callback(t, FEs, elapsed, x, best_f, out, in_opts)
    out = callback(t, FEs, toc(mytimer), x, best_f, out, in_opts)
    out["best_x"] = best_x
    out["best_f"] = best_f
    return out

def oneplusoneES_conf(fobj, opts, x0, callback):
    sigma0 = opts["sigma0"]
    coef=opts["coef"]
    mytimer = tic()
    x = x0.clone()
    fx = fobj(x)
    sigma = sigma0 * 1.0
    FEs = 1
    tau = x0.numel() ** 0.5
    out = callback(0, 1, toc(mytimer), x, fx[0]+coef*fx[1], {}, opts)

    elapsed = toc(mytimer)
    print(f"[1+1 ES_adaptive] FEs = {FEs}: f = {fx[0]+coef*fx[1]:g}, metric = {fx[2]:g}, corss_loss = {fx[0]:g}, confidence = {fx[1]:g}, s = {sigma:.2f}, elapsed = {elapsed:.1f}s")

    while FEs < opts["maxFEs"]:
        u = torch.randn_like(x)
        y = x + u * sigma
        fy = fobj(y)
        success=fy[0]+coef*fy[1]<fx[0]+coef*fx[1]
        sigma *= exp((1 if success else 0 - 0.2) / tau)
        if success:
            x, fx = y, fy

        FEs += 1
        elapsed = toc(mytimer)

        if FEs % opts["verbose"] == 1:
            print(f"[1+1 ES_adaptive] FEs = {FEs}: f = {fx[0]+coef*fx[1]:g}, metric = {fx[2]:g}, corss_loss = {fx[0]:g}, confidence = {fx[1]:g}, s = {sigma:.2f}, elapsed = {elapsed:.1f}s")

        out = callback(FEs, FEs, elapsed, x, fx[0]+coef*fx[1], out, opts)

    out = callback(FEs, FEs, toc(mytimer), x, fx[0]+coef*fx[1], out, opts)
    out["best_x"] = x
    out["best_f"] = fx[0]+coef*fx[1]
    return out

def ZOSignSGD(env, opts, x0, callback):
    popsize = opts["popsize"]
    radius = opts["radius"]
    beta = opts["beta"]
    opts["verbose"]=25
    lr=opts["lr"]   

    mytimer = tic()

    train_dataset = env.get_minibatch_dataloader2()

    x=x0.clone()
    train_f=env.obj_try_1(train_dataset, x)
    out=callback(0, 1, toc(mytimer), x, train_f[0], {}, opts)

    elapsed = toc(mytimer)

    FEs=1
    t = 0

    print(f"[zo-signsgd] FEs = {FEs}: f = {train_f[0]:g}, metric = {train_f[1]:g}, lr = {lr}, elapsed = {elapsed:.1f}s")
    while FEs < opts["maxFEs"]:
        t += 1
        g = 0
        favg = 0
        metric_avg=0
        for _ in range(popsize):
            samples = env.get_minibatch_dataloader2()
            u = torch.randn_like(x)
            y1 = x + u * radius
            fy1 = env.obj_try_1(samples, y1)
            y2 = x - u * radius
            fy2 = env.obj_try_1(samples, y2)

            # gaussian smoothing based gradient estimation
            g += (fy1[0] - fy2[0]) / radius / 2 / popsize * u
            favg += (fy1[0] + fy2[0]) / 2 / popsize
            FEs += 2
            metric_avg+= (fy1[1] + fy2[1]) / 2 / popsize

        # momentum
        if t == 1:
            m = g
            # moving average of the training loss/grad
            ftrain = favg
            gtrain = g.norm() ** 2
        else:
            m = beta * m + (1 - beta) * g
            ftrain = ftrain * 0.9 + 0.1 * favg
            gtrain = gtrain * 0.9 + 0.1 * g.norm() ** 2

        # normalization: Cutkosky, Ashok, and Harsh Mehta. “Momentum Improves Normalized SGD.” In Proceedings of the 37th International Conference on Machine Learning, 119:2260–68. ICML’20. JMLR.org, 2020.

        ss = lr / t**0.5 # step-size
        # ss = lr
        x -= ss * m.sign()
        elapsed = toc(mytimer)
        if FEs % opts["verbose"] == 1:
            print(f"[zo-signsgd] FEs = {FEs}: f = {favg:g}, metric = {metric_avg:g}, lr = {ss:g}, elapsed = {elapsed:.1f}s")
        out=callback(t, FEs, elapsed, x, favg, out, opts)
    out = callback(t, FEs, toc(mytimer), x, favg, out, opts)
    out["best_x"] = x
    out["best_f"] = favg
    return out

