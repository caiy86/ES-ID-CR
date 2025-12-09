# callback functions
# require: environment and validation set
from math import exp, log
import numpy as np

from termcolor import colored
from timeit import default_timer as timer

tic = timer
toc = lambda t: timer() - t

def make_valid_json(out):
    names = ['best_x', 'test_best_x','val_best_x','val_x']
    for n in names:
        if n in out:
            out[n] = out[n].tolist()
    return out

def get_coef(coef, t, T):
    # if it is a range
    if isinstance(coef, tuple):
        coef = exp((log(coef[1]) - log(coef[0])) * t / T + log(coef[0]))
    return coef

# is fitness(x) is less than or equal to fitness(y)
def is_less_eq(fx, fy, coef, t, T):
    coef = get_coef(coef, t, T)
    return fx[0] + coef * fx[1] <= fy[0] + coef * fy[1], coef

def compute_fitness(fobj, coef, t, T):
    coef = get_coef(coef, t, T)
    return [f[0] + coef * f[1] for f in fobj]

def record(env, val_dataset, test_dataset, t, FEs, elapsed, x, fx, out, opts):
    if not out:
        out = {"train": [], "val": [], "test": []}
    if t % opts["verbose"]:
        out["train"].append({"FEs": FEs, "elapsed": elapsed, "loss": fx})

    if opts["test_times"] > 0:
        test_interval = opts["maxFEs"] / opts["test_times"]
        if (
            not out["test"]
            or out["test"][-1]["FEs"] + test_interval < FEs
            or (FEs >= opts["maxFEs"] and not FEs == out["test"][-1]["FEs"])
        ):
            loss_pred, loss_conf, metric = env.objective2(test_dataset, x)
            out["test"].append(
                {
                    "FEs": FEs,
                    "elapsed": elapsed,
                    "loss_pred": loss_pred,
                    "loss_conf": loss_conf,
                    "metric": metric,
                }
            )
            i = np.argmax([o["metric"] for o in out["test"]])
            best_test_loss, best_test_metric = out["test"][i]["loss_pred"], out["test"][i]["metric"]
            if metric >= best_test_metric:
                out["test_best_x"] = x

            str = f"test iter = {t}, loss = ({loss_pred:g}, {loss_conf:g}), metric = {metric:.3f}, best loss = {best_test_loss :g}, metric = {best_test_metric:.3f}"
            print(colored(str, "red"))
        # if (FEs >= opts["maxFEs"] and not FEs == out["test"][-1]["FEs"]):
        #     loss_pred, loss_conf, metric = env.objective2(test_dataset, out["val_best_x"])
        #     out["test"].append(
        #         {
        #             "FEs": FEs,
        #             "elapsed": elapsed,
        #             "loss_pred": loss_pred,
        #             "loss_conf": loss_conf,
        #             "metric": metric,
        #         }
        #     )
        #     i = np.argmax([o["metric"] for o in out["test"]])
        #     best_test_loss, best_test_metric = out["test"][i]["loss_pred"], out["test"][i]["metric"]
        #     if metric >= best_test_metric:
        #         out["test_best_x"] = out["val_best_x"]

        #     str = f"test iter (val_best) = {t}, loss = ({loss_pred:g}, {loss_conf:g}), metric = {metric:.3f}, best loss = {best_test_loss :g}, metric = {best_test_metric:.3f}"
        #     print(colored(str, "red"))
        # elif (
        #     not out["test"]
        #     or out["test"][-1]["FEs"] + test_interval < FEs
        # ):
        #     loss_pred, loss_conf, metric = env.objective2(test_dataset, x)
        #     out["test"].append(
        #         {
        #             "FEs": FEs,
        #             "elapsed": elapsed,
        #             "loss_pred": loss_pred,
        #             "loss_conf": loss_conf,
        #             "metric": metric,
        #         }
        #     )
        #     i = np.argmax([o["metric"] for o in out["test"]])
        #     best_test_loss, best_test_metric = out["test"][i]["loss_pred"], out["test"][i]["metric"]
        #     if metric >= best_test_metric:
        #         out["test_best_x"] = x

        #     str = f"test iter (train_best) = {t}, loss = ({loss_pred:g}, {loss_conf:g}), metric = {metric:.3f}, best loss = {best_test_loss :g}, metric = {best_test_metric:.3f}"
        #     print(colored(str, "red"))
    return out

