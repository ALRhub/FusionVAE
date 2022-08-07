import torch
from config import cfg
from neural_operations import Conv2D


class TimeDist(torch.nn.Module):
    """Time-distributed layer"""
    def __init__(self, module, cell_type=None):
        super().__init__()
        self.module = module
        self.cell_type = cell_type

    def forward(self, x):
        assert len(x.size()) == 5
        y = self.module(x.contiguous().view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4)))
        y = y.contiguous().view(x.size(0), x.size(1), -1, y.size(-2), y.size(-1))
        return y


class EncCombinerCell(torch.nn.Module):
    def __init__(self, Cin1, Cin2, Cout, cell_type):
        super().__init__()
        self.cell_type = cell_type
        # Cin = Cin1 + Cin2
        if cfg.lat.convFBefAg or cfg.lat.convPBefAg:
            Cout = 2 * cfg.args.num_latent_per_group
        if cfg.lat.convFBefAg:
            self.conv_f3x = TimeDist(Conv2D(Cin1, Cout, kernel_size=1, stride=1, padding=0, bias=True))
            self.conv_f3y = Conv2D(Cin1, Cout, kernel_size=1, stride=1, padding=0, bias=True)

        if not cfg.nn.encComb == 'None':
            self.conv_s2xy = Conv2D(Cin2, Cout, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2, training=True, conv_f3x=None, conv_f3y=None):
        if cfg.nn.encComb == 'None':  # noEncCombiner
            return x1

        x2 = self.conv_s2xy(x2)

        if cfg.nn.encComb == 'add':
            if x1 is None:
                return x2
            return x1 + x2

        if x1 is None:  # no input image (neither noisy nor target image)
            return x2[:, None]

        if cfg.nn.encComb == 'multiAdd' and not cfg.lat.convFBefAg and conv_f3x is None and conv_f3y is None:
            return x1 + x2[:, None]

        if cfg.lat.convFBefAg:
            if conv_f3x is None:
                conv_f3x = self.conv_f3x
            if conv_f3y is None:
                conv_f3y = self.conv_f3y

        ftr_x, ftr_y = None, None

        if True:  # training:  # target image exist
            ftr_y = x1[:, -1]
            if x1.shape[1] > 1:  # at least one noisy image (and target image)
                ftr_x = x1[:, 0:-1]
        else:  # at least one noisy image and no target image
            ftr_x = x1

        if cfg.lat.convFBefAg or cfg.lat.convPBefAg:
            if True:  # training:  # target image exist
                ftr_y = conv_f3y(ftr_y)
            if x1.shape[1] > 1:  # at least one noisy image
                ftr_x = conv_f3x(ftr_x)

        if cfg.nn.encComb == 'multiAdd':
            if ftr_x is not None:
                if ftr_y is not None:
                    x1 = torch.cat([ftr_x, ftr_y[:, None]], dim=1)
                else:
                    x1 = ftr_x
            else:
                x1 = ftr_y[:, None]
            return x1 + x2[:, None]

        elif cfg.nn.encComb == 'agg':
            # aggregate all (instead of addition)
            if True:  # training:  # target image exist
                ftr_y = torch.cat([ftr_y[:, None], x2[:, None]], dim=1)
                mu_y, log_sig_y = torch.chunk(ftr_y, 2, dim=-3)
                mu_y, log_sig_y = aggregate_dist(method=cfg.lat.agg, mu=mu_y, log_sig=log_sig_y)
                ftr_y = torch.cat([mu_y, log_sig_y], dim=1)[:, None]

            if ftr_x is not None:  # at least one noisy image
                mu_x, log_sig_x = torch.chunk(ftr_x, 2, dim=-3)
                mu_x, log_sig_x = aggregate_dist(method=cfg.lat.agg, mu=mu_x, log_sig=log_sig_x)
                ftr_x = torch.cat([mu_x, log_sig_x], dim=1)[:, None]
                if ftr_y is not None:  # target image exist (together with noisy image(s))
                    param = torch.cat([ftr_x, ftr_y], dim=1)
                    return param
                else:  # no target image (but noisy ones)
                    return ftr_x
            else:  # no noisy image
                return ftr_y


@torch.jit.script
def bayesian_aggregation(mu: torch.Tensor, log_sig: torch.Tensor):
    """Bayesian aggregation over dimension 1"""

    if log_sig.shape[1] == 1:
        return mu[:, 0], log_sig[:, 0]
    if log_sig.shape[1] == 0:
        return mu, log_sig

    sig = torch.exp(log_sig)

    # init of Bayesian aggregation
    m = mu[:, 0]
    s = sig[:, 0]

    for i in range(1, sig.shape[1]):
        # kalman gain
        q_t = s / (s + sig[:, i])

        # kalman update
        m = m + q_t * (mu[:, i] - m)
        s = (1 - q_t) * s
    return m, torch.log(s)


@torch.jit.script
def aggregate_dist(method: str, mu: torch.Tensor, log_sig: torch.Tensor):
    if method.startswith('maxAg'):
        mu = torch.max(mu, dim=1)[0]
        log_sig = torch.max(log_sig, dim=1)[0]
    elif method.startswith('meaAg'):
        mu = torch.mean(mu, dim=1)
        log_sig = torch.mean(log_sig, dim=1)
    elif method.startswith('BayAg'):
        mu, log_sig = bayesian_aggregation(mu=mu, log_sig=log_sig)
    else:
        raise NotImplementedError
    return mu, log_sig


@torch.jit.script
def aggregate(method: str, ftr: torch.Tensor):
    if method.startswith('maxAg'):
        ftr = torch.max(ftr, dim=1)[0]
    elif method.startswith('meaAg'):
        ftr = torch.mean(ftr, dim=1)
    elif method.startswith('BayAg'):
        mu, log_sig = torch.chunk(ftr, 2, dim=-3)
        mu, log_sig = bayesian_aggregation(mu=mu, log_sig=log_sig)
        ftr = torch.cat([mu, log_sig], dim=1)
    else:
        raise NotImplementedError
    return ftr


def s_skip_aggregate(s, s_skip):
    if cfg.nn.skip_agg == 'maxAg' and s_skip is not None:
        s = torch.cat([s[:, None], s_skip], dim=1)
        s = torch.max(s, dim=1)[0]
    if cfg.nn.skip_agg == 'meaAg' and s_skip is not None:
        s = torch.cat([s[:, None], s_skip], dim=1)
        s = torch.mean(s, dim=1)
    elif cfg.nn.skip_agg == 'maxAgAdd' and s_skip is not None:
        s = s + torch.max(s_skip, dim=1)[0]
    elif cfg.nn.skip_agg == 'maxAgConcat':
        if s_skip is None:
            s_skip_agg = torch.zeros_like(s)
        else:
            s_skip_agg = torch.max(s_skip, dim=1)[0]
        s = torch.cat([s, s_skip_agg], dim=1)
    return s

