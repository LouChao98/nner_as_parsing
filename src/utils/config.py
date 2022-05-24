import inspect
from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class Config:
    @classmethod
    def build(cls, env, allow_unrecognized=False):
        if isinstance(env, dict):
            if 'cfg' in env and isinstance(env['cfg'], cls):
                return env['cfg']
            # https://stackoverflow.com/a/55096964
            matched = {k: v for k, v in env.items() if k in inspect.signature(cls).parameters}
            unmatched = {k: env[k] for k in env.keys() - matched.keys() if not k.startswith('n_')} # n_ is auto added.
            if unmatched and not allow_unrecognized:
                raise ValueError(f'Unrecognized cfg: {unmatched}')
            cfg = cls(**{k: v for k, v in env.items() if k in inspect.signature(cls).parameters})

            for key, value in cfg.__dict__.items():
                if not key.startswith('_'):
                    assert value is not MISSING, f'{key} is MISSING.'

            return cfg
        elif isinstance(env, cls):
            return env
        raise TypeError
