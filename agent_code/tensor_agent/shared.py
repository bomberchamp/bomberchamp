
from agent_code.tensor_agent.model import FullModel

models = []
online_pool_size = 2

def getModel(*args, **kwargs):
    if len(models) == 0:
        candidate = FullModel(*args, **kwargs)
        models.append([1, candidate])
        return candidate

    n, candidate = models[-1]
    if n < online_pool_size:
        models[-1][0] += 1
        return candidate.clone(share_online=True)

    candidate = candidate.clone(share_online=False)
    models.append([1, candidate])
    return candidate
