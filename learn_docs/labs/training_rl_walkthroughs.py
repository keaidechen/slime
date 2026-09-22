"""Executable arithmetic/state references for the training, serving and RL articles.

Run: python3 learn_docs/labs/training_rl_walkthroughs.py
Uses only Python's standard library; does not exercise CUDA, Ray or a real store.
"""
from copy import deepcopy
from math import ceil, exp, log, sqrt
from statistics import mean, stdev

from parallel_runtime_walkthroughs import close, finite_difference


def adam_step_and_skip():
    weights, m, v, step = [1., 2.], [0., 0.], [0., 0.], 0
    gradients = [scaled/8 for scaled in [24.,32.]]
    norm = sqrt(sum(g*g for g in gradients))
    close(norm, 5)
    gradients = [g*min(1, 2.5/norm) for g in gradients]
    close(gradients, [1.5,2])
    step += 1
    for i, g in enumerate(gradients):
        m[i] = .9*m[i]+.1*g
        v[i] = .99*v[i]+.01*g*g
        mh, vh = m[i]/(1-.9**step), v[i]/(1-.99**step)
        weights[i] = (1-.1*.1)*weights[i]-.1*mh/sqrt(vh)
    close(m, [.15,.2])
    close(v, [.0225,.04])
    close(weights, [.89,1.88])
    saved = deepcopy((weights,m,v,step))
    next_gradients = [1.,float('inf')]
    skip = any(not (-float('inf') < g < float('inf')) for g in next_gradients)
    assert skip and (weights,m,v,step) == saved
    close(sqrt(sum(min(2.5,g)**2 for g in [3,4])), sqrt(12.5))


def packing_checkpoint():
    sequences = [['BOS','a','b','EOS'], ['BOS','c','EOS']]
    flat, labels, masks, positions, visible = [], [], [], [], []
    offset = 0
    for sample in sequences:
        flat.extend(sample)
        labels.extend(sample[1:]+[None])
        masks.extend([1]*(len(sample)-1)+[0])
        positions.extend(range(len(sample)))
        visible.extend([set(range(offset, offset+i+1)) for i in range(len(sample))])
        offset += len(sample)
    assert labels == ['a','b','EOS',None,'c','EOS',None]
    assert masks == [1,1,1,0,1,1,0]
    assert positions == [0,1,2,3,0,1,2]
    assert visible[4] == {4} and visible[6] == {4,5,6}
    losses = [.1,.2,.4,0,.6,.8,0]
    close(sum(l*m for l,m in zip(losses,masks))/sum(masks), .42)
    masks[0] = 0
    close(sum(l*m for l,m in zip(losses,masks))/sum(masks), .5)
    close((mean(losses[:3])+mean(losses[4:6]))/2, 7/15)
    # Incomplete shards cannot publish a new recovery point.
    committed = {12: {'cursor':20,'shards':{0,1}}}
    pending = {'cursor':22,'shards':{0}}
    assert pending['shards'] != {0,1}
    assert max(committed) == 12 and committed[12]['cursor'] == 20
    pending['shards'].add(1)
    if pending['shards'] == {0,1}:
        committed[13] = deepcopy(pending)
    assert committed[max(committed)]['cursor'] == 22


def serving_speculation():
    lengths = [20,9,15]
    pages = [ceil(n/4) for n in lengths]
    next_pages = [ceil((n+1)/4) for n in lengths]
    assert pages == [5,3,4] and next_pages == [6,3,4]
    assert 24-sum(pages) == 12 and 24-sum(next_pages) == 11
    mib = 1024*128/1024
    transfer_ms = mib/(8*1024)*1000
    close(transfer_ms, 15.625)
    close(5+30+transfer_ms+2, 52.625)
    p, q = [.5,.3,.2], [.2,.5,.3]
    accepted = [proposal*min(1,target/proposal) for target,proposal in zip(p,q)]
    rejection = 1-sum(accepted)
    residual = [max(0,target-proposal) for target,proposal in zip(p,q)]
    output = [a+rejection*r/sum(residual) for a,r in zip(accepted,residual)]
    close(output, p)
    wrong = [a+rejection*t for a,t in zip(accepted,p)]
    close(wrong, [.35,.39,.26])
    close(10/((4+12)/2), 1.25)


def ppo_grpo_gae():
    rewards, lengths = [0.,1.,2.], [2,1,2]
    group_adv = [(r-mean(rewards))/stdev(rewards) for r in rewards]
    close(group_adv, [-1,0,1])
    advantages = [a for a,n in zip(group_adv,lengths) for _ in range(n)]
    old = [log(.2)]*5
    current = list(map(log,[.4,.1,.2,.3,.1]))
    def token_losses(logprobs):
        ratios = [exp(new-prev) for new,prev in zip(logprobs,old)]
        return [max(-r*a, -min(1.2,max(.8,r))*a) for r,a in zip(ratios,advantages)]
    losses = token_losses(current)
    close(losses, [2,.8,0,-1.2,-.5])
    close(mean(losses), .22)
    means, offset = [], 0
    for n in lengths:
        means.append(mean(losses[offset:offset+n]))
        offset += n
    close(mean(means), 11/60)
    close(finite_difference(lambda x: sum(token_losses(x)), current), [2,0,0,0,-.5])
    # Padding must not change a masked token mean.
    masks = [1]*5+[0]*3
    padded = losses+[7,8,9]
    close(sum(l*m for l,m in zip(padded,masks))/sum(masks), .22)
    values, reward = [.2,.4,0], [0,1]
    advantages, running = [0,0], 0
    for t in reversed(range(2)):
        delta = reward[t]+values[t+1]-values[t]
        running = delta+.5*running
        advantages[t] = running
    close(advantages, [.5,.6])
    close([a+v for a,v in zip(advantages,values)], [.7,1])


def async_queue():
    queue, ends, peaks = 0, [], []
    for _ in range(4):
        admitted = min(4, 6-queue)
        queue += admitted
        peaks.append(queue)
        queue -= min(2,queue)
        ends.append(queue)
    assert ends == [2,4,4,4] and peaks == [4,6,6,6]
    committed = set()
    records = [('G0',10,'complete'), ('G1',9,'complete'), ('G2',11,'aborted'),
               ('G0',10,'complete'), ('G3',13,'complete')]
    decisions = []
    for gid,version,status in records:
        if gid in committed:
            decision = 'duplicate'
        elif status != 'complete':
            decision = 'retry'
        elif version > 12:
            decision = 'invalid'
        elif 12-version > 2:
            decision = 'expired'
        else:
            decision = 'accept'
            committed.add(gid)  # Simulate a successful training/checkpoint commit.
        decisions.append(decision)
    assert decisions == ['accept','expired','retry','duplicate','invalid']
    # An in-place partial load invalidates service until a complete snapshot is installed.
    destination = [12,12,12]
    paused = True
    destination[0] = 13
    assert paused and len(set(destination)) == 2
    destination[:] = [13]*3
    ready = all(version == 13 for version in destination)
    assert ready


if __name__ == '__main__':
    for check in (adam_step_and_skip, packing_checkpoint, serving_speculation, ppo_grpo_gae, async_queue):
        check()
        print(f'PASS {check.__name__}')
    print('All training/serving/RL reference checks passed (CPU, teaching protocols).')
