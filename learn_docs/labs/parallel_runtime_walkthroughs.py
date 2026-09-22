"""CPU-only reference checks for 手工推演索引.md; no third-party dependencies.

Run from repository root: python3 learn_docs/labs/parallel_runtime_walkthroughs.py
These check mathematical/state-machine examples, not a distributed backend.
"""
from collections import Counter
from math import exp, isclose, log, sqrt


def close(a, b):
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), (a, b)
        for x, y in zip(a, b):
            close(x, y)
    else:
        assert isclose(a, b, rel_tol=1e-6, abs_tol=1e-7), (a, b)


def transpose(a):
    return list(map(list, zip(*a)))


def mm(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def add(a, b):
    return [[x + y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def finite_difference(fn, values):
    result = []
    for i in range(len(values)):
        plus, minus = list(values), list(values)
        plus[i] += 1e-5
        minus[i] -= 1e-5
        result.append((fn(plus) - fn(minus)) / 2e-5)
    return result


def ddp_fsdp():
    samples = [[1, 0], [0, 1], [1, 1], [2, 0]]
    weights = [1, 2]
    grads = [[sum(w*x for w, x in zip(weights, s)) * x for x in s] for s in samples]
    reference = [sum(g[j] for g in grads)/4 for j in range(2)]
    # Two ranks, unequal real-token counts; sum losses scaled by D/N before DDP mean.
    rank_grads = [[sum(grads[i][j] for i in ids)*2/4 for j in range(2)]
                  for ids in ([0, 1, 2], [3])]
    reduced = [sum(g[j] for g in rank_grads)/2 for j in range(2)]
    close(reduced, reference)
    close(reduced, [2, 1.25])
    close([w-.1*g for w, g in zip(weights, reduced)], [.8, 1.875])
    close(sum(sum(w*x for w, x in zip(weights, s))**2/2 for s in samples)/4, 2.25)
    wrong = [(sum(g[j] for g in grads[:3])/3+grads[3][j])/2 for j in range(2)]
    close(wrong, [8/3, 5/6])
    end = 0
    for readiness in [(2, 3), (6, 7)]:
        end = max(end, *readiness)+2
    assert end == 9
    full = [1, 2, 3, 4, 5, 6, 7, 8]
    local = [full, list(reversed(full))]
    shards = [[sum(g[i] for g in local)/2 for i in indexes]
              for indexes in (range(4), range(4, 8))]
    updated = [w-.1*g for w, g in zip(full, sum(shards, []))]
    close(updated, [w-.45 for w in full])
    assert 4*(2+4+4+8)+8*2+8*4 == 120


def tensor_parallel():
    x = [[1, 2], [3, 4]]
    w1 = [[1, 0, 1, 2], [0, 1, 1, 0]]
    w2 = [[1, 0], [0, 1], [1, 1], [2, -1]]
    a = mm(x, w1)  # All entries positive: ReLU derivative is 1 here.
    y = mm(a, w2)
    close(y, [[8, 3], [22, 5]])
    a_shards = [mm(x, [row[start:start+2] for row in w1]) for start in (0, 2)]
    partial = [mm(a_shards[r], w2[2*r:2*r+2]) for r in range(2)]
    close(add(*partial), y)
    g = [[1, 1], [1, 1]]
    da = [mm(g, transpose(w2[2*r:2*r+2])) for r in range(2)]
    dw2 = sum([mm(transpose(a_shards[r]), g) for r in range(2)], [])
    dw1_parts = [mm(transpose(x), d) for d in da]
    dw1 = [dw1_parts[0][i]+dw1_parts[1][i] for i in range(2)]
    dx = add(*[mm(da[r], transpose([row[2*r:2*r+2] for row in w1])) for r in range(2)])
    close(dw1, [[4, 4, 8, 4], [6, 6, 12, 6]])
    close(dw2, [[4, 4], [6, 6], [10, 10], [8, 8]])
    close(dx, [[5, 3], [5, 3]])
    def loss(xv, av, bv):
        hidden = mm([xv[:2], xv[2:]], [av[:4], av[4:]])
        hidden = [[max(0, v) for v in row] for row in hidden]
        return sum(map(sum, mm(hidden, [bv[i:i+2] for i in range(0, 8, 2)])))
    xv, av, bv = sum(x, []), sum(w1, []), sum(w2, [])
    close(finite_difference(lambda v: loss(v, av, bv), xv), sum(dx, []))
    close(finite_difference(lambda v: loss(xv, v, bv), av), sum(dw1, []))
    close(finite_difference(lambda v: loss(xv, av, v), bv), sum(dw2, []))


def pipeline():
    schedules = [
        [['F0','F1','F2','F3','-','-','B3','B2','B1','B0'],
         ['-','F0','F1','F2','F3','B3','B2','B1','B0','-']],
        [['F0','F1','-','B0','F2','B1','F3','B2','-','B3'],
         ['-','F0','B0','F1','B1','F2','B2','F3','B3','-']],
    ]
    for rows, expected in zip(schedules, ([4,4], [2,1])):
        completed, live, peak = set(), [set(), set()], [0, 0]
        for tick in range(10):
            new = []
            for stage in range(2):
                op = rows[stage][tick]
                if op == '-':
                    continue
                direction, batch = op[0], int(op[1])
                task = (stage, direction, batch)
                assert task not in completed
                dependencies = []
                if direction == 'F' and stage == 1:
                    dependencies.append((0, 'F', batch))
                if direction == 'B':
                    dependencies.append((stage, 'F', batch))
                    if stage == 0:
                        dependencies.append((1, 'B', batch))
                assert all(d in completed for d in dependencies), (tick, task)
                if direction == 'F':
                    live[stage].add(batch)
                else:
                    live[stage].remove(batch)
                peak[stage] = max(peak[stage], len(live[stage]))
                new.append(task)
            completed.update(new)  # Same-slot outputs cannot be consumed.
        assert len(completed) == 16 and not any(live)
        assert peak == expected, peak


def sequence_context():
    rows = [[1,3], [2,4], [3,5], [4,6]]
    normalized = []
    for row in rows:
        mean = sum(row)/2
        std = sqrt(sum((v-mean)**2 for v in row)/2)
        normalized.append([(v-mean)/std for v in row])
    close(normalized, [[-1,1]]*4)
    close([sum(row[j] for row in normalized) for j in range(2)], [-4,4])
    q, k, v = [1]*4, [log(i) for i in (1,2,3,4)], [1,2,4,8]
    def attention(qv, kv, vv, index):
        scores = [qv[index]*kv[j] for j in range(index+1)]
        weights = [exp(s-max(scores)) for s in scores]
        return sum(w*vv[j] for j, w in enumerate(weights))/sum(weights)
    close([attention(q,k,v,i) for i in range(4)], [1,5/3,17/6,4.9])
    # Independent block-wise online merge, including maximum rescaling.
    m, l, n = float('-inf'), 0, 0
    for indexes in ([2,3], [0,1]):
        bm = max(k[j] for j in indexes)
        bl = sum(exp(k[j]-bm) for j in indexes)
        bn = sum(exp(k[j]-bm)*v[j] for j in indexes)
        new_m = max(m, bm)
        l = exp(m-new_m)*l + exp(bm-new_m)*bl
        n = exp(m-new_m)*n + exp(bm-new_m)*bn
        m = new_m
    close(n/l, attention(q,k,v,3))
    p = [.1,.2,.3,.4]
    ds = [a*(b-4.9) for a,b in zip(p,v)]
    close(ds, [-.39,-.58,-.27,1.24])
    close(finite_difference(lambda z: attention(q,k,z,3), v), p)
    close(finite_difference(lambda z: attention(q,z,v,3), k), ds)
    close(finite_difference(lambda z: attention(z,k,v,3), q), [0,0,0,sum(a*b for a,b in zip(ds,k))])
    assert sum(i+1 for i in [0,1,6,7]) == sum(i+1 for i in [2,3,4,5]) == 18


def experts_mesh():
    x, gate, coefficients = [1,2,3,4], [.75,.25,.5,.25], [2,3]
    # Dispatch deliberately reorders tokens; combine must use original ids.
    expert_batches = [[(i,x[i],gate[i]) for i in [3,0,2,1]],
                      [(i,x[i],1-gate[i]) for i in [2,1,3,0]]]
    output, dc = [0]*4, [0]*2
    for expert, assignments in enumerate(expert_batches):
        for i, token, probability in assignments:
            output[i] += probability*coefficients[expert]*token
            dc[expert] += probability*token
    close(output, [2.25,5.5,7.5,11])
    close(dc, [3.75,6.25])
    ranks = {(p,d,t):4*p+2*d+t for p in range(2) for d in range(2) for t in range(2)}
    tp = [[ranks[p,d,t] for t in range(2)] for p in range(2) for d in range(2)]
    dp = [[ranks[p,d,t] for d in range(2)] for p in range(2) for t in range(2)]
    pp = [[ranks[p,d,t] for p in range(2)] for d in range(2) for t in range(2)]
    assert tp == [[0,1],[2,3],[4,5],[6,7]]
    assert dp == [[0,2],[1,3],[4,6],[5,7]]
    assert pp == [[0,4],[1,5],[2,6],[3,7]]


def kv_runtime():
    pages, tables, cache, in_flight = {}, {}, set(), set()
    free_counts = []
    def check():
        refs = Counter(p for table in tables.values() for p in table)
        assert set(pages) == set(refs) | cache
        assert all(0 < count <= 4 for count in pages.values())
        free_counts.append(6-len(pages))
    def release(request):
        assert request not in in_flight
        del tables[request]
        refs = {p for table in tables.values() for p in table}
        for p in list(pages):
            if p not in refs and p not in cache:
                del pages[p]
    pages.update({0:4,1:2}); tables['A']=[0,1]; cache.add(0); check()
    pages[2]=3; tables['B']=[0,2]; check()
    pages[1]=4; pages[3]=1; tables['A'].append(3); check()
    tables['C']=list(tables['B']); check()
    pages[4]=pages[2]+1; tables['C'][-1]=4; check()
    refs = Counter(p for t in tables.values() for p in t)
    assert refs[0] == 3 and refs[2] == refs[4] == 1
    assert sum(pages.values()) == 16 and len(pages)*4 == 20
    release('A'); check()
    in_flight.add('B'); check()  # Cancellation does not release outstanding device work.
    in_flight.remove('B'); release('B'); check()
    release('C'); check()
    cache.remove(0); del pages[0]; check()
    assert free_counts == [4,3,2,2,1,3,3,4,5,6], free_counts
    kv, prompt_left = {'A':0,'B':4,'C':0}, {'A':5,'B':0,'C':2}
    pending = {'A':False,'B':True,'C':False}
    for iteration in [[('B',1),('A',3),('C',2)], [('B',1),('C',1),('A',2)]]:
        assert sum(count for _, count in iteration) <= 6
        for request, count in iteration:
            if prompt_left[request]:
                assert count <= prompt_left[request]
                prompt_left[request] -= count
                pending[request] = prompt_left[request] == 0
            else:
                assert pending[request] and count == 1
            kv[request] += count
    assert kv == {'A':5,'B':6,'C':3} and all(pending.values())


if __name__ == '__main__':
    for check in (ddp_fsdp, tensor_parallel, pipeline, sequence_context, experts_mesh, kv_runtime):
        check()
        print(f'PASS {check.__name__}')
    print('All parallel/runtime reference checks passed (CPU, simulated collectives).')
