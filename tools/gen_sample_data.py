#!/usr/bin/env python3
"""
Generate sample datasets for GTS++ artifact evaluation.

Produces small-scale versions of the benchmark datasets in GTS text format:
  Line 1: <dim> <num_vectors> <distance_code>
  Lines 2+: space-separated float values (one vector per line)

Distance codes: 0=Linf, 1=L1, 2=L2, 5=Cosine, 6=Edit distance

Usage:
    python3 gen_sample_data.py          # Generate all sample datasets
    python3 gen_sample_data.py sift     # Generate only SIFT sample
"""

import os
import sys
import random
import math
import string

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample')
os.makedirs(OUTDIR, exist_ok=True)

random.seed(42)

def write_vector_dataset(name, dim, n, metric_code, gen_fn):
    """Write a vector dataset in GTS text format."""
    path = os.path.join(OUTDIR, f'{name}.txt')
    with open(path, 'w') as f:
        f.write(f'{dim} {n} {metric_code}\n')
        for i in range(n):
            vec = gen_fn(dim, i)
            f.write(' '.join(f'{v:.6f}' for v in vec) + '\n')
    print(f'  {path}: {dim}D x {n} vectors (metric={metric_code})')
    return path

def write_query_ids(name, n_data, n_queries):
    """Write query ID file (random data point IDs)."""
    path = os.path.join(OUTDIR, f'{name}_qid.txt')
    ids = random.sample(range(n_data), min(n_queries, n_data))
    with open(path, 'w') as f:
        f.write(f'{len(ids)}\n')
        for i in ids:
            f.write(f'{i}\n')
    print(f'  {path}: {len(ids)} query IDs')

def write_query_vectors(name, dim, n_queries, gen_fn):
    """Write query vectors in GTS text format (with header)."""
    path = os.path.join(OUTDIR, f'{name}_query.txt')
    with open(path, 'w') as f:
        f.write(f'{n_queries}\n')
        for i in range(n_queries):
            vec = gen_fn(dim, i + 100000)
            f.write(' '.join(f'{v:.6f}' for v in vec) + '\n')
    print(f'  {path}: {n_queries} query vectors')

def write_update_workload(name, n_data, n_ops, radius):
    """Write update workload file (insert/delete operations)."""
    path = os.path.join(OUTDIR, f'{name}_update.txt')
    with open(path, 'w') as f:
        f.write(f'{n_ops} {radius}\n')
        for i in range(n_ops):
            flag = 1 if random.random() < 0.5 else 0  # 50/50 insert/delete
            oid = random.randint(0, n_data - 1)
            f.write(f'{flag} {oid}\n')
    print(f'  {path}: {n_ops} update ops')

def write_groundtruth(name, n_data, dim, n_queries, k, metric_code, gen_fn):
    """Compute and write brute-force ground truth for query vectors."""
    path = os.path.join(OUTDIR, f'{name}_groundtruth.txt')

    # Generate data
    data = [gen_fn(dim, i) for i in range(n_data)]
    queries = [gen_fn(dim, i + 100000) for i in range(n_queries)]

    def dist(a, b, code):
        if code == 2:  # L2
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
        elif code == 1:  # L1
            return sum(abs(x - y) for x, y in zip(a, b))
        elif code == 0:  # Linf
            return max(abs(x - y) for x, y in zip(a, b))
        elif code == 5:  # Cosine angle
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na * nb < 1e-10:
                return 0.0
            cos_val = min(dot / (na * nb), 1.0)
            return abs(math.acos(cos_val) * 180 / math.pi)
        return 0.0

    with open(path, 'w') as f:
        for qi, q in enumerate(queries):
            dists = [(dist(q, d, metric_code), i) for i, d in enumerate(data)]
            dists.sort()
            # Write top-100 (or all if fewer) neighbor IDs
            top = min(100, len(dists))
            ids = [str(dists[j][1]) for j in range(top)]
            f.write(' '.join(ids) + '\n')
    print(f'  {path}: {n_queries} x top-{min(100, n_data)} ground truth')

def gen_uniform(dim, seed):
    """Generate a uniform random vector in [0, 1000]."""
    random.seed(seed * 137 + 7)
    return [random.uniform(0, 1000) for _ in range(dim)]

def gen_gaussian(dim, seed):
    """Generate a Gaussian random vector (mean=500, std=100)."""
    random.seed(seed * 137 + 7)
    return [random.gauss(500, 100) for _ in range(dim)]

def gen_unit(dim, seed):
    """Generate a unit-norm vector (for cosine distance)."""
    random.seed(seed * 137 + 7)
    vec = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / max(norm, 1e-8) for v in vec]

def write_word_dataset(name, n, n_queries):
    """Write an edit-distance string dataset."""
    path_data = os.path.join(OUTDIR, f'{name}.txt')
    max_len = 20
    random.seed(42)
    words = []
    for i in range(n):
        length = random.randint(3, max_len)
        word = ''.join(random.choices(string.ascii_lowercase, k=length))
        words.append(word)

    with open(path_data, 'w') as f:
        f.write(f'{max_len} {n} 6\n')
        for w in words:
            f.write(w + '\n')
    print(f'  {path_data}: {n} strings (edit distance)')

    write_query_ids(name, n, n_queries)


def main():
    targets = set(sys.argv[1:]) if len(sys.argv) > 1 else None

    print('Generating sample datasets for GTS++ artifact evaluation...\n')

    # 1. SIFT-like (128D, L2) - 10K sample
    if targets is None or 'sift' in targets:
        print('[SIFT-10K] 128D, L2:')
        n, dim, nq = 10000, 128, 100
        write_vector_dataset('sift_10k', dim, n, 2, gen_uniform)
        write_query_ids('sift_10k', n, nq)
        write_query_vectors('sift_10k', dim, nq, gen_uniform)
        write_groundtruth('sift_10k', n, dim, nq, 100, 2, gen_uniform)
        write_update_workload('sift_10k', n, 200, 0.82)
        print()

    # 2. GIST-like (960D, L2) - 1K sample
    if targets is None or 'gist' in targets:
        print('[GIST-1K] 960D, L2:')
        n, dim, nq = 1000, 960, 50
        write_vector_dataset('gist_1k', dim, n, 2, gen_gaussian)
        write_query_ids('gist_1k', n, nq)
        write_query_vectors('gist_1k', dim, nq, gen_gaussian)
        write_groundtruth('gist_1k', n, dim, nq, 100, 2, gen_gaussian)
        print()

    # 3. Deep-like (96D, L2) - 10K sample
    if targets is None or 'deep' in targets:
        print('[Deep-10K] 96D, L2:')
        n, dim, nq = 10000, 96, 100
        write_vector_dataset('deep_10k', dim, n, 2, gen_gaussian)
        write_query_ids('deep_10k', n, nq)
        write_query_vectors('deep_10k', dim, nq, gen_gaussian)
        write_groundtruth('deep_10k', n, dim, nq, 100, 2, gen_gaussian)
        print()

    # 4. Cosine (300D) - 1K sample
    if targets is None or 'cosine' in targets:
        print('[Cosine-1K] 300D, Cosine:')
        n, dim, nq = 1000, 300, 50
        write_vector_dataset('cosine_1k', dim, n, 5, gen_unit)
        write_query_ids('cosine_1k', n, nq)
        write_query_vectors('cosine_1k', dim, nq, gen_unit)
        print()

    # 5. Word (edit distance) - 5K sample
    if targets is None or 'word' in targets:
        print('[Word-5K] Edit distance:')
        write_word_dataset('word_5k', 5000, 100)
        print()

    print('Done! Sample data written to:', os.path.abspath(OUTDIR))


if __name__ == '__main__':
    main()
