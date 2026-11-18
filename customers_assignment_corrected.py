"""Exercise 8, Part C: k-nearest neighbours, k-means and PCA on your own bike-shop customers.

Foundations of Artificial Intelligence, Unit 8 (Modul University Vienna, 2026/27)
STARTER FILE.  1. Put your student number into STUDENT_NUMBER at the bottom.
2. Complete the four parts marked TODO (do not change anything else).
3. Run:  python3 customers_starter.py   and copy the whole output into your PDF.
Only the Python standard library is needed. It runs in a few seconds.
"""
import math
import random
import sys

FEATURES = ("km per year", "age", "rides per week")
TYPES = ("commuter", "leisure", "sport")
KS = (1, 3, 5, 9, 15, 25, 41)


# ----------------------------------------------------------------------------- data (given)
def make_customers(seed, n):
    """Your own customers: a list of (features, type) with features = (km per year, age, rides per week)."""
    rng = random.Random(seed)
    jit = lambda a, b: rng.uniform(a, b)
    centre = {"commuter": (jit(2200, 2800), jit(32, 38), jit(8.0, 10.0)),
              "leisure": (jit(1400, 2000), jit(44, 52), jit(2.0, 3.0)),
              "sport": (jit(4500, 5500), jit(34, 42), jit(4.0, 5.0))}
    spread = {"commuter": (900, 9, 1.4), "leisure": (900, 12, 1.1), "sport": (1400, 10, 1.4)}
    share = (jit(0.30, 0.40), jit(0.30, 0.40))
    rows = []
    for _ in range(n):
        u = rng.random()
        t = TYPES[0] if u < share[0] else TYPES[1] if u < share[0] + share[1] else TYPES[2]
        c, s = centre[t], spread[t]
        km = max(100, round(rng.gauss(c[0], s[0]), -1))
        age = min(80, max(16, round(rng.gauss(c[1], s[1]))))
        rides = max(0, round(rng.gauss(c[2], s[2])))
        rows.append(((km, age, rides), t))
    return rows


def split(seed):
    rows = make_customers(seed, 300)
    return rows[:180], rows[180:]          # 180 training customers, 120 test customers


# ----------------------------------------------------------------------------- TODO 1
def euclid(a, b):
    """Euclidean distance between two feature tuples of the same length:
    the square root of the sum of the squared differences (lecture, k-NN slides)."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ----------------------------------------------------------------------------- TODO 2
def knn_predict(train, q, k):
    """Type of the query q by a vote of its k nearest training customers (Euclidean distance).
    train: list of (features, type). Steps:
      1. compute (distance, index, type) for every training customer and sort this list,
         so that equal distances are broken by the index;
      2. keep the first k and count the votes for each type;
      3. return the type with the most votes. If several types are tied, return the one
         among them that appears first in the sorted list (the nearest)."""
    distances = [
        (euclid(features, q), index, customer_type)
        for index, (features, customer_type) in enumerate(train)
    ]
    nearest = sorted(distances)[:k]

    votes = {t: 0 for t in TYPES}
    for _, _, customer_type in nearest:
        votes[customer_type] += 1

    most_votes = max(votes.values())
    # In a vote tie, the first tied type in nearest-neighbour order wins.
    for _, _, customer_type in nearest:
        if votes[customer_type] == most_votes:
            return customer_type


# ----------------------------------------------------------------------------- TODO 3
def fit_scaler(xs):
    """Mean and POPULATION standard deviation (divide by n, not n - 1) of each column of xs,
    a list of feature tuples. Return the pair (mu, sd) of two lists."""
    n = len(xs)
    mu = [sum(x[j] for x in xs) / n for j in range(len(xs[0]))]
    sd = [
        math.sqrt(sum((x[j] - mu[j]) ** 2 for x in xs) / n)
        for j in range(len(xs[0]))
    ]
    return mu, sd


def scale(x, mu, sd):
    """z-scores of one feature tuple: (x_j - mu_j) / sd_j for every feature j. Return a tuple."""
    return tuple((x[j] - mu[j]) / sd[j] for j in range(len(x)))


# ----------------------------------------------------------------------------- TODO 4
def kmeans(points, centres):
    """Lloyd's algorithm from the given starting centres (a list of K feature tuples).
    Repeat:
      assign: label each point with the index of its nearest centre
              (if two centres are equally near, take the lower index);
      update: move each centre to the mean of its points (a centre with no points stays where it is);
    until the centres no longer change. Return (centres, labels, W), where
    W = the sum over all points of the squared distance to the centre of their cluster."""
    K = len(centres)
    while True:
        # Assign each point to its nearest centre; ties go to the lower index.
        labels = [
            min(range(K), key=lambda c: euclid(point, centres[c]))
            for point in points
        ]

        # Update each centre to the mean of its assigned points.
        new_centres = []
        for c in range(K):
            members = [point for point, label in zip(points, labels) if label == c]
            if members:
                new_centres.append(tuple(
                    sum(point[j] for point in members) / len(members)
                    for j in range(len(points[0]))
                ))
            else:
                # An empty cluster keeps its previous centre.
                new_centres.append(centres[c])

        if new_centres == centres:
            W = sum(
                sum((point[j] - centres[label][j]) ** 2 for j in range(len(point)))
                for point, label in zip(points, labels)
            )
            return centres, labels, W

        centres = new_centres


# ----------------------------------------------------------------------------- report (given)
def error(train, test, k):
    return sum(knn_predict(train, x, k) != y for x, y in test) / len(test)


def cv_error(train, k, folds=5):
    """5-fold cross-validation error of k-NN on the training customers (fold i = every 5th customer)."""
    wrong = 0
    for f in range(folds):
        val = train[f::folds]
        rest = [r for i, r in enumerate(train) if i % folds != f]
        wrong += sum(knn_predict(rest, x, k) != y for x, y in val)
    return wrong / len(train)


def best_run(points, K, seed, restarts):
    runs = []
    for r in range(restarts):
        rng = random.Random(seed * 100 + 7 * K + r)
        runs.append(kmeans(points, rng.sample(points, K)))
    return runs


def pca(points):
    """Principal components of a list of (already scaled) feature tuples, by the Jacobi method:
    eigenvalues (largest first) and the matching eigenvectors of the covariance matrix (1/N)."""
    n, m = len(points), len(points[0])
    mean = [sum(p[j] for p in points) / n for j in range(m)]
    A = [[sum((p[i] - mean[i]) * (p[j] - mean[j]) for p in points) / n for j in range(m)] for i in range(m)]
    V = [[float(i == j) for j in range(m)] for i in range(m)]
    for _ in range(100):
        off = max(abs(A[i][j]) for i in range(m) for j in range(m) if i != j)
        if off < 1e-12:
            break
        for p in range(m):
            for q in range(p + 1, m):
                if abs(A[p][q]) < 1e-15:
                    continue
                th = 0.5 * math.atan2(2 * A[p][q], A[q][q] - A[p][p])
                c, s = math.cos(th), math.sin(th)
                for k in range(m):                       # rotate columns p, q
                    akp, akq = A[k][p], A[k][q]
                    A[k][p], A[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(m):                       # rotate rows p, q
                    apk, aqk = A[p][k], A[q][k]
                    A[p][k], A[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(m):
                    vkp, vkq = V[k][p], V[k][q]
                    V[k][p], V[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    order = sorted(range(m), key=lambda i: -A[i][i])
    lam = [A[i][i] for i in order]
    vecs = []
    for i in order:
        v = [V[k][i] for k in range(m)]
        big = max(range(m), key=lambda k: abs(v[k]))
        vecs.append([x if v[big] > 0 else -x for x in v])   # sign: largest loading positive
    return lam, vecs, mean


def project(x, vecs, mean, n):
    return tuple(sum((x[j] - mean[j]) * v[j] for j in range(len(x))) for v in vecs[:n])


def report(seed):
    train, test = split(seed)
    print(f"Student number {seed}: {len(train)} training and {len(test)} test customers.")
    print("Training customers per type: " + ", ".join(f"{t} {sum(y == t for _, y in train)}" for t in TYPES))
    mu, sd = fit_scaler([x for x, _ in train])
    print("\nFeature          mean      standard deviation   (training customers)")
    for j, f in enumerate(FEATURES):
        print(f"  {f:15s} {mu[j]:8.1f}   {sd[j]:8.1f}")
    strain = [(scale(x, mu, sd), y) for x, y in train]
    stest = [(scale(x, mu, sd), y) for x, y in test]

    print("\nError rates of k-NN       raw units           scaled (z-scores)")
    print("    k                 training    test      training    test     5-fold CV (scaled)")
    cv = {}
    for k in KS:
        cv[k] = cv_error(strain, k)
        print(f"  {k:3d}               {error(train, train, k):7.3f} {error(train, test, k):7.3f}"
              f"     {error(strain, strain, k):7.3f} {error(strain, stest, k):7.3f}     {cv[k]:7.3f}")
    kbest = min(KS, key=lambda k: (cv[k], k))
    print(f"\nk chosen by 5-fold cross-validation: k = {kbest}; its test error (scaled): {error(strain, stest, kbest):.3f}")
    q = (3500, 45, 3)
    print(f"New customer {q}: raw k = {kbest} says {knn_predict(train, q, kbest)}, "
          f"scaled k = {kbest} says {knn_predict(strain, scale(q, mu, sd), kbest)}")

    pts = [x for x, _ in strain]
    runs = best_run(pts, 3, seed, 10)
    print("\nk-means on the scaled training customers (no types used), K = 3, 10 random starts:")
    print("  W per start: " + ", ".join(f"{w:.2f}" for _, _, w in runs))
    cen, lab, w = min(runs, key=lambda r: r[2])
    print(f"  lowest W = {w:.2f}; {len(set(round(r[2], 2) for r in runs))} different final values of W")
    print("  cluster x type      " + "".join(f"{t:>10s}" for t in TYPES) + "     centre in original units")
    for c in range(3):
        orig = tuple(cen[c][j] * sd[j] + mu[j] for j in range(3))
        print(f"  cluster {c + 1}           " + "".join(f"{sum(l == c and y == t for l, (_, y) in zip(lab, train)):10d}" for t in TYPES)
              + f"     ({orig[0]:.0f} km, {orig[1]:.0f} years, {orig[2]:.1f} rides)")
    print("\nElbow: lowest W of 10 random starts for each K")
    print("  " + "   ".join(f"K={K}: {min(r[2] for r in best_run(pts, K, seed, 10)):.1f}" for K in range(1, 8)))

    lam, vecs, pmean = pca(pts)
    tot = sum(lam)
    print("\nPCA of the scaled training customers (no types used):")
    print("  share of variance:  " + "   ".join(f"PC{i + 1} {l / tot:.3f}" for i, l in enumerate(lam)))
    print("  PC1 loadings:       " + ", ".join(f"{f} {v:+.2f}" for f, v in zip(FEATURES, vecs[0])))
    ptrain = [(project(x, vecs, pmean, 2), y) for x, y in strain]
    ptest = [(project(x, vecs, pmean, 2), y) for x, y in stest]
    p1train = [(project(x, vecs, pmean, 1), y) for x, y in strain]
    p1test = [(project(x, vecs, pmean, 1), y) for x, y in stest]
    print(f"  test error of k-NN with k = {kbest}: 3 scaled features {error(strain, stest, kbest):.3f}, "
          f"PC1 and PC2 {error(ptrain, ptest, kbest):.3f}, PC1 only {error(p1train, p1test, kbest):.3f}")


STUDENT_NUMBER = 62505390      # <-- your student number

if __name__ == "__main__":
    if STUDENT_NUMBER == 0:
        sys.exit("Please put your student number into STUDENT_NUMBER at the bottom of the file.")
    report(STUDENT_NUMBER)
