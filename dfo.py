import numpy as np


def dfo_minimise(f, lower, upper, n_flies=30, n_iter=300, delta=0.001, seed=0):
    """Minimise f(x) inside the box lower <= x <= upper.

    Parameters
    ----------
    f        : function taking a 1-D numpy array (one solution) and returning a number
    lower, upper : lists with the bounds of every dimension
    n_flies  : swarm size
    n_iter   : number of iterations
    delta    : disturbance threshold (probability of a random restart per coordinate)
    seed     : random seed, so every run gives the same result

    Returns
    -------
    best_x   : best solution found
    best_f   : its fitness value f(best_x)
    history  : best fitness after every iteration (to plot convergence)
    """
    rng = np.random.default_rng(seed)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    dims = len(lower)

    # 1. Initialise: every fly at a random point inside the bounds.
    flies = rng.uniform(lower, upper, size=(n_flies, dims))
    history = []

    for _ in range(n_iter):
        # 2. Evaluate every fly (lower fitness = better solution).
        fitness = np.array([f(x) for x in flies])
        best = int(np.argmin(fitness))
        history.append(fitness[best])

        new_flies = flies.copy()
        for i in range(n_flies):
            if i == best:
                continue                                   # 3. Elitism: the best fly stays

            # 4. Ring topology: compare left and right neighbour, keep the better one.
            left, right = (i - 1) % n_flies, (i + 1) % n_flies
            nb = left if fitness[left] < fitness[right] else right

            for d in range(dims):
                if rng.random() < delta:
                    # 5. Disturbance: restart this coordinate at random (exploration).
                    new_flies[i, d] = rng.uniform(lower[d], upper[d])
                else:
                    # 6. Main update: start at the best neighbour, move towards the swarm best.
                    u = rng.random()
                    new_flies[i, d] = flies[nb, d] + u * (flies[best, d] - flies[i, d])

                # 7. A fly that left the search space is put back at a random position.
                if not lower[d] <= new_flies[i, d] <= upper[d]:
                    new_flies[i, d] = rng.uniform(lower[d], upper[d])
        flies = new_flies

    fitness = np.array([f(x) for x in flies])
    best = int(np.argmin(fitness))
    return flies[best], fitness[best], np.array(history)


if __name__ == "__main__":
    # Quick self-test: the minimum of (x - 3)^2 + (y + 1)^2 is at (3, -1).
    x, fx, _ = dfo_minimise(lambda v: (v[0] - 3) ** 2 + (v[1] + 1) ** 2,
                            lower=[-10, -10], upper=[10, 10])
    print(f"DFO self-test: found x={x.round(3)}, f={fx:.6f} (expected (3, -1), 0)")
