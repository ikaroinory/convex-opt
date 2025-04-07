import torch

from optimizer import Optimizer


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class ShubertPiyavskiiSearch(Optimizer):
    def __init__(self, f, l=5, epsilon=None):
        super(ShubertPiyavskiiSearch, self).__init__(f, epsilon=epsilon)

        self.l = l

    def _get_sp_intersection(self, alpha: Point, beta: Point) -> Point:
        x = ((alpha.y - beta.y) + self.l * (alpha.x + beta.x)) / (2 * self.l)
        y = alpha.y - self.l * (x - alpha.x)

        return Point(x, y)

    def optimize(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        mid = (alpha + beta) / 2
        A, M, B = Point(alpha, self.f(alpha)), Point(mid, self.f(mid)), Point(beta, self.f(beta))
        pts = [A, self._get_sp_intersection(A, M), M, self._get_sp_intersection(M, B), B]

        diff = torch.inf
        while diff > self.epsilon:
            i = torch.argmin(torch.tensor([P.y for P in pts if P.x not in [alpha, mid, beta]]))
            P = Point(pts[i].x, self.f(pts[i].x))

            diff = P.y - pts[i].y

            P_prev = self._get_sp_intersection(pts[i - 1], P)
            P_next = self._get_sp_intersection(P, pts[i + 1])

            if (P_next.x - P_prev.x) < self.epsilon:
                return P.x

            del pts[i]

            pts.insert(i, P_next)
            pts.insert(i, P)
            pts.insert(i, P_prev)

            self.iterator_count += 1

        return None
