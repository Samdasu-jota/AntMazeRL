"""
A* 길찾기 (numpy만, 외부 그래프 라이브러리 불필요).

업계 표준 분리: 고전 플래너가 '서브목표(웨이포인트)'를, RL 정책이 '보행'을 담당.
미로 벽을 점유격자(occupancy grid)로 래스터화 → 개미 반경만큼 팽창(inflate) →
8방향 A* → Ramer–Douglas–Peucker로 코너 웨이포인트만 남김.

`plan((0,0),(0,6))` → 기둥을 우회하는 웨이포인트 리스트(예 [(1.0,0),(1.0,6),(0,6)]).
WaypointFollower(waypoints=plan())로 바로 주입. 고정 레이아웃에선 하드코딩
WAYPOINTS와 동등하지만, 랜덤 미로(Stage 2)로 일반화된다.
"""
import heapq
from collections import deque

import numpy as np

# (cx, cy, hx, hy) 절반길이 — make_maze_xml의 벽들과 동일
MAZE_WALLS = [
    (0.0, 3.0, 0.3, 3.0),    # wall_mid (가운데 기둥)
    (-4.0, 3.0, 0.3, 5.0),   # wall_left
    (4.0, 3.0, 0.3, 5.0),    # wall_right
    (0.0, 8.0, 4.0, 0.3),    # wall_top
    (0.0, -2.0, 4.0, 0.3),   # wall_bot
]
BOUNDS = (-4.0, 4.0, -2.0, 8.0)   # (xmin, xmax, ymin, ymax)


def build_occupancy(walls=MAZE_WALLS, bounds=BOUNDS, res=0.1, inflate=0.6):
    """벽을 격자에 그리고 inflate(개미 반경+여유)만큼 팽창. True=점유."""
    xmin, xmax, ymin, ymax = bounds
    nx = int(round((xmax - xmin) / res)) + 1
    ny = int(round((ymax - ymin) / res)) + 1
    grid = np.zeros((nx, ny), dtype=bool)
    for cx, cy, hx, hy in walls:
        x0, x1 = cx - hx - inflate, cx + hx + inflate
        y0, y1 = cy - hy - inflate, cy + hy + inflate
        ix0 = max(0, int(np.floor((x0 - xmin) / res)))
        ix1 = min(nx - 1, int(np.ceil((x1 - xmin) / res)))
        iy0 = max(0, int(np.floor((y0 - ymin) / res)))
        iy1 = min(ny - 1, int(np.ceil((y1 - ymin) / res)))
        grid[ix0:ix1 + 1, iy0:iy1 + 1] = True
    return grid, (xmin, ymin), res


def world_to_grid(xy, origin, res):
    return (int(round((xy[0] - origin[0]) / res)),
            int(round((xy[1] - origin[1]) / res)))


def grid_to_world(ij, origin, res):
    return np.array([origin[0] + ij[0] * res, origin[1] + ij[1] * res])


def snap_lateral(grid, ij):
    """같은 행(row)에서 좌우로 가장 가까운 자유 셀(+x 우선). 시작/목표가 기둥 끝에
    걸려 있을 때 '옆으로 비켜' 깔끔한 우회 시작점을 준다(뒤로 가는 snap 방지)."""
    nx, ny = grid.shape
    i0, j = ij
    if 0 <= i0 < nx and 0 <= j < ny and not grid[i0, j]:
        return ij
    for dx in range(1, nx):
        for i in (i0 + dx, i0 - dx):        # +x 먼저
            if 0 <= i < nx and not grid[i, j]:
                return (i, j)
    return _nearest_free(grid, ij)


def _nearest_free(grid, ij):
    nx, ny = grid.shape
    if not grid[ij]:
        return ij
    seen = {ij}
    q = deque([ij])
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < nx and 0 <= nj < ny and (ni, nj) not in seen:
                if not grid[ni, nj]:
                    return (ni, nj)
                seen.add((ni, nj))
                q.append((ni, nj))
    return ij


def astar(grid, start, goal):
    """8방향 A* (유클리드 휴리스틱). 격자 셀 리스트 또는 None."""
    nx, ny = grid.shape

    def h(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    open_h = [(h(start, goal), 0.0, start)]
    came, g = {}, {start: 0.0}
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while open_h:
        _, gc, cur = heapq.heappop(open_h)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if gc > g.get(cur, 1e18):
            continue
        for di, dj in nbrs:
            ni, nj = cur[0] + di, cur[1] + dj
            if not (0 <= ni < nx and 0 <= nj < ny) or grid[ni, nj]:
                continue
            ng = gc + np.hypot(di, dj)
            if ng < g.get((ni, nj), 1e18):
                g[(ni, nj)] = ng
                came[(ni, nj)] = cur
                heapq.heappush(open_h, (ng + h((ni, nj), goal), ng, (ni, nj)))
    return None


def simplify(pts, tol=0.25):
    """Ramer–Douglas–Peucker: 직선 구간을 코너로 압축."""
    if len(pts) < 3:
        return list(pts)
    a, b = pts[0], pts[-1]
    ab = b - a
    L = float(np.hypot(ab[0], ab[1]))
    if L < 1e-9:
        d = [float(np.hypot(*(p - a))) for p in pts]
    else:
        d = [abs(float(np.cross(ab, p - a))) / L for p in pts]
    idx = int(np.argmax(d))
    if d[idx] > tol:
        return simplify(pts[:idx + 1], tol)[:-1] + simplify(pts[idx:], tol)
    return [a, b]


def plan(start=(0.0, 0.0), goal=(0.0, 6.0), res=0.1, inflate=0.6, tol=0.25,
         pillar_half_len=3.0):
    """occupancy → A* → simplify → 웨이포인트. 마지막은 '진짜 목표'로 둔다.
    pillar_half_len: 중앙 기둥(wall_mid) 길이의 절반(기본 3.0=풀 6m). 축소 기둥(1.0)이면
    A*가 더 짧고 부드러운 우회 경로를 준다(make_maze_xml과 동일 파라미터)."""
    walls = [(0.0, 3.0, 0.3, pillar_half_len)] + MAZE_WALLS[1:]   # wall_mid 길이만 교체
    grid, origin, r = build_occupancy(walls=walls, res=res, inflate=inflate)
    s = snap_lateral(grid, world_to_grid(start, origin, r))
    go = snap_lateral(grid, world_to_grid(goal, origin, r))
    cells = astar(grid, s, go)
    if cells is None:
        raise RuntimeError("A*가 경로를 못 찾음 — bounds/inflate 확인")
    world = [grid_to_world(c, origin, r) for c in cells]
    corners = simplify(world, tol=tol)
    wps = [np.asarray(c, dtype=float) for c in corners]
    wps.append(np.asarray(goal, dtype=float))    # 정확한 최종 목표
    return wps


if __name__ == "__main__":
    wps = plan()
    print("A* 웨이포인트 (start (0,0) → goal (0,6), 기둥 우회):")
    for w in wps:
        print("   ", np.round(w, 2).tolist())
    grid, origin, r = build_occupancy(inflate=0.6)
    bad = [w for w in wps[:-1] if grid[world_to_grid(w, origin, r)]]
    print(f"inflated 벽 안 웨이포인트: {len(bad)} (0이어야 함)")
