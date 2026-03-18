class ROSA:
    """
    后缀自动机在线序列预测器

    用法：
        rosa = ROSA()
        结果 = rosa(序列, mode="value")       # 输出预测值列表
        结果 = rosa(序列, mode="index")       # 输出匹配位置索引列表
        结果 = rosa(序列, mode="both")        # 输出 (索引列表, 值列表)
        结果 = rosa(序列, mode="last_value")  # 只输出最后一个位置的预测值
        结果 = rosa(序列, mode="last_index")  # 只输出最后一个位置的匹配索引
        结果 = rosa(序列, mode="last_both")   # 只输出最后一个位置的 (索引, 值)

    支持输入：list, tuple, torch.Tensor (1D或2D), numpy.ndarray (1D或2D)
    2D 输入视为批量数据，第一维是 batch，第二维是序列长度
    """

    def __call__(self, x, mode="value"):
        batched_input, is_batched = self._to_batched_lists(x)
        batch_results = [self._process_single(seq, mode) for seq in batched_input]
        return self._pack_output(batch_results, is_batched, mode)

    def _to_batched_lists(self, x):
        # 尝试从 Tensor / ndarray 转成 python list
        if hasattr(x, "tolist"):
            x = x.tolist()
        # 判断是否是批量数据（二维）：内层元素还是 list/tuple
        if len(x) > 0 and isinstance(x[0], (list, tuple)):
            return [list(seq) for seq in x], True
        return [list(x)], False

    def _process_single(self, x, mode):
        n = len(x)
        if n == 0:
            return self._format_single(mode, [], [], 0)
        s = 2 * n + 1
        b = [None] * s
        c = [-1] * s
        d = [0] * s
        e = [-1] * s
        b[0] = {}
        g = 0
        z = 1
        indices = [-1] * n
        values = [-1] * n
        for i, t in enumerate(x):
            r, z, g = self._extend_automaton(b, c, d, e, g, z, t)
            indices[i], values[i] = self._query_prediction(c, d, e, x, g)
            self._update_endpos(c, e, g, i)
        return self._format_single(mode, indices, values, n)

    def _extend_automaton(self, b, c, d, e, g, z, t):
        r = z
        z += 1
        b[r] = {}
        d[r] = d[g] + 1
        p = g
        while p != -1 and t not in b[p]:
            b[p][t] = r
            p = c[p]
        if p == -1:
            c[r] = 0
        else:
            q = b[p][t]
            if d[p] + 1 == d[q]:
                c[r] = q
            else:
                u = z
                z += 1
                b[u] = b[q].copy()
                d[u] = d[p] + 1
                c[u] = c[q]
                e[u] = e[q]
                while p != -1 and b[p][t] == q:
                    b[p][t] = u
                    p = c[p]
                c[q] = c[r] = u
        return r, z, r

    def _query_prediction(self, c, d, e, x, g):
        v = g
        idx = -1
        val = -1
        while v != -1:
            if d[v] > 0 and e[v] >= 0:
                idx = e[v]
                val = x[e[v] + 1] if e[v] + 1 < len(x) else -1
                break
            v = c[v]
        return idx, val

    def _update_endpos(self, c, e, g, i):
        v = g
        while v != -1 and e[v] < i:
            e[v] = i
            v = c[v]

    def _format_single(self, mode, indices, values, n):
        if mode == "value":
            return values
        elif mode == "index":
            return indices
        elif mode == "both":
            return indices, values
        elif mode == "last_value":
            return values[n - 1] if n > 0 else -1
        elif mode == "last_index":
            return indices[n - 1] if n > 0 else -1
        elif mode == "last_both":
            return (indices[n - 1], values[n - 1]) if n > 0 else (-1, -1)
        else:
            raise ValueError(f"不支持的 mode: {mode}，可选: value, index, both, last_value, last_index, last_both")

    def _pack_output(self, batch_results, is_batched, mode):
        if not is_batched:
            return batch_results[0]
        # 批量模式：both 类的 mode 需要把 (indices, values) 拆开再分别聚合
        if mode in ("both", "last_both"):
            all_first = [r[0] for r in batch_results]
            all_second = [r[1] for r in batch_results]
            return all_first, all_second
        return batch_results


if __name__ == "__main__":
    rosa = ROSA()

    # ====== 单条序列 ======
    print(rosa([1, 2, 3, 1, 2, 3]))
    # [-1, -1, -1, -1, 1, 2]

    # ====== 批量（二维 list）======
    print(rosa([[1, 2, 1, 2], [3, 3, 3, 3]], mode="index"))
    # [[-1, -1, 0, 1], [-1, 0, 1, 2]]

    # ====== Tensor 输入 ======
    import torch

    t = torch.tensor([[5, 6, 5, 6, 5], [9, 9, 9, 9, 9]])
    print(rosa(t, mode="value"))
    # [[-1, -1, -1, 5, 6], [-1, 9, 9, 9, 9]]

    print(rosa(t, mode="last_value"))
    # [6, 9]

    # ====== numpy 输入 ======
    import numpy as np

    a = np.array([1, 2, 1, 2, 1])
    print(rosa(a, mode="both"))
    # ([-1, -1, 0, 1, 2], [-1, -1, -1, 1, 2])

    # ====== 1D Tensor 也行 ======
    print(rosa(torch.tensor([4, 4, 4]), mode="index"))
    # [-1, 0, 1]
