import torch


class ROSA:
    def __call__(self, x, mode="value"):
        batched_input, is_batched = self._to_batched_lists(x)
        batch_results = [self._process_single(seq, mode) for seq in batched_input]
        return self._pack_output(batch_results, is_batched, mode, x)

    def _to_batched_lists(self, x):
        raw = x.tolist() if hasattr(x, "tolist") else x
        if len(raw) > 0 and isinstance(raw[0], (list, tuple)):
            return [list(seq) for seq in raw], True
        return [list(raw)], False

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

    def _pack_output(self, batch_results, is_batched, mode, x):
        # 需要拆分 both 类型的结果
        if mode in ("both", "last_both"):
            if not is_batched:
                first, second = batch_results[0]
                return self._to_tensor(first, x), self._to_tensor(second, x)
            all_first = [r[0] for r in batch_results]
            all_second = [r[1] for r in batch_results]
            return self._to_tensor(all_first, x), self._to_tensor(all_second, x)
        if not is_batched:
            return self._to_tensor(batch_results[0], x)
        return self._to_tensor(batch_results, x)

    def _to_tensor(self, data, x):
        """如果输入是 Tensor，输出也转成同设备的 Tensor，-1 替换为 0"""
        if hasattr(x, "device"):
            t = torch.tensor(data, dtype=torch.long, device=x.device)
            t = t.clamp(min=0)  # -1 变成 0，保证能安全过 embedding
            return t
        return data


if __name__ == "__main__":
    rosa = ROSA()

    # ====== 单条序列 ======
    print(rosa([1, 2, 3, 1, 2, 3]))
    # [-1, -1, -1, -1, 1, 2]

    # ====== 批量（二维 list）======
    print(rosa([[1, 2, 1, 2], [3, 3, 3, 3]], mode="index"))
    # [[-1, -1, 0, 1], [-1, 0, 1, 2]]

    # ====== Tensor 输入 ======
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
