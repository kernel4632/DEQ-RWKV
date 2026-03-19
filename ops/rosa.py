import torch


class ROSA:
    def __init__(self, return_index=False, return_type=None):
        """
        return_index: 是否返回索引位置而非值，默认 False（返回值）
        return_type: 指定输出类型，可选 'list', 'str', 'tensor', None（自动与输入类型一致）
        """
        self.return_index = return_index
        self.return_type = return_type

    def _core(self, x):
        n = len(x)
        y_val = [-1] * n
        y_idx = [-1] * n
        s = 2 * n + 1
        b = [None] * s
        c = [-1] * s
        d = [0] * s
        e = [-1] * s
        b[0] = {}
        g = 0
        z = 1
        for i, t in enumerate(x):
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
            v = g = r
            a_val = -1
            a_idx = -1
            while v != -1:
                if d[v] > 0 and e[v] >= 0:
                    a_idx = e[v] + 1
                    a_val = x[a_idx]
                    break
                v = c[v]
            y_val[i] = a_val
            y_idx[i] = a_idx
            v = g
            while v != -1 and e[v] < i:
                e[v] = i
                v = c[v]
        return y_val, y_idx

    def _to_list(self, x):
        if isinstance(x, torch.Tensor):
            return x.tolist()
        if isinstance(x, str):
            return list(x)
        return list(x)

    def _is_batch(self, x):
        if isinstance(x, torch.Tensor):
            return x.dim() >= 2
        if isinstance(x, str):
            return False
        if isinstance(x, (list, tuple)) and len(x) > 0:
            return isinstance(x[0], (list, tuple, str, torch.Tensor))
        return False

    def _detect_input_type(self, x):
        if isinstance(x, torch.Tensor):
            return "tensor"
        if isinstance(x, str):
            return "str"
        return "list"

    def _convert_output(self, result, target_type):
        if target_type == "tensor":
            return torch.tensor(result)
        if target_type == "str":
            return "".join(str(v) for v in result)
        return result

    def __call__(self, x, return_index=None, return_type=None):
        """
        x: 输入数据，支持 str / list / Tensor，也支持批数据（二维 list / 2D Tensor）
        return_index: 是否返回索引（覆盖实例默认值）
        return_type: 输出类型（覆盖实例默认值），可选 'list', 'str', 'tensor', None
        """
        ri = return_index if return_index is not None else self.return_index
        rt = return_type if return_type is not None else self.return_type

        batch = self._is_batch(x)

        if batch:
            input_type = self._detect_input_type(x)
            # 对于 batch tensor，内层类型也是 tensor
            inner_type = "tensor" if input_type == "tensor" else None
            results = []
            items = x if not isinstance(x, torch.Tensor) else x
            for item in items:
                it = self._detect_input_type(item) if inner_type is None else inner_type
                seq = self._to_list(item)
                y_val, y_idx = self._core(seq)
                out = y_idx if ri else y_val
                out_type = rt if rt is not None else it
                results.append(self._convert_output(out, out_type))
            # batch 输出：如果目标是 tensor 则堆叠，否则返回 list of results
            if (rt or input_type) == "tensor":
                try:
                    return torch.stack(results)
                except Exception:
                    return results
            return results
        else:
            input_type = self._detect_input_type(x)
            seq = self._to_list(x)
            y_val, y_idx = self._core(seq)
            out = y_idx if ri else y_val
            out_type = rt if rt is not None else input_type
            return self._convert_output(out, out_type)


# ============ 测试 ============ #

if __name__ == "__main__":
    rosa = ROSA()

    # ========== 测试输入变量 ==========
    str_input = "abcabc"
    str_sentence = "the cat sat on the mat"
    list_input = [1, 2, 3, 1, 2]
    tensor_input = torch.tensor([1, 2, 3, 1, 2])
    batch_list = [[1, 2, 3, 1], [4, 5, 4, 5]]
    batch_tensor = torch.tensor([[1, 2, 3, 1], [4, 5, 4, 5]])
    batch_str = ["hello world", "abcabc", "the cat sat on the mat"]

    # ========== 自适应输入输出类型 ==========
    print("=== 自适应输入输出类型 ===")
    print(f"  str  输入: {str_input!r}")
    print(f"  str  输出: {rosa(str_input)!r}")
    print()
    print(f"  list 输入: {list_input}")
    print(f"  list 输出: {rosa(list_input)}")
    print()
    print(f"  Tensor 输入: {tensor_input}")
    print(f"  Tensor 输出: {rosa(tensor_input)}")
    print()

    # ========== 批数据 ==========
    print("=== 批数据 ===")
    print(f"  二维list 输入: {batch_list}")
    print(f"  二维list 输出: {rosa(batch_list)}")
    print()
    print(f"  2D Tensor 输入: {batch_tensor}")
    print(f"  2D Tensor 输出: {rosa(batch_tensor)}")
    print()
    print(f"  批量str 输入: {batch_str}")
    print(f"  批量str 输出: {rosa(batch_str)}")
    print()

    # ========== 返回索引 vs 返回值 ==========
    print("=== 返回索引 vs 返回值 ===")
    print(f"  输入: {list_input}")
    print(f"  返回值(默认): {rosa(list_input)}")
    print(f"  返回索引:     {rosa(list_input, return_index=True)}")
    print()

    # ========== 强制指定输出类型 ==========
    print("=== 强制指定输出类型 ===")
    print(f"  输入: {list_input}")
    print(f"  return_type='tensor': {rosa(list_input, return_type='tensor')}")
    print(f"  return_type='str':    {rosa(list_input, return_type='str')!r}")
    print()
    print(f"  输入: {str_sentence!r}")
    print(f"  return_type='list':   {rosa(str_sentence, return_type='list')}")
    print()

    # ========== 字符串句子 ==========
    print("=== 字符串句子 ===")
    print(f"  输入: {str_sentence!r}")
    print(f"  返回值(str):      {rosa(str_sentence)!r}")
    print(f"  返回值(list):     {rosa(str_sentence, return_type='list')}")
    print(f"  返回索引(list):   {rosa(str_sentence, return_index=True, return_type='list')}")
    print()

    # ========== 批量字符串指定输出类型 ==========
    print("=== 批量字符串指定输出类型 ===")
    print(f"  输入: {batch_str}")
    print(f"  return_type='list': {rosa(batch_str, return_type='list')}")
    print()

    # ========== 实例级别默认配置 ==========
    print("=== 实例级别默认配置 ===")
    rosa2 = ROSA(return_index=True, return_type="tensor")
    print("  ROSA(return_index=True, return_type='tensor')")
    print(f"  输入: {list_input}")
    print(f"  输出: {rosa2(list_input)}")