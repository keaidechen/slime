// Host-only layout operations; needs CUTLASS headers and the CUDA development toolkit.
// No kernel is launched. See chapter 07 for the include path and build command.
#include <cute/tensor.hpp>
#include <cassert>
#include <cstdio>

int main() {
    using namespace cute;
    float data[6] = {10, 11, 12, 20, 21, 22};
    auto row_layout = make_layout(make_shape(Int<2>{}, Int<3>{}), make_stride(Int<3>{}, Int<1>{}));
    auto view = make_tensor(data, row_layout);
    auto transpose_layout = make_layout(make_shape(Int<3>{}, Int<2>{}), make_stride(Int<1>{}, Int<3>{}));
    auto transposed = make_tensor(data, transpose_layout);
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 3; ++c) {
            int offset = row_layout(make_coord(r, c));
            assert(offset == r * 3 + c);
            assert(view(r, c) == transposed(c, r));
            std::printf("(%d,%d) -> offset %d -> %.0f\n", r, c, offset, view(r, c));
        }
    }
    transposed(2, 1) = 99;
    assert(data[5] == 99 && view(1, 2) == 99);
    std::puts("PASS CuTe: changing a view changes the same storage; no transpose copy was made");
}
