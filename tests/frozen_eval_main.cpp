// Frozen-parameter evaluation of a checkpoint, and nothing else. Builds in both
// configure modes (evaluate_frozen picks the device or CPU path itself), so the
// SAME checkpoint can be evaluated by both samplers for a cross-sampler check.
#include "../lib/checkpoint.h"
#include "../lib/descent.h"
#include "../lib/physics.h"
#include <cstdio>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: frozen_eval <checkpoint>\n"; return 2; }
    Ansatz a({64}, {64}, {64}, Activation::Gelu);
    load_checkpoint(argv[1], a);
    const DescentResult r = evaluate_frozen(a);
    std::printf("FROZEN %s E %.6f err %.6f var %.4f r_rms %.5f L2 %.5f node_hits %.0f\n",
#ifdef VMC_CUDA
                "gpu",
#else
                "cpu",
#endif
                r.El_exp, r.El_err, r.var, r.r_rms, r.L2, r.total_node_hits);
    return 0;
}
