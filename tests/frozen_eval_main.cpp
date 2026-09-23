// Frozen-parameter evaluation of a checkpoint, and nothing else. Builds in both
// configure modes (evaluate_frozen picks the device or CPU path itself), so the
// SAME checkpoint can be evaluated by both samplers for a cross-sampler check.
#include "../lib/checkpoint.h"
#include "../lib/descent.h"
#include "../lib/physics.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Hidden widths of one network from a checkpoint header "name L n0 n1 ... nL",
// so any architecture can be evaluated without rebuilding.
static std::vector<int> ckpt_hidden(const char* path, const char* name) {
    std::FILE* f = std::fopen(path, "r");
    if (!f) throw std::runtime_error(std::string("cannot open checkpoint ") + path);
    char line[4096];
    std::vector<int> hidden;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, name, std::strlen(name)) != 0 || line[std::strlen(name)] != ' ') continue;
        std::vector<int> v;
        for (char* tok = std::strtok(line + std::strlen(name), " \n"); tok; tok = std::strtok(nullptr, " \n")) v.push_back(std::atoi(tok));
        for (std::size_t i = 2; i + 1 < v.size(); i++) hidden.push_back(v[i]);
        break;
    }
    std::fclose(f);
    if (hidden.empty()) throw std::runtime_error(std::string("no ") + name + " header in " + path);
    return hidden;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: frozen_eval <checkpoint>\n"; return 2; }
    Ansatz a(ckpt_hidden(argv[1], "h_net"), ckpt_hidden(argv[1], "rho_net"), ckpt_hidden(argv[1], "orb_net"), Activation::Gelu);
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
