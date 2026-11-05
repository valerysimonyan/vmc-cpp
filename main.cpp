#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include "lib/config.h"
#include "lib/wavefunction.h"
#include "lib/train.h"
#include "lib/checkpoint.h"
#include "lib/constants.h"
#include "lib/network.h"

// ./main [run.ini]  -- the run file is optional; every setting it leaves out keeps its default (constants.h)
int main(int argc, char** argv) {
    if (argc > 2) { std::cerr << "usage: main [run.ini]\n"; return 2; }
    const RunConfig cfg = load_run_config(argc == 2 ? argv[1] : nullptr);
    {
        std::ofstream out("run_config.ini");   // the resolved settings, so this run directory records how it was made
        write_run_config(out, cfg);
    }
    std::cout << "=== Run settings" << (argc == 2 ? std::string(" (") + argv[1] + ")" : std::string(" (defaults)")) << " ===\n";
    write_run_config(std::cout, cfg);

    Ansatz ansatz(cfg.h_hidden, cfg.rho_hidden, cfg.orb_hidden, Activation::Gelu);
    validate_config(ansatz, cfg);

    CheckpointState start;
    if (!cfg.resume.empty()) {
        start = load_checkpoint(cfg.resume, ansatz);
        std::cout << "Resumed from " << cfg.resume << " at iteration " << start.iter << " (lambda " << start.lambda << ")\n";
    } else if (!cfg.transfer.empty()) {
        load_transfer(cfg.transfer, ansatz);
    }

    DescentResult result = train(ansatz, cfg, start);

    std::cout << "Final E_loc: " << result.El_exp << std::endl;
    std::cout << "Final error: " << result.El_err << std::endl;
    std::cout << "Final variance: " << result.var << std::endl;
    std::cout << "Final acceptance: " << result.acceptance << std::endl;

    // Evaluate the final parameters (the "best" checkpoint tends to be a lucky early iteration)
    DescentResult eval = evaluate_frozen(ansatz, cfg);

    std::cout << "Frozen-eval E: " << eval.El_exp << " +/- " << eval.El_err
              << ", var: " << eval.var << ", r_rms: " << eval.r_rms << std::endl;

    return 0;
}
     
