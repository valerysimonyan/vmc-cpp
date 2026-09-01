#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include "lib/monte_carlo.h"
#include "lib/physics.h"
#include "lib/descent.h"
#include "lib/constants.h"
#include "lib/network.h"



int main() {
    Ansatz ansatz({64}, {64}, {64}, Activation::Gelu);
    validate_config(ansatz);

    if (resume) {
        std::ifstream probe("best_checkpoint.txt");
        if (probe.good()) {
            probe.close();
            load_checkpoint("best_checkpoint.txt", ansatz);
            std::cout << "Resumed from best_checkpoint.txt\n";
        } else {
            std::cout << "resume=true but best_checkpoint.txt not found -- starting fresh.\n";
        }
    } else if (std::string(transfer_from) != "") {
        load_transfer(transfer_from, ansatz);
    }

    DescentResult result = descent(ansatz);
    std::cout << "Final E_loc: " << result.El_exp << std::endl;
    std::cout << "Final error: " << result.El_err << std::endl; 
    std::cout << "Final variance: " << result.var << std::endl; 
    std::cout << "Final acceptance: " << result.acceptance << std::endl;

    load_checkpoint("best_checkpoint.txt", ansatz);
    DescentResult eval = evaluate_frozen(ansatz);
    std::cout << "Frozen-eval E: " << eval.El_exp << " +/- " << eval.El_err
              << ", var: " << eval.var << ", r_rms: " << eval.r_rms << std::endl;

    return 0;
}
     
