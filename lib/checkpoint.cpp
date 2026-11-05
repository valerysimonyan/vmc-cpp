#include "checkpoint.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace{

// Write layers, their sizes, and parameters
void write_network(std::ostream& f, const char* name, const Network& net) {
    f << name << " " << net.layers.size() << " " << net.layers[0].input_size;
    for (const auto& l : net.layers) f << " " << l.output_size;
    f << "\n";
    f << std::setprecision(17);
    for (double p : net.params) f << p << '\n';
}

// Upload network into file
void read_network(std::istream& f, const char* name, Network& net, const std::string& path) {
    std::string tag;
    std::size_t n_layers = 0;

    // Throw error if read fails, wrong network is given, or there is a size mismatch in networks
    if (!(f >> tag >> n_layers)) throw std::runtime_error("load_checkpoint(" + path + "): failed reading " + name + " header");
    if (tag != name) throw std::runtime_error("load_checkpoint(" + path + "): expected network '" + std::string(name) + "', found '" + tag + "'");
    if (n_layers != net.layers.size()) throw std::runtime_error("load_checkpoint(" + path + "): " + name + " has " + std::to_string(n_layers) + " layers in file, expects " + std::to_string(net.layers.size()));
    
    // Throw error if any size mismatch
    int w = 0;
    f >> w; 
    if (w != net.layers[0].input_size) throw std::runtime_error("load_checkpoint(" + path + "): " + name + " input width mismatch (file=" + std::to_string(w) + ", expects=" + std::to_string(net.layers[0].input_size) + ")");
    for (std::size_t l = 0; l < net.layers.size(); l++) {
        f >> w;
        if (w != net.layers[l].output_size) throw std::runtime_error("load_checkpoint(" + path + "): " + name + " layer " + std::to_string(l) + " output width mismatch (file=" + std::to_string(w) + ", expects=" + std::to_string(net.layers[l].output_size) + ")");
    }
    for (double& p : net.params) f >> p;
}

// Read network partially going layer by layer until mismatch occurs
int read_network_partial(std::istream& f, const char* name, Network& net, const std::string& path) {
    std::string tag; 
    std::size_t n_layers_file = 0;

    // Throw error if read fails, or network mismatch
    if (!(f >> tag >> n_layers_file)) throw std::runtime_error("load_transfer(" + path + "): failed reading " + name + " header");
    if (tag != name) throw std::runtime_error("load_transfer(" + path + "): expected network '" + std::string(name) + "', found '" + tag + "'");
    
    // Read widths
    std::vector<int> widths_file(n_layers_file + 1);
    for (std::size_t i = 0; i <= n_layers_file; i++) f >> widths_file[i];

    // Check if sizes still matching
    bool still_matching = (n_layers_file > 0 && !net.layers.empty() && widths_file[0] == net.layers[0].input_size);
    int transferred = 0;

    // Go layer by layer checking if it matches, if matches write in old parameters, return number of transferred layers
    for (std::size_t l = 0; l < n_layers_file; l++) {
        int in_file = widths_file[l];
        int out_file = widths_file[l+1];
        std::size_t n_params_layer = (std::size_t)in_file * out_file + out_file;

        bool layer_matches = still_matching && l < net.layers.size() && net.layers[l].input_size == in_file && net.layers[l].output_size == out_file;

        if (layer_matches) {
            int w_off = net.layers[l].weight_offset, b_off = net.layers[l].bias_offset;
            for (int k = 0; k < in_file*out_file; k++) f >> net.params[w_off + k];
            for (int k = 0; k < out_file; k++) f >> net.params[b_off + k];
            transferred++;
        } else {
            still_matching = false;
            double discard;
            for (std::size_t k = 0; k < n_params_layer; k++) f >> discard;
        }
    }
    return transferred;    
}

}

// Read Jastrow parameters
static void read_jastrow(std::ifstream& f, Ansatz& a) {
    std::string tag; int n = 0;
    if (!(f >> tag) || tag != "jastrow") return;
    f >> n;
    for (int m = 0; m < n; m++) { 
        double c; 
        f >> c; 
        if (m < n_jas_par) a.jc[m] = c; 
    }
}

// Read header extracting state of training parameters, including iteration number, lambda, and best E_ucb
static CheckpointState read_header(std::ifstream& f, const std::string& path, const char* who) {
    std::string header;
    std::getline(f, header);
    CheckpointState st;
    // v1 has no header so return default state, v2 had header with training state
    if (header == "CHECKPOINT v1") return st; 
    if (header != "CHECKPOINT v2") throw std::runtime_error(std::string(who) + "(" + path + "): unrecognized header '" + header + "'");
    // String holding training params
    std::string tag, k_iter, k_lambda, k_best, v_lambda, v_best;
    // If header format is wrong, throw error
    if (!(f >> tag >> k_iter >> st.iter >> k_lambda >> v_lambda >> k_best >> v_best) || tag != "state" || k_iter != "iter" || k_lambda != "lambda" || k_best != "best_E_ucb") throw std::runtime_error(std::string(who) + "(" + path + "): malformed state line");
    
    // Write to state then return
    st.lambda = std::stod(v_lambda);
    st.best_E_ucb = std::stod(v_best);
    return st;
}

// Save parameters to file for full psi
void save_checkpoint(const std::string& path, const Ansatz& a, const CheckpointState& st) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("save_checkpoint: cannot open " + path);
    f << "CHECKPOINT v2\n";
    f << std::setprecision(17) << "state iter " << st.iter << " lambda " << st.lambda << " best_E_ucb " << st.best_E_ucb << "\n";

    write_network(f, "h_net", a.h_net);
    write_network(f, "rho_net", a.rho_net);
    write_network(f, "orb_net", a.orb_net);
    
    f << std::setprecision(17) << "alpha " << a.alpha << "\n";

    f << "jastrow " << n_jas_par;
    for (int m = 0; m < n_jas_par; m++) f << " " << a.jc[m];
    f << "\n";

}

// Load parameters to file for full psi
CheckpointState load_checkpoint(const std::string& path, Ansatz& a) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("load_checkpoint: cannot open " + path);
    const CheckpointState st = read_header(f, path, "load_checkpoint");

    read_network(f, "h_net", a.h_net, path);
    read_network(f, "rho_net", a.rho_net, path);
    read_network(f, "orb_net", a.orb_net, path);
    
    std::string tag;
    f >> tag >> a.alpha;
    if (tag != "alpha") throw std::runtime_error("load_checkpoint(" + path + "): expected 'alpha', found '" + tag + "'");
    read_jastrow(f, a);
    return st;
}

// Load transfer code
void load_transfer(const std::string& path, Ansatz& a) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("load_transfer: cannot open " + path);
    read_header(f, path, "load_transfer");   // a transfer starts a new run: the state is not used

    // Widths of these networks outputs do not depend on particle number by construction
    read_network(f, "h_net", a.h_net, path);
    read_network(f, "rho_net", a.rho_net, path);
    
    // Widths of orbs do, only transfer matching layers
    int n_transferred = read_network_partial(f, "orb_net", a.orb_net, path);
    std::cout << "orb_net: " << n_transferred << "/" << a.orb_net.layers.size() << " layers transferred";
    if ((std::size_t)n_transferred < a.orb_net.layers.size()) std::cout << ", " << (a.orb_net.layers.size() - n_transferred) << " layer(s) reinitialized";
    std::cout << "\n";

    std::string tag;
    f >> tag >> a.alpha;
    if (tag != "alpha") throw std::runtime_error("load_transfer(" + path + "): expected 'alpha', found '" + tag + "'");
    read_jastrow(f, a);
}