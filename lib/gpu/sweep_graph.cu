#include "sweep_graph.h"
#include "sampler_kernels.h"
#include "eval.h"
#include "prof.h"

#include <cstdio>
#include <stdexcept>
#include <string>

static constexpr std::size_t blas_ws_bytes = 32u << 20;   // cuBLAS's recommended graph workspace

SweepGraphs::~SweepGraphs() {
    if (coord) cudaGraphExecDestroy(coord);
    if (spin)  cudaGraphExecDestroy(spin);
    if (tau)   cudaGraphExecDestroy(tau);
    if (stream) cudaStreamDestroy(stream);
}

static void ensure_resources(SweepGraphs& g) {
    if (!g.stream) CUDA_CHECK(cudaStreamCreate(&g.stream));
    if (g.blas_ws.n == 0) g.blas_ws.alloc(blas_ws_bytes);
}

// Capture whatever `body` enqueues on g.stream into a new graph.
template <typename Body>
static cudaGraph_t capture(SweepGraphs& g, cublasHandle_t handle, const char* what, Body&& body, std::size_t& n_nodes) {
    // Bind stream and workspace now, so nothing inside capture changes the handle.
    if (cublasSetStream(handle, g.stream) != CUBLAS_STATUS_SUCCESS ||
        cublasSetWorkspace(handle, g.blas_ws.d, g.blas_ws.bytes()) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("sweep graph: binding cuBLAS stream/workspace failed before capturing ") + what);

    graph_capturing() = true;
    CUDA_CHECK(cudaStreamBeginCapture(g.stream, cudaStreamCaptureModeGlobal));
    cudaGraph_t graph = nullptr;
    try {
        body(g.stream);
    } catch (...) {
        cudaStreamEndCapture(g.stream, &graph);      // abandon the half-built graph
        if (graph) cudaGraphDestroy(graph);
        cudaGetLastError();
        graph_capturing() = false;
        throw;
    }
    const cudaError_t e = cudaStreamEndCapture(g.stream, &graph);
    graph_capturing() = false;
    if (e != cudaSuccess || !graph) {
        throw std::runtime_error(std::string("sweep graph: capture of ") + what + " failed: " + cudaGetErrorString(e)
                                 + " -- something inside the stage synchronised, allocated or used the legacy stream");
    }
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &n_nodes));
    g.n_captures++;
    return graph;
}

// Put `graph` into `exec`: update in place when the topology allows it, else
// (first time, or B changed the node count) instantiate afresh.
static void install(SweepGraphs& g, cudaGraphExec_t& exec, cudaGraph_t graph) {
    if (exec) {
        cudaGraphExecUpdateResultInfo info{};
        if (cudaGraphExecUpdate(exec, graph, &info) == cudaSuccess) {
            g.n_updates++;
            CUDA_CHECK(cudaGraphDestroy(graph));
            return;
        }
        cudaGetLastError();                          // clear the failed update's error
        CUDA_CHECK(cudaGraphExecDestroy(exec));
        exec = nullptr;
    }
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
}

static void launch(SweepGraphs& g, cudaGraphExec_t exec) {
    CUDA_CHECK(cudaGraphLaunch(exec, g.stream));
    // One API launch, however many kernels the graph holds: the launches column
    // counts what the host issues, which is what graphs reduce.
    if constexpr (prof_enabled) launch_counter().fetch_add(1, std::memory_order_relaxed);
}

void sweep_device_graph(DeviceState& ds, cublasHandle_t handle, int B, double step) {
    if (B <= 0) return;
    SweepGraphs& g = ds.graphs;
    ensure_resources(g);

    // Coordinate draw: propose -> eval -> accept. Depends on B and step.
    if (!g.coord || g.coord_B != B || g.coord_step != step) {
        cudaGraph_t gr = capture(g, handle, "coordinate draw", [&](cudaStream_t s) {
            propose_coord(ds, B, step, s);
            eval_logp_batch_prop(ds, handle, B, ds.x_prop.d, ds.S_prop.d, ds.logp_prop.d, s);
            accept_coord(ds, B, s);
        }, g.coord_nodes);
        install(g, g.coord, gr);
        g.coord_B = B;
        g.coord_step = step;
    }
    {
        VMC_PROF("coord_draws", 0);
        for (int j = 0; j < draws; j++) launch(g, g.coord);
    }

    const bool do_spin = (spin_mode == SpinMode::Sampled && N_u > 0 && N_d > 0);
    const bool do_tau  = (tau_mode  == TauMode::Sampled  && N_p > 0 && N_n > 0);
    if (!(do_spin || do_tau)) return;

    // The (s,t) table is built once per sweep: nothing to gain from a graph.
    {
        VMC_PROF("st_table", 0);
        build_st_table_batch(ds, handle, B, 0);
        S_from_table_batch(ds, handle, B, ds.s.d, ds.t.d, ds.S_cur.d, 0);
    }

    // Discrete rounds depend on B only.
    if (g.disc_B != B) {
        if (do_spin) {
            cudaGraph_t gr = capture(g, handle, "spin round", [&](cudaStream_t s) {
                propose_discrete(ds, B, true, s);
                S_from_table_batch(ds, handle, B, ds.s_prop.d, ds.t.d, ds.S_prop.d, s);
                accept_discrete(ds, B, true, s);
            }, g.spin_nodes);
            install(g, g.spin, gr);
        }
        if (do_tau) {
            cudaGraph_t gr = capture(g, handle, "isospin round", [&](cudaStream_t s) {
                propose_discrete(ds, B, false, s);
                S_from_table_batch(ds, handle, B, ds.s.d, ds.t_prop.d, ds.S_prop.d, s);
                accept_discrete(ds, B, false, s);
            }, g.tau_nodes);
            install(g, g.tau, gr);
        }
        g.disc_B = B;
    }
    VMC_PROF("discrete_block", 0);
    if (do_spin) for (int j = 0; j < spin_draws; j++) launch(g, g.spin);
    if (do_tau)  for (int j = 0; j < tau_draws;  j++) launch(g, g.tau);
}
