#pragma once

#include "util.h"

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <cstddef>


// Define set of activation functions
enum class Activation{Tanh, Gelu};

// Depending on activation function set response
template <typename T>
T apply_activation(Activation act, const T& x) {
    switch (act) {
        case Activation::Tanh:
            return tanh(x);

        case Activation::Gelu: {
            T x3 = x*x*x;
            T inner = x+T(0.044715) * x3;
            T scaled = T(0.7978845608) * inner;
            T t = tanh(scaled);
            return T(0.5) * x * (T(1.0) + t);
        }
    }
    assert(false);
    __builtin_unreachable();
}

// Exact gradients of activation functions
inline double act_grad(Activation act, double z) {
    switch (act) {
        case Activation::Tanh: {
            double t = std::tanh(z);
            return 1.0 - t * t;
        }
        case Activation::Gelu: {
            double s = 0.7978845608;
            double k = 0.044715;
            double g = s * (z + k * z * z * z);
            double gp = s * (1.0 + 3.0 * k * z * z);
            double t = std::tanh(g);
            return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * gp;
        }
    }
    assert(false);
    __builtin_unreachable();
}


struct ForwardCache {
    std::vector<double> a;
    std::vector<double> z;
};


// Define layer structure and indexing aids
struct Layer {
    int input_size, output_size;
    int weight_offset, bias_offset;
};

// Define network structure
struct Network {
    std::vector<Layer> layers;
    std::vector<double> params;
    Activation activation;

    std::vector<int> a_offset;
    std::vector<int> z_offset;
    int a_tot_size, z_tot_size;
    std::size_t max_width;

    // Automatically initialize network by tracking proper indexing and then populating the network with random numbers between [-1,1], everything between : and { is run first initializing object
    Network(int input_width, std::vector<int> hidden_widths, int output_width, Activation act): activation(act) {
        // Build network architecture
        std::vector<int> sizes;
        
        sizes.push_back(input_width);
        for (int h: hidden_widths) sizes.push_back(h);
        sizes.push_back(output_width);

        int tot_size = 0;
        for (std::size_t i = 0; i+1<sizes.size(); i++) {
            int in_size = sizes[i];
            int out_size = sizes[i+1];

            Layer layer;
            layer.input_size = in_size;
            layer.output_size = out_size;
            layer.weight_offset = tot_size;
            tot_size += in_size*out_size;
            layer.bias_offset = tot_size;
            tot_size += out_size;

            layers.push_back(layer);
        }

        // Storage for derivative, create and populate
        a_offset.resize(layers.size()+1);
        z_offset.resize(layers.size());

        int a_off = 0, z_off = 0;
        max_width = (std::size_t)layers[0].input_size;
        for (std::size_t l = 0; l < layers.size(); l++) {
            a_offset[l] = a_off;
            a_off += layers[l].input_size;
            z_offset[l] = z_off;
            z_off += layers[l].output_size;
            max_width = std::max(max_width, (std::size_t)layers[l].output_size);
        }
        a_offset[layers.size()] = a_off;
        a_tot_size = a_off + layers.back().output_size;
        z_tot_size = z_off;
    
        // Populate parameters of network with random numbers between [-1,1]
        params = std::vector<double>(tot_size);
        for (int i = 0; i < tot_size; i++) {
            params[i] = gen_uniform_sample(-1,1);
        }
    }
 
    // Returns weight of edge connecting node i in layer l to node j in layer l+1, & makes direct reference to value without making copy
    double& weight(int l, int i, int j) {
        return params[layers[l].weight_offset+layers[l].input_size*i+j]; 
    }

    // Returns bias of node i in layer l
    double& bias(int l, int i) {
        return params[layers[l].bias_offset + i];
    }

    // Forward pass using "ping pong" to update params
    template<typename T>
    std::vector<T>* forward_opt (const std::vector<T>& input, const std::vector<double>& param_values, std::vector<T>& buf_a, std::vector<T>& buf_b) const {
        // Find the widest layer, set the buffers to be the widest posible size
        if (buf_a.size() != max_width) buf_a.resize(max_width);
        if (buf_b.size() != max_width) buf_b.resize(max_width);

        // Assign an input to first input.size() elements of buf_a
        for (std::size_t i = 0; i < input.size(); i++) {
            buf_a[i] = input[i];
        }

        // Make pointers to elements of buf_a and buf_b
        std::vector<T>* cur = &buf_a;
        std::vector<T>* nxt = &buf_b;

        // Begin with input layer, iterate through layers
        for (std::size_t l = 0; l < layers.size(); l++) {
            int in_size = layers[l].input_size;
            int out_size = layers[l].output_size;
            bool is_out = (l+1 == layers.size());

            // Evaluate neurons
            for (int i = 0; i < out_size; i++) {
                T sum = T(param_values[layers[l].bias_offset+i]);
                for (int j = 0; j < in_size; j++) {
                    sum = sum + param_values[layers[l].weight_offset+in_size*i+j] * (*cur)[j];
                }
                // If not output layer add to array activation_function(sum), else add (sum)
                (*nxt)[i] = is_out ? sum : apply_activation(activation, sum);
            }
            // Swap values of cur and nxt then rewrite over nxt using former values of cur
            std::swap(cur, nxt);
        }
        return cur;
    }

    // Use cache structure for forward pass
    void forward_cached(const std::vector<double>& input, ForwardCache& cache, std::vector<double>& output) const {
        if((int)cache.a.size() != a_tot_size) cache.a.resize(a_tot_size);
        if((int)cache.z.size() != z_tot_size) cache.z.resize(z_tot_size); 

        for (std::size_t i = 0; i < input.size(); i++) {
            cache.a[a_offset[0] + i] = input[i];
        }

        for (std::size_t l = 0; l < layers.size(); l++) {
            int in_size = layers[l].input_size;
            int out_size = layers[l].output_size;
            bool is_out = (l+1 == layers.size());

            for (int i = 0; i < out_size; i++) {
                double sum = params[layers[l].bias_offset+i];
                for (int j = 0; j < in_size; j++) {
                    sum += params[layers[l].weight_offset + in_size * i + j] * cache.a[a_offset[l]+j];
                }
                cache.a[a_offset[l + 1] + i] = is_out ? sum : apply_activation(activation, sum);
                cache.z[z_offset[l]+i] = sum;
            }
        }

        int out_size = layers.back().output_size;
        if ((int)output.size() != out_size) output.resize(out_size);
        for (int i = 0; i < out_size; i++) {
            output[i] = cache.a[a_offset[layers.size()]+i];
        }
    }


    // Backwards pass for network, if Accumulate is false we do normal backprop, if true (say you have one network with multiple seed), you preserve dtheta and add on top of it, template is so compiler treats it as two different functions w/ first line already resolved
    template<bool Accumulate>
    void backprop_impl(const ForwardCache& cache, const std::vector<double>& seed, std::vector<double>& dtheta, std::vector<double>& delta_a, std::vector<double>& delta_b, std::vector<double>* dinput = nullptr) const {
        // Zero for first call, otherwise set to parameter size
        if constexpr (!Accumulate) dtheta.assign(params.size(), 0.0);

        std::size_t max_width = std::max(seed.size(), (std::size_t)layers[0].input_size);
        for (auto& l : layers) {
            max_width = std::max(max_width, (std::size_t)l.output_size);
            max_width = std::max(max_width, (std::size_t)l.input_size);
        }
        if (delta_a.size() != max_width) delta_a.resize(max_width);
        if (delta_b.size() != max_width) delta_b.resize(max_width);

        for (std::size_t i = 0; i < seed.size(); i++) {
            delta_a[i] = seed[i];
        }

        // Make pointers to elements of delta_a and delta_b
        std::vector<double>* cur = &delta_a;
        std::vector<double>* nxt = &delta_b;

        int n_layers = (int)layers.size();
        // Start form outmost layer
        for (int l = n_layers-1; l >= 0; l--) {
            int in_size = layers[l].input_size;
            int out_size = layers[l].output_size;

            for (int i = 0; i < in_size; i++) {
                (*nxt)[i] = 0.0;
            }

            // Bias derivatives are just the derivative of the next node, weigth derivatives are rescaled by the weight parameter
            for (int i = 0; i < out_size; i++) {
                if constexpr (Accumulate) dtheta[layers[l].bias_offset+i] += (*cur)[i];
                else dtheta[layers[l].bias_offset+i] = (*cur)[i];
            
                for (int j = 0; j < in_size; j++) {
                    if constexpr (Accumulate) dtheta[layers[l].weight_offset + in_size * i + j] += (*cur)[i] * cache.a[a_offset[l] + j];
                    else dtheta[layers[l].weight_offset + in_size * i + j] = (*cur)[i] * cache.a[a_offset[l] + j];
                
                    (*nxt)[j] += params[layers[l].weight_offset + in_size * i + j] * (*cur)[i];
                }    
            }
            
            // Also include derivative of activation function
            if (l > 0) {
                for (int i = 0; i < in_size; i++) {
                    (*nxt)[i] *= act_grad(activation, cache.z[z_offset[l-1]+i]);
                }
            }
            std::swap(cur,nxt);
        }
        
        // As the output of xi is also a parameter we include this to track dtheta, have option for nullptr as not all outputs are parameters
        if (dinput != nullptr) {
            int in0 = layers[0].input_size;
            if((int)dinput -> size() != in0) dinput->resize(in0);
            for(int i = 0; i < in0; i++) {
                (*dinput)[i] = (*cur)[i];
            }
        }
    }
    
    void backprop(const ForwardCache& cache, const std::vector<double>& seed, std::vector<double>& dtheta, std::vector<double>& delta_a, std::vector<double>& delta_b, std::vector<double>* dinput = nullptr) const {
        backprop_impl<false>(cache, seed, dtheta, delta_a, delta_b, dinput);
    }

    // Same as backprop but adds into dtheta_accum instead of overwriting, so per-particle gradients from a shared-weight embedding network can be summed in place
    void backprop_acc(const ForwardCache& cache, const std::vector<double>& seed, std::vector<double>& dtheta, std::vector<double>& delta_a, std::vector<double>& delta_b, std::vector<double>* dinput = nullptr) const {
        backprop_impl<true>(cache, seed, dtheta, delta_a, delta_b, dinput);
    }

};
