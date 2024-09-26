#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include "lib.cpp"

#include <iostream>
#include <memory>

#define DEBUG 0

int main() {
    std::string modelPath = "../model/ChordPredictor.ts";

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(modelPath);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    float sampleRate = 44100.0;
    torch::Tensor chromaVector = torch::zeros({6, 12});
    int fft_size = 512;
    // Create a random frequency output
    for(size_t i=0;i<6;i++){
        auto fft_result = torch::rand({fft_size}); // Assume some random FFT
        auto chroma = calculate_chroma_spectrum(fft_result, sampleRate);
        chromaVector[i] = chroma;
        #if DEBUG
            std::cout << "Chroma: " << chroma << '\n';
        #endif
    }
    chromaVector = torch::stack({at::unsqueeze(chromaVector.transpose(0, 1), 0), at::unsqueeze(chromaVector.transpose(0, 1), 0)});
    #if DEBUG
        std::cout << "Chroma Shape: " << chromaVector.sizes() << '\n';
    #endif

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(chromaVector);
    std::cout << "Input Shape: " << inputs[0].toTensor().sizes() << '\n';

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor()[0];
    std::string actualKey = predictChord(output, CHORD_TEMPLATE);
    std::cout << "Output: " << actualKey << '\n';
}