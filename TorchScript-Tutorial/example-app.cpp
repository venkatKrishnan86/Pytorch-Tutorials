#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

#include <iostream>
#include <memory>

int main() {
    std::string args = "../python/model/linear_regressor.ts";

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(args);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1}) * 0.43);
    std::cout << "Input: "<< inputs << '\n';

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << "Output: " << output << '\n';

    at::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    std::cout << "ok\n";

    at::Tensor new_tensor = torch::randn({2, 1, tensor.sizes()[1]});
    std::cout << new_tensor << std::endl;
    std::cout << new_tensor[0] << std::endl;
    std::cout << "ok\n";

    at::Tensor probabilities = torch::nn::functional::softmax(torch::randn({2, 3}), torch::nn::functional::SoftmaxFuncOptions(1));
    std::cout << probabilities << std::endl;
    std::cout << probabilities[0].argmax(0) << std::endl;
    std::cout << "ok\n";
}