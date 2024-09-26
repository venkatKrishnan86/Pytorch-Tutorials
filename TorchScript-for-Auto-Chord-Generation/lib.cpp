#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

#include<iostream>
#include <unordered_map> 

const std::unordered_map<std::string, std::vector<int>> CHORD_TEMPLATE = {
    {"A", {0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}},
    {"A#", {0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0}},
    {"B", {0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1}},
    {"C", {1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0}},
    {"C#", {0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}},
    {"D", {0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0}},
    {"D#", {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0}},
    {"E", {0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1}},
    {"F", {1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0}},
    {"F#", {0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0}},
    {"G", {0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1}},
    {"G#", {1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0}},
    {"Am", {1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}},
    {"A#m", {0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}},
    {"Bm", {0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1}},
    {"Cm", {1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0}},
    {"C#m", {0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}},
    {"Dm", {0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0}},
    {"D#m", {0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0}},
    {"Em", {0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1}},
    {"Fm", {1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}},
    {"F#m", {0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0}},
    {"Gm", {0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0}},
    {"G#m", {0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1}},
    {"Adim", {1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0}},
    {"A#dim", {0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0}},
    {"Bdim", {0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1}},
    {"Cdim", {1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0}},
    {"C#dim", {0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0}},
    {"Ddim", {0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0}},
    {"D#dim", {0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0}},
    {"Edim", {0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0}},
    {"Fdim", {0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1}},
    {"F#dim", {1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0}},
    {"Gdim", {0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0}},
    {"G#dim", {0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1}},
    {"Aaug", {0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0}},
    {"A#aug", {0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0}},
    {"Baug", {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}},
    {"Caug", {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}},
    {"C#aug", {0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0}},
    {"Daug", {0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0}},
    {"D#aug", {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}},
    {"Eaug", {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}},
    {"Faug", {0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0}},
    {"F#aug", {0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0}},
    {"Gaug", {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}},
    {"G#aug", {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}},
};

// Function to convert frequency to pitch class
int frequency_to_pitch_class(float frequency) {
    if (frequency <= 0) return -1; // Invalid frequency
    int pitch_class = static_cast<int>(std::round(12 * std::log2(frequency / 440.0) + 69)) % 12;
    return pitch_class;
}

// Function to predict the chord given the output from the model. Expected shape: (12, )
std::string predictChord(at::Tensor output, std::unordered_map<std::string, std::vector<int>> chordTemplate)
{
    std::unordered_map<std::string, std::vector<int>>::iterator it;
    std::string actualKey = "";
    at::Tensor minDistance = torch::tensor(10000.0); 
    for (it = chordTemplate.begin(); it != chordTemplate.end(); it++) {
        torch::Tensor chord = torch::tensor(it->second, {torch::kFloat64});
        at::Tensor distance = torch::norm(chord - output);
        if(minDistance.item<double>() > distance.item<double>()) {
            minDistance = distance;
            actualKey = it->first;
        }
    }
    return actualKey;
}

// Function to calculate the chroma spectrum
torch::Tensor calculate_chroma_spectrum(const torch::Tensor& fft_output, float sample_rate) {
    // Assume fft_output is a 1D tensor with the magnitudes of the FFT
    int n_bins = 12;
    auto chroma = torch::zeros({n_bins});

    // Calculate the frequencies corresponding to each bin in the FFT output
    int fft_size = fft_output.size(0);
    float freq_bin = sample_rate / fft_size;

    for (int i = 0; i < fft_size / 2; ++i) { // Only need up to Nyquist frequency
        float frequency = i * freq_bin;
        int pitch_class = frequency_to_pitch_class(frequency);
        if (pitch_class >= 0) {
            chroma[pitch_class] += fft_output[i];
        }
    }

    return chroma;
}