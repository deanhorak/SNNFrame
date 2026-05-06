#include "snnfw/adapters/InterneuronAdapters.h"
#include "snnfw/Neuron.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using snnfw::Neuron;
using snnfw::adapters::BaseAdapter;
using snnfw::adapters::InterneuronReceiverAdapter;
using snnfw::adapters::InterneuronTransmitterAdapter;
using snnfw::adapters::SensoryAdapter;

namespace {

std::string getArg(int argc, char* argv[], const std::string& key, const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == key) return argv[i + 1];
    }
    return def;
}

int getArgInt(int argc, char* argv[], const std::string& key, int def) {
    return std::atoi(getArg(argc, argv, key, std::to_string(def)).c_str());
}

void runRx(int argc, char* argv[]) {
    BaseAdapter::Config cfg;
    cfg.type = "interneuron_rx";
    cfg.name = "rx_demo";
    cfg.temporalWindow = 10.0;
    cfg.stringParams["bind_host"] = getArg(argc, argv, "--host", "0.0.0.0");
    cfg.intParams["bind_port"] = getArgInt(argc, argv, "--port", 5000);
    cfg.intParams["neuron_count"] = getArgInt(argc, argv, "--channels", 16);
    cfg.intParams["receive_timeout_ms"] = 5;

    InterneuronReceiverAdapter rx(cfg);
    if (!rx.initialize()) {
        std::cerr << "RX init failed\n";
        std::exit(1);
    }

    const int durationMs = getArgInt(argc, argv, "--duration-ms", 4000);
    const int pollMs = getArgInt(argc, argv, "--poll-ms", 10);
    const auto t0 = std::chrono::steady_clock::now();
    int frames = 0;

    while (true) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - t0)
                                 .count();
        if (elapsed > durationMs) break;

        SensoryAdapter::DataSample sample;
        auto pattern = rx.processData(sample);
        std::vector<int> active;
        for (size_t i = 0; i < pattern.spikeTimes.size(); ++i) {
            if (!pattern.spikeTimes[i].empty()) active.push_back(static_cast<int>(i));
        }
        if (!active.empty()) {
            ++frames;
            std::cout << "RX frame " << frames << " active=";
            for (size_t i = 0; i < active.size(); ++i) {
                if (i) std::cout << ",";
                std::cout << active[i];
            }
            std::cout << "\n";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(pollMs));
    }

    std::cout << "RX done frames=" << frames << "\n";
    rx.shutdown();
}

void runTx(int argc, char* argv[]) {
    const int channels = getArgInt(argc, argv, "--channels", 16);
    const int frames = getArgInt(argc, argv, "--frames", 20);
    const int intervalMs = getArgInt(argc, argv, "--interval-ms", 10);

    BaseAdapter::Config cfg;
    cfg.type = "interneuron_tx";
    cfg.name = "tx_demo";
    cfg.temporalWindow = 10.0;
    cfg.stringParams["remote_host"] = getArg(argc, argv, "--host", "127.0.0.1");
    cfg.intParams["remote_port"] = getArgInt(argc, argv, "--port", 5000);
    cfg.doubleParams["update_interval_ms"] = static_cast<double>(intervalMs);
    cfg.intParams["transmit_neuron_ids"] = 0;

    InterneuronTransmitterAdapter tx(cfg);
    if (!tx.initialize()) {
        std::cerr << "TX init failed\n";
        std::exit(1);
    }

    std::vector<std::shared_ptr<Neuron>> neurons;
    neurons.reserve(static_cast<size_t>(channels));
    for (int i = 0; i < channels; ++i) {
        neurons.push_back(std::make_shared<Neuron>(200.0, 0.7, 1, static_cast<uint64_t>(i)));
    }

    double currentMs = 0.0;
    for (int f = 0; f < frames; ++f) {
        currentMs += intervalMs;
        const int a = f % channels;
        const int b = (f + 3) % channels;
        neurons[static_cast<size_t>(a)]->insertSpike(currentMs - 1.0);
        neurons[static_cast<size_t>(b)]->insertSpike(currentMs - 1.0);
        tx.processNeurons(neurons, currentMs);
        std::cout << "TX frame " << (f + 1) << " active=" << a << "," << b << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(intervalMs));
    }

    tx.shutdown();
}

} // namespace

int main(int argc, char* argv[]) {
    const std::string mode = getArg(argc, argv, "--mode", "rx");
    if (mode == "rx") {
        runRx(argc, argv);
        return 0;
    }
    if (mode == "tx") {
        runTx(argc, argv);
        return 0;
    }

    std::cerr << "Usage: --mode rx|tx [--host H] [--port P] [--channels N]"
                 " [--duration-ms M | --frames N --interval-ms M]\n";
    return 2;
}
