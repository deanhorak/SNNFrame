#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Cluster.h"
#include "snnfw/Neuron.h"
#include <iostream>
#include <memory>
#include <map>

int main() {
    std::cout << "=== SNNFW Cluster Example ===" << std::endl;

    // Create factory for object creation
    snnfw::NeuralObjectFactory factory;

    // Create a cluster
    auto cluster = factory.createCluster();
    std::cout << "\nCreated cluster with ID: " << cluster->getId() << std::endl;

    // Create some neurons and store them in a map for later access
    std::map<uint64_t, std::shared_ptr<snnfw::Neuron>> neuronMap;

    auto neuron1 = factory.createNeuron(50.0, 0.95, 20);
    auto neuron2 = factory.createNeuron(50.0, 0.95, 20);
    auto neuron3 = factory.createNeuron(50.0, 0.95, 20);

    neuronMap[neuron1->getId()] = neuron1;
    neuronMap[neuron2->getId()] = neuron2;
    neuronMap[neuron3->getId()] = neuron3;

    std::cout << "\nCreated 3 neurons with IDs: "
              << neuron1->getId() << ", "
              << neuron2->getId() << ", "
              << neuron3->getId() << std::endl;

    // Add neurons to cluster by ID
    cluster->addNeuron(neuron1->getId());
    cluster->addNeuron(neuron2->getId());
    cluster->addNeuron(neuron3->getId());

    std::cout << "\nAdded neuron IDs to cluster" << std::endl;

    // Print cluster info
    std::cout << "\n=== Cluster Information ===" << std::endl;
    cluster->printInfo();

    // Test neuron functionality by looking up from cluster
    std::cout << "\n=== Testing Neuron Functionality ===" << std::endl;
    uint64_t testNeuronId = cluster->getNeuronId(0);
    if (testNeuronId != 0 && neuronMap.count(testNeuronId)) {
        auto testNeuron = neuronMap[testNeuronId];
        std::cout << "Testing neuron with ID: " << testNeuron->getId() << std::endl;
        testNeuron->insertSpike(10.0);
        testNeuron->insertSpike(20.0);
        testNeuron->insertSpike(30.0);
        testNeuron->learnCurrentPattern();

        std::cout << "\nNeuron spikes:" << std::endl;
        testNeuron->printSpikes();
    }

    // Test cluster size
    std::cout << "\n=== Cluster Size ===" << std::endl;
    std::cout << "Cluster contains " << cluster->size() << " neuron IDs" << std::endl;

    // Remove a neuron by ID
    std::cout << "\n=== Removing Neuron ===" << std::endl;
    if (cluster->removeNeuron(neuron2->getId())) {
        std::cout << "Successfully removed neuron with ID " << neuron2->getId() << std::endl;
        std::cout << "Cluster now contains " << cluster->size() << " neuron IDs" << std::endl;
        cluster->printInfo();
    }

    return 0;
}

